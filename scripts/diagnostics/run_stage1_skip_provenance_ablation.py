#!/usr/bin/env python3
"""
scripts/diagnostics/run_stage1_skip_provenance_ablation.py

Phase 6 Diagnostic: Stage-1 Skip Provenance Ablation (Zero-Training)
Target Cohort: Exactly the 43 wider-gap bridge events (D_gap > 5.0 px) from Phase 6 Diagnostic D.
Target Model: Candidate B Baseline (B2ConvNeXtViTUNet, Setting A, Tile 448).

Stratification:
- Group A: 5.0 < D_gap <= 8.0 px (N = 11, negative control for stride-8 resolution limit)
- Group B: D_gap > 8.0 px (N = 32, primary causal provenance cohort)

Probes:
- Probe A0: Stage-1 SAGE-off Counterfactual (DIRECT CAUSAL TEST)
    F_post = F_pre + 0.1E  -->  F_SAGE-off = F_pre
    Evaluated for:
      (1) Stage-1 Skip Only (skips[1] = F_pre, rest of encoder/bottleneck unperturbed)
      (2) Stage-1 Global (both skip and downstream encoder unperturbed)
- Probe A: Pre- vs Post-SAGE Representation Contrast
    Measure Cosine Sim(Corr, Crack) and Sim(Corr, BG) on F_pre and F_post
- Probe B: Stage-1 Skip Activation Patching
    B0: Baseline S1
    B1: Local background replacement
    B2: Matched true-crack segment replacement (aligned from same image)
    B3: Spatial shuffle of corridor feature cells
- Probe C: Endpoint-Preserving Interior Replacement
    Keep crack endpoints CA and CB 100% unchanged; patch strictly interior gap with local background.
"""

import collections
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Ensure SAGE_LITE and project root are on sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sage_lite_dir = os.path.join(project_root, "SAGE_LITE")
for p in [project_root, sage_lite_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from tools.run_phase6_c_topology_diagnostic import load_model_from_checkpoint


def find_matched_control_segment(target_bin: np.ndarray, target_length_px: float) -> Optional[Dict[str, Any]]:
    """
    Finds a matched true crack continuation segment in target_bin
    with Euclidean distance between endpoints approximately equal to target_length_px.
    """
    skel = skeletonize(target_bin > 0).astype(np.uint8)
    pts = np.argwhere(skel)
    if len(pts) < 10:
        return None

    np.random.seed(42)
    sample_indices = np.random.choice(len(pts), size=min(100, len(pts)), replace=False)
    
    best_pair = None
    best_diff = float('inf')

    for idx in sample_indices:
        p0 = pts[idx]
        dists = np.linalg.norm(pts - p0, axis=1)
        diffs = np.abs(dists - target_length_px)
        min_i = np.argmin(diffs)
        if diffs[min_i] < best_diff:
            p1 = pts[min_i]
            line_mask = np.zeros_like(target_bin)
            cv2.line(line_mask, (int(p0[1]), int(p0[0])), (int(p1[1]), int(p1[0])), 1, thickness=3)
            overlap = np.sum((line_mask == 1) & (target_bin == 1)) / max(np.sum(line_mask == 1), 1)
            if overlap > 0.65:
                best_diff = diffs[min_i]
                best_pair = (p0, p1)
                if best_diff < 1.0:
                    break

    if best_pair is None:
        p0 = pts[len(pts) // 2]
        dists = np.linalg.norm(pts - p0, axis=1)
        min_i = np.argmin(np.abs(dists - target_length_px))
        best_pair = (p0, pts[min_i])

    p0, p1 = best_pair
    line_mask = np.zeros_like(target_bin)
    cv2.line(line_mask, (int(p0[1]), int(p0[0])), (int(p1[1]), int(p1[0])), 1, thickness=3)
    ctrl_corridor = (line_mask == 1) & (target_bin == 1)

    return {
        'corridor_mask': ctrl_corridor,
        'endpoints': (p0, p1),
        'length_px': float(np.linalg.norm(p0 - p1))
    }


def compute_cosine_similarity(feat_map: torch.Tensor, mask1: np.ndarray, mask2: np.ndarray, coords: List[Tuple[int, int]], H: int, W: int, S: int = 8) -> float:
    """
    Computes cosine similarity between average feature vector of region mask1 and region mask2.
    feat_map: (N_tiles, C, 56, 56)
    """
    v1_list, v2_list = [], []
    for j, (y_tile, x_tile) in enumerate(coords):
        for r_idx in range(56):
            for c_idx in range(56):
                gy0, gy1 = y_tile + r_idx * S, y_tile + (r_idx + 1) * S
                gx0, gx1 = x_tile + c_idx * S, x_tile + (c_idx + 1) * S
                cy0, cy1 = max(0, min(H, gy0)), max(0, min(H, gy1))
                cx0, cx1 = max(0, min(W, gx0)), max(0, min(W, gx1))
                if cy1 > cy0 and cx1 > cx0:
                    if np.sum(mask1[cy0:cy1, cx0:cx1]) > 0:
                        v1_list.append(feat_map[j, :, r_idx, c_idx])
                    if np.sum(mask2[cy0:cy1, cx0:cx1]) > 0:
                        v2_list.append(feat_map[j, :, r_idx, c_idx])

    if len(v1_list) == 0 or len(v2_list) == 0:
        return 0.0

    mean_v1 = torch.stack(v1_list).mean(dim=0)
    mean_v2 = torch.stack(v2_list).mean(dim=0)

    sim = F.cosine_similarity(mean_v1.unsqueeze(0), mean_v2.unsqueeze(0)).item()
    return float(sim)


def render_provenance_panel(
    img_rgb: np.ndarray,
    target_bin: np.ndarray,
    corridor_mask: np.ndarray,
    base_logits: np.ndarray,
    probe_logits: Dict[str, np.ndarray],
    out_path: str,
    event_id: int,
    case_name: str,
    d_gap: float,
    d_sage: float,
    d_bg: float,
    d_crack: float,
    d_int: float
):
    """
    Renders multi-probe visual panel:
    1. RGB + Corridor Contour
    2. Base Logits / Probability
    3. Probe A0 (SAGE-off)
    4. Probe B1 (Local BG Replacement)
    5. Probe B2 (True Crack Replacement)
    6. Probe C (Interior Replacement)
    """
    H, W = img_rgb.shape[:2]
    corr_pts = np.argwhere(corridor_mask)
    if len(corr_pts) == 0:
        return

    ymin, xmin = np.min(corr_pts, axis=0)
    ymax, xmax = np.max(corr_pts, axis=0)
    pad = 40
    ymin, xmin = max(0, ymin - pad), max(0, xmin - pad)
    ymax, xmax = min(H, ymax + pad + 1), min(W, xmax + pad + 1)

    crop_rgb = img_rgb[ymin:ymax, xmin:xmax].copy()
    crop_gt = target_bin[ymin:ymax, xmin:xmax]
    crop_corr = corridor_mask[ymin:ymax, xmin:xmax]
    contours, _ = cv2.findContours(crop_corr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    titles = [
        '1. Baseline Prob',
        f'2. A0: SAGE-off d={d_sage:+.2f}',
        f'3. B1: BG-Patch d={d_bg:+.2f}',
        f'4. B2: Crack-Patch d={d_crack:+.2f}',
        f'5. C: Interior-Patch d={d_int:+.2f}'
    ]
    logit_keys = ['base', 'sage_off', 'b1_bg', 'b2_crack', 'c_int']

    panels = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    for key, title in zip(logit_keys, titles):
        l_crop = probe_logits[key][ymin:ymax, xmin:xmax]
        prob = 1.0 / (1.0 + np.exp(-l_crop.astype(np.float64)))
        heat = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        blended = (0.65 * heat + 0.35 * gray_3ch).astype(np.uint8)

        cv2.drawContours(blended, contours, -1, (0, 255, 255), 1)
        cv2.putText(blended, title, (4, 12), font, 0.30, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(blended)

    strip = np.hstack(panels)
    banner_h = 24
    banner = np.zeros((banner_h, strip.shape[1], 3), dtype=np.uint8)
    banner_title = f"Event #{event_id:03d} ({case_name}) | Gap: {d_gap:.1f}px | SAGE d={d_sage:+.2f} | Interior d={d_int:+.2f}"
    cv2.putText(banner, banner_title, (10, 16), font, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

    final_img = np.vstack([banner, strip])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    t0_start = time.time()
    print("=" * 80)
    print("PHASE 6 DIAGNOSTIC: STAGE-1 SKIP PROVENANCE ABLATION (ZERO-TRAINING)")
    print("Target Cohort: 43 Wider-Gap Events (D_gap > 5.0 px) | Checkpoint: Candidate B")
    print("=" * 80)

    out_dir = os.path.join(project_root, 'results', 'diagnostics', 'phase6_stage1_provenance')
    viz_dir = os.path.join(out_dir, 'visualizations')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    config_path = os.path.join(project_root, 'results', 'configs', 'b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml')
    ckpt_path = os.path.join(project_root, 'results', 'checkpoints', 'P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = False

    print(f"Device: {device} (CuDNN disabled for numerical precision)")
    print("Loading Candidate B model...")
    model = load_model_from_checkpoint(config_path, ckpt_path, device)
    model.eval()

    csv_118 = os.path.join(project_root, 'results', 'diagnostics', 'phase6_bottleneck_path', 'bottleneck_path_118events.csv')
    df_118 = pd.read_csv(csv_118)
    df_43 = df_118[df_118['gap_category'] == 'Wider_Gap (> 5 px)'].copy().reset_index(drop=True)
    assert len(df_43) == 43, f"Expected exactly 43 wider gap events, found {len(df_43)}"
    print(f"Cohort loaded: {len(df_43)} events across {df_43['case_name'].nunique()} unique images.")

    data_root = os.path.join(project_root, 'datasets', 'Crack500_ready')
    val_img_dir = os.path.join(data_root, 'val', 'images')
    val_mask_dir = os.path.join(data_root, 'val', 'masks')

    event_records = []
    grouped = df_43.groupby('case_name')
    pbar = tqdm(total=len(df_43), desc="Running Stage-1 Provenance Diagnostic")

    for case_name, group_events in grouped:
        ip = os.path.join(val_img_dir, case_name + '.jpg')
        if not os.path.exists(ip):
            ip = os.path.join(val_img_dir, case_name + '.png')
        mp = os.path.join(val_mask_dir, case_name + '.png')
        if not os.path.exists(mp):
            mp = os.path.join(val_mask_dir, case_name + '.jpg')

        img = cv2.imread(ip)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        target_bin = (target > 127).astype(np.uint8)
        H, W = target_bin.shape[:2]

        tile_size = 448
        pad_h = (tile_size - (H % tile_size)) % tile_size
        pad_w = (tile_size - (W % tile_size)) % tile_size
        padded_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        pH, pW = padded_img.shape[:2]

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        patches, coords = [], []
        for y in range(0, pH, tile_size):
            for x in range(0, pW, tile_size):
                p = padded_img[y:y+tile_size, x:x+tile_size]
                pt = torch.from_numpy(p).permute(2, 0, 1).float() / 255.0
                pt = (pt - mean) / std
                patches.append(pt)
                coords.append((y, x))

        batch = torch.stack(patches).to(device)

        # Forward backbone and extract Stage 1 F_pre and F_post
        stage1 = model.backbone.convnext.stages[1]
        with torch.no_grad():
            x_stem = model.backbone.convnext.stem(batch)
            s0_feat = model.backbone.convnext.stages[0](x_stem)
            f_pre = stage1._execute_main_path(s0_feat)
            f_post = stage1(s0_feat)

            feat_dict = model.backbone(batch)
            base_logits_tiles = model.decoder(feat_dict['bottleneck'], feat_dict['skips'], target_size=(448, 448))

        base_logits_np = np.zeros((pH, pW), dtype=np.float32)
        for j, (y, x) in enumerate(coords):
            base_logits_np[y:y+448, x:x+448] = base_logits_tiles[j, 0].cpu().numpy()
        base_logits_np = base_logits_np[:H, :W]
        base_pred_bin = (base_logits_np > 0.0).astype(np.uint8)

        num_gt_cc, gt_labels = cv2.connectedComponents(target_bin, connectivity=8)
        num_pred_cc, base_pred_labels = cv2.connectedComponents(base_pred_bin, connectivity=8)

        # Process each bridge event
        for _, ev_row in group_events.iterrows():
            ev_id = int(ev_row['event_id'])
            p_id = int(ev_row['pred_cc_id'])
            gA = int(ev_row['primary_gt_A'])
            gB = int(ev_row['primary_gt_B'])
            d_gap = float(ev_row['d_gap_px'])
            cohort_group = 'Group_A (5-8px)' if d_gap <= 8.0 else 'Group_B (>8px)'

            cc_fp = (base_pred_labels == p_id) & (target_bin == 0)
            r = max(int(np.ceil(d_gap / 2.0)) + 2, 3)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
            dil_A = cv2.dilate((gt_labels == gA).astype(np.uint8), k)
            dil_B = cv2.dilate((gt_labels == gB).astype(np.uint8), k)
            corridor = (dil_A > 0) & (dil_B > 0) & cc_fp
            if np.sum(corridor) == 0:
                corridor = cc_fp

            # Local background ring
            k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))
            local_ring = cv2.dilate(corridor.astype(np.uint8), k_bg) > 0
            local_bg = local_ring & (target_bin == 0) & (base_pred_bin == 0)

            # Strict interior corridor (excluding 2px dilation around GT crack endpoints)
            k_end = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dil_endpoints = cv2.dilate(target_bin, k_end) > 0
            interior_corridor = corridor & (~dil_endpoints)
            if np.sum(interior_corridor) == 0:
                interior_corridor = corridor

            # Matched true crack control segment
            ctrl_info = find_matched_control_segment(target_bin, d_gap)
            ctrl_mask = ctrl_info['corridor_mask'] if ctrl_info is not None else None

            # Baseline continuous metrics
            z00_vals = base_logits_np[corridor]
            z00_med = float(np.median(z00_vals))
            z00_p10 = float(np.percentile(z00_vals, 10))
            z00_prob = float(1.0 / (1.0 + np.exp(-z00_med)))

            # =================================================================
            # PROBE A0: Stage-1 SAGE-off Counterfactual
            # =================================================================
            # (1) Stage-1 Skip Only
            skips_sage_off = list(feat_dict['skips'])
            skips_sage_off[1] = f_pre
            with torch.no_grad():
                out_sage_off = model.decoder(feat_dict['bottleneck'], skips_sage_off, target_size=(448, 448))
            sage_off_np = np.zeros((pH, pW), dtype=np.float32)
            for j, (y, x) in enumerate(coords):
                sage_off_np[y:y+448, x:x+448] = out_sage_off[j, 0].cpu().numpy()
            sage_off_np = sage_off_np[:H, :W]

            z_sage_off_vals = sage_off_np[corridor]
            z_sage_off_med = float(np.median(z_sage_off_vals))
            delta_sage_off = z00_med - z_sage_off_med

            # Topology check for SAGE-off
            so_bin = (sage_off_np > 0.0).astype(np.uint8)
            _, so_labels = cv2.connectedComponents(so_bin, connectivity=8)
            gA_so = so_labels[gt_labels == gA]; gA_so = gA_so[gA_so > 0]
            gB_so = so_labels[gt_labels == gB]; gB_so = gB_so[gB_so > 0]
            cured_sage_off = bool((len(gA_so) > 0 and len(gB_so) > 0) and (np.bincount(gA_so).argmax() != np.bincount(gB_so).argmax()))

            # =================================================================
            # PROBE A: Pre- vs Post-SAGE Representation Contrast
            # =================================================================
            sim_pre_crack = compute_cosine_similarity(f_pre, corridor, target_bin, coords, H, W, S=8)
            sim_pre_bg = compute_cosine_similarity(f_pre, corridor, local_bg, coords, H, W, S=8)
            sim_post_crack = compute_cosine_similarity(f_post, corridor, target_bin, coords, H, W, S=8)
            sim_post_bg = compute_cosine_similarity(f_post, corridor, local_bg, coords, H, W, S=8)

            norm_f_pre = float(torch.norm(f_pre).item())
            norm_f_post = float(torch.norm(f_post).item())
            norm_diff = float(torch.norm(f_post - f_pre).item())

            # =================================================================
            # PROBE B: Stage-1 Skip Activation Patching
            # =================================================================
            S = 8
            s1_feat = feat_dict['skips'][1]  # (B, 96, 56, 56)

            # Compute local background mean vector
            bg_vectors = []
            for j, (y_tile, x_tile) in enumerate(coords):
                for r_idx in range(56):
                    for c_idx in range(56):
                        gy0, gy1 = y_tile + r_idx * S, y_tile + (r_idx + 1) * S
                        gx0, gx1 = x_tile + c_idx * S, x_tile + (c_idx + 1) * S
                        cy0, cy1 = max(0, min(H, gy0)), max(0, min(H, gy1))
                        cx0, cx1 = max(0, min(W, gx0)), max(0, min(W, gx1))
                        if cy1 > cy0 and cx1 > cx0 and np.sum(local_bg[cy0:cy1, cx0:cx1]) > (S * S * 0.5):
                            bg_vectors.append(s1_feat[j, :, r_idx, c_idx])
            mean_bg_vec = torch.stack(bg_vectors).mean(dim=0).view(1, 96, 1, 1) if len(bg_vectors) > 0 else torch.zeros((1, 96, 1, 1), device=device)

            # Compute matched true-crack mean vector
            crack_vectors = []
            if ctrl_mask is not None and np.sum(ctrl_mask) > 0:
                for j, (y_tile, x_tile) in enumerate(coords):
                    for r_idx in range(56):
                        for c_idx in range(56):
                            gy0, gy1 = y_tile + r_idx * S, y_tile + (r_idx + 1) * S
                            gx0, gx1 = x_tile + c_idx * S, x_tile + (c_idx + 1) * S
                            cy0, cy1 = max(0, min(H, gy0)), max(0, min(H, gy1))
                            cx0, cx1 = max(0, min(W, gx0)), max(0, min(W, gx1))
                            if cy1 > cy0 and cx1 > cx0 and np.sum(ctrl_mask[cy0:cy1, cx0:cx1]) > 0:
                                crack_vectors.append(s1_feat[j, :, r_idx, c_idx])
            mean_crack_vec = torch.stack(crack_vectors).mean(dim=0).view(1, 96, 1, 1) if len(crack_vectors) > 0 else mean_bg_vec

            # B1: Local BG Replacement
            s1_b1 = s1_feat.clone()
            # B2: True Crack Replacement
            s1_b2 = s1_feat.clone()
            # B3: Spatial Shuffle
            s1_b3 = s1_feat.clone()

            corr_cell_indices = []
            for j, (y_tile, x_tile) in enumerate(coords):
                for r_idx in range(56):
                    for c_idx in range(56):
                        gy0, gy1 = y_tile + r_idx * S, y_tile + (r_idx + 1) * S
                        gx0, gx1 = x_tile + c_idx * S, x_tile + (c_idx + 1) * S
                        cy0, cy1 = max(0, min(H, gy0)), max(0, min(H, gy1))
                        cx0, cx1 = max(0, min(W, gx0)), max(0, min(W, gx1))
                        if cy1 > cy0 and cx1 > cx0 and np.sum(corridor[cy0:cy1, cx0:cx1]) > 0:
                            s1_b1[j, :, r_idx, c_idx] = mean_bg_vec[0, :, 0, 0]
                            s1_b2[j, :, r_idx, c_idx] = mean_crack_vec[0, :, 0, 0]
                            corr_cell_indices.append((j, r_idx, c_idx))

            # Shuffle B3
            if len(corr_cell_indices) > 1:
                shuffled_idx = np.random.RandomState(42).permutation(len(corr_cell_indices))
                shuffled_vals = [s1_feat[corr_cell_indices[k][0], :, corr_cell_indices[k][1], corr_cell_indices[k][2]].clone() for k in shuffled_idx]
                for k, (j, r_idx, c_idx) in enumerate(corr_cell_indices):
                    s1_b3[j, :, r_idx, c_idx] = shuffled_vals[k]

            def forward_with_s1(s1_tensor):
                sk = list(feat_dict['skips'])
                sk[1] = s1_tensor
                with torch.no_grad():
                    out = model.decoder(feat_dict['bottleneck'], sk, target_size=(448, 448))
                out_np = np.zeros((pH, pW), dtype=np.float32)
                for j, (y, x) in enumerate(coords):
                    out_np[y:y+448, x:x+448] = out[j, 0].cpu().numpy()
                out_np = out_np[:H, :W]
                return out_np

            b1_np = forward_with_s1(s1_b1)
            b2_np = forward_with_s1(s1_b2)
            b3_np = forward_with_s1(s1_b3)

            z_b1_med = float(np.median(b1_np[corridor]))
            z_b2_med = float(np.median(b2_np[corridor]))
            z_b3_med = float(np.median(b3_np[corridor]))

            delta_b1 = z00_med - z_b1_med
            delta_b2 = z00_med - z_b2_med
            delta_b3 = z00_med - z_b3_med

            # =================================================================
            # PROBE C: Endpoint-Preserving Interior Replacement
            # =================================================================
            s1_c = s1_feat.clone()
            n_int_cells_patched = 0
            for j, (y_tile, x_tile) in enumerate(coords):
                for r_idx in range(56):
                    for c_idx in range(56):
                        gy0, gy1 = y_tile + r_idx * S, y_tile + (r_idx + 1) * S
                        gx0, gx1 = x_tile + c_idx * S, x_tile + (c_idx + 1) * S
                        cy0, cy1 = max(0, min(H, gy0)), max(0, min(H, gy1))
                        cx0, cx1 = max(0, min(W, gx0)), max(0, min(W, gx1))
                        if cy1 > cy0 and cx1 > cx0:
                            # Strict interior: touches interior corridor AND zero true crack
                            if np.sum(interior_corridor[cy0:cy1, cx0:cx1]) > 0 and np.sum(target_bin[cy0:cy1, cx0:cx1]) == 0:
                                s1_c[j, :, r_idx, c_idx] = mean_bg_vec[0, :, 0, 0]
                                n_int_cells_patched += 1

            c_np = forward_with_s1(s1_c)
            z_c_med = float(np.median(c_np[corridor]))
            delta_c = z00_med - z_c_med

            # Topology check for Probe C
            c_bin = (c_np > 0.0).astype(np.uint8)
            _, c_labels = cv2.connectedComponents(c_bin, connectivity=8)
            gA_c = c_labels[gt_labels == gA]; gA_c = gA_c[gA_c > 0]
            gB_c = c_labels[gt_labels == gB]; gB_c = gB_c[gB_c > 0]
            cured_c = bool((len(gA_c) > 0 and len(gB_c) > 0) and (np.bincount(gA_c).argmax() != np.bincount(gB_c).argmax()))

            # Record
            rec = {
                'event_id': ev_id,
                'case_name': case_name,
                'd_gap_px': d_gap,
                'cohort_group': cohort_group,
                'n_corridor_pixels': int(np.sum(corridor)),
                'n_interior_pixels': int(np.sum(interior_corridor)),
                'n_interior_cells_patched': n_int_cells_patched,
                'z00_baseline_logit': z00_med,
                'z00_baseline_prob': z00_prob,
                # Probe A0
                'probe_a0_sage_off_logit': z_sage_off_med,
                'probe_a0_delta_sage_off': delta_sage_off,
                'probe_a0_cured': cured_sage_off,
                # Probe A
                'probe_a_sim_pre_crack': sim_pre_crack,
                'probe_a_sim_pre_bg': sim_pre_bg,
                'probe_a_sim_post_crack': sim_post_crack,
                'probe_a_sim_post_bg': sim_post_bg,
                'probe_a_delta_sim_crack': sim_post_crack - sim_pre_crack,
                'probe_a_delta_sim_bg': sim_post_bg - sim_pre_bg,
                'probe_a_norm_pre': norm_f_pre,
                'probe_a_norm_post': norm_f_post,
                'probe_a_norm_diff': norm_diff,
                # Probe B
                'probe_b1_bg_logit': z_b1_med,
                'probe_b1_delta_bg': delta_b1,
                'probe_b2_crack_logit': z_b2_med,
                'probe_b2_delta_crack': delta_b2,
                'probe_b3_shuffle_logit': z_b3_med,
                'probe_b3_delta_shuffle': delta_b3,
                # Probe C
                'probe_c_interior_logit': z_c_med,
                'probe_c_delta_interior': delta_c,
                'probe_c_cured': cured_c
            }

            # Render visualization for key representative cases
            if ev_id in [2, 3, 4, 10, 13, 14, 16, 20]:
                v_path = os.path.join(viz_dir, f"event_{ev_id:03d}_{case_name}_gap{d_gap:.1f}px_provenance.png")
                probe_map = {
                    'base': base_logits_np,
                    'sage_off': sage_off_np,
                    'b1_bg': b1_np,
                    'b2_crack': b2_np,
                    'c_int': c_np
                }
                render_provenance_panel(
                    img, target_bin, corridor, base_logits_np, probe_map,
                    v_path, ev_id, case_name, d_gap, delta_sage_off, delta_b1, delta_b2, delta_c
                )

            event_records.append(rec)
            pbar.update(1)

    pbar.close()

    df_out = pd.DataFrame(event_records)
    csv_out = os.path.join(out_dir, 'stage1_provenance_measurements_43events.csv')
    df_out.to_csv(csv_out, index=False)
    print(f"\nSaved per-event measurements: {csv_out}")

    # Build stratified summaries
    groups = {
        'Overall (All 43)': df_out,
        'Group_A (5.0 < D_gap <= 8.0 px)': df_out[df_out['cohort_group'] == 'Group_A (5-8px)'],
        'Group_B (D_gap > 8.0 px)': df_out[df_out['cohort_group'] == 'Group_B (>8px)']
    }

    summary_rows = []
    summary_dict = {}

    for g_name, g_df in groups.items():
        row = {
            'cohort_group': g_name,
            'count': len(g_df),
            'median_d_gap': float(g_df['d_gap_px'].median()),
            'baseline_logit': float(g_df['z00_baseline_logit'].median()),
            # Probe A0 SAGE-off
            'sage_off_logit': float(g_df['probe_a0_sage_off_logit'].median()),
            'median_delta_sage_off': float(g_df['probe_a0_delta_sage_off'].median()),
            'mean_delta_sage_off': float(g_df['probe_a0_delta_sage_off'].mean()),
            'cured_sage_off_count': int(g_df['probe_a0_cured'].sum()),
            # Probe A Representation
            'sim_pre_crack': float(g_df['probe_a_sim_pre_crack'].median()),
            'sim_pre_bg': float(g_df['probe_a_sim_pre_bg'].median()),
            'sim_post_crack': float(g_df['probe_a_sim_post_crack'].median()),
            'sim_post_bg': float(g_df['probe_a_sim_post_bg'].median()),
            # Probe B Patching
            'median_delta_b1_bg': float(g_df['probe_b1_delta_bg'].median()),
            'median_delta_b2_crack': float(g_df['probe_b2_delta_crack'].median()),
            'median_delta_b3_shuffle': float(g_df['probe_b3_delta_shuffle'].median()),
            # Probe C Interior Patch
            'interior_logit': float(g_df['probe_c_interior_logit'].median()),
            'median_delta_c_interior': float(g_df['probe_c_delta_interior'].median()),
            'mean_delta_c_interior': float(g_df['probe_c_delta_interior'].mean()),
            'cured_c_interior_count': int(g_df['probe_c_cured'].sum()),
            'cure_rate_c_pct': float(g_df['probe_c_cured'].mean() * 100.0)
        }
        summary_rows.append(row)
        summary_dict[g_name] = {k: v for k, v in row.items() if k != 'cohort_group'}

    df_sum = pd.DataFrame(summary_rows)
    sum_csv_out = os.path.join(out_dir, 'stage1_provenance_summary.csv')
    df_sum.to_csv(sum_csv_out, index=False)
    print(f"Saved summary CSV: {sum_csv_out}")

    json_out = os.path.join(out_dir, 'stage1_provenance_summary.json')
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Saved summary JSON: {json_out}")

    print("\n" + "=" * 80)
    print("STAGE-1 SKIP PROVENANCE ABLATION SUMMARY TABLE (STRATIFIED)")
    print("=" * 80)
    print(df_sum.to_string(index=False))

    el_time = time.time() - t0_start
    print(f"\nExecution finished in {el_time:.1f}s")


if __name__ == '__main__':
    main()
