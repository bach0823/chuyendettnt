#!/usr/bin/env python3
"""
scripts/diagnostics/run_stage1_representation_characterization.py

Zero-Training Characterization of Stage-1 Main Path Representation (F_preSAGE)
Target Cohort: Exactly the 43 wider-gap bridge events (D_gap > 5.0 px) from Phase 6 Diagnostic D.
Target Model: Candidate B Baseline (B2ConvNeXtViTUNet, Setting A, Tile 448).

Investigates:
What is the nature of the representation that Stage-1 main path uses to represent
false-bridge corridors: is it generic continuity/line representation, or genuinely
close to semantic true-crack representation?

Stratification:
- Group A: 5.0 < D_gap <= 8.0 px (N = 11, negative control for stride-8 resolution limit)
- Group B: D_gap > 8.0 px (N = 32, primary causal provenance cohort)
- Overall (N = 43)

Key Metrics:
1. Normalized semantic crack projection: alpha = <v_corr - v_bg, v_crack - v_bg> / ||v_crack - v_bg||^2
2. Spatial decomposition: alpha_full, alpha_interior, alpha_endpoint
3. Channel-wise activation contrast correlation: r_channel = Pearson(v_corr - v_bg, v_crack - v_bg)
4. Centered cosine similarity: cos(theta) = <Delta v_corr, Delta v_crack> / (||Delta v_corr|| * ||Delta v_crack||)
5. Relative displacement magnitude: M = ||Delta v_corr|| / ||Delta v_crack||
6. Upstream progression: Stage 0 (stride 4, 48ch) vs Stage 1 (stride 8, 96ch)
7. Direct SAGE-0 contribution: Stage 1 main path with Stage 0 SAGE active vs bypassed
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


def extract_masked_feature_vector(
    feat_map: torch.Tensor,
    mask: np.ndarray,
    coords: List[Tuple[int, int]],
    H: int,
    W: int,
    stride: int
) -> Optional[torch.Tensor]:
    """
    Computes weighted mean feature vector over a spatial binary mask.
    feat_map: (N_tiles, C, H_s, W_s)
    mask: (H, W) boolean/uint8
    stride: 4 (Stage 0) or 8 (Stage 1)
    """
    N_tiles, C, H_s, W_s = feat_map.shape
    accum_vec = torch.zeros(C, device=feat_map.device, dtype=torch.float32)
    total_weight = 0.0

    for j, (y_tile, x_tile) in enumerate(coords):
        for r_idx in range(H_s):
            for c_idx in range(W_s):
                gy0, gy1 = y_tile + r_idx * stride, y_tile + (r_idx + 1) * stride
                gx0, gx1 = x_tile + c_idx * stride, x_tile + (c_idx + 1) * stride
                cy0, cy1 = max(0, min(H, gy0)), max(0, min(H, gy1))
                cx0, cx1 = max(0, min(W, gx0)), max(0, min(W, gx1))
                if cy1 > cy0 and cx1 > cx0:
                    w = float(np.sum(mask[cy0:cy1, cx0:cx1]))
                    if w > 0:
                        accum_vec += w * feat_map[j, :, r_idx, c_idx]
                        total_weight += w

    if total_weight == 0.0:
        return None

    return accum_vec / total_weight


def compute_representation_metrics(
    v_corr: torch.Tensor,
    v_crack: torch.Tensor,
    v_bg: torch.Tensor
) -> Dict[str, float]:
    """
    Computes rigorous representation geometry metrics:
    - Delta v_corr = v_corr - v_bg
    - Delta v_crack = v_crack - v_bg
    - alpha = <Delta v_corr, Delta v_crack> / ||Delta v_crack||^2
    - cos(theta) = centered cosine similarity (directional alignment)
    - M = ||Delta v_corr|| / ||Delta v_crack|| (relative magnitude)
    - r_channel = Pearson correlation across channels of Delta v_corr and Delta v_crack
    """
    with torch.no_grad():
        norm_corr = float(torch.norm(v_corr, p=2).item())
        norm_crack = float(torch.norm(v_crack, p=2).item())
        norm_bg = float(torch.norm(v_bg, p=2).item())

        raw_sim_corr_crack = float(F.cosine_similarity(v_corr.unsqueeze(0), v_crack.unsqueeze(0)).item())
        raw_sim_corr_bg = float(F.cosine_similarity(v_corr.unsqueeze(0), v_bg.unsqueeze(0)).item())
        raw_sim_crack_bg = float(F.cosine_similarity(v_crack.unsqueeze(0), v_bg.unsqueeze(0)).item())

        delta_corr = v_corr - v_bg
        delta_crack = v_crack - v_bg

        norm_delta_corr = float(torch.norm(delta_corr, p=2).item())
        norm_delta_crack = float(torch.norm(delta_crack, p=2).item())

        if norm_delta_crack < 1e-8:
            return {
                'norm_corr': norm_corr, 'norm_crack': norm_crack, 'norm_bg': norm_bg,
                'raw_sim_corr_crack': raw_sim_corr_crack, 'raw_sim_corr_bg': raw_sim_corr_bg, 'raw_sim_crack_bg': raw_sim_crack_bg,
                'norm_delta_corr': norm_delta_corr, 'norm_delta_crack': norm_delta_crack,
                'magnitude_ratio': 0.0, 'centered_cos': 0.0, 'alpha': 0.0, 'r_channel': 0.0
            }

        magnitude_ratio = norm_delta_corr / norm_delta_crack
        dot_product = float(torch.dot(delta_corr, delta_crack).item())
        
        if norm_delta_corr > 1e-8 and norm_delta_crack > 1e-8:
            centered_cos = dot_product / (norm_delta_corr * norm_delta_crack)
        else:
            centered_cos = 0.0

        alpha = dot_product / (norm_delta_crack ** 2)

        # Pearson correlation across channels
        mu_c = torch.mean(delta_corr)
        mu_k = torch.mean(delta_crack)
        zc = delta_corr - mu_c
        zk = delta_crack - mu_k
        zc_norm = torch.norm(zc, p=2).item()
        zk_norm = torch.norm(zk, p=2).item()
        if zc_norm > 1e-8 and zk_norm > 1e-8:
            r_channel = float(torch.dot(zc, zk).item() / (zc_norm * zk_norm))
        else:
            r_channel = 0.0

    return {
        'norm_corr': norm_corr,
        'norm_crack': norm_crack,
        'norm_bg': norm_bg,
        'raw_sim_corr_crack': raw_sim_corr_crack,
        'raw_sim_corr_bg': raw_sim_corr_bg,
        'raw_sim_crack_bg': raw_sim_crack_bg,
        'norm_delta_corr': norm_delta_corr,
        'norm_delta_crack': norm_delta_crack,
        'magnitude_ratio': float(magnitude_ratio),
        'centered_cos': float(centered_cos),
        'alpha': float(alpha),
        'r_channel': float(r_channel)
    }


def render_characterization_panel(
    img_rgb: np.ndarray,
    target_bin: np.ndarray,
    corridor_mask: np.ndarray,
    interior_mask: np.ndarray,
    base_logits: np.ndarray,
    out_path: str,
    event_id: int,
    case_name: str,
    d_gap: float,
    metrics_s1_pre: Dict[str, float],
    metrics_s1_int: Dict[str, float],
    metrics_s0_pre: Dict[str, float],
    delta_corr_s1: np.ndarray,
    delta_crack_s1: np.ndarray
):
    """
    Renders diagnostic visualization panel:
    1. RGB + Ground Truth (green) + Corridor (yellow) + Interior (cyan)
    2. Baseline Probability Heatmap
    3. Stage-0 vs Stage-1 Representation Metrics Summary
    4. Top Channel Activation Profiles (Corridor vs True Crack)
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
    crop_int = interior_mask[ymin:ymax, xmin:xmax]

    # Panel 1: RGB + Annotations
    p1 = crop_rgb.copy()
    gt_contours, _ = cv2.findContours(crop_gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    corr_contours, _ = cv2.findContours(crop_corr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    int_contours, _ = cv2.findContours(crop_int.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(p1, gt_contours, -1, (0, 255, 0), 1)
    cv2.drawContours(p1, corr_contours, -1, (255, 255, 0), 1)
    cv2.drawContours(p1, int_contours, -1, (0, 255, 255), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(p1, "1. RGB (GT:grn, Corr:yel, Int:cyn)", (4, 12), font, 0.30, (255, 255, 255), 1, cv2.LINE_AA)

    # Panel 2: Baseline Probability
    l_crop = base_logits[ymin:ymax, xmin:xmax]
    prob = 1.0 / (1.0 + np.exp(-l_crop.astype(np.float64)))
    heat = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    p2 = (0.65 * heat + 0.35 * gray_3ch).astype(np.uint8)
    cv2.drawContours(p2, corr_contours, -1, (255, 255, 255), 1)
    cv2.putText(p2, f"2. Baseline Prob (med={np.median(prob[crop_corr > 0]):.2f})", (4, 12), font, 0.30, (255, 255, 255), 1, cv2.LINE_AA)

    # Resize panels to common height if needed
    h_target = max(p1.shape[0], 120)
    w_target = max(p1.shape[1], 120)
    p1_resized = cv2.resize(p1, (w_target, h_target), interpolation=cv2.INTER_NEAREST)
    p2_resized = cv2.resize(p2, (w_target, h_target), interpolation=cv2.INTER_NEAREST)

    # Panel 3: Text summary panel
    p3 = np.zeros((h_target, 240, 3), dtype=np.uint8)
    p3[:] = (25, 25, 25)
    lines = [
        f"Event #{event_id:03d} | Gap: {d_gap:.1f}px",
        f"S1 alpha (Full):    {metrics_s1_pre['alpha']:.3f}",
        f"S1 alpha (Interior):{metrics_s1_int['alpha']:.3f}",
        f"S1 cos(theta):      {metrics_s1_pre['centered_cos']:.3f}",
        f"S1 Mag Ratio M:     {metrics_s1_pre['magnitude_ratio']:.3f}",
        f"S1 r_channel:       {metrics_s1_pre['r_channel']:.3f}",
        f"S0 alpha (preSAGE): {metrics_s0_pre['alpha']:.3f}",
        f"S0 r_channel:       {metrics_s0_pre['r_channel']:.3f}",
    ]
    y_text = 16
    for line in lines:
        cv2.putText(p3, line, (8, y_text), font, 0.32, (200, 220, 255), 1, cv2.LINE_AA)
        y_text += 14

    # Panel 4: Top 15 Channel Profile
    p4 = np.zeros((h_target, 240, 3), dtype=np.uint8)
    p4[:] = (15, 15, 20)
    cv2.putText(p4, "Top-10 Crack Delta-Channels", (8, 14), font, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
    top_channels = np.argsort(np.abs(delta_crack_s1))[::-1][:8]
    y_bar = 28
    for ch in top_channels:
        val_k = delta_crack_s1[ch]
        val_c = delta_corr_s1[ch]
        # Bar chart
        bar_len_k = int(np.clip(val_k * 40, -50, 50))
        bar_len_c = int(np.clip(val_c * 40, -50, 50))
        x_center = 120
        # Draw crack (green) and corridor (yellow)
        cv2.line(p4, (x_center, y_bar), (x_center + bar_len_k, y_bar), (0, 220, 0), 3)
        cv2.line(p4, (x_center, y_bar + 4), (x_center + bar_len_c, y_bar + 4), (0, 220, 255), 3)
        cv2.putText(p4, f"C{ch:02d}", (8, y_bar + 3), font, 0.28, (180, 180, 180), 1, cv2.LINE_AA)
        y_bar += 11

    cv2.putText(p4, "Grn: True Crack | Yel: Corridor", (8, h_target - 6), font, 0.28, (150, 255, 150), 1, cv2.LINE_AA)

    strip = np.hstack([p1_resized, p2_resized, p3, p4])
    banner_h = 24
    banner = np.zeros((banner_h, strip.shape[1], 3), dtype=np.uint8)
    banner_title = f"Phase 6: Stage-1 Representation Characterization | Event #{event_id:03d} ({case_name}) | Gap {d_gap:.1f}px"
    cv2.putText(banner, banner_title, (10, 16), font, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

    final_img = np.vstack([banner, strip])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    t0_start = time.time()
    print("=" * 80)
    print("PHASE 6: STAGE-1 MAIN PATH REPRESENTATION CHARACTERIZATION (ZERO-TRAINING)")
    print("Target Cohort: 43 Wider-Gap Events (D_gap > 5.0 px) | Checkpoint: Candidate B")
    print("=" * 80)

    out_dir = os.path.join(project_root, 'results', 'diagnostics', 'stage1_characterization')
    viz_dir = os.path.join(out_dir, 'visualizations')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    config_path = os.path.join(project_root, 'results', 'configs', 'b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml')
    ckpt_path = os.path.join(project_root, 'results', 'checkpoints', 'P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = False

    print(f"Device: {device} (CuDNN disabled for deterministic numerical precision)")
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
    pbar = tqdm(total=len(df_43), desc="Characterizing Stage-1 Representation")

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

        # Forward backbone stages
        stage0 = model.backbone.convnext.stages[0]
        stage1 = model.backbone.convnext.stages[1]
        with torch.no_grad():
            x_stem = model.backbone.convnext.stem(batch)
            s0_pre = stage0._execute_main_path(x_stem)
            s0_post = stage0(x_stem)

            # Stage 1 main path conditioned on standard s0_post
            s1_pre = stage1._execute_main_path(s0_post)
            s1_post = stage1(s0_post)

            # Stage 1 main path counterfactual: what if Stage 0 SAGE was also bypassed?
            s1_pre_s0_off = stage1._execute_main_path(s0_pre)

            # Model prediction logits
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
            if np.sum(local_bg) == 0:
                local_bg = (target_bin == 0) & (base_pred_bin == 0)

            # Spatial decomposition: strict interior corridor vs endpoint transition
            k_end = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dil_endpoints = cv2.dilate(target_bin, k_end) > 0
            interior_corridor = corridor & (~dil_endpoints)
            if np.sum(interior_corridor) == 0:
                interior_corridor = corridor.copy()
            endpoint_corridor = corridor & dil_endpoints
            if np.sum(endpoint_corridor) == 0:
                endpoint_corridor = corridor.copy()

            # Matched true crack control segment
            ctrl_info = find_matched_control_segment(target_bin, d_gap)
            if ctrl_info is not None and np.sum(ctrl_info['corridor_mask']) > 0:
                crack_mask = ctrl_info['corridor_mask']
                ctrl_len = ctrl_info['length_px']
            else:
                crack_mask = target_bin > 0
                ctrl_len = d_gap

            # -------------------------------------------------------------
            # Extract Feature Vectors at Stage 1 (stride = 8, C = 96)
            # -------------------------------------------------------------
            v_bg_s1 = extract_masked_feature_vector(s1_pre, local_bg, coords, H, W, stride=8)
            v_crack_s1 = extract_masked_feature_vector(s1_pre, crack_mask, coords, H, W, stride=8)
            v_corr_s1 = extract_masked_feature_vector(s1_pre, corridor, coords, H, W, stride=8)
            v_int_s1 = extract_masked_feature_vector(s1_pre, interior_corridor, coords, H, W, stride=8)
            v_end_s1 = extract_masked_feature_vector(s1_pre, endpoint_corridor, coords, H, W, stride=8)

            # Metrics for Stage 1 preSAGE (primary object of inquiry)
            m_s1_pre = compute_representation_metrics(v_corr_s1, v_crack_s1, v_bg_s1)
            m_s1_int = compute_representation_metrics(v_int_s1, v_crack_s1, v_bg_s1)
            m_s1_end = compute_representation_metrics(v_end_s1, v_crack_s1, v_bg_s1)

            # Stage 1 postSAGE (to verify if SAGE alters the representation)
            v_bg_s1_post = extract_masked_feature_vector(s1_post, local_bg, coords, H, W, stride=8)
            v_crack_s1_post = extract_masked_feature_vector(s1_post, crack_mask, coords, H, W, stride=8)
            v_corr_s1_post = extract_masked_feature_vector(s1_post, corridor, coords, H, W, stride=8)
            m_s1_post = compute_representation_metrics(v_corr_s1_post, v_crack_s1_post, v_bg_s1_post)

            # Stage 1 preSAGE when Stage 0 SAGE is bypassed (S0-off counterfactual)
            v_bg_s1_s0off = extract_masked_feature_vector(s1_pre_s0_off, local_bg, coords, H, W, stride=8)
            v_crack_s1_s0off = extract_masked_feature_vector(s1_pre_s0_off, crack_mask, coords, H, W, stride=8)
            v_corr_s1_s0off = extract_masked_feature_vector(s1_pre_s0_off, corridor, coords, H, W, stride=8)
            m_s1_s0off = compute_representation_metrics(v_corr_s1_s0off, v_crack_s1_s0off, v_bg_s1_s0off)

            # -------------------------------------------------------------
            # Extract Feature Vectors at Stage 0 (stride = 4, C = 48)
            # -------------------------------------------------------------
            v_bg_s0_pre = extract_masked_feature_vector(s0_pre, local_bg, coords, H, W, stride=4)
            v_crack_s0_pre = extract_masked_feature_vector(s0_pre, crack_mask, coords, H, W, stride=4)
            v_corr_s0_pre = extract_masked_feature_vector(s0_pre, corridor, coords, H, W, stride=4)
            m_s0_pre = compute_representation_metrics(v_corr_s0_pre, v_crack_s0_pre, v_bg_s0_pre)

            v_bg_s0_post = extract_masked_feature_vector(s0_post, local_bg, coords, H, W, stride=4)
            v_crack_s0_post = extract_masked_feature_vector(s0_post, crack_mask, coords, H, W, stride=4)
            v_corr_s0_post = extract_masked_feature_vector(s0_post, corridor, coords, H, W, stride=4)
            m_s0_post = compute_representation_metrics(v_corr_s0_post, v_crack_s0_post, v_bg_s0_post)

            # Baseline logit stats on corridor
            z_vals = base_logits_np[corridor]
            z_med = float(np.median(z_vals))
            z_prob = float(1.0 / (1.0 + np.exp(-z_med)))

            record = {
                'event_id': ev_id,
                'case_name': case_name,
                'cohort_group': cohort_group,
                'd_gap_px': d_gap,
                'ctrl_length_px': ctrl_len,
                'n_corridor_px': int(np.sum(corridor)),
                'n_interior_px': int(np.sum(interior_corridor)),
                'n_endpoint_px': int(np.sum(endpoint_corridor)),
                'base_logit_med': z_med,
                'base_prob_med': z_prob,

                # Stage 1 preSAGE (Full Corridor)
                's1_pre_alpha': m_s1_pre['alpha'],
                's1_pre_centered_cos': m_s1_pre['centered_cos'],
                's1_pre_mag_ratio': m_s1_pre['magnitude_ratio'],
                's1_pre_r_channel': m_s1_pre['r_channel'],
                's1_pre_raw_cos_crack': m_s1_pre['raw_sim_corr_crack'],
                's1_pre_raw_cos_bg': m_s1_pre['raw_sim_corr_bg'],
                's1_pre_norm_delta_corr': m_s1_pre['norm_delta_corr'],
                's1_pre_norm_delta_crack': m_s1_pre['norm_delta_crack'],

                # Spatial Decomposition: Interior vs Endpoint
                's1_int_alpha': m_s1_int['alpha'],
                's1_int_centered_cos': m_s1_int['centered_cos'],
                's1_int_mag_ratio': m_s1_int['magnitude_ratio'],
                's1_int_r_channel': m_s1_int['r_channel'],
                's1_end_alpha': m_s1_end['alpha'],
                's1_end_centered_cos': m_s1_end['centered_cos'],
                's1_end_mag_ratio': m_s1_end['magnitude_ratio'],
                's1_end_r_channel': m_s1_end['r_channel'],

                # Stage 1 postSAGE
                's1_post_alpha': m_s1_post['alpha'],
                's1_post_centered_cos': m_s1_post['centered_cos'],
                's1_post_mag_ratio': m_s1_post['magnitude_ratio'],
                's1_post_r_channel': m_s1_post['r_channel'],

                # Stage 1 conditioned on S0-off
                's1_s0off_alpha': m_s1_s0off['alpha'],
                's1_s0off_centered_cos': m_s1_s0off['centered_cos'],
                's1_s0off_mag_ratio': m_s1_s0off['magnitude_ratio'],
                's1_s0off_r_channel': m_s1_s0off['r_channel'],

                # Stage 0 preSAGE & postSAGE
                's0_pre_alpha': m_s0_pre['alpha'],
                's0_pre_centered_cos': m_s0_pre['centered_cos'],
                's0_pre_mag_ratio': m_s0_pre['magnitude_ratio'],
                's0_pre_r_channel': m_s0_pre['r_channel'],
                's0_post_alpha': m_s0_post['alpha'],
                's0_post_centered_cos': m_s0_post['centered_cos'],
                's0_post_mag_ratio': m_s0_post['magnitude_ratio'],
                's0_post_r_channel': m_s0_post['r_channel']
            }
            event_records.append(record)

            # Render visualization for key representative cases
            if ev_id in [2, 12, 19, 21, 22, 28, 43, 62]:
                viz_path = os.path.join(viz_dir, f"event_{ev_id:03d}_{case_name}_gap{d_gap:.1f}px_representation.png")
                delta_corr_np = (v_corr_s1 - v_bg_s1).cpu().numpy()
                delta_crack_np = (v_crack_s1 - v_bg_s1).cpu().numpy()
                render_characterization_panel(
                    img_rgb=img,
                    target_bin=target_bin,
                    corridor_mask=corridor,
                    interior_mask=interior_corridor,
                    base_logits=base_logits_np,
                    out_path=viz_path,
                    event_id=ev_id,
                    case_name=case_name,
                    d_gap=d_gap,
                    metrics_s1_pre=m_s1_pre,
                    metrics_s1_int=m_s1_int,
                    metrics_s0_pre=m_s0_pre,
                    delta_corr_s1=delta_corr_np,
                    delta_crack_s1=delta_crack_np
                )

            pbar.update(1)

    pbar.close()

    df_res = pd.DataFrame(event_records)
    csv_out = os.path.join(out_dir, 'stage1_representation_measurements_43events.csv')
    df_res.to_csv(csv_out, index=False)
    print(f"\nSaved raw per-event measurements: {csv_out}")

    # Generate Stratified Summaries
    groups = {
        'Group_A (5-8px)': df_res[df_res['cohort_group'] == 'Group_A (5-8px)'],
        'Group_B (>8px)': df_res[df_res['cohort_group'] == 'Group_B (>8px)'],
        'Overall (All 43)': df_res
    }

    summary_rows = []
    summary_dict = {}

    for g_name, g_df in groups.items():
        n = len(g_df)
        g_stats = {
            'cohort': g_name,
            'N': n,
            'mean_d_gap': float(g_df['d_gap_px'].mean()),
            # Stage 1 preSAGE
            's1_pre_alpha_mean': float(g_df['s1_pre_alpha'].mean()),
            's1_pre_alpha_median': float(g_df['s1_pre_alpha'].median()),
            's1_pre_centered_cos_mean': float(g_df['s1_pre_centered_cos'].mean()),
            's1_pre_mag_ratio_mean': float(g_df['s1_pre_mag_ratio'].mean()),
            's1_pre_r_channel_mean': float(g_df['s1_pre_r_channel'].mean()),
            # Interior vs Endpoint
            's1_int_alpha_mean': float(g_df['s1_int_alpha'].mean()),
            's1_int_alpha_median': float(g_df['s1_int_alpha'].median()),
            's1_int_r_channel_mean': float(g_df['s1_int_r_channel'].mean()),
            's1_end_alpha_mean': float(g_df['s1_end_alpha'].mean()),
            's1_end_alpha_median': float(g_df['s1_end_alpha'].median()),
            # SAGE-0 Off counterfactual at Stage 1
            's1_s0off_alpha_mean': float(g_df['s1_s0off_alpha'].mean()),
            's1_s0off_r_channel_mean': float(g_df['s1_s0off_r_channel'].mean()),
            # Stage 0 preSAGE
            's0_pre_alpha_mean': float(g_df['s0_pre_alpha'].mean()),
            's0_pre_alpha_median': float(g_df['s0_pre_alpha'].median()),
            's0_pre_r_channel_mean': float(g_df['s0_pre_r_channel'].mean()),
            's0_post_alpha_mean': float(g_df['s0_post_alpha'].mean()),
        }
        summary_rows.append(g_stats)
        summary_dict[g_name] = g_stats

    df_summary = pd.DataFrame(summary_rows)
    sum_csv = os.path.join(out_dir, 'stage1_representation_summary.csv')
    df_summary.to_csv(sum_csv, index=False)
    print(f"Saved stratified summary CSV: {sum_csv}")

    sum_json = os.path.join(out_dir, 'stage1_representation_summary.json')
    with open(sum_json, 'w', encoding='utf-8') as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Saved stratified summary JSON: {sum_json}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Cohort':<18} | {'N':<3} | {'Gap(px)':<7} | {'S1 alpha':<8} | {'S1 cos_th':<9} | {'S1 Mag_M':<8} | {'S1 r_ch':<7} | {'S1 Int_a':<8} | {'S0 alpha':<8}")
    print("-" * 90)
    for row in summary_rows:
        print(f"{row['cohort']:<18} | {row['N']:<3} | {row['mean_d_gap']:<7.1f} | {row['s1_pre_alpha_mean']:<8.4f} | {row['s1_pre_centered_cos_mean']:<9.4f} | {row['s1_pre_mag_ratio_mean']:<8.4f} | {row['s1_pre_r_channel_mean']:<7.4f} | {row['s1_int_alpha_mean']:<8.4f} | {row['s0_pre_alpha_mean']:<8.4f}")
    print("=" * 90)

    elapsed = time.time() - t0_start
    print(f"Completed Stage-1 Representation Characterization in {elapsed:.1f}s.")


if __name__ == '__main__':
    main()
