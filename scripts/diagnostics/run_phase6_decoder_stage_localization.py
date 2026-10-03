#!/usr/bin/env python3
"""
scripts/diagnostics/run_phase6_decoder_stage_localization.py

Phase 6 Diagnostic: Decoder-Stage False Bridge Localization & Progression.
Target Cohort: Exactly the 43 wider-gap bridge events (D_gap > 5 px) from Phase 6 Diagnostic D.
Evaluates Candidate B on Crack500 validation set (Setting A, Tile 448).

Stages Hooked:
1. Bottleneck (14x14) - deep ViT/CNN representation
2. DecoderBlock 0 (28x28) - 1st upsampling stage (skips Stage 2, 192 ch)
3. DecoderBlock 1 (56x56) - 2nd upsampling stage (skips Stage 1, 96 ch)
4. DecoderBlock 2 (112x112) - shallowest decoder stage (skips Stage 0, 48 ch)
5. Segmentation Head (112x112) - pre-upsample 1x1 conv logits
6. Final Logits (448x448) - 4x bilinear upsampled logits

Control Group:
- Matched True Crack Continuations / Junctions within the same image with matching length (~D_gap).

Outputs:
- results/diagnostics/decoder_localization_43events/decoder_stage_measurements_43events.csv
- results/diagnostics/decoder_localization_43events/decoder_stage_summary.csv
- results/diagnostics/decoder_localization_43events/decoder_stage_summary.json
- results/diagnostics/decoder_localization_43events/visualizations/*.png
- results/diagnostics/decoder_localization_43events/DECODER_STAGE_LOCALIZATION_REPORT.md
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

    # Sample candidates to find pair with distance closest to target_length_px
    # Pick a reference point near median
    np.random.seed(42)
    sample_indices = np.random.choice(len(pts), size=min(100, len(pts)), replace=False)
    
    best_pair = None
    best_diff = float('inf')

    for idx in sample_indices:
        p0 = pts[idx]
        dists = np.linalg.norm(pts - p0, axis=1)
        # We want points with dist close to target_length_px
        diffs = np.abs(dists - target_length_px)
        min_i = np.argmin(diffs)
        if diffs[min_i] < best_diff and dists[min_i] > 3.0:
            # Check if straight/dilated line between p0 and p1 is mostly crack
            p1 = pts[min_i]
            line_mask = np.zeros_like(target_bin)
            cv2.line(line_mask, (int(p0[1]), int(p0[0])), (int(p1[1]), int(p1[0])), 1, thickness=3)
            # Crack overlap ratio
            overlap = np.sum((line_mask == 1) & (target_bin == 1)) / max(np.sum(line_mask == 1), 1)
            if overlap > 0.65:
                best_diff = diffs[min_i]
                best_pair = (p0, p1)
                if best_diff < 1.0:
                    break

    if best_pair is None:
        # Fallback to closest pair
        p0 = pts[len(pts) // 2]
        dists = np.linalg.norm(pts - p0, axis=1)
        min_i = np.argmin(np.abs(dists - target_length_px))
        best_pair = (p0, pts[min_i])

    p0, p1 = best_pair
    line_mask = np.zeros_like(target_bin)
    cv2.line(line_mask, (int(p0[1]), int(p0[0])), (int(p1[1]), int(p1[0])), 1, thickness=3)
    ctrl_corridor = (line_mask == 1) & (target_bin == 1)

    # Endpoints masks (radius 3 around p0 and p1)
    end_mask = np.zeros_like(target_bin)
    cv2.circle(end_mask, (int(p0[1]), int(p0[0])), 3, 1, -1)
    cv2.circle(end_mask, (int(p1[1]), int(p1[0])), 3, 1, -1)
    ctrl_endpoints = (end_mask == 1) & (target_bin == 1)

    # Local background around control segment
    k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))
    ctrl_ring = cv2.dilate(ctrl_corridor.astype(np.uint8), k_bg) > 0
    ctrl_bg = ctrl_ring & (target_bin == 0)

    return {
        'corridor_mask': ctrl_corridor,
        'endpoints_mask': ctrl_endpoints,
        'bg_mask': ctrl_bg,
        'length_px': float(np.linalg.norm(p0 - p1))
    }


def render_progression_visualization(
    img_rgb: np.ndarray,
    target_bin: np.ndarray,
    pred_bin: np.ndarray,
    stage_maps: Dict[str, np.ndarray],
    corridor_mask: np.ndarray,
    out_path: str,
    event_id: int,
    case_name: str,
    gap_px: float,
    first_emergence: str
):
    """
    Renders multi-stage spatial progression heatmap panel for representative cases:
    1. Raw RGB + GT / Pred overlay
    2. Bottleneck 14x14 heatmap
    3. Decoder 28x28 heatmap
    4. Decoder 56x56 heatmap
    5. Decoder 112x112 heatmap
    6. Head 112x112 / Final 448x448 probability heatmap
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
    crop_pred = pred_bin[ymin:ymax, xmin:xmax]
    crop_corr = corridor_mask[ymin:ymax, xmin:xmax]
    cH, cW = crop_rgb.shape[:2]

    # Panel 1: RGB with overlays
    p1 = crop_rgb.copy()
    p1[(crop_pred == 1) & (crop_gt == 1)] = (0.5 * p1[(crop_pred == 1) & (crop_gt == 1)] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    p1[(crop_pred == 1) & (crop_gt == 0)] = (0.4 * p1[(crop_pred == 1) & (crop_gt == 0)] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
    # Outline corridor in yellow
    contours, _ = cv2.findContours(crop_corr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(p1, contours, -1, (0, 255, 255), 1)

    panels = [p1]
    stage_order = ['bottleneck_14', 'decoder_28', 'decoder_56', 'decoder_112', 'final_448']
    stage_titles = ['1. RGB Overlay', '2. Bottleneck 14x14', '3. Dec 28x28', '4. Dec 56x56', '5. Dec 112x112', '6. Final Prob 448']

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    cv2.putText(p1, stage_titles[0], (4, 12), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    for idx, s_name in enumerate(stage_order):
        act_crop = stage_maps[s_name][ymin:ymax, xmin:xmax]
        # Normalize to 0-255 for colormap
        if s_name == 'final_448':
            norm_val = np.clip(act_crop, 0.0, 1.0)
            norm_u8 = (norm_val * 255).astype(np.uint8)
        else:
            vmin, vmax = np.percentile(act_crop, 5), np.percentile(act_crop, 98)
            norm_val = np.clip((act_crop - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
            norm_u8 = (norm_val * 255).astype(np.uint8)

        heat = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        # Blend slightly with grayscale RGB for anatomical reference
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        blended = (0.65 * heat + 0.35 * gray_3ch).astype(np.uint8)
        cv2.drawContours(blended, contours, -1, (255, 255, 255), 1)

        cv2.putText(blended, stage_titles[idx + 1], (4, 12), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(blended)

    strip = np.hstack(panels)
    banner_h = 24
    banner = np.zeros((banner_h, strip.shape[1], 3), dtype=np.uint8)
    title = f"Event #{event_id:03d} ({case_name}) | Gap: {gap_px:.1f}px | First Emergence: {first_emergence}"
    cv2.putText(banner, title, (10, 16), font, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

    final_img = np.vstack([banner, strip])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    t0_start = time.time()
    print("=" * 80)
    print("PHASE 6 DIAGNOSTIC: DECODER-STAGE LOCALIZATION (43 WIDER-GAP EVENTS)")
    print("Cohort: Gap > 5.0 px from Phase 6 Diagnostic D | Checkpoint: Candidate B")
    print("=" * 80)

    out_dir = os.path.join(project_root, 'results', 'diagnostics', 'decoder_localization_43events')
    viz_dir = os.path.join(out_dir, 'visualizations')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    config_path = os.path.join(project_root, 'results', 'configs', 'b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml')
    ckpt_path = os.path.join(project_root, 'results', 'checkpoints', 'P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = False

    print(f"Device: {device} (CuDNN disabled for numerical stability)")
    print("Loading Candidate B model...")
    model = load_model_from_checkpoint(config_path, ckpt_path, device)
    model.eval()

    # Load exactly the 43 wider gap events from Diagnostic D
    csv_118 = os.path.join(project_root, 'results', 'diagnostics', 'phase6_bottleneck_path', 'bottleneck_path_118events.csv')
    df_118 = pd.read_csv(csv_118)
    df_43 = df_118[df_118['gap_category'] == 'Wider_Gap (> 5 px)'].copy().reset_index(drop=True)
    assert len(df_43) == 43, f"Expected exactly 43 wider gap events, found {len(df_43)}"
    print(f"Cohort loaded: {len(df_43)} events across {df_43['case_name'].nunique()} unique images.")

    data_root = os.path.join(project_root, 'datasets', 'Crack500_ready')
    val_img_dir = os.path.join(data_root, 'val', 'images')
    val_mask_dir = os.path.join(data_root, 'val', 'masks')

    event_measurements = []
    stage_emergence_counter = collections.Counter()
    positive_emergence_counter = collections.Counter()

    # Group events by case_name so we run inference only once per image
    grouped = df_43.groupby('case_name')

    pbar = tqdm(total=len(df_43), desc="Analyzing Decoder Stages")

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

        # Register forward hooks on intermediate decoder stages
        cache = {}
        def hook_s3(m, i, o): cache['s3_14'] = i[0].detach()
        def hook_s2(m, i, o): cache['s2_28'] = o.detach()
        def hook_s1(m, i, o): cache['s1_56'] = o.detach()
        def hook_s0(m, i, o): cache['s0_112'] = o.detach()
        def hook_hd(m, i, o): cache['head_112'] = o.detach()

        h_s3 = model.decoder.decoder_blocks[0].register_forward_hook(hook_s3)
        h_s2 = model.decoder.decoder_blocks[0].register_forward_hook(hook_s2)
        h_s1 = model.decoder.decoder_blocks[1].register_forward_hook(hook_s1)
        h_s0 = model.decoder.decoder_blocks[2].register_forward_hook(hook_s0)
        h_hd = model.decoder.segmentation_head.register_forward_hook(hook_hd)

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
        with torch.no_grad():
            logits_final = model(batch)

        h_s3.remove(); h_s2.remove(); h_s1.remove(); h_s0.remove(); h_hd.remove()

        # Reconstruct full-image feature maps for each stage
        stages_info = [
            ('bottleneck_14', cache['s3_14'], 32),
            ('decoder_28', cache['s2_28'], 16),
            ('decoder_56', cache['s1_56'], 8),
            ('decoder_112', cache['s0_112'], 4),
            ('head_112', cache['head_112'], 4),
            ('final_448', logits_final, 1)
        ]

        # Stitched 448 prediction
        logits_448_np = np.zeros((pH, pW), dtype=np.float32)
        for j, (y, x) in enumerate(coords):
            logits_448_np[y:y+448, x:x+448] = logits_final[j, 0].cpu().numpy()
        logits_448_np = logits_448_np[:H, :W]
        prob_448_np = 1.0 / (1.0 + np.exp(-logits_448_np.astype(np.float64)))
        pred_bin = (logits_448_np > 0.0).astype(np.uint8)

        num_gt_cc, gt_labels = cv2.connectedComponents(target_bin, connectivity=8)
        num_pred_cc, pred_labels = cv2.connectedComponents(pred_bin, connectivity=8)

        # Bilinearly interpolate all feature maps to (H, W)
        interpolated_stage_maps = {}
        interpolated_feature_tensors = {}

        for st_name, st_tensor, down_scale in stages_info:
            C_dim = st_tensor.shape[1]
            tH, tW = 448 // down_scale, 448 // down_scale
            stitched = np.zeros((C_dim, pH // down_scale, pW // down_scale), dtype=np.float32)
            for j, (y, x) in enumerate(coords):
                sy, sx = y // down_scale, x // down_scale
                stitched[:, sy:sy+tH, sx:sx+tW] = st_tensor[j].cpu().numpy()
            stitched = stitched[:, :H // down_scale, :W // down_scale]

            # Convert to torch tensor and interpolate to (H, W)
            s_tensor = torch.from_numpy(stitched).unsqueeze(0)
            s_448 = F.interpolate(s_tensor, size=(H, W), mode='bilinear', align_corners=False).squeeze(0).numpy()
            interpolated_feature_tensors[st_name] = s_448

            # Spatial energy / activation map
            if st_name in ('head_112', 'final_448'):
                interpolated_stage_maps[st_name] = s_448[0] # raw logit
            else:
                interpolated_stage_maps[st_name] = np.mean(np.abs(s_448), axis=0)

        # Process each bridge event in this image
        for _, ev_row in group_events.iterrows():
            ev_id = int(ev_row['event_id'])
            p_id = int(ev_row['pred_cc_id'])
            gA = int(ev_row['primary_gt_A'])
            gB = int(ev_row['primary_gt_B'])
            d_gap = float(ev_row['d_gap_px'])

            mask_A = (gt_labels == gA) & (pred_labels == p_id)
            mask_B = (gt_labels == gB) & (pred_labels == p_id)
            crack_endpoints = mask_A | mask_B
            cc_fp = (pred_labels == p_id) & (target_bin == 0)

            # Isolate false corridor
            dt_A = distance_transform_edt(gt_labels != gA)
            r = max(int(np.ceil(d_gap / 2.0)) + 2, 3)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
            dil_A = cv2.dilate((gt_labels == gA).astype(np.uint8), k)
            dil_B = cv2.dilate((gt_labels == gB).astype(np.uint8), k)
            corridor = (dil_A > 0) & (dil_B > 0) & cc_fp
            if np.sum(corridor) == 0:
                corridor = cc_fp

            # Local background (ring within 24 px excluding crack & pred)
            k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))
            local_ring = cv2.dilate(corridor.astype(np.uint8), k_bg) > 0
            local_bg = local_ring & (target_bin == 0) & (pred_bin == 0)
            if np.sum(local_bg) == 0:
                local_bg = (target_bin == 0) & (pred_bin == 0)

            # Matched True Crack Control
            ctrl_dict = find_matched_control_segment(target_bin, d_gap)

            record = {
                'event_id': ev_id,
                'case_name': case_name,
                'd_gap_px': round(d_gap, 2),
                'p_bottleneck_final': float(ev_row['p_bottleneck']),
                'n_corridor_pixels': int(np.sum(corridor)),
                'n_crack_pixels': int(np.sum(crack_endpoints)),
                'n_bg_pixels': int(np.sum(local_bg)),
            }

            first_crack_dominated_stage = "none"
            first_positive_logit_stage = "none"

            # Evaluate each stage
            for st_name, _, _ in stages_info:
                feat = interpolated_feature_tensors[st_name]
                act_map = interpolated_stage_maps[st_name]

                # False Bridge Measurements
                e_corr = float(np.mean(act_map[corridor]))
                e_crack = float(np.mean(act_map[crack_endpoints]))
                e_bg = float(np.mean(act_map[local_bg]))
                contrast = float((e_corr - e_bg) / max(e_crack - e_bg, 1e-6))

                # Semantic Cosine Similarity
                v_corr = np.mean(feat[:, corridor], axis=1)
                v_crack = np.mean(feat[:, crack_endpoints], axis=1)
                v_bg = np.mean(feat[:, local_bg], axis=1)

                sim_crack = float(np.dot(v_corr, v_crack) / (np.linalg.norm(v_corr) * np.linalg.norm(v_crack) + 1e-8))
                sim_bg = float(np.dot(v_corr, v_bg) / (np.linalg.norm(v_corr) * np.linalg.norm(v_bg) + 1e-8))

                record[f'{st_name}_energy_corr'] = round(e_corr, 4)
                record[f'{st_name}_energy_crack'] = round(e_crack, 4)
                record[f'{st_name}_energy_bg'] = round(e_bg, 4)
                record[f'{st_name}_contrast'] = round(contrast, 4)
                record[f'{st_name}_sim_crack'] = round(sim_crack, 4)
                record[f'{st_name}_sim_bg'] = round(sim_bg, 4)

                # Check if crack-dominated at this stage
                if first_crack_dominated_stage == "none" and sim_crack > sim_bg and contrast >= 0.50:
                    first_crack_dominated_stage = st_name

                # Check if positive logit / prob > 0.5
                if st_name in ('head_112', 'final_448'):
                    logit_mean = float(np.mean(feat[0, corridor]))
                    prob_mean = float(1.0 / (1.0 + np.exp(-logit_mean)))
                    record[f'{st_name}_logit_corr'] = round(logit_mean, 4)
                    record[f'{st_name}_prob_corr'] = round(prob_mean, 4)
                    if first_positive_logit_stage == "none" and logit_mean > 0.0:
                        first_positive_logit_stage = st_name

                # Matched True Crack Control Measurements
                if ctrl_dict is not None:
                    c_corr = ctrl_dict['corridor_mask']
                    c_end = ctrl_dict['endpoints_mask']
                    c_bg = ctrl_dict['bg_mask']
                    if np.sum(c_corr) > 0 and np.sum(c_end) > 0 and np.sum(c_bg) > 0:
                        ce_corr = float(np.mean(act_map[c_corr]))
                        ce_crack = float(np.mean(act_map[c_end]))
                        ce_bg = float(np.mean(act_map[c_bg]))
                        c_contrast = float((ce_corr - ce_bg) / max(ce_crack - ce_bg, 1e-6))

                        cv_corr = np.mean(feat[:, c_corr], axis=1)
                        cv_crack = np.mean(feat[:, c_end], axis=1)
                        cv_bg = np.mean(feat[:, c_bg], axis=1)
                        c_sim_crack = float(np.dot(cv_corr, cv_crack) / (np.linalg.norm(cv_corr) * np.linalg.norm(cv_crack) + 1e-8))
                        c_sim_bg = float(np.dot(cv_corr, cv_bg) / (np.linalg.norm(cv_corr) * np.linalg.norm(cv_bg) + 1e-8))

                        record[f'{st_name}_ctrl_contrast'] = round(c_contrast, 4)
                        record[f'{st_name}_ctrl_sim_crack'] = round(c_sim_crack, 4)
                        record[f'{st_name}_ctrl_sim_bg'] = round(c_sim_bg, 4)

            record['first_crack_dominated_stage'] = first_crack_dominated_stage
            record['first_positive_logit_stage'] = first_positive_logit_stage
            stage_emergence_counter[first_crack_dominated_stage] += 1
            positive_emergence_counter[first_positive_logit_stage] += 1
            event_measurements.append(record)

            # Render visualization for select representative cases (first 5 events)
            if ev_id in [2, 3, 4, 5, 10]:
                viz_name = f"event_{ev_id:03d}_{case_name}_gap{d_gap:.1f}px.png"
                viz_path = os.path.join(viz_dir, viz_name)
                # For visualization, compute probability maps for final
                stage_maps_for_viz = dict(interpolated_stage_maps)
                stage_maps_for_viz['final_448'] = prob_448_np
                render_progression_visualization(
                    img_rgb=img,
                    target_bin=target_bin,
                    pred_bin=pred_bin,
                    stage_maps=stage_maps_for_viz,
                    corridor_mask=corridor,
                    out_path=viz_path,
                    event_id=ev_id,
                    case_name=case_name,
                    gap_px=d_gap,
                    first_emergence=first_crack_dominated_stage
                )

            pbar.update(1)

    pbar.close()

    # 1. Save raw per-event measurements CSV
    df_results = pd.DataFrame(event_measurements)
    csv_raw = os.path.join(out_dir, 'decoder_stage_measurements_43events.csv')
    df_results.to_csv(csv_raw, index=False)
    print(f"Saved: {csv_raw}")

    # 2. Stage-by-Stage Summary Table
    stage_names = ['bottleneck_14', 'decoder_28', 'decoder_56', 'decoder_112', 'head_112', 'final_448']
    summary_rows = []

    for s in stage_names:
        c_series = df_results[f'{s}_contrast']
        sim_crack = df_results[f'{s}_sim_crack']
        sim_bg = df_results[f'{s}_sim_bg']

        ctrl_contrast = df_results.get(f'{s}_ctrl_contrast', pd.Series([]))
        ctrl_sim_crack = df_results.get(f'{s}_ctrl_sim_crack', pd.Series([]))

        n_first = stage_emergence_counter.get(s, 0)
        pct_first = (n_first / 43.0) * 100.0

        row_dict = {
            'decoder_stage': s,
            'first_emergence_count': n_first,
            'first_emergence_pct': round(pct_first, 1),
            'median_contrast_bridge': round(float(c_series.median()), 4),
            'p25_contrast_bridge': round(float(c_series.quantile(0.25)), 4),
            'p75_contrast_bridge': round(float(c_series.quantile(0.75)), 4),
            'mean_contrast_bridge': round(float(c_series.mean()), 4),
            'median_sim_to_crack': round(float(sim_crack.median()), 4),
            'median_sim_to_bg': round(float(sim_bg.median()), 4),
            'median_contrast_ctrl': round(float(ctrl_contrast.median()), 4) if len(ctrl_contrast) > 0 else np.nan,
            'median_sim_ctrl_to_crack': round(float(ctrl_sim_crack.median()), 4) if len(ctrl_sim_crack) > 0 else np.nan,
        }
        if f'{s}_logit_corr' in df_results:
            row_dict['median_logit_corr'] = round(float(df_results[f'{s}_logit_corr'].median()), 4)
            row_dict['median_prob_corr'] = round(float(df_results[f'{s}_prob_corr'].median()), 4)
        summary_rows.append(row_dict)

    df_summary = pd.DataFrame(summary_rows)
    csv_summary = os.path.join(out_dir, 'decoder_stage_summary.csv')
    df_summary.to_csv(csv_summary, index=False)
    print(f"Saved: {csv_summary}")

    # 3. Save JSON Summary
    summary_json = {
        'metadata': {
            'target_model': 'Candidate B (Baseline B2)',
            'cohort_size': 43,
            'gap_criteria': 'D_gap > 5.0 px',
            'execution_time_seconds': round(time.time() - t0_start, 2)
        },
        'first_emergence_distribution': dict(stage_emergence_counter),
        'positive_logit_emergence_distribution': dict(positive_emergence_counter),
        'stage_summary_table': summary_rows
    }
    json_path = os.path.join(out_dir, 'decoder_stage_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=2)
    print(f"Saved: {json_path}")

    print("\n" + "=" * 80)
    print("DECODER STAGE LOCALIZATION SUMMARY (N = 43 EVENTS):")
    print("=" * 80)
    print(df_summary[['decoder_stage', 'first_emergence_count', 'first_emergence_pct', 'median_contrast_bridge', 'median_sim_to_crack', 'median_sim_to_bg']].to_string())
    print("\nFirst Positive Logit Distribution:")
    for k, v in positive_emergence_counter.items():
        print(f"  {k:15s}: {v} / 43 ({v/43*100:.1f}%)")
    print(f"\nExecution finished in {time.time() - t0_start:.1f}s")


if __name__ == '__main__':
    main()
