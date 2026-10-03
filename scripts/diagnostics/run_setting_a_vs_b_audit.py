#!/usr/bin/env python3
"""
scripts/diagnostics/run_setting_a_vs_b_audit.py

Phase 6 Audit: Setting A (non-overlap 448x448) vs Setting B (50% overlap 448x448, stride 224, avg probs)
Evaluates Candidate B checkpoint on all 348 validation images of Crack500.

Determines:
- Does Candidate B's false bridge significantly depend on non-overlap tile boundary?
- Bridge transition: A -> B (cured vs persistent vs created)
- Tile boundary proximity analysis for all A bridge events
- Probability difference map: near tile boundaries vs elsewhere
- Empirical view count distribution in Setting B
"""

import collections
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import torch
from tqdm import tqdm

# Ensure SAGE_LITE and project root are on sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sage_lite_dir = os.path.join(project_root, "SAGE_LITE")
for p in [project_root, sage_lite_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from tools.run_phase6_c_topology_diagnostic import load_model_from_checkpoint
from scripts.evaluate_crack_official import (
    predict_full_image_tiling_setting_a,
    predict_full_image_tiling_setting_b
)


def compute_topology_and_pixel_metrics(pred_bin: np.ndarray, target_bin: np.ndarray) -> Dict[str, Any]:
    """
    Computes standard pixel metrics and topology metrics:
    - dice, precision, recall, tp, fp, fn
    - gt_cc, pred_cc
    - bridge_events, merged_gt_count, has_bridge
    - break_events (fragmented_gt_components), extra_fragments, has_break
    - cldice, tprec, tsens
    """
    assert pred_bin.shape == target_bin.shape, f"Shape mismatch: {pred_bin.shape} vs {target_bin.shape}"
    H, W = target_bin.shape[:2]

    num_gt_cc, gt_labels = cv2.connectedComponents(target_bin.astype(np.uint8), connectivity=8)
    num_pred_cc, pred_labels = cv2.connectedComponents(pred_bin.astype(np.uint8), connectivity=8)

    gt_cc = int(max(num_gt_cc - 1, 0))
    pred_cc = int(max(num_pred_cc - 1, 0))

    # Pixel metrics
    tp = int(np.sum((pred_bin == 1) & (target_bin == 1)))
    fp = int(np.sum((pred_bin == 1) & (target_bin == 0)))
    fn = int(np.sum((pred_bin == 0) & (target_bin == 1)))
    dice = float((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8))
    precision = float(tp / (tp + fp + 1e-8))
    recall = float(tp / (tp + fn + 1e-8))

    # 1. False Bridge / Merge
    bridge_events = 0
    merged_gt_set = set()
    for p_id in range(1, pred_cc + 1):
        overlapping_gt = np.unique(gt_labels[pred_labels == p_id])
        overlapping_gt = overlapping_gt[overlapping_gt > 0]
        if len(overlapping_gt) >= 2:
            bridge_events += 1
            for g_id in overlapping_gt:
                merged_gt_set.add(int(g_id))
    has_bridge = int(bridge_events > 0)

    # 2. Breakage / Fragmentation
    fragmented_gt_components = 0
    extra_pred_fragments = 0
    for g_id in range(1, gt_cc + 1):
        overlapping_pred = np.unique(pred_labels[gt_labels == g_id])
        overlapping_pred = overlapping_pred[overlapping_pred > 0]
        if len(overlapping_pred) >= 2:
            fragmented_gt_components += 1
            extra_pred_fragments += int(len(overlapping_pred) - 1)
    has_break = int(fragmented_gt_components > 0)

    # 3. Centerline Dice (clDice)
    if np.sum(target_bin) == 0 and np.sum(pred_bin) == 0:
        cldice, tprec, tsens = 1.0, 1.0, 1.0
    elif np.sum(target_bin) == 0 or np.sum(pred_bin) == 0:
        cldice, tprec, tsens = 0.0, 0.0, 0.0
    else:
        s_target = skeletonize(target_bin.astype(bool))
        s_pred = skeletonize(pred_bin.astype(bool))
        s_gt_sum = np.sum(s_target)
        s_pred_sum = np.sum(s_pred)
        if s_gt_sum == 0 or s_pred_sum == 0:
            cldice, tprec, tsens = 0.0, 0.0, 0.0
        else:
            tprec_val = np.sum(s_pred & (target_bin == 1)) / (s_pred_sum + 1e-8)
            tsens_val = np.sum(s_target & (pred_bin == 1)) / (s_gt_sum + 1e-8)
            tprec = float(tprec_val)
            tsens = float(tsens_val)
            cldice = float((2.0 * tprec * tsens) / (tprec + tsens + 1e-8))

    return {
        'gt_cc': gt_cc,
        'pred_cc': pred_cc,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'bridge_events': bridge_events,
        'has_bridge': has_bridge,
        'merged_gt_count': len(merged_gt_set),
        'break_events': fragmented_gt_components,
        'has_break': has_break,
        'extra_fragments': extra_pred_fragments,
        'cldice': cldice,
        'tprec': tprec,
        'tsens': tsens
    }


def compute_setting_b_count_map(H: int, W: int, tile_size: int = 448, stride: int = 224) -> np.ndarray:
    """
    Computes the exact number_of_tile_views_per_pixel count_map for Setting B.
    """
    if H < tile_size:
        pad_h = tile_size - H
    else:
        rem_h = (H - tile_size) % stride
        pad_h = (stride - rem_h) % stride if rem_h != 0 else 0

    if W < tile_size:
        pad_w = tile_size - W
    else:
        rem_w = (W - tile_size) % stride
        pad_w = (stride - rem_w) % stride if rem_w != 0 else 0

    pH = H + pad_h
    pW = W + pad_w

    count_map = np.zeros((pH, pW), dtype=np.int32)
    for y in range(0, pH - tile_size + 1, stride):
        for x in range(0, pW - tile_size + 1, stride):
            count_map[y:y+tile_size, x:x+tile_size] += 1

    return count_map[:H, :W]


def isolate_bridge_corridors_and_distance(
    pred_bin_a: np.ndarray,
    target_bin: np.ndarray,
    tile_size: int = 448
) -> List[Dict[str, Any]]:
    """
    Isolates each bridge event in Setting A and computes distance to Setting A tile boundaries.
    Tile boundary lines in Setting A:
      x = 448 * k
      y = 448 * k
    Inside image [0, H) x [0, W).
    """
    H, W = target_bin.shape[:2]
    num_gt_cc, gt_labels = cv2.connectedComponents(target_bin.astype(np.uint8), connectivity=8)
    num_pred_cc, pred_labels = cv2.connectedComponents(pred_bin_a.astype(np.uint8), connectivity=8)

    # Precompute distance to tile boundary grids
    # 1. Any tile boundary: x = 0, 448, 896... and y = 0, 448, 896...
    # 2. Internal tile boundaries: x = 448, 896... and y = 448, 896... (where 0 < coord < H or W)
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # Distance to nearest 448*k in x and y
    dx_any = np.minimum(x_coords % tile_size, tile_size - (x_coords % tile_size))
    dy_any = np.minimum(y_coords % tile_size, tile_size - (y_coords % tile_size))
    dist_any_boundary = np.minimum(dx_any, dy_any).astype(np.float32)

    # Internal tile boundaries
    internal_x_lines = [k * tile_size for k in range(1, (W // tile_size) + 1) if k * tile_size < W]
    internal_y_lines = [k * tile_size for k in range(1, (H // tile_size) + 1) if k * tile_size < H]

    if internal_x_lines or internal_y_lines:
        dist_x_internal = np.full((H, W), 999999.0, dtype=np.float32)
        for xl in internal_x_lines:
            dist_x_internal = np.minimum(dist_x_internal, np.abs(x_coords - xl))

        dist_y_internal = np.full((H, W), 999999.0, dtype=np.float32)
        for yl in internal_y_lines:
            dist_y_internal = np.minimum(dist_y_internal, np.abs(y_coords - yl))

        dist_internal_boundary = np.minimum(dist_x_internal, dist_y_internal).astype(np.float32)
    else:
        dist_internal_boundary = np.full((H, W), 999999.0, dtype=np.float32)

    events_info = []
    event_counter = 0

    for p_id in range(1, num_pred_cc):
        p_mask = (pred_labels == p_id)
        overlapping_gt = np.unique(gt_labels[p_mask])
        overlapping_gt = [int(g) for g in overlapping_gt if g > 0]
        if len(overlapping_gt) < 2:
            continue

        event_counter += 1
        # Bridge False Positive corridor pixels: inside this CC but target == 0
        cc_fp = p_mask & (target_bin == 0)

        # Pairwise corridors to find closest pair and corridor neck
        dt_dict = {g: distance_transform_edt(gt_labels != g) for g in overlapping_gt}
        min_gap = float('inf')
        pair_min = (overlapping_gt[0], overlapping_gt[1])
        for i in range(len(overlapping_gt)):
            gi = overlapping_gt[i]
            for j in range(i + 1, len(overlapping_gt)):
                gj = overlapping_gt[j]
                d = float(np.min(dt_dict[gi][gt_labels == gj]))
                if d < min_gap:
                    min_gap = d
                    pair_min = (gi, gj)

        gA, gB = pair_min
        # Isolate neck via dilation
        r = max(int(np.ceil(min_gap / 2.0)) + 2, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        dil_A = cv2.dilate((gt_labels == gA).astype(np.uint8), kernel)
        dil_B = cv2.dilate((gt_labels == gB).astype(np.uint8), kernel)
        corridor = (dil_A > 0) & (dil_B > 0) & cc_fp
        if np.sum(corridor) == 0:
            corridor = cc_fp

        corr_pts = np.argwhere(corridor)
        if len(corr_pts) > 0:
            c_ymin, c_xmin = np.min(corr_pts, axis=0)
            c_ymax, c_xmax = np.max(corr_pts, axis=0)
            c_center_y, c_center_x = float(np.mean(corr_pts[:, 0])), float(np.mean(corr_pts[:, 1]))

            # Distances from corridor pixels to tile boundaries
            min_dist_any = float(np.min(dist_any_boundary[corridor]))
            mean_dist_any = float(np.mean(dist_any_boundary[corridor]))
            min_dist_internal = float(np.min(dist_internal_boundary[corridor]))
            mean_dist_internal = float(np.mean(dist_internal_boundary[corridor]))
        else:
            min_dist_any = np.nan
            mean_dist_any = np.nan
            min_dist_internal = np.nan
            mean_dist_internal = np.nan
            c_center_y, c_center_x = np.nan, np.nan

        # Stratify according to protocol: 0-16, 16-32, 32-64, >64
        def stratify_dist(d):
            if np.isnan(d) or d > 900000:
                return "no_internal_boundary"
            if d <= 16.0:
                return "0-16 px"
            elif d <= 32.0:
                return "16-32 px"
            elif d <= 64.0:
                return "32-64 px"
            else:
                return ">64 px"

        events_info.append({
            'pred_cc_id': p_id,
            'event_index_in_img': event_counter,
            'num_merged_gt': len(overlapping_gt),
            'merged_gt_ids': str(overlapping_gt),
            'primary_gt_A': gA,
            'primary_gt_B': gB,
            'd_gap_px': round(min_gap, 2),
            'n_corridor_pixels': int(np.sum(corridor)),
            'corridor_center_y': round(c_center_y, 1),
            'corridor_center_x': round(c_center_x, 1),
            'min_dist_to_internal_tile_boundary': round(min_dist_internal, 2) if min_dist_internal < 900000 else np.nan,
            'mean_dist_to_internal_tile_boundary': round(mean_dist_internal, 2) if mean_dist_internal < 900000 else np.nan,
            'min_dist_to_any_tile_boundary': round(min_dist_any, 2),
            'mean_dist_to_any_tile_boundary': round(mean_dist_any, 2),
            'bin_internal_boundary': stratify_dist(min_dist_internal),
            'bin_any_boundary': stratify_dist(min_dist_any)
        })

    return events_info


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    t0_start = time.time()
    print("=" * 80)
    print("PHASE 6 AUDIT: SETTING A (NON-OVERLAP) VS SETTING B (50% OVERLAP BLENDING)")
    print("Dataset: Crack500 Val (N=348), Setting A: 448x448, Setting B: stride 224, blend_mode='probs'")
    print("=" * 80)

    out_dir = os.path.join(project_root, 'results', 'diagnostics', 'setting_a_vs_b')
    os.makedirs(out_dir, exist_ok=True)

    config_path = os.path.join(project_root, 'results', 'configs', 'b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml')
    ckpt_path = os.path.join(project_root, 'results', 'checkpoints', 'P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = False

    print(f"Device: {device} (CuDNN disabled for numerical stability)")
    print("Loading Candidate B model...")
    model = load_model_from_checkpoint(config_path, ckpt_path, device)
    model.eval()

    # Load consensus / cohort metadata if available
    consensus_csv = os.path.join(project_root, 'results', 'diagnostics', 'phase6_bridge_localization', 'bridge_consensus_per_image.csv')
    df_consensus = pd.read_csv(consensus_csv) if os.path.exists(consensus_csv) else None

    gate_csv = os.path.join(project_root, 'results', 'diagnostics', 'phase6_t2_spatial_gateability', 't2_spatial_statistics_per_sample.csv')
    df_gate = pd.read_csv(gate_csv) if os.path.exists(gate_csv) else None

    # Load 348 validation images
    data_root = os.path.join(project_root, 'datasets', 'Crack500_ready')
    val_img_dir = os.path.join(data_root, 'val', 'images')
    val_mask_dir = os.path.join(data_root, 'val', 'masks')

    img_files = sorted(os.listdir(val_img_dir))
    case_names = [os.path.splitext(f)[0] for f in img_files if f.lower().endswith(('.jpg', '.png'))]
    assert len(case_names) == 348, f"Expected 348 validation images, got {len(case_names)}"
    print(f"Loaded {len(case_names)} validation samples.")

    per_image_rows = []
    tile_boundary_rows = []
    view_count_counter = collections.Counter()
    all_abs_diff_pixels = []
    bridge110_abs_diff_pixels = []

    # Map difference near boundary vs far
    abs_diff_near_boundary = []
    abs_diff_far_boundary = []

    for case_name in tqdm(case_names, desc="Evaluating Setting A vs Setting B"):
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

        # 1. Setting A Inference
        with torch.no_grad():
            logits_a = predict_full_image_tiling_setting_a(model, img, device, tile_size=448, batch_size=8)
        prob_a = 1.0 / (1.0 + np.exp(-logits_a.astype(np.float64)))
        pred_bin_a = (logits_a > 0.0).astype(np.uint8)[:H, :W]
        prob_a = prob_a[:H, :W]

        # 2. Setting B Inference (stride 224, blend_mode='probs')
        with torch.no_grad():
            prob_b = predict_full_image_tiling_setting_b(
                model, img, device, tile_size=448, stride=224, batch_size=8, blend_mode='probs'
            )
        pred_bin_b = (prob_b > 0.5).astype(np.uint8)[:H, :W]
        prob_b = prob_b[:H, :W]

        # 3. View Count Map for Setting B
        count_map = compute_setting_b_count_map(H, W, tile_size=448, stride=224)
        for v in count_map.ravel():
            view_count_counter[int(v)] += 1

        # 4. Topology & pixel metrics for Setting A and Setting B
        metrics_a = compute_topology_and_pixel_metrics(pred_bin_a, target_bin)
        metrics_b = compute_topology_and_pixel_metrics(pred_bin_b, target_bin)

        # 5. Probability difference
        abs_diff = np.abs(prob_a - prob_b)
        mean_abs_diff_img = float(np.mean(abs_diff))

        # Check internal boundary distance map for image
        internal_x = [k * 448 for k in range(1, (W // 448) + 1) if k * 448 < W]
        internal_y = [k * 448 for k in range(1, (H // 448) + 1) if k * 448 < H]
        y_g, x_g = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        if internal_x or internal_y:
            d_x = np.full((H, W), 999999.0, dtype=np.float32)
            for xl in internal_x:
                d_x = np.minimum(d_x, np.abs(x_g - xl))
            d_y = np.full((H, W), 999999.0, dtype=np.float32)
            for yl in internal_y:
                d_y = np.minimum(d_y, np.abs(y_g - yl))
            d_internal = np.minimum(d_x, d_y)
            near_mask = (d_internal <= 32.0)
            abs_diff_near_boundary.extend(abs_diff[near_mask].tolist())
            abs_diff_far_boundary.extend(abs_diff[~near_mask].tolist())

        all_abs_diff_pixels.extend(abs_diff.ravel().tolist())
        if metrics_a['bridge_events'] > 0:
            bridge110_abs_diff_pixels.extend(abs_diff.ravel().tolist())

        # 6. Status classification
        if metrics_a['has_bridge'] and metrics_b['has_bridge']:
            trans_status = "A_bridge_to_B_bridge"
        elif metrics_a['has_bridge'] and not metrics_b['has_bridge']:
            trans_status = "A_bridge_to_B_cured"
        elif not metrics_a['has_bridge'] and metrics_b['has_bridge']:
            trans_status = "A_clean_to_B_bridge"
        else:
            trans_status = "Clean_A_and_B"

        # Cohort mapping
        c7 = np.nan
        subgroup = "Unassigned"
        if df_consensus is not None:
            match = df_consensus[df_consensus['image_id'] == case_name]
            if len(match) > 0:
                c7 = int(match['n_models_with_bridge'].values[0] == 7)
        if df_gate is not None:
            match = df_gate[df_gate['image_id'] == case_name]
            if len(match) > 0:
                subgroup = match['subgroup'].values[0]

        row_record = {
            'image_id': case_name,
            'cohort_subgroup': subgroup,
            'is_consensus_7of7': c7,
            'image_h': H,
            'image_w': W,
            # Setting A
            'A_bridge_status': metrics_a['has_bridge'],
            'A_bridge_events': metrics_a['bridge_events'],
            'A_break_events': metrics_a['break_events'],
            'A_dice': round(metrics_a['dice'], 4),
            'A_recall': round(metrics_a['recall'], 4),
            'A_precision': round(metrics_a['precision'], 4),
            'A_cldice': round(metrics_a['cldice'], 4),
            # Setting B
            'B_bridge_status': metrics_b['has_bridge'],
            'B_bridge_events': metrics_b['bridge_events'],
            'B_break_events': metrics_b['break_events'],
            'B_dice': round(metrics_b['dice'], 4),
            'B_recall': round(metrics_b['recall'], 4),
            'B_precision': round(metrics_b['precision'], 4),
            'B_cldice': round(metrics_b['cldice'], 4),
            # Deltas (B - A)
            'delta_bridge_events': metrics_b['bridge_events'] - metrics_a['bridge_events'],
            'delta_break_events': metrics_b['break_events'] - metrics_a['break_events'],
            'delta_dice': round(metrics_b['dice'] - metrics_a['dice'], 4),
            'delta_recall': round(metrics_b['recall'] - metrics_a['recall'], 4),
            'delta_precision': round(metrics_b['precision'] - metrics_a['precision'], 4),
            'delta_cldice': round(metrics_b['cldice'] - metrics_a['cldice'], 4),
            'transition_category': trans_status,
            'mean_abs_prob_diff': round(mean_abs_diff_img, 4)
        }
        per_image_rows.append(row_record)

        # 7. If this image has A-bridge events, extract per-event boundary distance
        if metrics_a['bridge_events'] > 0:
            events_info = isolate_bridge_corridors_and_distance(pred_bin_a, target_bin, tile_size=448)
            for ev in events_info:
                tile_boundary_rows.append({
                    'image_id': case_name,
                    'transition_category': trans_status,
                    'is_image_cured_by_b': int(trans_status == "A_bridge_to_B_cured"),
                    'A_bridge_events': metrics_a['bridge_events'],
                    'B_bridge_events': metrics_b['bridge_events'],
                    **ev
                })

    df_per_image = pd.DataFrame(per_image_rows)
    df_boundary = pd.DataFrame(tile_boundary_rows)

    # 1. Save per-image CSV
    csv_per_image = os.path.join(out_dir, 'setting_a_vs_b_per_image.csv')
    df_per_image.to_csv(csv_per_image, index=False)
    print(f"Saved: {csv_per_image}")

    # 2. Save tile boundary analysis CSV
    csv_boundary = os.path.join(out_dir, 'tile_boundary_bridge_analysis.csv')
    df_boundary.to_csv(csv_boundary, index=False)
    print(f"Saved: {csv_boundary}")

    # 3. Bridge Transition Summary Table
    # A bridge cohort: 110 images, 118 events
    a_bridge_mask = df_per_image['A_bridge_status'] == 1
    total_a_bridge_imgs = int(a_bridge_mask.sum())
    total_a_bridge_evs = int(df_per_image['A_bridge_events'].sum())

    total_b_bridge_imgs = int((df_per_image['B_bridge_status'] == 1).sum())
    total_b_bridge_evs = int(df_per_image['B_bridge_events'].sum())

    cured_imgs = int(((df_per_image['A_bridge_status'] == 1) & (df_per_image['B_bridge_status'] == 0)).sum())
    persistent_imgs = int(((df_per_image['A_bridge_status'] == 1) & (df_per_image['B_bridge_status'] == 1)).sum())
    created_imgs = int(((df_per_image['A_bridge_status'] == 0) & (df_per_image['B_bridge_status'] == 1)).sum())

    # Event level transition on the 110 A-bridge images
    df_a_bridge = df_per_image[a_bridge_mask]
    a_events_in_cured_imgs = int(df_per_image[df_per_image['transition_category'] == "A_bridge_to_B_cured"]['A_bridge_events'].sum())
    a_events_in_persistent_imgs = int(df_per_image[df_per_image['transition_category'] == "A_bridge_to_B_bridge"]['A_bridge_events'].sum())
    b_events_in_persistent_imgs = int(df_per_image[df_per_image['transition_category'] == "A_bridge_to_B_bridge"]['B_bridge_events'].sum())
    b_events_in_created_imgs = int(df_per_image[df_per_image['transition_category'] == "A_clean_to_B_bridge"]['B_bridge_events'].sum())

    # Transition dataframe
    trans_summary = [
        {'Metric': 'Bridge Images (Total)', 'Setting_A': total_a_bridge_imgs, 'Setting_B': total_b_bridge_imgs, 'Delta_B_minus_A': total_b_bridge_imgs - total_a_bridge_imgs},
        {'Metric': 'Bridge Events (Total)', 'Setting_A': total_a_bridge_evs, 'Setting_B': total_b_bridge_evs, 'Delta_B_minus_A': total_b_bridge_evs - total_a_bridge_evs},
        {'Metric': 'Break Events (Total)', 'Setting_A': int(df_per_image['A_break_events'].sum()), 'Setting_B': int(df_per_image['B_break_events'].sum()), 'Delta_B_minus_A': int(df_per_image['B_break_events'].sum()) - int(df_per_image['A_break_events'].sum())},
        {'Metric': 'Break Images (Total)', 'Setting_A': int((df_per_image['A_break_events'] > 0).sum()), 'Setting_B': int((df_per_image['B_break_events'] > 0).sum()), 'Delta_B_minus_A': int((df_per_image['B_break_events'] > 0).sum()) - int((df_per_image['A_break_events'] > 0).sum())},
        {'Metric': 'Mean Dice', 'Setting_A': round(float(df_per_image['A_dice'].mean()), 4), 'Setting_B': round(float(df_per_image['B_dice'].mean()), 4), 'Delta_B_minus_A': round(float(df_per_image['B_dice'].mean() - df_per_image['A_dice'].mean()), 4)},
        {'Metric': 'Mean Recall', 'Setting_A': round(float(df_per_image['A_recall'].mean()), 4), 'Setting_B': round(float(df_per_image['B_recall'].mean()), 4), 'Delta_B_minus_A': round(float(df_per_image['B_recall'].mean() - df_per_image['A_recall'].mean()), 4)},
        {'Metric': 'Mean Precision', 'Setting_A': round(float(df_per_image['A_precision'].mean()), 4), 'Setting_B': round(float(df_per_image['B_precision'].mean()), 4), 'Delta_B_minus_A': round(float(df_per_image['B_precision'].mean() - df_per_image['A_precision'].mean()), 4)},
        {'Metric': 'Mean clDice', 'Setting_A': round(float(df_per_image['A_cldice'].mean()), 4), 'Setting_B': round(float(df_per_image['B_cldice'].mean()), 4), 'Delta_B_minus_A': round(float(df_per_image['B_cldice'].mean() - df_per_image['A_cldice'].mean()), 4)},
    ]
    df_trans = pd.DataFrame(trans_summary)
    csv_trans = os.path.join(out_dir, 'bridge_transition_a_to_b.csv')
    df_trans.to_csv(csv_trans, index=False)
    print(f"Saved: {csv_trans}")

    # 4. Cohort breakdowns
    cohort_stats = {}
    for c_name, c_sub in [
        ('All_348', df_per_image),
        ('Consensus_7of7_101', df_per_image[df_per_image['is_consensus_7of7'] == 1]),
        ('Consensus_Resistant_82', df_per_image[df_per_image['cohort_subgroup'] == 'Consensus_Resistant']),
        ('Clean_Control_Base_238', df_per_image[df_per_image['A_bridge_status'] == 0]),
        ('Clean_Control_Strict_56', df_per_image[df_per_image['cohort_subgroup'] == 'Clean_Control'])
    ]:
        if len(c_sub) > 0:
            cohort_stats[c_name] = {
                'N_images': len(c_sub),
                'A_bridge_imgs': int(c_sub['A_bridge_status'].sum()),
                'B_bridge_imgs': int(c_sub['B_bridge_status'].sum()),
                'A_bridge_evs': int(c_sub['A_bridge_events'].sum()),
                'B_bridge_evs': int(c_sub['B_bridge_events'].sum()),
                'A_break_evs': int(c_sub['A_break_events'].sum()),
                'B_break_evs': int(c_sub['B_break_events'].sum()),
                'A_dice': round(float(c_sub['A_dice'].mean()), 4),
                'B_dice': round(float(c_sub['B_dice'].mean()), 4),
                'delta_dice': round(float(c_sub['B_dice'].mean() - c_sub['A_dice'].mean()), 4),
                'A_recall': round(float(c_sub['A_recall'].mean()), 4),
                'B_recall': round(float(c_sub['B_recall'].mean()), 4),
                'A_cldice': round(float(c_sub['A_cldice'].mean()), 4),
                'B_cldice': round(float(c_sub['B_cldice'].mean()), 4)
            }

    # 5. Boundary Stratification Breakdown for A-bridge events
    boundary_strat = {}
    for col_bin in ['bin_internal_boundary', 'bin_any_boundary']:
        b_grouped = df_boundary.groupby(col_bin).agg(
            total_events=('event_index_in_img', 'count'),
            cured_events=('is_image_cured_by_b', 'sum')
        ).reset_index()
        b_grouped['cure_rate_pct'] = (b_grouped['cured_events'] / b_grouped['total_events']) * 100.0
        boundary_strat[col_bin] = b_grouped.to_dict(orient='records')

    # 6. View Count Distribution
    total_views = sum(view_count_counter.values())
    view_dist = {
        f"{k}_views": {
            'count_pixels': v,
            'percentage': round((v / total_views) * 100.0, 2)
        }
        for k, v in sorted(view_count_counter.items())
    }

    # 7. Summary JSON
    summary_json = {
        'evaluation_metadata': {
            'checkpoint': 'P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth',
            'dataset': 'Crack500 Validation (N=348)',
            'setting_a_spec': 'tile 448x448, stride 448 (non-overlap), logits > 0.0',
            'setting_b_spec': 'tile 448x448, stride 224 (50% overlap), blend_mode="probs", probs > 0.5',
            'execution_time_seconds': round(time.time() - t0_start, 2)
        },
        'overall_results': {
            'setting_a': {
                'dice': round(float(df_per_image['A_dice'].mean()), 4),
                'recall': round(float(df_per_image['A_recall'].mean()), 4),
                'precision': round(float(df_per_image['A_precision'].mean()), 4),
                'cldice': round(float(df_per_image['A_cldice'].mean()), 4),
                'bridge_images': total_a_bridge_imgs,
                'bridge_events': total_a_bridge_evs,
                'break_events': int(df_per_image['A_break_events'].sum()),
                'break_images': int((df_per_image['A_break_events'] > 0).sum())
            },
            'setting_b': {
                'dice': round(float(df_per_image['B_dice'].mean()), 4),
                'recall': round(float(df_per_image['B_recall'].mean()), 4),
                'precision': round(float(df_per_image['B_precision'].mean()), 4),
                'cldice': round(float(df_per_image['B_cldice'].mean()), 4),
                'bridge_images': total_b_bridge_imgs,
                'bridge_events': total_b_bridge_evs,
                'break_events': int(df_per_image['B_break_events'].sum()),
                'break_images': int((df_per_image['B_break_events'] > 0).sum())
            },
            'delta_b_minus_a': {
                'dice': round(float(df_per_image['B_dice'].mean() - df_per_image['A_dice'].mean()), 4),
                'recall': round(float(df_per_image['B_recall'].mean() - df_per_image['A_recall'].mean()), 4),
                'precision': round(float(df_per_image['B_precision'].mean() - df_per_image['A_precision'].mean()), 4),
                'cldice': round(float(df_per_image['B_cldice'].mean() - df_per_image['A_cldice'].mean()), 4),
                'bridge_images': total_b_bridge_imgs - total_a_bridge_imgs,
                'bridge_events': total_b_bridge_evs - total_a_bridge_evs,
                'break_events': int(df_per_image['B_break_events'].sum()) - int(df_per_image['A_break_events'].sum()),
                'break_images': int((df_per_image['B_break_events'] > 0).sum()) - int((df_per_image['A_break_events'] > 0).sum())
            }
        },
        'bridge_transition_counts': {
            'a_bridge_images': total_a_bridge_imgs,
            'b_bridge_images': total_b_bridge_imgs,
            'a_to_b_cured_images': cured_imgs,
            'a_to_b_persistent_images': persistent_imgs,
            'a_clean_to_b_created_images': created_imgs,
            'a_bridge_events': total_a_bridge_evs,
            'b_bridge_events': total_b_bridge_evs,
            'net_event_change': total_b_bridge_evs - total_a_bridge_evs
        },
        'cohort_breakdowns': cohort_stats,
        'tile_boundary_stratification': boundary_strat,
        'setting_b_view_count_distribution': view_dist,
        'probability_difference': {
            'mean_abs_diff_all348': round(float(np.mean(all_abs_diff_pixels)), 4),
            'mean_abs_diff_bridge110': round(float(np.mean(bridge110_abs_diff_pixels)), 4),
            'mean_abs_diff_near_internal_boundary_le32px': round(float(np.mean(abs_diff_near_boundary)), 4) if abs_diff_near_boundary else np.nan,
            'mean_abs_diff_far_internal_boundary_gt32px': round(float(np.mean(abs_diff_far_boundary)), 4) if abs_diff_far_boundary else np.nan
        }
    }

    json_path = os.path.join(out_dir, 'setting_a_vs_b_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=2)
    print(f"Saved: {json_path}")

    print("\n" + "=" * 80)
    print("SETTING A VS SETTING B SUMMARY RESULTS:")
    print("=" * 80)
    print(f"Setting A: Dice={summary_json['overall_results']['setting_a']['dice']:.4f} | "
          f"clDice={summary_json['overall_results']['setting_a']['cldice']:.4f} | "
          f"Bridge Imgs={total_a_bridge_imgs} | Bridge Evs={total_a_bridge_evs} | "
          f"Break Evs={summary_json['overall_results']['setting_a']['break_events']}")
    print(f"Setting B: Dice={summary_json['overall_results']['setting_b']['dice']:.4f} | "
          f"clDice={summary_json['overall_results']['setting_b']['cldice']:.4f} | "
          f"Bridge Imgs={total_b_bridge_imgs} | Bridge Evs={total_b_bridge_evs} | "
          f"Break Evs={summary_json['overall_results']['setting_b']['break_events']}")
    print(f"Transition: Cured Imgs={cured_imgs} | Persistent Imgs={persistent_imgs} | Created Imgs={created_imgs}")
    print(f"Total execution time: {time.time() - t0_start:.1f}s")


if __name__ == '__main__':
    main()
