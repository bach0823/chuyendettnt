#!/usr/bin/env python3
"""
scripts/diagnostics/run_phase6_diagnostic_a_and_d.py

Phase 6 Diagnostic A & Diagnostic D Protocol:
1. Diagnostic A: Maximum-Bottleneck / Widest Path Diagnostic inside pred_cc.
   Formulation:
       P* = argmax_{P subset C_pred: S_A -> S_B} min_{v in P} P(v)
       P_bottleneck = min_{v in P*} P(v)
   Also extracts secondary profile via min-cost Dijkstra with C(x) = -log(P(x) + eps).
   Unit of analysis: Exactly 118 bridge events across the 110 bridge images of Candidate B on Crack500 val set.

2. Diagnostic D: Annotation QC & Gap Stratification.
   Measures Euclidean gap distance D_gap between bridged GT components.
   Stratifies into:
     - <= 3 px (narrow ambiguity candidate)
     - 3 < D_gap <= 5 px (intermediate ambiguity candidate)
     - > 5 px (wider-gap bridge candidate)
   Exports high-resolution visual crop panels (RGB, GT overlay, Pred overlay, Widest Path & Bottleneck).

Outputs:
  - results/diagnostics/phase6_bottleneck_path/bottleneck_path_118events.csv
  - results/diagnostics/phase6_bottleneck_path/bottleneck_path_mst_edges.csv
  - results/diagnostics/phase6_bottleneck_path/bottleneck_path_summary.json
  - results/diagnostics/phase6_annotation_qc/gap_stratification_summary.csv
  - results/diagnostics/phase6_annotation_qc/crops/*.png
"""

import heapq
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
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
from scripts.evaluate_crack_official import predict_full_image_tiling_setting_a


def find_maximum_bottleneck_path(
    prob_map: np.ndarray,
    pred_mask: np.ndarray,
    mask_A: np.ndarray,
    mask_B: np.ndarray
) -> Optional[List[Tuple[int, int]]]:
    """
    Finds the path P* from mask_A to mask_B strictly inside pred_mask
    maximizing min_{v in P} prob_map[v].
    Ties broken by:
      1. Higher sum of probabilities (average confidence)
      2. Shorter path length
    """
    H, W = prob_map.shape
    best_bottleneck = np.full((H, W), -1.0, dtype=np.float32)
    best_length = np.full((H, W), 1_000_000, dtype=np.int32)
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    pq: List[Tuple[float, int, int, int]] = []
    pts_A = np.argwhere(mask_A)
    for y, x in pts_A:
        p = float(prob_map[y, x])
        best_bottleneck[y, x] = p
        best_length[y, x] = 0
        heapq.heappush(pq, (-p, 0, int(y), int(x)))

    target_found: Optional[Tuple[int, int]] = None
    target_pts = set((int(y), int(x)) for y, x in np.argwhere(mask_B))
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while pq:
        neg_b, length, y, x = heapq.heappop(pq)
        cur_b = -neg_b

        if cur_b < best_bottleneck[y, x]:
            continue
        if cur_b == best_bottleneck[y, x] and length > best_length[y, x]:
            continue

        if (y, x) in target_pts:
            target_found = (y, x)
            break

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and pred_mask[ny, nx]:
                p_next = float(prob_map[ny, nx])
                next_b = min(cur_b, p_next)
                next_l = length + 1

                improved = False
                if next_b > best_bottleneck[ny, nx] + 1e-7:
                    improved = True
                elif abs(next_b - best_bottleneck[ny, nx]) <= 1e-7 and next_l < best_length[ny, nx]:
                    improved = True

                if improved:
                    best_bottleneck[ny, nx] = next_b
                    best_length[ny, nx] = next_l
                    parent[(ny, nx)] = (y, x)
                    heapq.heappush(pq, (-next_b, next_l, ny, nx))

    if target_found is None:
        return None

    path = [target_found]
    curr = target_found
    while curr in parent:
        curr = parent[curr]
        path.append(curr)
        if mask_A[curr[0], curr[1]]:
            break
    path.reverse()
    return path


def find_min_cost_path_nlogp(
    prob_map: np.ndarray,
    pred_mask: np.ndarray,
    mask_A: np.ndarray,
    mask_B: np.ndarray,
    eps: float = 1e-7
) -> Optional[List[Tuple[int, int]]]:
    """
    Finds the minimum cost path from mask_A to mask_B strictly inside pred_mask
    with cost C(x) = -log(prob_map[x] + eps).
    """
    H, W = prob_map.shape
    cost_map = -np.log(prob_map + eps).astype(np.float32)
    best_cost = np.full((H, W), np.inf, dtype=np.float32)
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    pq: List[Tuple[float, int, int]] = []
    pts_A = np.argwhere(mask_A)
    for y, x in pts_A:
        c = float(cost_map[y, x])
        best_cost[y, x] = c
        heapq.heappush(pq, (c, int(y), int(x)))

    target_found: Optional[Tuple[int, int]] = None
    target_pts = set((int(y), int(x)) for y, x in np.argwhere(mask_B))
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while pq:
        cur_c, y, x = heapq.heappop(pq)
        if cur_c > best_cost[y, x]:
            continue

        if (y, x) in target_pts:
            target_found = (y, x)
            break

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and pred_mask[ny, nx]:
                next_c = cur_c + float(cost_map[ny, nx])
                if next_c < best_cost[ny, nx]:
                    best_cost[ny, nx] = next_c
                    parent[(ny, nx)] = (y, x)
                    heapq.heappush(pq, (next_c, ny, nx))

    if target_found is None:
        return None

    path = [target_found]
    curr = target_found
    while curr in parent:
        curr = parent[curr]
        path.append(curr)
        if mask_A[curr[0], curr[1]]:
            break
    path.reverse()
    return path


def compute_mst_edges(gt_labels: np.ndarray, overlapping_gt: List[int]) -> List[Tuple[float, int, int]]:
    """
    Computes Minimum Spanning Tree (MST) edges between overlapping GT components
    weighted by minimum Euclidean distance between components.
    Returns list of (gap_dist, g_i, g_j) for the k-1 tree edges.
    """
    k = len(overlapping_gt)
    if k < 2:
        return []

    # Distance transforms for each GT component
    dt_dict = {}
    for g in overlapping_gt:
        dt_dict[g] = distance_transform_edt(gt_labels != g)

    # All pairwise edges
    all_edges = []
    for i in range(k):
        gi = overlapping_gt[i]
        for j in range(i + 1, k):
            gj = overlapping_gt[j]
            d = float(np.min(dt_dict[gi][gt_labels == gj]))
            all_edges.append((d, gi, gj))

    # Kruskal's algorithm for MST
    all_edges.sort(key=lambda x: x[0])
    parent_map = {g: g for g in overlapping_gt}

    def find_p(u):
        if parent_map[u] != u:
            parent_map[u] = find_p(parent_map[u])
        return parent_map[u]

    def union_p(u, v):
        ru, rv = find_p(u), find_p(v)
        if ru != rv:
            parent_map[ru] = rv
            return True
        return False

    mst_edges = []
    for d, u, v in all_edges:
        if union_p(u, v):
            mst_edges.append((d, u, v))
            if len(mst_edges) == k - 1:
                break

    return mst_edges


def render_visual_crop_panel(
    img_rgb: np.ndarray,
    target_bin: np.ndarray,
    pred_bin: np.ndarray,
    prob_map: np.ndarray,
    path: List[Tuple[int, int]],
    gA: int,
    gB: int,
    gt_labels: np.ndarray,
    d_gap: float,
    p_bottleneck: float,
    gap_cat: str,
    out_path: str
):
    """
    Renders and saves a 4-panel visual verification crop for an event:
    Panel 1: Raw RGB
    Panel 2: GT Overlay (gA, gB, and other GT components)
    Panel 3: Prediction & TP/FP Overlay
    Panel 4: Path & Bottleneck Saddle Localization
    """
    H, W = img_rgb.shape[:2]
    path_pts = np.array(path)
    ymin, xmin = np.min(path_pts, axis=0)
    ymax, xmax = np.max(path_pts, axis=0)

    # Pad bounding box by 32 pixels
    pad = 32
    ymin = max(0, ymin - pad)
    xmin = max(0, xmin - pad)
    ymax = min(H, ymax + pad + 1)
    xmax = min(W, xmax + pad + 1)

    crop_rgb = img_rgb[ymin:ymax, xmin:xmax].copy()
    crop_gt = target_bin[ymin:ymax, xmin:xmax]
    crop_gt_labels = gt_labels[ymin:ymax, xmin:xmax]
    crop_pred = pred_bin[ymin:ymax, xmin:xmax]
    crop_prob = prob_map[ymin:ymax, xmin:xmax]

    cH, cW = crop_rgb.shape[:2]

    # Panel 1: RGB
    p1 = crop_rgb.copy()

    # Panel 2: GT Components
    p2 = crop_rgb.copy()
    # Mask A: Cyan, Mask B: Yellow, Other GT: Green
    mask_a_crop = (crop_gt_labels == gA)
    mask_b_crop = (crop_gt_labels == gB)
    other_gt_crop = (crop_gt > 0) & (~mask_a_crop) & (~mask_b_crop)
    
    p2[mask_a_crop] = (0.5 * p2[mask_a_crop] + 0.5 * np.array([0, 255, 255])).astype(np.uint8)
    p2[mask_b_crop] = (0.5 * p2[mask_b_crop] + 0.5 * np.array([255, 255, 0])).astype(np.uint8)
    p2[other_gt_crop] = (0.5 * p2[other_gt_crop] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)

    # Panel 3: Pred & TP/FP
    p3 = crop_rgb.copy()
    tp_crop = (crop_pred == 1) & (crop_gt == 1)
    fp_crop = (crop_pred == 1) & (crop_gt == 0)
    p3[tp_crop] = (0.5 * p3[tp_crop] + 0.5 * np.array([0, 255, 0])).astype(np.uint8) # Green TP
    p3[fp_crop] = (0.4 * p3[fp_crop] + 0.6 * np.array([255, 0, 0])).astype(np.uint8) # Red FP bridge

    # Panel 4: Widest Path & Bottleneck
    p4 = crop_rgb.copy()
    # Draw prediction boundary in white
    contours, _ = cv2.findContours(crop_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(p4, contours, -1, (200, 200, 200), 1)

    # Draw path points in magenta (RGB: 255, 0, 255)
    for py, px in path:
        cy, cx = py - ymin, px - xmin
        if 0 <= cy < cH and 0 <= cx < cW:
            p4[cy, cx] = [255, 0, 255]

    # Find bottleneck pixel on path
    probs = [prob_map[py, px] for py, px in path]
    min_idx = int(np.argmin(probs))
    by, bx = path[min_idx]
    cby, cbx = by - ymin, bx - xmin

    # Draw circle and crosshair around bottleneck
    if 0 <= cby < cH and 0 <= cbx < cW:
        cv2.circle(p4, (cbx, cby), 4, (0, 255, 255), 1) # Yellow ring
        p4[cby, cbx] = [255, 255, 255] # White center

    # Put labels on panels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (255, 255, 255)
    cv2.putText(p1, "1. Raw RGB", (5, 14), font, font_scale, color, 1, cv2.LINE_AA)
    cv2.putText(p2, f"2. GT (A:Cyan, B:Yel)", (5, 14), font, font_scale, color, 1, cv2.LINE_AA)
    cv2.putText(p3, "3. Pred (TP:Grn, FP:Red)", (5, 14), font, font_scale, color, 1, cv2.LINE_AA)
    cv2.putText(p4, f"4. Widest Path (P_b={p_bottleneck:.3f})", (5, 14), font, font_scale, color, 1, cv2.LINE_AA)

    # Combine into 1x4 strip
    strip = np.hstack([p1, p2, p3, p4])

    # Add header banner
    banner_h = 24
    banner = np.zeros((banner_h, strip.shape[1], 3), dtype=np.uint8)
    title = f"Gap: {d_gap:.2f}px [{gap_cat}] | P_bottleneck: {p_bottleneck:.4f} | Path Len: {len(path)}px"
    cv2.putText(banner, title, (10, 16), font, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    final_crop = np.vstack([banner, strip])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(final_crop, cv2.COLOR_RGB2BGR))


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    t0_start = time.time()
    print("=" * 80)
    print("PHASE 6: DIAGNOSTIC A (WIDEST PATH BOTTLENECK) & DIAGNOSTIC D (GAP QC)")
    print("=" * 80)

    out_dir_a = os.path.join(project_root, 'results', 'diagnostics', 'phase6_bottleneck_path')
    out_dir_d = os.path.join(project_root, 'results', 'diagnostics', 'phase6_annotation_qc')
    crop_dir = os.path.join(out_dir_d, 'crops')
    os.makedirs(out_dir_a, exist_ok=True)
    os.makedirs(out_dir_d, exist_ok=True)
    os.makedirs(crop_dir, exist_ok=True)

    config_path = os.path.join(project_root, 'results', 'configs', 'b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml')
    ckpt_path = os.path.join(project_root, 'results', 'checkpoints', 'P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = False

    print(f"Device: {device} (CuDNN disabled for numerical stability)")
    print(f"Loading Candidate B model...")
    model = load_model_from_checkpoint(config_path, ckpt_path, device)
    model.eval()

    # Load paired topology metadata to select exactly the 110 bridge cases
    topo_csv = os.path.join(project_root, 'results', 'diagnostics', 'phase6_c_topology', 'topology_per_sample_paired.csv')
    df_topo = pd.read_csv(topo_csv)
    df_bridge = df_topo[df_topo['Base_bridge_events'] > 0].copy()
    assert len(df_bridge) == 110, f"Expected 110 bridge images, got {len(df_bridge)}"
    assert df_bridge['Base_bridge_events'].sum() == 118, f"Expected 118 bridge events, got {df_bridge['Base_bridge_events'].sum()}"
    print(f"Validated bridge cohort: 110 images, exactly 118 bridge events.")

    data_root = os.path.join(project_root, 'datasets', 'Crack500_ready')
    val_img_dir = os.path.join(data_root, 'val', 'images')
    val_mask_dir = os.path.join(data_root, 'val', 'masks')

    event_rows = []
    mst_rows = []

    global_event_counter = 0

    for _, row in tqdm(df_bridge.iterrows(), total=len(df_bridge), desc="Running Diagnostic A & D"):
        case_name = row['case_name']
        cat = row['candidate_b_category']

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

        with torch.no_grad():
            logits_np = predict_full_image_tiling_setting_a(
                model, img, device, tile_size=448, batch_size=8
            )

        prob_np = 1.0 / (1.0 + np.exp(-logits_np.astype(np.float64)))
        pred_bin = (logits_np > 0.0).astype(np.uint8)[:target_bin.shape[0], :target_bin.shape[1]]
        prob_np = prob_np[:target_bin.shape[0], :target_bin.shape[1]]

        num_gt_cc, gt_labels = cv2.connectedComponents(target_bin, connectivity=8)
        num_pred_cc, pred_labels = cv2.connectedComponents(pred_bin, connectivity=8)

        # Iterate over all predicted connected components
        for p_id in range(1, num_pred_cc):
            p_mask = (pred_labels == p_id)
            overlapping_gt = np.unique(gt_labels[p_mask])
            overlapping_gt = [int(g) for g in overlapping_gt if g > 0]

            if len(overlapping_gt) < 2:
                continue

            # Found a bridge event
            global_event_counter += 1
            event_id = global_event_counter

            # Compute Minimum Spanning Tree of the merged GT components
            mst_edges = compute_mst_edges(gt_labels, overlapping_gt)

            # Closest pair is the minimum distance edge in the MST
            min_gap_edge = min(mst_edges, key=lambda x: x[0])
            d_gap_primary, gA_primary, gB_primary = min_gap_edge

            # Stratification category (Diagnostic D)
            if d_gap_primary <= 3.0:
                gap_cat = "Narrow_Ambiguity (<= 3 px)"
                gap_cat_code = "narrow_le3"
            elif d_gap_primary <= 5.0:
                gap_cat = "Intermediate_Ambiguity (3 < D <= 5 px)"
                gap_cat_code = "intermediate_3to5"
            else:
                gap_cat = "Wider_Gap (> 5 px)"
                gap_cat_code = "wider_gt5"

            # Compute Diagnostic A Widest Path for primary pair
            mask_A = p_mask & (gt_labels == gA_primary)
            mask_B = p_mask & (gt_labels == gB_primary)

            path_mb = find_maximum_bottleneck_path(prob_np, p_mask, mask_A, mask_B)
            path_cost = find_min_cost_path_nlogp(prob_np, p_mask, mask_A, mask_B)

            assert path_mb is not None, f"Failed to find widest path for {case_name}, event {event_id}"
            assert path_cost is not None, f"Failed to find min-cost path for {case_name}, event {event_id}"

            probs_mb = np.array([prob_np[y, x] for y, x in path_mb], dtype=np.float32)
            probs_cost = np.array([prob_np[y, x] for y, x in path_cost], dtype=np.float32)

            p_bottleneck = float(np.min(probs_mb))
            p_median = float(np.median(probs_mb))
            p10 = float(np.percentile(probs_mb, 10))
            p25 = float(np.percentile(probs_mb, 25))
            p50 = float(np.percentile(probs_mb, 50))
            p75 = float(np.percentile(probs_mb, 75))
            p90 = float(np.percentile(probs_mb, 90))
            p_max = float(np.max(probs_mb))
            path_len = int(len(path_mb))

            # Endpoints average and valley depth
            prob_endpoints_avg = float(0.5 * (np.mean(prob_np[mask_A]) + np.mean(prob_np[mask_B])))
            valley_depth = float(prob_endpoints_avg - p_bottleneck)

            # FP pixels on path
            fp_indices = [i for i, (y, x) in enumerate(path_mb) if target_bin[y, x] == 0]
            num_fp_on_path = int(len(fp_indices))
            if num_fp_on_path > 0:
                fp_probs = probs_mb[fp_indices]
                fp_p_min = float(np.min(fp_probs))
                fp_p_med = float(np.median(fp_probs))
                fp_p_max = float(np.max(fp_probs))
            else:
                fp_p_min = np.nan
                fp_p_med = np.nan
                fp_p_max = np.nan

            # Secondary profile (-log P min-cost path)
            sec_p_min = float(np.min(probs_cost))
            sec_p_med = float(np.median(probs_cost))
            sec_path_len = int(len(path_cost))

            # Render Visual Crop Panel for Diagnostic D
            crop_filename = f"{case_name}_ev{event_id:03d}_d{d_gap_primary:.1f}px_{gap_cat_code}.png"
            crop_path = os.path.join(crop_dir, crop_filename)
            render_visual_crop_panel(
                img_rgb=img,
                target_bin=target_bin,
                pred_bin=pred_bin,
                prob_map=prob_np,
                path=path_mb,
                gA=gA_primary,
                gB=gB_primary,
                gt_labels=gt_labels,
                d_gap=d_gap_primary,
                p_bottleneck=p_bottleneck,
                gap_cat=gap_cat,
                out_path=crop_path
            )

            event_record = {
                'event_id': event_id,
                'case_name': case_name,
                'candidate_b_category': cat,
                'pred_cc_id': p_id,
                'num_merged_gt': len(overlapping_gt),
                'merged_gt_ids': str(overlapping_gt),
                'primary_gt_A': gA_primary,
                'primary_gt_B': gB_primary,
                'd_gap_px': round(d_gap_primary, 4),
                'gap_category': gap_cat,
                'gap_cat_code': gap_cat_code,
                # Primary Diagnostic A Metrics (Maximum Bottleneck Path)
                'p_bottleneck': round(p_bottleneck, 4),
                'p_10': round(p10, 4),
                'p_25': round(p25, 4),
                'p_50_median': round(p50, 4),
                'p_75': round(p75, 4),
                'p_90': round(p90, 4),
                'p_max': round(p_max, 4),
                'prob_endpoints_avg': round(prob_endpoints_avg, 4),
                'valley_depth': round(valley_depth, 4),
                'path_length': path_len,
                'num_fp_on_path': num_fp_on_path,
                'fp_prob_min': round(fp_p_min, 4) if not np.isnan(fp_p_min) else np.nan,
                'fp_prob_median': round(fp_p_med, 4) if not np.isnan(fp_p_med) else np.nan,
                'fp_prob_max': round(fp_p_max, 4) if not np.isnan(fp_p_max) else np.nan,
                # Secondary Profile (-log P min-cost)
                'sec_p_min': round(sec_p_min, 4),
                'sec_p_median': round(sec_p_med, 4),
                'sec_path_length': sec_path_len,
                'crop_file': crop_filename
            }
            event_rows.append(event_record)

            # Record all MST edges for multi-component merges
            for edge_idx, (edge_d, u, v) in enumerate(mst_edges):
                mst_rows.append({
                    'event_id': event_id,
                    'case_name': case_name,
                    'pred_cc_id': p_id,
                    'edge_idx': edge_idx,
                    'gt_u': u,
                    'gt_v': v,
                    'd_gap_px': round(edge_d, 4),
                    'is_primary_edge': int((u == gA_primary and v == gB_primary) or (u == gB_primary and v == gA_primary))
                })

    assert global_event_counter == 118, f"Expected exactly 118 events, processed {global_event_counter}"
    print(f"\nAll 118 bridge events successfully processed and analyzed!")

    # 1. Save event table
    df_events = pd.DataFrame(event_rows)
    csv_events = os.path.join(out_dir_a, 'bottleneck_path_118events.csv')
    df_events.to_csv(csv_events, index=False)
    print(f"Saved: {csv_events}")

    # 2. Save MST edges table
    df_mst = pd.DataFrame(mst_rows)
    csv_mst = os.path.join(out_dir_a, 'bottleneck_path_mst_edges.csv')
    df_mst.to_csv(csv_mst, index=False)
    print(f"Saved: {csv_mst}")

    # 3. Diagnostic D Stratification Table
    strat_summary = []
    for cat_name in ["Narrow_Ambiguity (<= 3 px)", "Intermediate_Ambiguity (3 < D <= 5 px)", "Wider_Gap (> 5 px)"]:
        sub = df_events[df_events['gap_category'] == cat_name]
        cnt = len(sub)
        pct = (cnt / 118.0) * 100.0
        b_min = float(sub['p_bottleneck'].min()) if cnt > 0 else np.nan
        b_med = float(sub['p_bottleneck'].median()) if cnt > 0 else np.nan
        b_p25 = float(sub['p_bottleneck'].quantile(0.25)) if cnt > 0 else np.nan
        b_p75 = float(sub['p_bottleneck'].quantile(0.75)) if cnt > 0 else np.nan
        v_med = float(sub['valley_depth'].median()) if cnt > 0 else np.nan
        l_med = float(sub['path_length'].median()) if cnt > 0 else np.nan

        strat_summary.append({
            'gap_category': cat_name,
            'count': cnt,
            'pct_of_118': round(pct, 2),
            'd_gap_median': round(float(sub['d_gap_px'].median()), 2) if cnt > 0 else np.nan,
            'p_bottleneck_p25': round(b_p25, 4),
            'p_bottleneck_median': round(b_med, 4),
            'p_bottleneck_p75': round(b_p75, 4),
            'valley_depth_median': round(v_med, 4),
            'path_length_median': round(l_med, 2),
            'pct_bottleneck_ge75': round(float((sub['p_bottleneck'] >= 0.75).sum() / cnt * 100.0), 2) if cnt > 0 else np.nan,
            'pct_bottleneck_ge90': round(float((sub['p_bottleneck'] >= 0.90).sum() / cnt * 100.0), 2) if cnt > 0 else np.nan,
        })
    df_strat = pd.DataFrame(strat_summary)
    csv_strat = os.path.join(out_dir_d, 'gap_stratification_summary.csv')
    df_strat.to_csv(csv_strat, index=False)
    print(f"Saved: {csv_strat}")

    # 4. Global Distribution Summary JSON
    summary_dict = {
        'total_bridge_images': 110,
        'total_bridge_events': 118,
        'total_mst_edges': len(df_mst),
        'execution_time_seconds': round(time.time() - t0_start, 2),
        'p_bottleneck_distribution': {
            'min': float(df_events['p_bottleneck'].min()),
            'P05': float(df_events['p_bottleneck'].quantile(0.05)),
            'P10': float(df_events['p_bottleneck'].quantile(0.10)),
            'P25': float(df_events['p_bottleneck'].quantile(0.25)),
            'P50_median': float(df_events['p_bottleneck'].median()),
            'P75': float(df_events['p_bottleneck'].quantile(0.75)),
            'P90': float(df_events['p_bottleneck'].quantile(0.90)),
            'P95': float(df_events['p_bottleneck'].quantile(0.95)),
            'max': float(df_events['p_bottleneck'].max()),
            'mean': float(df_events['p_bottleneck'].mean()),
            'std': float(df_events['p_bottleneck'].std())
        },
        'valley_depth_distribution': {
            'min': float(df_events['valley_depth'].min()),
            'P25': float(df_events['valley_depth'].quantile(0.25)),
            'P50_median': float(df_events['valley_depth'].median()),
            'P75': float(df_events['valley_depth'].quantile(0.75)),
            'max': float(df_events['valley_depth'].max()),
            'mean': float(df_events['valley_depth'].mean()),
        },
        'd_gap_distribution': {
            'min': float(df_events['d_gap_px'].min()),
            'P25': float(df_events['d_gap_px'].quantile(0.25)),
            'P50_median': float(df_events['d_gap_px'].median()),
            'P75': float(df_events['d_gap_px'].quantile(0.75)),
            'max': float(df_events['d_gap_px'].max()),
        },
        'gap_stratification_counts': {
            row['gap_category']: {
                'count': row['count'],
                'pct': row['pct_of_118'],
                'p_bottleneck_median': row['p_bottleneck_median']
            }
            for row in strat_summary
        }
    }
    json_summary = os.path.join(out_dir_a, 'bottleneck_path_summary.json')
    with open(json_summary, 'w', encoding='utf-8') as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Saved: {json_summary}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC A & D EXECUTION SUMMARY:")
    print("=" * 80)
    print(f"P_bottleneck distribution across 118 events:")
    print(f"  Min:    {summary_dict['p_bottleneck_distribution']['min']:.4f}")
    print(f"  P10:    {summary_dict['p_bottleneck_distribution']['P10']:.4f}")
    print(f"  P25:    {summary_dict['p_bottleneck_distribution']['P25']:.4f}")
    print(f"  Median: {summary_dict['p_bottleneck_distribution']['P50_median']:.4f}")
    print(f"  P75:    {summary_dict['p_bottleneck_distribution']['P75']:.4f}")
    print(f"  P90:    {summary_dict['p_bottleneck_distribution']['P90']:.4f}")
    print(f"  Max:    {summary_dict['p_bottleneck_distribution']['max']:.4f}")
    print(f"  Mean:   {summary_dict['p_bottleneck_distribution']['mean']:.4f} +/- {summary_dict['p_bottleneck_distribution']['std']:.4f}")
    print("\nGap Stratification:")
    for row in strat_summary:
        print(f"  - {row['gap_category']:42s}: N={row['count']:2d} ({row['pct_of_118']:5.1f}%) | "
              f"Gap Med: {row['d_gap_median']:4.1f}px | P_b Med: {row['p_bottleneck_median']:.4f} | "
              f"P_b >= 0.75: {row['pct_bottleneck_ge75']:5.1f}%")
    print(f"\nTotal execution time: {time.time() - t0_start:.1f}s")


if __name__ == '__main__':
    main()
