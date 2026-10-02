#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_bridge_connector_probability.py

Task B: Bridge-Neck / Connector Probability Diagnostic
Answers:
- Are the minimal connector/neck pixels (the corridor directly joining distinct GT CCs)
  marginally positive near threshold 0.5 (e.g. 0.50-0.60), or confidently positive (>= 0.75)?
- Does probability drop at the neck compared to the broader bridge-FP and TP-crack regions?
- How does the neck probability distribution differ between the 101 consensus cases
  and the 7 cases that D2 was able to cure?
"""

import glob
import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
import torch
from tqdm import tqdm

sys.path.insert(0, 'SAGE_LITE')
from tools.run_phase6_c_topology_diagnostic import load_model_from_checkpoint
from scripts.evaluate_crack_official import predict_full_image_tiling_setting_a

def compute_stats(arr: np.ndarray) -> dict:
    if len(arr) == 0:
        return {
            'N_pixels': 0, 'mean': np.nan, 'std': np.nan, 'median': np.nan,
            'P10': np.nan, 'P25': np.nan, 'P50': np.nan, 'P75': np.nan, 'P90': np.nan,
            'min': np.nan, 'max': np.nan
        }
    return {
        'N_pixels': int(len(arr)),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'median': float(np.median(arr)),
        'P10': float(np.percentile(arr, 10)),
        'P25': float(np.percentile(arr, 25)),
        'P50': float(np.percentile(arr, 50)),
        'P75': float(np.percentile(arr, 75)),
        'P90': float(np.percentile(arr, 90)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr))
    }

def isolate_bridge_connector_neck(pred_bin: np.ndarray, target_bin: np.ndarray):
    """
    Isolates:
    1. bridge_fp_mask: all FP pixels in CCs merging >= 2 GT CCs
    2. connector_neck_mask: the minimal neck/corridor directly spanning between merged GT CCs
    3. non_neck_bridge_fp: bridge_fp_mask & (~connector_neck_mask)
    """
    num_gt_cc, gt_labels = cv2.connectedComponents(target_bin.astype(np.uint8), connectivity=8)
    num_pred_cc, pred_labels = cv2.connectedComponents(pred_bin.astype(np.uint8), connectivity=8)
    pred_cc = int(max(num_pred_cc - 1, 0))

    bridge_fp_mask = np.zeros_like(target_bin, dtype=np.uint8)
    connector_neck_mask = np.zeros_like(target_bin, dtype=np.uint8)

    for p_id in range(1, pred_cc + 1):
        p_mask = (pred_labels == p_id)
        overlapping_gt = np.unique(gt_labels[p_mask])
        overlapping_gt = overlapping_gt[overlapping_gt > 0]

        if len(overlapping_gt) < 2:
            continue

        # This is a bridge CC
        cc_fp = p_mask & (target_bin == 0)
        bridge_fp_mask = bridge_fp_mask | cc_fp.astype(np.uint8)

        # For every pair of merged GT CCs, isolate the corridor
        gt_list = list(overlapping_gt)
        for i_idx in range(len(gt_list)):
            gi = gt_list[i_idx]
            mask_i = (gt_labels == gi)
            dt_i = distance_transform_edt(1 - mask_i)

            for j_idx in range(i_idx + 1, len(gt_list)):
                gj = gt_list[j_idx]
                mask_j = (gt_labels == gj)
                
                # Gap distance between gi and gj
                gap_dist = float(np.min(dt_i[mask_j]))
                radius = int(np.ceil(gap_dist / 2.0)) + 2
                radius = max(radius, 3)

                ksize = 2 * radius + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

                dil_i = cv2.dilate(mask_i.astype(np.uint8), kernel)
                dil_j = cv2.dilate(mask_j.astype(np.uint8), kernel)

                corridor = (dil_i > 0) & (dil_j > 0)
                neck = cc_fp & corridor
                connector_neck_mask = connector_neck_mask | neck.astype(np.uint8)

    # In case connector_neck_mask is empty because of complex geometry, fallback to bridge_fp_mask
    if np.sum(bridge_fp_mask) > 0 and np.sum(connector_neck_mask) == 0:
        connector_neck_mask = bridge_fp_mask.copy()

    non_neck_bridge_fp = bridge_fp_mask & (connector_neck_mask == 0)

    return bridge_fp_mask, connector_neck_mask, non_neck_bridge_fp

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    out_dir = 'results/diagnostics/phase6_bridge_localization_v2'
    os.makedirs(out_dir, exist_ok=True)

    config_path = 'results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml'
    ckpt_path = 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'
    consensus_csv = 'results/diagnostics/phase6_bridge_localization/bridge_consensus_per_image.csv'
    df_meta = pd.read_csv(consensus_csv)

    # Target all 110 Base bridge cases
    df_base_bridge = df_meta[df_meta['Base_bridge'] == 1].copy()
    assert len(df_base_bridge) == 110, f"Expected 110 Base bridge cases, found {len(df_base_bridge)}"

    data_root = 'datasets/Crack500_ready'
    val_img_dir = os.path.join(data_root, 'val', 'images')
    val_mask_dir = os.path.join(data_root, 'val', 'masks')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = False

    print("=" * 80)
    print("TASK B: Loading Candidate B model for Bridge-Neck Probability Analysis...")
    print("=" * 80)
    model = load_model_from_checkpoint(config_path, ckpt_path, device)
    model.eval()

    all_neck_probs = []
    all_whole_bridge_probs = []
    all_non_neck_bridge_probs = []
    all_tp_probs = []

    # Stratified collectors
    c7_neck_probs = []
    cured_neck_probs = []
    persistent_neck_probs = []

    per_image_rows = []

    t0 = time.time()
    for _, meta_row in tqdm(df_base_bridge.iterrows(), total=len(df_base_bridge), desc="Base Connector Probability"):
        case_name = meta_row['image_id']
        trans_type = meta_row['d2_transition_type']
        n_models = meta_row['n_models_with_bridge']
        is_c7 = (n_models == 7)

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

        # Isolate connector neck
        whole_bridge_fp, neck_mask, non_neck_fp = isolate_bridge_connector_neck(pred_bin, target_bin)
        tp_mask = (pred_bin == 1) & (target_bin == 1)

        neck_p = prob_np[neck_mask > 0]
        whole_bridge_p = prob_np[whole_bridge_fp > 0]
        non_neck_p = prob_np[non_neck_fp > 0]
        tp_p = prob_np[tp_mask]

        all_neck_probs.append(neck_p.astype(np.float32))
        all_whole_bridge_probs.append(whole_bridge_p.astype(np.float32))
        all_non_neck_bridge_probs.append(non_neck_p.astype(np.float32))
        all_tp_probs.append(tp_p.astype(np.float32))

        if is_c7:
            c7_neck_probs.append(neck_p.astype(np.float32))
        if trans_type == 'Cured_by_D2':
            cured_neck_probs.append(neck_p.astype(np.float32))
        elif trans_type == 'Persistent_Base_to_D2':
            persistent_neck_probs.append(neck_p.astype(np.float32))

        # Per image metrics
        n_neck = len(neck_p)
        n_whole = len(whole_bridge_p)
        med_neck = float(np.median(neck_p)) if n_neck > 0 else np.nan
        mean_neck = float(np.mean(neck_p)) if n_neck > 0 else np.nan
        pct_neck_ge75 = float(np.sum(neck_p >= 0.75) / n_neck * 100.0) if n_neck > 0 else np.nan
        pct_neck_lt60 = float(np.sum(neck_p < 0.60) / n_neck * 100.0) if n_neck > 0 else np.nan

        per_image_rows.append({
            'image_id': case_name,
            'is_7_consensus': int(is_c7),
            'd2_transition': trans_type,
            'n_models_with_bridge': n_models,
            'gt_cc': meta_row['gt_cc'],
            'min_gap_px': meta_row.get('min_gap', np.nan),
            'neck_pixel_count': n_neck,
            'whole_bridge_pixel_count': n_whole,
            'neck_to_whole_ratio': round(float(n_neck / n_whole), 4) if n_whole > 0 else np.nan,
            'neck_prob_mean': round(mean_neck, 4),
            'neck_prob_median': round(med_neck, 4),
            'neck_pct_ge75': round(pct_neck_ge75, 2),
            'neck_pct_lt60': round(pct_neck_lt60, 2),
            'whole_bridge_prob_median': round(float(np.median(whole_bridge_p)), 4) if n_whole > 0 else np.nan,
            'tp_prob_median': round(float(np.median(tp_p)), 4) if len(tp_p) > 0 else np.nan,
        })

    dt = time.time() - t0
    print(f"Finished probability extraction in {dt:.1f}s")

    # Concatenate pixel arrays
    flat_neck = np.concatenate(all_neck_probs) if all_neck_probs else np.array([])
    flat_whole = np.concatenate(all_whole_bridge_probs) if all_whole_bridge_probs else np.array([])
    flat_non_neck = np.concatenate(all_non_neck_bridge_probs) if all_non_neck_bridge_probs else np.array([])
    flat_tp = np.concatenate(all_tp_probs) if all_tp_probs else np.array([])

    flat_c7_neck = np.concatenate(c7_neck_probs) if c7_neck_probs else np.array([])
    flat_cured_neck = np.concatenate(cured_neck_probs) if cured_neck_probs else np.array([])
    flat_persistent_neck = np.concatenate(persistent_neck_probs) if persistent_neck_probs else np.array([])

    # 1. Global summary statistics
    regions = [
        ('Minimal_Connector_Neck_All110', flat_neck),
        ('Whole_Bridge_FP_All110', flat_whole),
        ('Non_Neck_Bridge_FP_All110', flat_non_neck),
        ('TP_Crack_BridgeImages', flat_tp),
        ('Neck_7of7_Consensus101', flat_c7_neck),
        ('Neck_Cured_by_D2_7cases', flat_cured_neck),
        ('Neck_Persistent_Base_to_D2_103cases', flat_persistent_neck),
    ]

    global_rows = []
    for r_name, arr in regions:
        st = compute_stats(arr)
        st['region'] = r_name
        # Reorder so region is first
        row = {'region': r_name}
        row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in st.items() if k != 'region'})
        global_rows.append(row)

    df_global = pd.DataFrame(global_rows)
    out_global_path = os.path.join(out_dir, 'bridge_connector_probability_global.csv')
    df_global.to_csv(out_global_path, index=False)
    print(f"Saved: {out_global_path}")

    # 2. Per-image table
    df_per_image = pd.DataFrame(per_image_rows)
    out_per_image_path = os.path.join(out_dir, 'bridge_connector_probability_per_image.csv')
    df_per_image.to_csv(out_per_image_path, index=False)
    print(f"Saved: {out_per_image_path}")

    # 3. Bin distribution
    bins = [
        (0.50, 0.60, 'Marginal [0.50, 0.60)'),
        (0.60, 0.70, 'Low-Intermediate [0.60, 0.70)'),
        (0.70, 0.80, 'High-Intermediate [0.70, 0.80)'),
        (0.80, 0.90, 'Confident [0.80, 0.90)'),
        (0.90, 1.0001, 'Highly Confident [0.90, 1.00]'),
    ]

    bin_rows = []
    for r_name, arr in [
        ('Minimal_Connector_Neck_All110', flat_neck),
        ('Whole_Bridge_FP_All110', flat_whole),
        ('TP_Crack_BridgeImages', flat_tp),
        ('Neck_7of7_Consensus101', flat_c7_neck),
        ('Neck_Cured_by_D2_7cases', flat_cured_neck),
    ]:
        n_tot = len(arr)
        for b_low, b_high, b_label in bins:
            count = int(np.sum((arr >= b_low) & (arr < b_high)))
            pct = round(count / n_tot * 100.0, 2) if n_tot > 0 else 0.0
            bin_rows.append({
                'region': r_name,
                'bin_label': b_label,
                'bin_low': b_low,
                'bin_high': min(b_high, 1.0),
                'pixel_count': count,
                'pct_of_region': pct,
            })

    df_bins = pd.DataFrame(bin_rows)
    out_bins_path = os.path.join(out_dir, 'bridge_connector_probability_bins.csv')
    df_bins.to_csv(out_bins_path, index=False)
    print(f"Saved: {out_bins_path}")

    print("\n" + "=" * 80)
    print("TASK B SUMMARY:")
    print("=" * 80)
    print(df_global[['region', 'N_pixels', 'mean', 'median', 'P25', 'P75']].to_string(index=False))

if __name__ == '__main__':
    main()
