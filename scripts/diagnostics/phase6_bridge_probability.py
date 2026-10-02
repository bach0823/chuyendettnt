#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_bridge_probability.py

Task B: Base Bridge-Pixel Probability Diagnostic
Answers: Are false-bridge pixels marginally positive near threshold 0.5, or confidently positive?

Protocol:
- Evaluates Candidate B official checkpoint under exact Setting A tiling (448x448, stride 448).
- NO RETRAINING. NO THRESHOLD CHANGE IN OFFICIAL CHECKPOINT EVALUATION.
- Extracts unquantized raw probability p = sigmoid(raw_logit).
- Partitions predicted foreground (threshold 0.5) into:
  1. TP-crack: GT = 1, Base pred = 1
  2. Bridge-FP: GT = 0, Base pred = 1, part of multi-GT-merging CC
  3. Other-FP: GT = 0, Base pred = 1, NOT part of multi-GT-merging CC
"""

import glob
import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
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

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    config_path = 'results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml'
    ckpt_path = 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'
    data_root = 'datasets/Crack500_ready'
    out_dir = 'results/diagnostics/phase6_bridge_localization'
    os.makedirs(out_dir, exist_ok=True)

    out_global_stats = os.path.join(out_dir, 'bridge_probability_global_stats.csv')
    out_per_image = os.path.join(out_dir, 'bridge_probability_per_image.csv')
    out_histogram = os.path.join(out_dir, 'bridge_probability_histogram.csv')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = False

    print("=" * 80)
    print(f"TASK B: Loading Candidate B model on {device}...")
    print("=" * 80)
    model = load_model_from_checkpoint(config_path, ckpt_path, device)

    val_img_dir = os.path.join(data_root, 'val', 'images')
    val_mask_dir = os.path.join(data_root, 'val', 'masks')
    img_paths = sorted(glob.glob(os.path.join(val_img_dir, '*')))
    img_paths = [p for p in img_paths if os.path.splitext(p)[1].lower() in ['.jpg', '.jpeg', '.png']]

    val_pairs = []
    for ip in img_paths:
        stem = os.path.splitext(os.path.basename(ip))[0]
        for ext in ['.png', '.jpg', '.jpeg']:
            mp = os.path.join(val_mask_dir, stem + ext)
            if os.path.exists(mp):
                val_pairs.append((ip, mp, stem))
                break

    assert len(val_pairs) == 348, f"Expected 348 validation samples, found {len(val_pairs)}!"
    print(f"Running Setting A probability diagnostic on {len(val_pairs)} samples...")

    all_tp_probs = []
    all_bridge_probs = []
    all_other_fp_probs = []

    per_image_rows = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    bridge_sample_count = 0

    t0 = time.time()
    for ip, mp, stem in tqdm(val_pairs, desc="Base Probability Extraction"):
        img = cv2.imread(ip)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        target_bin = (target > 127).astype(np.uint8)

        with torch.no_grad():
            logits_np = predict_full_image_tiling_setting_a(
                model, img, device, tile_size=448, batch_size=8
            )

        # Exact unquantized probability
        prob_np = 1.0 / (1.0 + np.exp(-logits_np.astype(np.float64)))
        pred_bin = (logits_np > 0.0).astype(np.uint8)[:target_bin.shape[0], :target_bin.shape[1]]
        prob_np = prob_np[:target_bin.shape[0], :target_bin.shape[1]]

        # Connected components for Bridge definition
        num_gt_cc, gt_labels = cv2.connectedComponents(target_bin, connectivity=8)
        num_pred_cc, pred_labels = cv2.connectedComponents(pred_bin, connectivity=8)
        pred_cc = int(max(num_pred_cc - 1, 0))

        bridge_pred_ids = set()
        for p_id in range(1, pred_cc + 1):
            overlapping_gt = np.unique(gt_labels[pred_labels == p_id])
            overlapping_gt = overlapping_gt[overlapping_gt > 0]
            if len(overlapping_gt) >= 2:
                bridge_pred_ids.add(p_id)

        is_bridge_comp = np.isin(pred_labels, list(bridge_pred_ids))
        has_bridge = len(bridge_pred_ids) > 0
        if has_bridge:
            bridge_sample_count += 1

        # Pixel group masks
        tp_mask = (pred_bin == 1) & (target_bin == 1)
        bridge_fp_mask = (pred_bin == 1) & (target_bin == 0) & is_bridge_comp
        other_fp_mask = (pred_bin == 1) & (target_bin == 0) & (~is_bridge_comp)
        fn_mask = (pred_bin == 0) & (target_bin == 1)

        tp_cnt = int(np.sum(tp_mask))
        bridge_fp_cnt = int(np.sum(bridge_fp_mask))
        other_fp_cnt = int(np.sum(other_fp_mask))
        fn_cnt = int(np.sum(fn_mask))

        total_tp += tp_cnt
        total_fp += (bridge_fp_cnt + other_fp_cnt)
        total_fn += fn_cnt

        tp_p = prob_np[tp_mask]
        bridge_p = prob_np[bridge_fp_mask]
        other_fp_p = prob_np[other_fp_mask]

        all_tp_probs.append(tp_p.astype(np.float32))
        all_bridge_probs.append(bridge_p.astype(np.float32))
        all_other_fp_probs.append(other_fp_p.astype(np.float32))

        # Record per-image record
        bridge_p_cnt = len(bridge_p)
        if bridge_p_cnt > 0:
            b_mean = float(np.mean(bridge_p))
            b_med = float(np.median(bridge_p))
            b_p25 = float(np.percentile(bridge_p, 25))
            b_p75 = float(np.percentile(bridge_p, 75))
            b_min = float(np.min(bridge_p))
            b_max = float(np.max(bridge_p))
            if b_med < 0.60:
                phenotype = 'Marginal_Bridge (med < 0.60)'
            elif b_med < 0.75:
                phenotype = 'Intermediate_Bridge (0.60 <= med < 0.75)'
            else:
                phenotype = 'Confident_Bridge (med >= 0.75)'
        else:
            b_mean = np.nan
            b_med = np.nan
            b_p25 = np.nan
            b_p75 = np.nan
            b_min = np.nan
            b_max = np.nan
            phenotype = 'No_Bridge'

        tp_med = float(np.median(tp_p)) if len(tp_p) > 0 else np.nan
        other_fp_med = float(np.median(other_fp_p)) if len(other_fp_p) > 0 else np.nan

        sample_dice = float((2.0 * tp_cnt) / (2.0 * tp_cnt + bridge_fp_cnt + other_fp_cnt + fn_cnt + 1e-8))
        sample_prec = float(tp_cnt / (tp_cnt + bridge_fp_cnt + other_fp_cnt + 1e-8))
        sample_rec = float(tp_cnt / (tp_cnt + fn_cnt + 1e-8))

        per_image_rows.append({
            'image_id': stem,
            'has_bridge': int(has_bridge),
            'sample_dice': round(sample_dice, 4),
            'sample_precision': round(sample_prec, 4),
            'sample_recall': round(sample_rec, 4),
            'bridge_pixel_count': bridge_p_cnt,
            'bridge_prob_mean': round(b_mean, 4) if not np.isnan(b_mean) else None,
            'bridge_prob_median': round(b_med, 4) if not np.isnan(b_med) else None,
            'bridge_prob_p25': round(b_p25, 4) if not np.isnan(b_p25) else None,
            'bridge_prob_p75': round(b_p75, 4) if not np.isnan(b_p75) else None,
            'bridge_prob_min': round(b_min, 4) if not np.isnan(b_min) else None,
            'bridge_prob_max': round(b_max, 4) if not np.isnan(b_max) else None,
            'tp_crack_prob_median': round(tp_med, 4) if not np.isnan(tp_med) else None,
            'other_fp_prob_median': round(other_fp_med, 4) if not np.isnan(other_fp_med) else None,
            'bridge_phenotype': phenotype
        })

    t1 = time.time()
    print(f"Inference completed in {t1 - t0:.1f}s.")

    # Sanity checks
    mean_sample_dice = float(np.mean([r['sample_dice'] for r in per_image_rows]))
    mean_sample_prec = float(np.mean([r['sample_precision'] for r in per_image_rows]))
    mean_sample_rec = float(np.mean([r['sample_recall'] for r in per_image_rows]))
    pixel_global_dice = float((2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn))

    print("\n" + "=" * 80)
    print("TASK B SANITY CHECKS (BASE OFFICIAL REFERENCE COMPARISON)")
    print("=" * 80)
    print(f"Sample-Mean Dice:      {mean_sample_dice:.4f} (Expected = 0.7641, diff = {mean_sample_dice - 0.7641:+.4f})")
    print(f"Sample-Mean Precision: {mean_sample_prec:.4f} (Expected = 0.7338, diff = {mean_sample_prec - 0.7338:+.4f})")
    print(f"Sample-Mean Recall:    {mean_sample_rec:.4f} (Expected = 0.8477, diff = {mean_sample_rec - 0.8477:+.4f})")
    print(f"Pixel-Global Dice:     {pixel_global_dice:.4f} (micro-averaged across all pixels)")
    print(f"Bridge Images:         {bridge_sample_count}/348 (Expected = 110, diff = {bridge_sample_count - 110:+d})")
    assert bridge_sample_count == 110, f"Expected 110 bridge images, got {bridge_sample_count}!"
    assert abs(mean_sample_dice - 0.7641) < 0.001, f"Mean Sample Dice {mean_sample_dice:.4f} deviates from official 0.7641!"
    assert abs(mean_sample_prec - 0.7338) < 0.001, f"Mean Sample Prec {mean_sample_prec:.4f} deviates from official 0.7338!"
    assert abs(mean_sample_rec - 0.8477) < 0.001, f"Mean Sample Rec {mean_sample_rec:.4f} deviates from official 0.8477!"
    print("[PASS] All Task B Sanity Checks Succeeded with EXACT match to official baseline!")

    # 1. Save Per-image Analysis
    df_per_image = pd.DataFrame(per_image_rows)
    df_per_image.to_csv(out_per_image, index=False)
    print(f"Saved per-image probability table: {out_per_image}")

    # Phenotype distribution on the 110 bridge images
    df_bridge_only = df_per_image[df_per_image['has_bridge'] == 1]
    pheno_counts = df_bridge_only['bridge_phenotype'].value_counts()
    print("\n" + "=" * 80)
    print("110 BASE BRIDGE IMAGES PHENOTYPE DISTRIBUTION")
    print("=" * 80)
    for p_name, p_cnt in pheno_counts.items():
        print(f"  - {p_name:40s}: {p_cnt:3d}/110 ({p_cnt/110*100:5.2f}%)")

    # 2. Global Pixel Statistics
    all_tp = np.concatenate(all_tp_probs) if len(all_tp_probs) > 0 else np.array([])
    all_bridge = np.concatenate(all_bridge_probs) if len(all_bridge_probs) > 0 else np.array([])
    all_other_fp = np.concatenate(all_other_fp_probs) if len(all_other_fp_probs) > 0 else np.array([])

    stats_tp = compute_stats(all_tp)
    stats_bridge = compute_stats(all_bridge)
    stats_other_fp = compute_stats(all_other_fp)

    stats_rows = [
        {'group': 'TP-crack', **stats_tp},
        {'group': 'Bridge-FP', **stats_bridge},
        {'group': 'Other-FP', **stats_other_fp},
    ]
    df_stats = pd.DataFrame(stats_rows)

    # Add separation stats
    sep_stats = {
        'metric': [
            'mean(Bridge-FP) - mean(TP-crack)',
            'median(Bridge-FP) - median(TP-crack)',
            'mean(Bridge-FP) - mean(Other-FP)',
            'median(Bridge-FP) - median(Other-FP)',
        ],
        'value': [
            stats_bridge['mean'] - stats_tp['mean'],
            stats_bridge['median'] - stats_tp['median'],
            stats_bridge['mean'] - stats_other_fp['mean'],
            stats_bridge['median'] - stats_other_fp['median'],
        ]
    }
    df_sep = pd.DataFrame(sep_stats)

    with open(out_global_stats, 'w', encoding='utf-8') as f:
        f.write("# Task B: Base Global Pixel Probability Statistics\n\n")
        f.write("## 1. Pixel Groups Summary\n")
        df_stats.to_csv(f, index=False)
        f.write("\n## 2. Separation Statistics\n")
        df_sep.to_csv(f, index=False)

    print(f"\nSaved global probability stats: {out_global_stats}")
    print("\nGlobal Pixel Summary Table:")
    print(df_stats[['group', 'N_pixels', 'mean', 'median', 'std', 'P10', 'P25', 'P75', 'P90', 'min', 'max']].to_string(index=False))

    print("\nSeparation Statistics:")
    print(df_sep.to_string(index=False))

    # 3. Histogram Bins
    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.0000001]
    bin_labels = [
        '[0.50, 0.55)', '[0.55, 0.60)', '[0.60, 0.65)', '[0.65, 0.70)',
        '[0.70, 0.75)', '[0.75, 0.80)', '[0.80, 0.90)', '[0.90, 1.00]'
    ]

    hist_rows = []
    for b_idx in range(len(bin_labels)):
        low = bins[b_idx]
        high = bins[b_idx + 1]
        label = bin_labels[b_idx]

        tp_in_bin = np.sum((all_tp >= low) & (all_tp < high)) if b_idx < len(bin_labels) - 1 else np.sum((all_tp >= low) & (all_tp <= 1.0))
        bridge_in_bin = np.sum((all_bridge >= low) & (all_bridge < high)) if b_idx < len(bin_labels) - 1 else np.sum((all_bridge >= low) & (all_bridge <= 1.0))
        other_in_bin = np.sum((all_other_fp >= low) & (all_other_fp < high)) if b_idx < len(bin_labels) - 1 else np.sum((all_other_fp >= low) & (all_other_fp <= 1.0))

        hist_rows.append({
            'bin_range': label,
            'Bridge-FP_count': int(bridge_in_bin),
            'Bridge-FP_pct': round(float(bridge_in_bin / len(all_bridge) * 100.0), 2) if len(all_bridge) > 0 else 0,
            'TP-crack_count': int(tp_in_bin),
            'TP-crack_pct': round(float(tp_in_bin / len(all_tp) * 100.0), 2) if len(all_tp) > 0 else 0,
            'Other-FP_count': int(other_in_bin),
            'Other-FP_pct': round(float(other_in_bin / len(all_other_fp) * 100.0), 2) if len(all_other_fp) > 0 else 0,
        })

    df_hist = pd.DataFrame(hist_rows)
    df_hist.to_csv(out_histogram, index=False)
    print(f"\nSaved histogram table: {out_histogram}")
    print("\nHistogram Distribution Table (% of group pixels):")
    print(df_hist[['bin_range', 'Bridge-FP_count', 'Bridge-FP_pct', 'TP-crack_count', 'TP-crack_pct', 'Other-FP_count', 'Other-FP_pct']].to_string(index=False))

if __name__ == '__main__':
    main()
