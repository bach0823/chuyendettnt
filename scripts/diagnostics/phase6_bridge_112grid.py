#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_bridge_112grid.py

Task C: 112-Grid Representability Diagnostic
Answers: Are nearby GT components still represented as separate components when mapped
onto the 112x112 decoder-resolution grid?

Evaluation:
- All 176 validation images with >=2 GT connected components
- Subgroups:
  * All 110 Base bridge images
  * 103 Base -> D2 persistent bridges
  * 7 D2-cured cases
  * 6 D2-created cases
  * 101 D1 -> D2 persistent bridges (7/7 consensus)

Grid mappings evaluated:
1. Stride-4 decoder resolution grid (H//4, W//4): direct spatial footprint of 4x downsampling (448 -> 112)
2. Fixed 112x112 resize: nearest-neighbor downsampling to exactly 112x112
"""

import glob
import os
import sys
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

def compute_pairwise_min_gap(labels: np.ndarray, num_cc: int) -> float:
    """Computes minimum Euclidean gap between distinct components in a labeled mask."""
    if num_cc < 2:
        return np.nan
    min_dist = float('inf')
    # Use distance transform for efficiency
    for i in range(1, num_cc + 1):
        mask_i = (labels == i)
        if not np.any(mask_i):
            continue
        dt = distance_transform_edt(~mask_i)
        # Check distance to all other components
        other_mask = (labels > 0) & (labels != i)
        if np.any(other_mask):
            d = np.min(dt[other_mask])
            if d < min_dist:
                min_dist = float(d)
    return min_dist if min_dist != float('inf') else np.nan

def analyze_mapping(target_bin: np.ndarray, orig_labels: np.ndarray, orig_cc: int, target_size: tuple):
    """Maps mask and labels to target_size using nearest-neighbor and checks for CC merges."""
    W_out, H_out = target_size
    mask_down = cv2.resize(target_bin, (W_out, H_out), interpolation=cv2.INTER_NEAREST)
    labels_down_mapped = cv2.resize(orig_labels.astype(np.int32), (W_out, H_out), interpolation=cv2.INTER_NEAREST)

    num_down_cc, down_cc_labels = cv2.connectedComponents(mask_down, connectivity=8)
    down_cc_count = int(max(num_down_cc - 1, 0))

    has_merge = False
    merged_orig_pairs = set()
    for p in range(1, down_cc_count + 1):
        orig_in_p = np.unique(labels_down_mapped[down_cc_labels == p])
        orig_in_p = orig_in_p[orig_in_p > 0]
        if len(orig_in_p) >= 2:
            has_merge = True
            for i in range(len(orig_in_p)):
                for j in range(i + 1, len(orig_in_p)):
                    merged_orig_pairs.add((int(orig_in_p[i]), int(orig_in_p[j])))

    surviving_orig = set(np.unique(labels_down_mapped)) - {0}
    surviving_count = len(surviving_orig)
    preserved_separation = (not has_merge) and (surviving_count >= 2)

    # Compute min gap on downsampled grid
    down_gap = compute_pairwise_min_gap(down_cc_labels, down_cc_count)

    return {
        'down_cc_count': down_cc_count,
        'has_merge': int(has_merge),
        'preserved_separation': int(preserved_separation),
        'surviving_orig_count': surviving_count,
        'down_min_gap': round(down_gap, 2) if not np.isnan(down_gap) else None
    }

def main():
    masks_dir = 'datasets/Crack500_ready/val/masks'
    consensus_csv = 'results/diagnostics/phase6_bridge_localization/bridge_consensus_per_image.csv'
    out_dir = 'results/diagnostics/phase6_bridge_localization'
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'bridge_112grid_representability.csv')

    df_meta = pd.read_csv(consensus_csv)

    mask_files = sorted(glob.glob(os.path.join(masks_dir, '*')))
    mask_files = [f for f in mask_files if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg']]
    assert len(mask_files) == 348, f"Expected 348 masks, found {len(mask_files)}"

    records = []
    print("=" * 80)
    print("TASK C: 112-GRID REPRESENTABILITY DIAGNOSTIC")
    print("=" * 80)

    for mf in mask_files:
        stem = os.path.splitext(os.path.basename(mf))[0]
        target = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
        target_bin = (target > 127).astype(np.uint8)
        H, W = target_bin.shape

        num_gt_cc, gt_labels = cv2.connectedComponents(target_bin, connectivity=8)
        orig_cc = int(max(num_gt_cc - 1, 0))

        if orig_cc < 2:
            continue

        # 1. Stride-4 resolution grid: (W//4, H//4)
        s4_res = analyze_mapping(target_bin, gt_labels, orig_cc, (W // 4, H // 4))
        # 2. Fixed 112x112 resize: (112, 112)
        f112_res = analyze_mapping(target_bin, gt_labels, orig_cc, (112, 112))

        # Look up metadata
        row_meta = df_meta[df_meta['image_id'] == stem]
        if len(row_meta) > 0:
            row_meta = row_meta.iloc[0]
            cat = row_meta['category']
            base_br = int(row_meta['Base_bridge'])
            d2_br = int(row_meta['D2_bridge'])
            d1_br = int(row_meta['D1_bridge'])
            trans = row_meta['d2_transition_type']
            orig_min_gap = row_meta['min_gap']
            n_models = int(row_meta['n_models_with_bridge'])
        else:
            cat = 'unknown'
            base_br = 0
            d2_br = 0
            d1_br = 0
            trans = 'unknown'
            orig_min_gap = np.nan
            n_models = 0

        records.append({
            'image_id': stem,
            'orig_H': H,
            'orig_W': W,
            'orig_cc_count': orig_cc,
            'orig_min_gap': orig_min_gap,
            'category': cat,
            'Base_bridge': base_br,
            'D1_bridge': d1_br,
            'D2_bridge': d2_br,
            'n_models_with_bridge': n_models,
            'd2_transition_type': trans,
            # Stride-4 (H//4, W//4)
            's4_cc_count': s4_res['down_cc_count'],
            's4_has_merge': s4_res['has_merge'],
            's4_preserved_separation': s4_res['preserved_separation'],
            's4_surviving_cc_count': s4_res['surviving_orig_count'],
            's4_min_gap': s4_res['down_min_gap'],
            # Fixed 112x112
            'f112_cc_count': f112_res['down_cc_count'],
            'f112_has_merge': f112_res['has_merge'],
            'f112_preserved_separation': f112_res['preserved_separation'],
            'f112_surviving_cc_count': f112_res['surviving_orig_count'],
            'f112_min_gap': f112_res['down_min_gap'],
        })

    df_out = pd.DataFrame(records)
    df_out.to_csv(out_csv, index=False)
    print(f"Evaluated {len(df_out)} multi-component images (out of 348 total).")
    print(f"Saved: {out_csv}\n")

    # Sanity check: verify original CC count matches geometry diagnostic (176 images)
    assert len(df_out) == 176, f"Expected 176 images with >=2 GT CCs, found {len(df_out)}"
    print("[PASS] Multi-component count matches official geometry diagnostic: 176 images (100% of Base bridge images).")

    # Subgroups Analysis
    subgroups = [
        ("All Multi-Component Validation Images (N=176)", df_out),
        ("All Base Bridge Images (N=110)", df_out[df_out['Base_bridge'] == 1]),
        ("Base -> D2 Persistent Bridges (N=103)", df_out[df_out['d2_transition_type'] == 'Persistent_Base_to_D2']),
        ("Base -> D2 Cured Bridges (N=7)", df_out[df_out['d2_transition_type'] == 'Cured_by_D2']),
        ("Base Clean -> D2 Created Bridges (N=6)", df_out[df_out['d2_transition_type'] == 'Created_by_D2']),
        ("7/7 All-Model Consensus Bridges (N=101)", df_out[df_out['n_models_with_bridge'] == 7]),
    ]

    print("=" * 120)
    print("TASK C: 112-GRID REPRESENTABILITY SUBGROUP BREAKDOWN")
    print("=" * 120)
    print(f"{'Subgroup':45s} | {'N':3s} | {'Stride-4 Merge%':15s} | {'Preserved%':10s} | {'F112 Merge%':11s} | {'Orig Gap (med/min/max)':22s} | {'S4 Gap med':10s}")
    print("-" * 120)

    for name, sub in subgroups:
        n = len(sub)
        if n == 0:
            continue
        s4_m_cnt = sub['s4_has_merge'].sum()
        s4_m_pct = s4_m_cnt / n * 100.0
        s4_p_pct = sub['s4_preserved_separation'].mean() * 100.0

        f112_m_cnt = sub['f112_has_merge'].sum()
        f112_m_pct = f112_m_cnt / n * 100.0

        g_med = sub['orig_min_gap'].median()
        g_min = sub['orig_min_gap'].min()
        g_max = sub['orig_min_gap'].max()

        s4_g_med = sub['s4_min_gap'].dropna().median() if len(sub['s4_min_gap'].dropna()) > 0 else np.nan

        gap_str = f"{g_med:.1f} / {g_min:.1f} / {g_max:.1f}"
        s4_gap_str = f"{s4_g_med:.1f} px" if not np.isnan(s4_g_med) else "N/A"

        print(f"{name:45s} | {n:3d} | {s4_m_cnt:2d}/{n:2d} ({s4_m_pct:4.1f}%)   | {s4_p_pct:5.1f}%    | {f112_m_cnt:2d}/{n:2d} ({f112_m_pct:4.1f}%) | {gap_str:22s} | {s4_gap_str:10s}")

    print("=" * 120)

if __name__ == '__main__':
    main()
