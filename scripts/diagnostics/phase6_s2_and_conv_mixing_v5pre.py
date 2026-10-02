#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_s2_and_conv_mixing_v5pre.py

Phase 6 False-Bridge Causal Localization v5-pre:
Investigates:
1. Task A: S2 28x28 Ground-Truth Representability across all 176 multi-component images
   and focused subgroups (Consensus 7/7, D2-cured, D2-created, Base bridges).
2. Task B: Decoder Block 1 Conv1 & Conv2 Spatial Weight Decomposition (Center vs Off-Center).
3. Task C: Activation-Level Spatial-Mixing Analysis on the 25 representative validation cases.
4. Task D: Synthesis Mechanistic Matrix connecting S2 Representability with Conv1/Conv2 Mixing.

STRICT CONSTRAINTS:
- Diagnostic-Only: Zero training, zero parameter updates, zero model mutations.
- Checkpoints, configs, and threshold (tau=0.5) remain strictly unchanged.
- Test set (N=1124) is strictly sealed and untouched.
"""

import os
import sys
import glob
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, 'SAGE_LITE')
from tools.run_phase6_c_topology_diagnostic import load_model_from_checkpoint

# -----------------------------------------------------------------------------
# Task A Helper: Pairwise Gap and CC Analysis
# -----------------------------------------------------------------------------

def compute_pairwise_min_gap(labels: np.ndarray, num_cc: int) -> float:
    """Computes minimum Euclidean gap between distinct components in a labeled mask."""
    if num_cc < 2:
        return np.nan
    min_dist = float('inf')
    for i in range(1, num_cc + 1):
        mask_i = (labels == i)
        if not np.any(mask_i):
            continue
        dt = distance_transform_edt(~mask_i)
        other_mask = (labels > 0) & (labels != i)
        if np.any(other_mask):
            d = np.min(dt[other_mask])
            if d < min_dist:
                min_dist = float(d)
    return min_dist if min_dist != float('inf') else np.nan


def analyze_grid_mapping(target_bin: np.ndarray, orig_labels: np.ndarray, orig_cc: int, target_size: Tuple[int, int]) -> Dict[str, Any]:
    """Maps binary mask and CC labels to target_size using nearest-neighbor semantics."""
    W_out, H_out = target_size
    mask_down = cv2.resize(target_bin, (W_out, H_out), interpolation=cv2.INTER_NEAREST)
    labels_down_mapped = cv2.resize(orig_labels.astype(np.int32), (W_out, H_out), interpolation=cv2.INTER_NEAREST)

    num_down_cc, down_cc_labels = cv2.connectedComponents(mask_down, connectivity=8)
    down_cc_count = int(max(num_down_cc - 1, 0))

    has_merge = False
    merged_pairs = set()
    for p in range(1, down_cc_count + 1):
        orig_in_p = np.unique(labels_down_mapped[down_cc_labels == p])
        orig_in_p = orig_in_p[orig_in_p > 0]
        if len(orig_in_p) >= 2:
            has_merge = True
            for i in range(len(orig_in_p)):
                for j in range(i + 1, len(orig_in_p)):
                    merged_pairs.add((int(orig_in_p[i]), int(orig_in_p[j])))

    surviving_orig = set(np.unique(labels_down_mapped)) - {0}
    surviving_count = len(surviving_orig)
    lost_orig_count = max(0, orig_cc - surviving_count)
    preserved_separation = (not has_merge) and (surviving_count >= 2)

    down_gap = compute_pairwise_min_gap(down_cc_labels, down_cc_count)

    return {
        'down_cc_count': down_cc_count,
        'has_merge': int(has_merge),
        'preserved_separation': int(preserved_separation),
        'surviving_orig_count': surviving_count,
        'lost_orig_count': lost_orig_count,
        'down_min_gap': round(down_gap, 2) if not np.isnan(down_gap) else np.nan,
        'n_merged_pairs': len(merged_pairs)
    }

# -----------------------------------------------------------------------------
# Corridor & Region Isolation Helpers (Reused exactly from v2/v3/v4)
# -----------------------------------------------------------------------------

def isolate_base_bridge_corridor(pred_bin: np.ndarray, target_bin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Identifies bridge false-positive area and geometric corridor neck mask."""
    num_gt_cc, gt_labels = cv2.connectedComponents(target_bin.astype(np.uint8), connectivity=8)
    num_pred_cc, pred_labels = cv2.connectedComponents(pred_bin.astype(np.uint8), connectivity=8)
    gt_cc = int(max(num_gt_cc - 1, 0))
    pred_cc = int(max(num_pred_cc - 1, 0))

    bridge_fp_mask = np.zeros_like(pred_bin, dtype=np.uint8)
    connector_neck_mask = np.zeros_like(pred_bin, dtype=np.uint8)

    if gt_cc < 2 or pred_cc < 1:
        return bridge_fp_mask, connector_neck_mask

    for p_id in range(1, pred_cc + 1):
        cc_pred = (pred_labels == p_id)
        overlapping_gt = np.unique(gt_labels[cc_pred])
        overlapping_gt = overlapping_gt[overlapping_gt > 0]

        if len(overlapping_gt) >= 2:
            cc_fp = cc_pred & (target_bin == 0)
            bridge_fp_mask = bridge_fp_mask | cc_fp.astype(np.uint8)

            for i in range(len(overlapping_gt)):
                for j in range(i + 1, len(overlapping_gt)):
                    g_i = overlapping_gt[i]
                    g_j = overlapping_gt[j]
                    mask_i = (gt_labels == g_i)
                    mask_j = (gt_labels == g_j)

                    dt_i = distance_transform_edt(~mask_i)
                    gap_dist = float(np.min(dt_i[mask_j]))
                    radius = max(int(np.ceil(gap_dist / 2.0)) + 2, 3)
                    ksize = 2 * radius + 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

                    dil_i = cv2.dilate(mask_i.astype(np.uint8), kernel)
                    dil_j = cv2.dilate(mask_j.astype(np.uint8), kernel)

                    corridor = (dil_i > 0) & (dil_j > 0)
                    neck = cc_fp & corridor
                    connector_neck_mask = connector_neck_mask | neck.astype(np.uint8)

    if np.sum(bridge_fp_mask) > 0 and np.sum(connector_neck_mask) == 0:
        connector_neck_mask = bridge_fp_mask.copy()

    return bridge_fp_mask, connector_neck_mask

# -----------------------------------------------------------------------------
# TASK A: S2 GT Representability
# -----------------------------------------------------------------------------

def run_task_a(out_dir: str):
    print("\n" + "=" * 80)
    print("TASK A: S2 (28x28) GROUND-TRUTH GEOMETRIC REPRESENTABILITY")
    print("=" * 80)

    masks_dir = 'datasets/Crack500_ready/val/masks'
    consensus_csv = 'results/diagnostics/phase6_bridge_localization/bridge_consensus_per_image.csv'
    df_meta = pd.read_csv(consensus_csv)

    mask_files = sorted(glob.glob(os.path.join(masks_dir, '*')))
    mask_files = [f for f in mask_files if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg']]
    assert len(mask_files) == 348, f"Expected 348 masks, found {len(mask_files)}"

    records = []

    for mf in mask_files:
        stem = os.path.splitext(os.path.basename(mf))[0]
        target = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
        target_bin = (target > 127).astype(np.uint8)
        H, W = target_bin.shape

        num_gt_cc, gt_labels = cv2.connectedComponents(target_bin, connectivity=8)
        orig_cc = int(max(num_gt_cc - 1, 0))

        if orig_cc < 2:
            continue

        orig_min_gap = compute_pairwise_min_gap(gt_labels, orig_cc)

        # 1. Fixed 28x28 grid (direct tile-level downsampling 448 -> 28, factor 16)
        f28_res = analyze_grid_mapping(target_bin, gt_labels, orig_cc, (28, 28))

        # 2. Stride-16 resolution grid (H // 16, W // 16)
        s16_res = analyze_grid_mapping(target_bin, gt_labels, orig_cc, (W // 16, H // 16))

        # Lookup metadata
        row_meta = df_meta[df_meta['image_id'] == stem]
        if len(row_meta) > 0:
            rm = row_meta.iloc[0]
            cat = rm['category']
            base_br = int(rm['Base_bridge'])
            d1_br = int(rm['D1_bridge'])
            d2_br = int(rm['D2_bridge'])
            n_models = int(rm['n_models_with_bridge'])
            trans = rm['d2_transition_type']
        else:
            cat = 'unknown'
            base_br = d1_br = d2_br = n_models = 0
            trans = 'unknown'

        records.append({
            'image_id': stem,
            'orig_H': H,
            'orig_W': W,
            'orig_cc_count': orig_cc,
            'orig_min_gap': round(orig_min_gap, 2) if not np.isnan(orig_min_gap) else np.nan,
            'category': cat,
            'Base_bridge': base_br,
            'D1_bridge': d1_br,
            'D2_bridge': d2_br,
            'n_models_with_bridge': n_models,
            'd2_transition_type': trans,
            # Fixed 28x28
            'f28_cc_count': f28_res['down_cc_count'],
            'f28_has_merge': f28_res['has_merge'],
            'f28_preserved_separation': f28_res['preserved_separation'],
            'f28_surviving_cc_count': f28_res['surviving_orig_count'],
            'f28_lost_orig_count': f28_res['lost_orig_count'],
            'f28_min_gap': f28_res['down_min_gap'],
            'f28_merged_pairs': f28_res['n_merged_pairs'],
            # Stride-16 (H//16, W//16)
            's16_cc_count': s16_res['down_cc_count'],
            's16_has_merge': s16_res['has_merge'],
            's16_preserved_separation': s16_res['preserved_separation'],
            's16_surviving_cc_count': s16_res['surviving_orig_count'],
            's16_lost_orig_count': s16_res['lost_orig_count'],
            's16_min_gap': s16_res['down_min_gap'],
            's16_merged_pairs': s16_res['n_merged_pairs'],
        })

    df_out = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, 's2_gt_representability.csv')
    df_out.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} (N={len(df_out)} multi-component images)")

    # Compute Subgroup Statistics
    subgroups = [
        ('All_176_Multi_CC', df_out),
        ('Base_110_Bridges', df_out[df_out['Base_bridge'] == 1]),
        ('Persistent_Base_to_D2_103', df_out[df_out['d2_transition_type'] == 'Persistent_Base_to_D2']),
        ('Cured_by_D2_7', df_out[df_out['d2_transition_type'] == 'Cured_by_D2']),
        ('Created_by_D2_6', df_out[df_out['d2_transition_type'] == 'Created_by_D2']),
        ('Consensus_7of7_101', df_out[df_out['n_models_with_bridge'] == 7]),
    ]

    summary_rows = []
    for grp_name, grp_df in subgroups:
        n_cases = len(grp_df)
        if n_cases == 0:
            continue

        # Fixed 28x28 metrics
        f28_merge_pct = float(grp_df['f28_has_merge'].mean() * 100.0)
        f28_sep_pct = float(grp_df['f28_preserved_separation'].mean() * 100.0)
        f28_sep_df = grp_df[grp_df['f28_preserved_separation'] == 1]
        f28_med_gap = float(f28_sep_df['f28_min_gap'].median()) if len(f28_sep_df) > 0 else np.nan
        f28_mean_gap = float(f28_sep_df['f28_min_gap'].mean()) if len(f28_sep_df) > 0 else np.nan

        # Stride-16 metrics
        s16_merge_pct = float(grp_df['s16_has_merge'].mean() * 100.0)
        s16_sep_pct = float(grp_df['s16_preserved_separation'].mean() * 100.0)
        s16_sep_df = grp_df[grp_df['s16_preserved_separation'] == 1]
        s16_med_gap = float(s16_sep_df['s16_min_gap'].median()) if len(s16_sep_df) > 0 else np.nan
        s16_mean_gap = float(s16_sep_df['s16_min_gap'].mean()) if len(s16_sep_df) > 0 else np.nan

        # Original gap
        orig_med_gap = float(grp_df['orig_min_gap'].median())
        orig_mean_gap = float(grp_df['orig_min_gap'].mean())
        orig_min_gap = float(grp_df['orig_min_gap'].min())
        orig_max_gap = float(grp_df['orig_min_gap'].max())

        total_orig_cc = int(grp_df['orig_cc_count'].sum())
        f28_lost_cc = int(grp_df['f28_lost_orig_count'].sum())
        s16_lost_cc = int(grp_df['s16_lost_orig_count'].sum())

        summary_rows.append({
            'subgroup': grp_name,
            'N': n_cases,
            'orig_min_gap_median': round(orig_med_gap, 2),
            'orig_min_gap_mean': round(orig_mean_gap, 2),
            'orig_min_gap_min': round(orig_min_gap, 2),
            'orig_min_gap_max': round(orig_max_gap, 2),
            # Fixed 28x28
            'f28_pct_merged': round(f28_merge_pct, 2),
            'f28_pct_still_separated': round(f28_sep_pct, 2),
            'f28_median_gap_when_separated': round(f28_med_gap, 2),
            'f28_mean_gap_when_separated': round(f28_mean_gap, 2),
            'f28_total_components_lost': f28_lost_cc,
            # Stride-16
            's16_pct_merged': round(s16_merge_pct, 2),
            's16_pct_still_separated': round(s16_sep_pct, 2),
            's16_median_gap_when_separated': round(s16_med_gap, 2),
            's16_mean_gap_when_separated': round(s16_mean_gap, 2),
            's16_total_components_lost': s16_lost_cc,
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(out_dir, 's2_gt_representability_summary.csv')
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")
    print(df_summary[['subgroup', 'N', 'orig_min_gap_median', 'f28_pct_merged', 'f28_pct_still_separated', 's16_pct_merged', 's16_pct_still_separated']].to_string(index=False))

    return df_out, df_summary

# -----------------------------------------------------------------------------
# TASK B: Conv Spatial Weight Decomposition
# -----------------------------------------------------------------------------

def run_task_b(model: nn.Module, out_dir: str):
    print("\n" + "=" * 80)
    print("TASK B: DECODER BLOCK 1 CONV1 & CONV2 WEIGHT DECOMPOSITION")
    print("=" * 80)

    block1 = model.decoder.decoder_blocks[1]
    conv1 = block1.conv1[0]
    conv2 = block1.conv2[0]

    layers = [('Conv1', conv1), ('Conv2', conv2)]
    records = []

    for name, layer in layers:
        w = layer.weight.detach().cpu().numpy()  # shape (out_ch, in_ch, 3, 3)
        out_ch, in_ch, kh, kw = w.shape
        assert kh == 3 and kw == 3, f"Expected 3x3 kernel, got ({kh}, {kw})"

        for oc in range(out_ch):
            w_center = w[oc, :, 1, 1]  # shape (in_ch,)
            norm_center = float(np.linalg.norm(w_center))

            # Off-center: all 8 positions except (1, 1)
            off_mask = np.ones((3, 3), dtype=bool)
            off_mask[1, 1] = False
            w_off = w[oc, :, off_mask]  # shape (in_ch, 8)
            norm_off = float(np.linalg.norm(w_off))

            ratio = norm_off / (norm_center + 1e-8)
            total_norm = float(np.linalg.norm(w[oc]))

            records.append({
                'layer': name,
                'out_channel': oc,
                'in_channels': in_ch,
                'center_l2_norm': round(norm_center, 6),
                'offcenter_l2_norm': round(norm_off, 6),
                'total_l2_norm': round(total_norm, 6),
                'offcenter_over_center_ratio': round(ratio, 6),
                'offcenter_energy_fraction': round((norm_off**2) / (total_norm**2 + 1e-8), 6)
            })

    df_weights = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, 'conv_spatial_weight_decomposition.csv')
    df_weights.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} (N={len(df_weights)} channels across Conv1 and Conv2)")

    # Aggregate statistics
    for name in ['Conv1', 'Conv2']:
        sub = df_weights[df_weights['layer'] == name]
        r = sub['offcenter_over_center_ratio']
        print(f"\n--- {name} Spatial Weight Decomposition (N={len(sub)} channels) ---")
        print(f"  Center L2 Norm: Mean={sub['center_l2_norm'].mean():.4f}, Median={sub['center_l2_norm'].median():.4f}")
        print(f"  Off-Center L2 Norm: Mean={sub['offcenter_l2_norm'].mean():.4f}, Median={sub['offcenter_l2_norm'].median():.4f}")
        print(f"  Ratio (||Off|| / ||Center||): Mean={r.mean():.4f}, Median={r.median():.4f}, P25={r.quantile(0.25):.4f}, P75={r.quantile(0.75):.4f}, P90={r.quantile(0.90):.4f}, Max={r.max():.4f}")
        print(f"  Off-Center Energy Fraction: Mean={sub['offcenter_energy_fraction'].mean()*100:.2f}% (Expect ~88.9% for uniform 8/9)")

    return df_weights

# -----------------------------------------------------------------------------
# TASK C: Activation-Level Spatial-Mixing Analysis
# -----------------------------------------------------------------------------

def run_task_c(model: nn.Module, device: torch.device, out_dir: str):
    print("\n" + "=" * 80)
    print("TASK C: ACTIVATION-LEVEL SPATIAL-MIXING ANALYSIS")
    print("=" * 80)

    # 25 Deterministic cases from v4
    sample_cases_meta = [
        ('20160222_080850_1281_721', 'Consensus_7of7'),
        ('20160222_165225_1921_721', 'Consensus_7of7'),
        ('20160307_145059_1281_361', 'Consensus_7of7'),
        ('20160316_143527_1281_721', 'Consensus_7of7'),
        ('20160321_185557_1921_361', 'Consensus_7of7'),
        ('20160324_170347_1281_1',   'Consensus_7of7'),
        ('20160328_151244_1921_721', 'Consensus_7of7'),
        ('20160328_153620_1_361',   'Consensus_7of7'),
        ('20160328_153620_641_721', 'Consensus_7of7'),
        ('20160330_172309_1281_721', 'Consensus_7of7'),
        ('20160222_115837_1281_361', 'Cured_by_D2'),
        ('20160307_145059_1281_721', 'Cured_by_D2'),
        ('20160316_143527_1921_361', 'Cured_by_D2'),
        ('20160316_143547_1281_1081','Cured_by_D2'),
        ('20160318_181632_1_361',   'Cured_by_D2'),
        ('20160302_155857_1281_1',   'Created_by_D2'),
        ('20160307_162438_1921_1081','Created_by_D2'),
        ('20160321_185557_1281_721', 'Created_by_D2'),
        ('20160328_151244_1281_721', 'Created_by_D2'),
        ('20160402_150314_1921_1',   'Created_by_D2'),
        ('20160222_115224_641_721', 'Low_Consensus'),
        ('20160326_141808_1281_1',   'Low_Consensus'),
        ('20160328_152305_1921_361', 'Low_Consensus'),
        ('20160328_153620_641_361', 'Low_Consensus'),
        ('IMG_2941_1297_969',        'Low_Consensus'),
    ]

    data_root = 'datasets/Crack500_ready'
    val_img_dir = os.path.join(data_root, 'val', 'images')
    val_mask_dir = os.path.join(data_root, 'val', 'masks')

    block1 = model.decoder.decoder_blocks[1]
    conv1_layer = block1.conv1[0]
    conv2_layer = block1.conv2[0]

    # Pre-extract weights for center and off-center
    w1 = conv1_layer.weight  # (96, 288, 3, 3)
    b1 = conv1_layer.bias    # (96,)
    w1_center = w1[:, :, 1:2, 1:2].clone()
    w1_off = w1.clone()
    w1_off[:, :, 1, 1] = 0.0

    w2 = conv2_layer.weight  # (96, 96, 3, 3)
    b2 = conv2_layer.bias    # (96,)
    w2_center = w2[:, :, 1:2, 1:2].clone()
    w2_off = w2.clone()
    w2_off[:, :, 1, 1] = 0.0

    tile_size = 448
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    decomp_rows = []

    for c_id, stratum in tqdm(sample_cases_meta, desc="Task C: Decomposing Spatial Mixing"):
        ip = os.path.join(val_img_dir, c_id + '.png')
        if not os.path.exists(ip):
            ip = os.path.join(val_img_dir, c_id + '.jpg')
        mp = os.path.join(val_mask_dir, c_id + '.png')
        if not os.path.exists(mp):
            mp = os.path.join(val_mask_dir, c_id + '.jpg')

        img = cv2.imread(ip)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        target_bin = (target > 127).astype(np.uint8)
        H, W = target_bin.shape

        pad_h = (tile_size - (H % tile_size)) % tile_size
        pad_w = (tile_size - (W % tile_size)) % tile_size
        padded_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        pH, pW = padded_img.shape[:2]

        patches = []
        coords = []
        for y in range(0, pH, tile_size):
            for x in range(0, pW, tile_size):
                patch = padded_img[y:y+tile_size, x:x+tile_size]
                patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
                patch_tensor = (patch_tensor - mean.squeeze(0).cpu()) / std.squeeze(0).cpu()
                patches.append(patch_tensor)
                coords.append((y, x))

        # Arrays to accumulate spatial outputs at 56x56 resolution
        H_s1, W_s1 = pH // 8, pW // 8
        c1_center_map = np.zeros((96, H_s1, W_s1), dtype=np.float32)
        c1_off_map    = np.zeros((96, H_s1, W_s1), dtype=np.float32)
        c2_center_map = np.zeros((96, H_s1, W_s1), dtype=np.float32)
        c2_off_map    = np.zeros((96, H_s1, W_s1), dtype=np.float32)
        pred_logits_map = np.zeros((pH, pW), dtype=np.float32)

        # Hook to capture T2 (conv1 input) and T3 (conv2 input)
        captured = {}
        def hook_b1(module, inputs, output):
            # inputs[0] is x (S2), inputs[1] is skip (E1)
            pass

        # We execute patch by patch
        model.eval()
        with torch.no_grad():
            for patch_t, (py, px) in zip(patches, coords):
                inp = patch_t.unsqueeze(0).to(device)

                # Forward through backbone
                feat_dict = model.backbone(inp)
                skips = feat_dict["skips"]
                bottleneck = feat_dict["bottleneck"]

                # Decoder block 0
                reversed_skips = list(reversed(skips))
                x_dec = bottleneck
                s2 = model.decoder.decoder_blocks[0](x_dec, reversed_skips[0]) # (1, 192, 28, 28)

                # Decoder block 1 internal operations
                e1 = reversed_skips[1]
                t1 = block1.upsample(s2)
                if t1.shape[2:] != e1.shape[2:]:
                    t1 = F.interpolate(t1, size=e1.shape[2:], mode="bilinear", align_corners=False)
                t2 = torch.cat([t1, e1], dim=1) # (1, 288, 56, 56)

                # Conv1 Decomposition
                # Center contribution: 1x1 conv with center tap
                act_c1_center = F.conv2d(t2, w1_center) # (1, 96, 56, 56)
                # Off-center contribution: conv with center zeroed
                act_c1_off = F.conv2d(t2, w1_off, padding=1) # (1, 96, 56, 56)

                # Sanity check exact reproduction
                full_c1 = conv1_layer(t2)
                assert torch.allclose(act_c1_center + act_c1_off + b1.view(1, -1, 1, 1), full_c1, atol=1e-5), \
                    "Sanity check failed: center + offcenter != full conv1!"

                # Post Conv1 + BN + ReLU = T3
                t3 = block1.conv1[1:](full_c1) # BN + ReLU

                # Conv2 Decomposition
                act_c2_center = F.conv2d(t3, w2_center)
                act_c2_off = F.conv2d(t3, w2_off, padding=1)
                full_c2 = conv2_layer(t3)
                assert torch.allclose(act_c2_center + act_c2_off + b2.view(1, -1, 1, 1), full_c2, atol=1e-5), \
                    "Sanity check failed: center + offcenter != full conv2!"

                t4 = block1.conv2[1:](full_c2) # BN + ReLU

                # Complete forward for segmentation logits
                x_dec_sub = t4
                for blk, sk in zip(model.decoder.decoder_blocks[2:], reversed_skips[2:]):
                    x_dec_sub = blk(x_dec_sub, sk)
                logits = model.decoder.segmentation_head(x_dec_sub)
                if logits.shape[2:] != (tile_size, tile_size):
                    logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)

                # Accumulate
                sy, sx = py // 8, px // 8
                sh, sw = 56, 56
                c1_center_map[:, sy:sy+sh, sx:sx+sw] = act_c1_center.squeeze(0).cpu().numpy()
                c1_off_map[:, sy:sy+sh, sx:sx+sw]    = act_c1_off.squeeze(0).cpu().numpy()
                c2_center_map[:, sy:sy+sh, sx:sx+sw] = act_c2_center.squeeze(0).cpu().numpy()
                c2_off_map[:, sy:sy+sh, sx:sx+sw]    = act_c2_off.squeeze(0).cpu().numpy()
                pred_logits_map[py:py+tile_size, px:px+tile_size] = logits.squeeze().cpu().numpy()

        # Crop back to original resolution
        h_s1_dim = H // 8
        w_s1_dim = W // 8
        c1_c = c1_center_map[:, :h_s1_dim, :w_s1_dim]
        c1_o = c1_off_map[:, :h_s1_dim, :w_s1_dim]
        c2_c = c2_center_map[:, :h_s1_dim, :w_s1_dim]
        c2_o = c2_off_map[:, :h_s1_dim, :w_s1_dim]
        logits_crop = pred_logits_map[:H, :W]
        prob_crop = 1.0 / (1.0 + np.exp(-logits_crop))
        pred_bin = (logits_crop > 0.0).astype(np.uint8)

        # Extract regions
        whole_bridge_fp, neck_mask = isolate_base_bridge_corridor(pred_bin, target_bin)
        crack_mask = (pred_bin == 1) & (target_bin == 1)
        bg_mask = (pred_bin == 0) & (target_bin == 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dil_target = cv2.dilate(target_bin, kernel)
        nearby_bg_mask = (dil_target > 0) & (target_bin == 0) & (pred_bin == 0)
        if np.sum(nearby_bg_mask) == 0:
            nearby_bg_mask = bg_mask

        # Connector probability
        if np.sum(neck_mask) > 0:
            neck_prob_median = float(np.median(prob_crop[neck_mask > 0]))
            neck_prob_mean   = float(np.mean(prob_crop[neck_mask > 0]))
        else:
            neck_prob_median = np.nan
            neck_prob_mean   = np.nan

        # Downsample masks to S1 (56x56 scale)
        neck_s1 = (cv2.resize(neck_mask.astype(np.float32), (w_s1_dim, h_s1_dim), interpolation=cv2.INTER_AREA) > 0.05)
        crack_s1 = (cv2.resize(crack_mask.astype(np.float32), (w_s1_dim, h_s1_dim), interpolation=cv2.INTER_AREA) > 0.05)
        bg_s1 = (cv2.resize(nearby_bg_mask.astype(np.float32), (w_s1_dim, h_s1_dim), interpolation=cv2.INTER_AREA) > 0.3)

        # For Conv1 and Conv2: compute regional mean absolute activations & signed means
        # L2 or L1 norm across channels at each spatial pixel:
        # Here we compute channel-mean absolute magnitude: mean_c |act[c, h, w]|
        for layer_name, (c_map, o_map) in [('Conv1', (c1_c, c1_o)), ('Conv2', (c2_c, c2_o))]:
            abs_center = np.mean(np.abs(c_map), axis=0) # (H_s1, W_s1)
            abs_off    = np.mean(np.abs(o_map), axis=0) # (H_s1, W_s1)
            signed_center = np.mean(c_map, axis=0)
            signed_off    = np.mean(o_map, axis=0)

            # Neck
            neck_abs_c = float(np.mean(abs_center[neck_s1])) if np.sum(neck_s1) > 0 else np.nan
            neck_abs_o = float(np.mean(abs_off[neck_s1])) if np.sum(neck_s1) > 0 else np.nan
            neck_ratio = neck_abs_o / (neck_abs_c + 1e-8) if not np.isnan(neck_abs_c) else np.nan

            neck_sgn_c = float(np.mean(signed_center[neck_s1])) if np.sum(neck_s1) > 0 else np.nan
            neck_sgn_o = float(np.mean(signed_off[neck_s1])) if np.sum(neck_s1) > 0 else np.nan

            # BG
            bg_abs_c = float(np.mean(abs_center[bg_s1])) if np.sum(bg_s1) > 0 else np.nan
            bg_abs_o = float(np.mean(abs_off[bg_s1])) if np.sum(bg_s1) > 0 else np.nan
            bg_ratio = bg_abs_o / (bg_abs_c + 1e-8) if not np.isnan(bg_abs_c) else np.nan

            # Crack
            crack_abs_c = float(np.mean(abs_center[crack_s1])) if np.sum(crack_s1) > 0 else np.nan
            crack_abs_o = float(np.mean(abs_off[crack_s1])) if np.sum(crack_s1) > 0 else np.nan

            # Contrast: Neck over BG
            contrast_center = neck_abs_c / (bg_abs_c + 1e-8) if not np.isnan(neck_abs_c) and not np.isnan(bg_abs_c) else np.nan
            contrast_off    = neck_abs_o / (bg_abs_o + 1e-8) if not np.isnan(neck_abs_o) and not np.isnan(bg_abs_o) else np.nan

            decomp_rows.append({
                'image_id': c_id,
                'stratum': stratum,
                'layer': layer_name,
                'neck_pixels_s1': int(np.sum(neck_s1)),
                'bg_pixels_s1': int(np.sum(bg_s1)),
                'neck_center_abs': round(neck_abs_c, 5),
                'neck_offcenter_abs': round(neck_abs_o, 5),
                'neck_off_over_center_ratio': round(neck_ratio, 5),
                'neck_center_signed': round(neck_sgn_c, 5),
                'neck_offcenter_signed': round(neck_sgn_o, 5),
                'bg_center_abs': round(bg_abs_c, 5),
                'bg_offcenter_abs': round(bg_abs_o, 5),
                'bg_off_over_center_ratio': round(bg_ratio, 5),
                'crack_center_abs': round(crack_abs_c, 5),
                'crack_offcenter_abs': round(crack_abs_o, 5),
                'contrast_center_neck_over_bg': round(contrast_center, 5),
                'contrast_offcenter_neck_over_bg': round(contrast_off, 5),
                'diff_center_neck_minus_bg': round(neck_abs_c - bg_abs_c, 5) if not np.isnan(neck_abs_c) else np.nan,
                'diff_offcenter_neck_minus_bg': round(neck_abs_o - bg_abs_o, 5) if not np.isnan(neck_abs_o) else np.nan,
                'connector_prob_median': round(neck_prob_median, 4),
                'connector_prob_mean': round(neck_prob_mean, 4),
            })

    df_act = pd.DataFrame(decomp_rows)
    csv_path = os.path.join(out_dir, 'conv_spatial_activation_decomposition.csv')
    df_act.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} (N={len(df_act)} sample-layer combinations)")

    # Aggregate summary by stratum and layer
    summary_rows = []
    for (strat, layer), grp in df_act.groupby(['stratum', 'layer']):
        summary_rows.append({
            'stratum': strat,
            'layer': layer,
            'N': len(grp),
            'neck_center_abs_mean': round(float(grp['neck_center_abs'].dropna().mean()), 4),
            'neck_offcenter_abs_mean': round(float(grp['neck_offcenter_abs'].dropna().mean()), 4),
            'neck_off_over_center_ratio_mean': round(float(grp['neck_off_over_center_ratio'].dropna().mean()), 4),
            'bg_center_abs_mean': round(float(grp['bg_center_abs'].dropna().mean()), 4),
            'bg_offcenter_abs_mean': round(float(grp['bg_offcenter_abs'].dropna().mean()), 4),
            'contrast_center_mean': round(float(grp['contrast_center_neck_over_bg'].dropna().mean()), 4),
            'contrast_offcenter_mean': round(float(grp['contrast_offcenter_neck_over_bg'].dropna().mean()), 4),
            'diff_center_mean': round(float(grp['diff_center_neck_minus_bg'].dropna().mean()), 4),
            'diff_offcenter_mean': round(float(grp['diff_offcenter_neck_minus_bg'].dropna().mean()), 4),
        })

    df_act_summary = pd.DataFrame(summary_rows)
    sum_csv = os.path.join(out_dir, 'conv_spatial_activation_summary.csv')
    df_act_summary.to_csv(sum_csv, index=False)
    print(f"Saved: {sum_csv}")
    print(df_act_summary[['stratum', 'layer', 'neck_off_over_center_ratio_mean', 'contrast_center_mean', 'contrast_offcenter_mean', 'diff_center_mean', 'diff_offcenter_mean']].to_string(index=False))

    return df_act, df_act_summary

# -----------------------------------------------------------------------------
# TASK D: Synthesis Mechanistic Matrix
# -----------------------------------------------------------------------------

def run_task_d(df_s2: pd.DataFrame, df_act: pd.DataFrame, out_dir: str):
    print("\n" + "=" * 80)
    print("TASK D: SYNTHESIS MECHANISTIC MATRIX")
    print("=" * 80)

    # Load v4 stage statistics to get T0, T3, T4 contrast
    v4_stats_csv = 'results/diagnostics/phase6_bridge_localization_v4/decoder_block1_stage_statistics.csv'
    df_v4 = pd.read_csv(v4_stats_csv)

    matrix_rows = []
    cases = df_act['image_id'].unique()

    for c_id in cases:
        sub_act = df_act[df_act['image_id'] == c_id]
        strat = sub_act['stratum'].iloc[0]

        # S2 representability
        s2_row = df_s2[df_s2['image_id'] == c_id]
        if len(s2_row) > 0:
            sr = s2_row.iloc[0]
            orig_min_gap = sr['orig_min_gap']
            f28_min_gap = sr['f28_min_gap']
            f28_has_merge = int(sr['f28_has_merge'])
            s16_min_gap = sr['s16_min_gap']
            s16_has_merge = int(sr['s16_has_merge'])
        else:
            orig_min_gap = np.nan
            f28_min_gap = np.nan
            f28_has_merge = 0
            s16_min_gap = np.nan
            s16_has_merge = 0

        # Conv1 and Conv2 act decomposition
        act_c1 = sub_act[sub_act['layer'] == 'Conv1'].iloc[0]
        act_c2 = sub_act[sub_act['layer'] == 'Conv2'].iloc[0]

        c1_ratio = act_c1['neck_off_over_center_ratio']
        c2_ratio = act_c2['neck_off_over_center_ratio']
        c1_contrast_off = act_c1['contrast_offcenter_neck_over_bg']
        c1_contrast_center = act_c1['contrast_center_neck_over_bg']
        c2_contrast_off = act_c2['contrast_offcenter_neck_over_bg']
        c2_contrast_center = act_c2['contrast_center_neck_over_bg']
        conn_prob = act_c1['connector_prob_median']

        # v4 contrasts
        sub_v4 = df_v4[df_v4['case_name'] == c_id]
        t0_row = sub_v4[sub_v4['stage'] == 'T0_S2_In_28x28']
        t3_row = sub_v4[sub_v4['stage'] == 'T3_PostConv1_56x56']
        t4_row = sub_v4[sub_v4['stage'] == 'T4_PostConv2_56x56']

        t0_contrast = float(t0_row['contrast_neck_over_bg'].iloc[0]) if len(t0_row) > 0 else np.nan
        t3_contrast = float(t3_row['contrast_neck_over_bg'].iloc[0]) if len(t3_row) > 0 else np.nan
        t4_contrast = float(t4_row['contrast_neck_over_bg'].iloc[0]) if len(t4_row) > 0 else np.nan

        matrix_rows.append({
            'image_id': c_id,
            'group': strat,
            'original_min_gap': orig_min_gap,
            'f28_gap_if_separate': f28_min_gap,
            'f28_components_merged': f28_has_merge,
            's16_gap_if_separate': s16_min_gap,
            's16_components_merged': s16_has_merge,
            't0_neck_bg_contrast': round(t0_contrast, 4),
            't3_contrast': round(t3_contrast, 4),
            't4_contrast': round(t4_contrast, 4),
            'conv1_offcenter_over_center_ratio': c1_ratio,
            'conv1_offcenter_contrast': c1_contrast_off,
            'conv1_center_contrast': c1_contrast_center,
            'conv2_offcenter_over_center_ratio': c2_ratio,
            'conv2_offcenter_contrast': c2_contrast_off,
            'conv2_center_contrast': c2_contrast_center,
            'connector_probability': conn_prob
        })

    df_matrix = pd.DataFrame(matrix_rows)
    csv_path = os.path.join(out_dir, 'v5pre_mechanistic_matrix.csv')
    df_matrix.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} (N={len(df_matrix)} representative cases)")

    # Print comparative table by group
    grp_summary = df_matrix.groupby('group').agg({
        'original_min_gap': 'median',
        'f28_components_merged': 'mean',
        't0_neck_bg_contrast': 'mean',
        't3_contrast': 'mean',
        't4_contrast': 'mean',
        'conv1_offcenter_over_center_ratio': 'mean',
        'conv1_offcenter_contrast': 'mean',
        'conv1_center_contrast': 'mean',
        'connector_probability': 'median'
    }).reset_index()
    print("\n--- Group Comparison in Mechanistic Matrix ---")
    print(grp_summary.to_string(index=False))

    return df_matrix

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    out_dir = 'results/diagnostics/phase6_bridge_localization_v5pre'
    os.makedirs(out_dir, exist_ok=True)

    config_path = 'results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml'
    ckpt_path = 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # TASK A: S2 GT Representability (CPU)
    df_s2, df_s2_summary = run_task_a(out_dir)

    # Load Model for Tasks B & C
    print(f"\nLoading Candidate B model from: {ckpt_path}")
    model = load_model_from_checkpoint(config_path, ckpt_path, device=device)
    model.eval()

    # TASK B: Conv Spatial Weight Decomposition
    df_weights = run_task_b(model, out_dir)

    # TASK C: Activation-Level Spatial-Mixing Analysis
    df_act, df_act_summary = run_task_c(model, device, out_dir)

    # TASK D: Synthesis Mechanistic Matrix
    df_matrix = run_task_d(df_s2, df_act, out_dir)

    print("\n" + "=" * 80)
    print("ALL V5-PRE DIAGNOSTICS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    main()
