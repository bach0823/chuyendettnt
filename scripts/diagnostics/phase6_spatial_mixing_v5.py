#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_spatial_mixing_v5.py

Phase 6 V5 Counterfactual Spatial-Mixing Causality Test:
Investigates:
Is off-center spatial mixing inside Decoder Block 1 necessary for the false bridge?

Experimental Conditions:
- C0: Normal Baseline (alpha1=1.0, alpha2=1.0)
- C1: Conv1 spatial mixing removed (alpha1=0.0, alpha2=1.0)
- C2: Conv2 spatial mixing removed (alpha1=1.0, alpha2=0.0)
- C3: Both spatial mixing removed (alpha1=0.0, alpha2=0.0)
- C1_half: Intermediate Conv1 attenuation (alpha1=0.5, alpha2=1.0)
- C2_half: Intermediate Conv2 attenuation (alpha1=1.0, alpha2=0.5)
- C3_half: Intermediate Both attenuation (alpha1=0.5, alpha2=0.5)

STRICT CONSTRAINTS:
- Diagnostic-Only: Zero training, zero parameter updates, zero model mutations saved to disk.
- Checkpoints, configs, and threshold (tau=0.5) remain strictly unchanged.
- Test set (N=1124) is strictly sealed and untouched.
- Setting A evaluation path preserved exactly.
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
from skimage.morphology import skeletonize
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, 'SAGE_LITE')
from tools.run_phase6_c_topology_diagnostic import load_model_from_checkpoint

# -----------------------------------------------------------------------------
# Metric Utilities (Standardized across Phase 6)
# -----------------------------------------------------------------------------

def compute_topology_and_global_metrics(pred_bin: np.ndarray, target_bin: np.ndarray) -> Dict[str, Any]:
    assert pred_bin.shape == target_bin.shape
    H, W = target_bin.shape

    num_gt_cc, gt_labels = cv2.connectedComponents(target_bin.astype(np.uint8), connectivity=8)
    num_pred_cc, pred_labels = cv2.connectedComponents(pred_bin.astype(np.uint8), connectivity=8)

    gt_cc = int(max(num_gt_cc - 1, 0))
    pred_cc = int(max(num_pred_cc - 1, 0))

    tp = int(np.sum((pred_bin == 1) & (target_bin == 1)))
    fp = int(np.sum((pred_bin == 1) & (target_bin == 0)))
    fn = int(np.sum((pred_bin == 0) & (target_bin == 1)))
    dice = float((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8))
    precision = float(tp / (tp + fp + 1e-8))
    recall = float(tp / (tp + fn + 1e-8))
    gt_area = int(np.sum(target_bin == 1))
    pred_area = int(np.sum(pred_bin == 1))
    area_excess = float((pred_area - gt_area) / max(gt_area, 1) * 100.0)

    # False Bridge
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

    # Breakage / Fragmentation
    fragmented_gt_components = 0
    for g_id in range(1, gt_cc + 1):
        overlapping_pred = np.unique(pred_labels[gt_labels == g_id])
        overlapping_pred = overlapping_pred[overlapping_pred > 0]
        if len(overlapping_pred) >= 2:
            fragmented_gt_components += 1

    has_breakage = int(fragmented_gt_components > 0)

    # clDice
    s_gt = skeletonize(target_bin > 0)
    s_pred = skeletonize(pred_bin > 0)
    len_s_gt = int(np.sum(s_gt))
    len_s_pred = int(np.sum(s_pred))

    if len_s_pred == 0:
        tprec = 1.0 if len_s_gt == 0 else 0.0
    else:
        tprec = float(np.sum(s_pred & (target_bin > 0))) / float(len_s_pred)

    if len_s_gt == 0:
        tsens = 1.0 if len_s_pred == 0 else 0.0
    else:
        tsens = float(np.sum(s_gt & (pred_bin > 0))) / float(len_s_gt)

    if tprec + tsens == 0.0:
        cldice = 0.0
    else:
        cldice = float(2.0 * tprec * tsens / (tprec + tsens))

    return {
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'area_excess': area_excess,
        'pred_area': pred_area,
        'gt_area': gt_area,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'has_bridge': has_bridge,
        'bridge_events': bridge_events,
        'merged_gt_components': len(merged_gt_set),
        'has_breakage': has_breakage,
        'fragmented_gt_components': fragmented_gt_components,
        'tprec': tprec,
        'tsens': tsens,
        'cldice': cldice,
    }


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
# Spatial Mixing Attenuation Helper
# -----------------------------------------------------------------------------

def make_attenuated_weight(orig_weight: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Computes W' = W_center + alpha * W_off.
    For a 3x3 conv:
    Center tap is [:, :, 1, 1] (alpha=1.0 retains center exactly).
    All 8 off-center taps [:, :, (i,j) != (1,1)] are multiplied by alpha.
    """
    w_new = orig_weight.clone()
    for i in range(3):
        for j in range(3):
            if (i, j) != (1, 1):
                w_new[:, :, i, j] *= alpha
    return w_new

# -----------------------------------------------------------------------------
# Setting A Tiled Inference Engine
# -----------------------------------------------------------------------------

def run_setting_a_inference(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    tile_size: int = 448
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Executes Setting A tiling inference:
    tile=448, stride=448, reflect padding, exact cropping back.
    Returns: (pred_bin, pred_prob)
    """
    H, W = image.shape[:2]
    pad_h = (tile_size - (H % tile_size)) % tile_size
    pad_w = (tile_size - (W % tile_size)) % tile_size
    padded_img = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    pH, pW = padded_img.shape[:2]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    patches = []
    coords = []
    for y in range(0, pH, tile_size):
        for x in range(0, pW, tile_size):
            patch = padded_img[y:y+tile_size, x:x+tile_size]
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            patch_tensor = (patch_tensor - mean) / std
            patches.append(patch_tensor)
            coords.append((y, x))

    final_logits = np.zeros((pH, pW), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for patch_t, (py, px) in zip(patches, coords):
            inp = patch_t.unsqueeze(0).to(device)
            logits = model(inp)
            if logits.shape[2:] != (tile_size, tile_size):
                logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
            final_logits[py:py+tile_size, px:px+tile_size] = logits.squeeze().cpu().numpy()

    logits_cropped = final_logits[:H, :W]
    prob_cropped = 1.0 / (1.0 + np.exp(-logits_cropped))
    pred_bin = (logits_cropped > 0.0).astype(np.uint8)

    return pred_bin, prob_cropped

# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    out_dir = 'results/diagnostics/phase6_spatial_mixing_v5'
    os.makedirs(out_dir, exist_ok=True)

    config_path = 'results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml'
    ckpt_path = 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    print(f"\nLoading Candidate B model from: {ckpt_path}")
    model = load_model_from_checkpoint(config_path, ckpt_path, device=device)
    model.eval()

    # Parameter Count Sanity Check
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Candidate B Parameter count: {total_params:,} (Expected: 10,118,955)")
    assert total_params == 10118955, f"Parameter count mismatch: {total_params}"

    block1 = model.decoder.decoder_blocks[1]
    conv1 = block1.conv1[0]
    conv2 = block1.conv2[0]

    # Save pristine in-memory weights
    orig_w1 = conv1.weight.data.clone()
    orig_w2 = conv2.weight.data.clone()

    # Load exact 25 representative validation samples from v4
    case_ids_csv = 'results/diagnostics/phase6_bridge_localization_v4/selected_case_ids.csv'
    df_cases = pd.read_csv(case_ids_csv)
    sample_cases_meta = list(zip(df_cases['image_id'], df_cases['stratum']))
    assert len(sample_cases_meta) == 25, f"Expected 25 cases, got {len(sample_cases_meta)}"

    data_root = 'datasets/Crack500_ready'
    val_img_dir = os.path.join(data_root, 'val', 'images')
    val_mask_dir = os.path.join(data_root, 'val', 'masks')

    # Load all images and targets in memory
    samples = []
    for c_id, grp in sample_cases_meta:
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

        samples.append({
            'case_name': c_id,
            'stratum': grp,
            'img': img,
            'target_bin': target_bin,
        })

    # Define experimental conditions:
    # 4 Primary conditions: C0, C1, C2, C3
    # 3 Intermediate conditions: C1_half, C2_half, C3_half
    conditions = [
        ('C0_Normal',     1.0, 1.0, 'Normal Candidate B baseline'),
        ('C1_Conv1_Zero', 0.0, 1.0, 'Conv1 spatial mixing removed (W1_off = 0)'),
        ('C2_Conv2_Zero', 1.0, 0.0, 'Conv2 spatial mixing removed (W2_off = 0)'),
        ('C3_Both_Zero',  0.0, 0.0, 'Both Conv1 and Conv2 spatial mixing removed'),
        ('C1_Conv1_Half', 0.5, 1.0, 'Conv1 intermediate attenuation (alpha1 = 0.5)'),
        ('C2_Conv2_Half', 1.0, 0.5, 'Conv2 intermediate attenuation (alpha2 = 0.5)'),
        ('C3_Both_Half',  0.5, 0.5, 'Both intermediate attenuation (alpha1=alpha2=0.5)'),
    ]

    all_per_sample_rows = []
    c0_bridge_corridors = {}  # Store C0 neck mask for each sample

    print("\n" + "=" * 80)
    print("EXECUTING PHASE 6 V5 COUNTERFACTUAL SPATIAL-MIXING INFERENCE")
    print("=" * 80)

    for cond_name, a1, a2, desc in conditions:
        print(f"\n--- Running Condition: {cond_name} (alpha1={a1}, alpha2={a2}) ---")
        print(f"    Description: {desc}")

        # Compute modified weights
        w1_mod = make_attenuated_weight(orig_w1, a1)
        w2_mod = make_attenuated_weight(orig_w2, a2)

        # Numerical verification of intervention integrity
        assert torch.equal(w1_mod[:, :, 1, 1], orig_w1[:, :, 1, 1]), "Conv1 center tap altered!"
        assert torch.equal(w2_mod[:, :, 1, 1], orig_w2[:, :, 1, 1]), "Conv2 center tap altered!"
        if a1 == 0.0:
            assert torch.all(w1_mod[:, :, [0,0,0,1,1,2,2,2], [0,1,2,0,2,0,1,2]] == 0.0), "Conv1 off-center not zeroed!"
        if a2 == 0.0:
            assert torch.all(w2_mod[:, :, [0,0,0,1,1,2,2,2], [0,1,2,0,2,0,1,2]] == 0.0), "Conv2 off-center not zeroed!"

        try:
            # Apply intervention in memory
            conv1.weight.data.copy_(w1_mod)
            conv2.weight.data.copy_(w2_mod)

            # Evaluate on all 25 samples
            for s in tqdm(samples, desc=f"Inference {cond_name}"):
                c_id = s['case_name']
                grp = s['stratum']
                target_bin = s['target_bin']

                pred_bin, pred_prob = run_setting_a_inference(model, s['img'], device, tile_size=448)
                metrics = compute_topology_and_global_metrics(pred_bin, target_bin)
                whole_bridge_fp, neck_mask = isolate_base_bridge_corridor(pred_bin, target_bin)

                if cond_name == 'C0_Normal':
                    c0_bridge_corridors[c_id] = neck_mask.copy()

                # Connector probability (on condition's own neck mask)
                if np.sum(neck_mask) > 0:
                    neck_prob_median = float(np.median(pred_prob[neck_mask > 0]))
                    neck_prob_mean   = float(np.mean(pred_prob[neck_mask > 0]))
                    neck_area        = int(np.sum(neck_mask > 0))
                else:
                    neck_prob_median = np.nan
                    neck_prob_mean   = np.nan
                    neck_area        = 0

                # Fixed C0 corridor probability (to track attenuation over constant coordinates)
                c0_neck = c0_bridge_corridors.get(c_id, np.zeros_like(pred_bin))
                if np.sum(c0_neck) > 0:
                    c0_neck_prob_median = float(np.median(pred_prob[c0_neck > 0]))
                    c0_neck_prob_mean   = float(np.mean(pred_prob[c0_neck > 0]))
                else:
                    c0_neck_prob_median = np.nan
                    c0_neck_prob_mean   = np.nan

                all_per_sample_rows.append({
                    'condition': cond_name,
                    'alpha1': a1,
                    'alpha2': a2,
                    'image_id': c_id,
                    'stratum': grp,
                    'dice': round(metrics['dice'], 5),
                    'precision': round(metrics['precision'], 5),
                    'recall': round(metrics['recall'], 5),
                    'area_excess': round(metrics['area_excess'], 2),
                    'cldice': round(metrics['cldice'], 5),
                    'has_bridge': metrics['has_bridge'],
                    'bridge_events': metrics['bridge_events'],
                    'has_breakage': metrics['has_breakage'],
                    'fragmented_gt_components': metrics['fragmented_gt_components'],
                    'neck_area_px': neck_area,
                    'neck_prob_median': round(neck_prob_median, 4) if not np.isnan(neck_prob_median) else np.nan,
                    'c0_neck_prob_median': round(c0_neck_prob_median, 4) if not np.isnan(c0_neck_prob_median) else np.nan,
                    'c0_neck_prob_mean': round(c0_neck_prob_mean, 4) if not np.isnan(c0_neck_prob_mean) else np.nan,
                })

        finally:
            # Unconditional restoration
            conv1.weight.data.copy_(orig_w1)
            conv2.weight.data.copy_(orig_w2)

    # Verify model is perfectly restored
    assert torch.equal(conv1.weight.data, orig_w1), "Model Conv1 weight restoration failed!"
    assert torch.equal(conv2.weight.data, orig_w2), "Model Conv2 weight restoration failed!"
    print("\nSanity Check Passed: In-memory weights verified 100% restored to baseline.")

    df_per_sample = pd.DataFrame(all_per_sample_rows)
    per_sample_csv = os.path.join(out_dir, 'v5_per_sample_results.csv')
    df_per_sample.to_csv(per_sample_csv, index=False)
    print(f"Saved: {per_sample_csv} (N={len(df_per_sample)} rows)")

    # -------------------------------------------------------------------------
    # C0 Reproduction Sanity Check
    # -------------------------------------------------------------------------
    df_c0 = df_per_sample[df_per_sample['condition'] == 'C0_Normal']
    c0_consensus = df_c0[df_c0['stratum'] == 'Consensus_7of7']
    c0_cured = df_c0[df_c0['stratum'] == 'Cured_by_D2']
    c0_created = df_c0[df_c0['stratum'] == 'Created_by_D2']
    c0_low = df_c0[df_c0['stratum'] == 'Low_Consensus']

    print("\n--- C0 BASELINE REPRODUCTION VERIFICATION ---")
    print(f"Consensus 7/7 Bridges: {c0_consensus['has_bridge'].sum()} / {len(c0_consensus)} (Expected: 10/10)")
    print(f"Cured by D2 Bridges:   {c0_cured['has_bridge'].sum()} / {len(c0_cured)} (Expected: 5/5 in Base)")
    print(f"Created by D2 Bridges: {c0_created['has_bridge'].sum()} / {len(c0_created)} (Expected: 0/5 in Base)")
    print(f"Low Consensus Bridges: {c0_low['has_bridge'].sum()} / {len(c0_low)} (Expected: 1/5 in Base)")
    print(f"C0 Consensus Mean Dice: {c0_consensus['dice'].mean():.4f} (Expected: 0.8181)")
    print(f"C0 Consensus Mean Recall: {c0_consensus['recall'].mean():.4f} (Expected: 0.9139)")

    assert c0_consensus['has_bridge'].sum() == 10, "C0 failed to reproduce 10/10 consensus bridges!"
    assert c0_cured['has_bridge'].sum() == 5, "C0 failed to reproduce 5/5 cured base bridges!"
    assert c0_created['has_bridge'].sum() == 0, "C0 failed to reproduce 0/5 created base bridges!"

    # -------------------------------------------------------------------------
    # Transition Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PAIRED TRANSITION ANALYSIS RELATIVE TO C0")
    print("=" * 80)

    # Reference C0 states
    c0_map = dict(zip(df_c0['image_id'], df_c0['has_bridge']))

    transition_rows = []
    eval_conds = [c for c in df_per_sample['condition'].unique() if c != 'C0_Normal']

    for cond_name in eval_conds:
        df_sub = df_per_sample[df_per_sample['condition'] == cond_name]

        for strat in ['Consensus_7of7', 'Cured_by_D2', 'All_25_Representative']:
            if strat == 'All_25_Representative':
                strat_df = df_sub
                strat_c0 = df_c0
            else:
                strat_df = df_sub[df_sub['stratum'] == strat]
                strat_c0 = df_c0[df_c0['stratum'] == strat]

            n_cases = len(strat_df)
            if n_cases == 0:
                continue

            c0_bridges = int(strat_c0['has_bridge'].sum())
            cond_bridges = int(strat_df['has_bridge'].sum())

            # Paired transitions
            cured_count = 0
            persistent_count = 0
            created_count = 0

            for _, row in strat_df.iterrows():
                cid = row['image_id']
                was_br = c0_map[cid]
                is_br = row['has_bridge']

                if was_br == 1 and is_br == 0:
                    cured_count += 1
                elif was_br == 1 and is_br == 1:
                    persistent_count += 1
                elif was_br == 0 and is_br == 1:
                    created_count += 1

            transition_rows.append({
                'condition': cond_name,
                'stratum': strat,
                'N': n_cases,
                'c0_bridges': c0_bridges,
                'cond_bridges': cond_bridges,
                'cured_vs_c0': cured_count,
                'persistent_vs_c0': persistent_count,
                'created_vs_c0': created_count,
                'bridge_persistence_pct': round((persistent_count / max(c0_bridges, 1)) * 100.0, 1),
                'mean_dice': round(float(strat_df['dice'].mean()), 4),
                'mean_recall': round(float(strat_df['recall'].mean()), 4),
                'mean_precision': round(float(strat_df['precision'].mean()), 4),
                'mean_cldice': round(float(strat_df['cldice'].mean()), 4),
                'breakage_cases': int(strat_df['has_breakage'].sum()),
                'breakage_pct': round((strat_df['has_breakage'].sum() / n_cases) * 100.0, 1),
                'mean_c0_neck_prob': round(float(strat_df['c0_neck_prob_median'].dropna().mean()), 4) if len(strat_df['c0_neck_prob_median'].dropna()) > 0 else np.nan,
            })

    df_trans = pd.DataFrame(transition_rows)
    trans_csv = os.path.join(out_dir, 'v5_transition_analysis.csv')
    df_trans.to_csv(trans_csv, index=False)
    print(f"Saved: {trans_csv}")

    # Display transition table for Consensus 7/7
    c7_trans = df_trans[df_trans['stratum'] == 'Consensus_7of7']
    print("\n--- Consensus 7/7 Transitions ---")
    print(c7_trans[['condition', 'c0_bridges', 'cond_bridges', 'cured_vs_c0', 'persistent_vs_c0', 'created_vs_c0', 'mean_dice', 'mean_recall', 'breakage_cases', 'mean_c0_neck_prob']].to_string(index=False))

    # -------------------------------------------------------------------------
    # Connector Probability Analysis
    # -------------------------------------------------------------------------
    prob_rows = []
    for cond_name in df_per_sample['condition'].unique():
        sub = df_per_sample[df_per_sample['condition'] == cond_name]
        for strat in ['Consensus_7of7', 'Cured_by_D2', 'All_25_Representative']:
            if strat == 'All_25_Representative':
                s_df = sub
            else:
                s_df = sub[sub['stratum'] == strat]

            p_vals = s_df['c0_neck_prob_median'].dropna()
            prob_rows.append({
                'condition': cond_name,
                'stratum': strat,
                'N_measured': len(p_vals),
                'prob_mean': round(float(p_vals.mean()), 4) if len(p_vals) > 0 else np.nan,
                'prob_median': round(float(p_vals.median()), 4) if len(p_vals) > 0 else np.nan,
                'prob_p25': round(float(np.percentile(p_vals, 25)), 4) if len(p_vals) > 0 else np.nan,
                'prob_p75': round(float(np.percentile(p_vals, 75)), 4) if len(p_vals) > 0 else np.nan,
                'prob_min': round(float(np.min(p_vals)), 4) if len(p_vals) > 0 else np.nan,
                'prob_max': round(float(np.max(p_vals)), 4) if len(p_vals) > 0 else np.nan,
                'pct_ge_50': round(float(np.mean(p_vals >= 0.50) * 100.0), 2) if len(p_vals) > 0 else np.nan,
            })

    df_prob = pd.DataFrame(prob_rows)
    prob_csv = os.path.join(out_dir, 'v5_connector_probability.csv')
    df_prob.to_csv(prob_csv, index=False)
    print(f"\nSaved: {prob_csv}")

    # -------------------------------------------------------------------------
    # Overall Condition Summary
    # -------------------------------------------------------------------------
    sum_rows = []
    for cond_name in df_per_sample['condition'].unique():
        sub = df_per_sample[df_per_sample['condition'] == cond_name]
        c7_sub = sub[sub['stratum'] == 'Consensus_7of7']
        cured_sub = sub[sub['stratum'] == 'Cured_by_D2']

        sum_rows.append({
            'condition': cond_name,
            'c7_bridges_remaining': f"{c7_sub['has_bridge'].sum()}/10",
            'c7_cured': f"{10 - c7_sub['has_bridge'].sum()}/10",
            'c7_mean_dice': round(float(c7_sub['dice'].mean()), 4),
            'c7_mean_recall': round(float(c7_sub['recall'].mean()), 4),
            'c7_breakage': f"{c7_sub['has_breakage'].sum()}/10",
            'c7_c0_neck_prob_median': round(float(c7_sub['c0_neck_prob_median'].dropna().median()), 4),
            'cured_bridges_remaining': f"{cured_sub['has_bridge'].sum()}/5",
            'all25_total_bridges': f"{sub['has_bridge'].sum()}/25",
            'all25_mean_dice': round(float(sub['dice'].mean()), 4),
            'all25_mean_recall': round(float(sub['recall'].mean()), 4),
            'all25_breakage': f"{sub['has_breakage'].sum()}/25",
        })

    df_summary = pd.DataFrame(sum_rows)
    sum_csv = os.path.join(out_dir, 'counterfactual_v5_summary.csv')
    df_summary.to_csv(sum_csv, index=False)
    print(f"\nSaved: {sum_csv}")
    print("\n--- MASTER CONDITION SUMMARY ---")
    print(df_summary[['condition', 'c7_bridges_remaining', 'c7_mean_dice', 'c7_mean_recall', 'c7_breakage', 'c7_c0_neck_prob_median']].to_string(index=False))

    print("\n" + "=" * 80)
    print("PHASE 6 V5 COUNTERFACTUAL EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    main()
