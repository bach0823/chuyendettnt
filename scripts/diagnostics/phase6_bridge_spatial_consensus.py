#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_bridge_spatial_consensus.py

Task A: Spatial Bridge Consistency Diagnostic Across 7 Models
Answers:
- Do the 101 consensus failure cases fail in the exact same spatial location and geometry?
- Is there a tight pixel-level bridge-core shared across all models, or do different models bridge different gaps?
- What is the spatial overlap between Base and D2 on the 103 persistent cases?
- What happened spatially in the 7 cured cases and 6 created cases?

Target subsets:
- 101 7/7 consensus cases
- 103 Base∩D2 persistent cases
- 7 D2 cured cases
- 6 D2 created cases
Total unique cases = 116.
"""

import gc
import glob
import itertools
import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add paths
sys.path.insert(0, 'SAGE_LITE')
from tools.run_phase6_c_topology_diagnostic import load_model_from_checkpoint
from scripts.evaluate_crack_official import predict_full_image_tiling_setting_a

def extract_bridge_masks(pred_bin: np.ndarray, target_bin: np.ndarray):
    """
    Identifies bridge connected components and extracts:
    - bridge_fp_mask: (pred == 1) & (gt == 0) within CCs overlapping >=2 GT CCs
    - bridge_comp_mask: (pred == 1) within CCs overlapping >=2 GT CCs
    - num_bridge_events: number of distinct predicted CCs that bridge >= 2 GT CCs
    - num_merged_gt: number of distinct GT CCs merged by bridge CCs
    """
    num_gt_cc, gt_labels = cv2.connectedComponents(target_bin, connectivity=8)
    num_pred_cc, pred_labels = cv2.connectedComponents(pred_bin, connectivity=8)
    pred_cc = int(max(num_pred_cc - 1, 0))

    bridge_pred_ids = set()
    merged_gt_set = set()
    for p_id in range(1, pred_cc + 1):
        comp_mask = (pred_labels == p_id)
        overlapping_gt = np.unique(gt_labels[comp_mask])
        overlapping_gt = overlapping_gt[overlapping_gt > 0]
        if len(overlapping_gt) >= 2:
            bridge_pred_ids.add(p_id)
            for g_id in overlapping_gt:
                merged_gt_set.add(int(g_id))

    is_bridge_comp = np.isin(pred_labels, list(bridge_pred_ids))
    bridge_comp_mask = is_bridge_comp.astype(np.uint8)
    bridge_fp_mask = (is_bridge_comp & (target_bin == 0)).astype(np.uint8)

    return bridge_fp_mask, bridge_comp_mask, len(bridge_pred_ids), len(merged_gt_set)

def compute_pairwise_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = np.sum((mask1 > 0) & (mask2 > 0))
    union = np.sum((mask1 > 0) | (mask2 > 0))
    if union == 0:
        return 1.0 if np.sum(mask1) == 0 and np.sum(mask2) == 0 else 0.0
    return float(inter / union)

def compute_pairwise_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = np.sum((mask1 > 0) & (mask2 > 0))
    tot = np.sum(mask1 > 0) + np.sum(mask2 > 0)
    if tot == 0:
        return 1.0 if np.sum(mask1) == 0 and np.sum(mask2) == 0 else 0.0
    return float(2.0 * inter / tot)

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    out_dir = 'results/diagnostics/phase6_bridge_localization_v2'
    os.makedirs(out_dir, exist_ok=True)

    consensus_csv = 'results/diagnostics/phase6_bridge_localization/bridge_consensus_per_image.csv'
    df_meta = pd.read_csv(consensus_csv)

    # Filter target 116 images
    c7_cases = set(df_meta[df_meta['n_models_with_bridge'] == 7]['image_id'])
    persistent_cases = set(df_meta[df_meta['d2_transition_type'] == 'Persistent_Base_to_D2']['image_id'])
    cured_cases = set(df_meta[df_meta['d2_transition_type'] == 'Cured_by_D2']['image_id'])
    created_cases = set(df_meta[df_meta['d2_transition_type'] == 'Created_by_D2']['image_id'])

    target_case_set = c7_cases.union(persistent_cases).union(cured_cases).union(created_cases)
    target_cases = sorted(list(target_case_set))
    assert len(target_cases) == 116, f"Expected 116 cases, found {len(target_cases)}"

    print(f"Targeting {len(target_cases)} unique validation cases:")
    print(f"  - 7/7 consensus: {len(c7_cases)}")
    print(f"  - Persistent Base->D2: {len(persistent_cases)}")
    print(f"  - Cured by D2: {len(cured_cases)}")
    print(f"  - Created by D2: {len(created_cases)}")

    # Model definitions
    model_configs = {
        'Base': ('results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml',
                 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'),
        'A1': ('results/configs/b2_p3_run_c_d4_k2_h64_phase6_a1_boundary_iou.yaml',
               'results/checkpoints/P3_C_Phase6_A1_BoundaryIoU_D4_K2_H64_best_model_b2_global.pth'),
        'A2': ('results/configs/b2_p3_run_c_d4_k2_h64_phase6_a2_pure_plu.yaml',
               'results/checkpoints/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_best_model_b2_global.pth'),
        'B1': ('results/configs/b2_p3_run_c_d4_k2_h64_phase6_b1_ab_bpl.yaml',
               'results/checkpoints/P3_C_Phase6_B1_AB_BPL_D4_K2_H64_best_model_b2_global.pth'),
        'C1': ('results/configs/b2_p3_run_c_d4_k2_h64_phase6_c1_cldice.yaml',
               'results/checkpoints/P3_C_Phase6_C1_clDice_D4_K2_H64_best_model_b2_global.pth'),
        'D1': ('results/configs/b2_p3_run_c_d4_k2_h64_phase6_d1_plu_abbpl.yaml',
               'results/checkpoints/P3_C_Phase6_D1_PLU_ABBPL_D4_K2_H64_best_model_b2_global.pth'),
        'D2': ('results/configs/b2_p3_run_c_d4_k2_h64_phase6_d2_pure_sep.yaml',
               'results/checkpoints/P3_C_Phase6_D2_Pure_Sep_D4_K2_H64_best_model_b2_global.pth'),
    }
    models = list(model_configs.keys())

    data_root = 'datasets/Crack500_ready'
    val_img_dir = os.path.join(data_root, 'val', 'images')
    val_mask_dir = os.path.join(data_root, 'val', 'masks')

    # Load image and GT paths
    case_files = {}
    for c in target_cases:
        ip = os.path.join(val_img_dir, c + '.jpg')
        if not os.path.exists(ip):
            ip = os.path.join(val_img_dir, c + '.png')
        mp = os.path.join(val_mask_dir, c + '.png')
        if not os.path.exists(mp):
            mp = os.path.join(val_mask_dir, c + '.jpg')
        assert os.path.exists(ip), f"Missing image {ip}"
        assert os.path.exists(mp), f"Missing mask {mp}"
        case_files[c] = (ip, mp)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = False

    # Store bridge masks: bridge_fp_masks[model][case_name] = np.ndarray
    bridge_fp_masks = {m: {} for m in models}
    bridge_comp_masks = {m: {} for m in models}
    bridge_event_counts = {m: {} for m in models}

    # Run inference per model
    print("\n" + "=" * 80)
    print("STEP 1: Running Setting A inference across 7 models on 116 target cases...")
    print("=" * 80)

    for m_idx, m_name in enumerate(models):
        cfg_path, ckpt_path = model_configs[m_name]
        print(f"\n[{m_idx + 1}/7] Loading model '{m_name}' from {ckpt_path}...")
        model = load_model_from_checkpoint(cfg_path, ckpt_path, device)
        model.eval()

        t0 = time.time()
        for case_name in tqdm(target_cases, desc=f"Inference {m_name}"):
            ip, mp = case_files[case_name]
            img = cv2.imread(ip)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            target = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            target_bin = (target > 127).astype(np.uint8)

            with torch.no_grad():
                logits_np = predict_full_image_tiling_setting_a(
                    model, img, device, tile_size=448, batch_size=8
                )

            pred_bin = (logits_np > 0.0).astype(np.uint8)[:target_bin.shape[0], :target_bin.shape[1]]
            fp_m, comp_m, n_ev, n_mgt = extract_bridge_masks(pred_bin, target_bin)

            bridge_fp_masks[m_name][case_name] = fp_m
            bridge_comp_masks[m_name][case_name] = comp_m
            bridge_event_counts[m_name][case_name] = n_ev

        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        dt = time.time() - t0
        print(f"Finished {m_name} in {dt:.1f}s")

    # STEP 2: Compute spatial consensus and overlaps
    print("\n" + "=" * 80)
    print("STEP 2: Computing pairwise and multi-model spatial consensus metrics...")
    print("=" * 80)

    per_image_rows = []
    pairs = list(itertools.combinations(models, 2))  # 21 pairs

    for case_name in target_cases:
        ip, mp = case_files[case_name]
        target = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        target_bin = (target > 127).astype(np.uint8)
        H, W = target_bin.shape

        is_c7 = case_name in c7_cases
        is_persistent = case_name in persistent_cases
        is_cured = case_name in cured_cases
        is_created = case_name in created_cases

        # Transition category
        if is_cured:
            trans_cat = "Cured_by_D2"
        elif is_created:
            trans_cat = "Created_by_D2"
        elif is_persistent:
            trans_cat = "Persistent_Base_to_D2"
        else:
            trans_cat = "Other"

        row = {
            'image_id': case_name,
            'is_7_consensus': int(is_c7),
            'd2_transition': trans_cat,
            'image_height': H,
            'image_width': W,
            'gt_pixels': int(np.sum(target_bin)),
        }

        # Individual model bridge pixel counts
        for m in models:
            fp_m = bridge_fp_masks[m][case_name]
            row[f'{m}_bridge_fp_pixels'] = int(np.sum(fp_m))
            row[f'{m}_bridge_events'] = bridge_event_counts[m][case_name]

        # Pairwise IoUs of bridge-FP
        pairwise_ious = []
        for m1, m2 in pairs:
            fp1 = bridge_fp_masks[m1][case_name]
            fp2 = bridge_fp_masks[m2][case_name]
            p_iou = compute_pairwise_iou(fp1, fp2)
            row[f'iou_{m1}_vs_{m2}'] = round(p_iou, 4)
            # Only include in mean if at least one model bridged
            if np.sum(fp1) > 0 or np.sum(fp2) > 0:
                pairwise_ious.append(p_iou)

        row['mean_pairwise_bridge_iou'] = round(float(np.mean(pairwise_ious)), 4) if pairwise_ious else np.nan
        row['base_vs_d2_bridge_iou'] = round(compute_pairwise_iou(bridge_fp_masks['Base'][case_name], bridge_fp_masks['D2'][case_name]), 4)
        row['base_vs_d2_bridge_dice'] = round(compute_pairwise_dice(bridge_fp_masks['Base'][case_name], bridge_fp_masks['D2'][case_name]), 4)

        # 7-way spatial intersection and union
        stack_masks = np.stack([bridge_fp_masks[m][case_name] for m in models], axis=0)  # (7, H, W)
        sum_mask = np.sum(stack_masks, axis=0)  # pixel values in [0, 7]

        inter_7 = int(np.sum(sum_mask == 7))
        union_7 = int(np.sum(sum_mask >= 1))
        inter_ge4 = int(np.sum(sum_mask >= 4))
        inter_ge5 = int(np.sum(sum_mask >= 5))
        inter_ge6 = int(np.sum(sum_mask >= 6))

        row['pixels_7way_intersection'] = inter_7
        row['pixels_7way_union'] = union_7
        row['iou_7way_strict'] = round(float(inter_7 / union_7), 4) if union_7 > 0 else (1.0 if union_7 == 0 else 0.0)
        row['ratio_core_ge4_over_union'] = round(float(inter_ge4 / union_7), 4) if union_7 > 0 else np.nan
        row['ratio_core_ge5_over_union'] = round(float(inter_ge5 / union_7), 4) if union_7 > 0 else np.nan
        row['ratio_core_ge6_over_union'] = round(float(inter_ge6 / union_7), 4) if union_7 > 0 else np.nan

        # Pixel agreement histogram (1..7)
        for k in range(1, 8):
            row[f'pixels_with_{k}_models'] = int(np.sum(sum_mask == k))

        per_image_rows.append(row)

    df_per_image = pd.DataFrame(per_image_rows)
    out_per_image_path = os.path.join(out_dir, 'bridge_spatial_consensus_per_image.csv')
    df_per_image.to_csv(out_per_image_path, index=False)
    print(f"Saved: {out_per_image_path} ({len(df_per_image)} rows)")

    # STEP 3: Summary table by stratum
    print("\n" + "=" * 80)
    print("STEP 3: Aggregating spatial consensus summary...")
    print("=" * 80)

    summary_rows = []
    strata = [
        ('Consensus_7of7', df_per_image[df_per_image['is_7_consensus'] == 1]),
        ('Persistent_Base_to_D2', df_per_image[df_per_image['d2_transition'] == 'Persistent_Base_to_D2']),
        ('Cured_by_D2', df_per_image[df_per_image['d2_transition'] == 'Cured_by_D2']),
        ('Created_by_D2', df_per_image[df_per_image['d2_transition'] == 'Created_by_D2']),
        ('All_116_Target_Cases', df_per_image),
    ]

    for s_name, sub in strata:
        n = len(sub)
        if n == 0:
            continue

        mean_inter7 = sub['pixels_7way_intersection'].mean()
        mean_union7 = sub['pixels_7way_union'].mean()
        mean_iou7 = sub['iou_7way_strict'].mean()
        med_iou7 = sub['iou_7way_strict'].median()
        mean_pair_iou = sub['mean_pairwise_bridge_iou'].mean()
        med_pair_iou = sub['mean_pairwise_bridge_iou'].median()
        mean_base_d2_iou = sub['base_vs_d2_bridge_iou'].mean()
        med_base_d2_iou = sub['base_vs_d2_bridge_iou'].median()
        mean_base_d2_dice = sub['base_vs_d2_bridge_dice'].mean()

        mean_ge4_ratio = sub['ratio_core_ge4_over_union'].mean()
        mean_ge5_ratio = sub['ratio_core_ge5_over_union'].mean()
        mean_ge6_ratio = sub['ratio_core_ge6_over_union'].mean()

        summary_rows.append({
            'stratum': s_name,
            'N_cases': n,
            'mean_pairwise_iou': round(mean_pair_iou, 4),
            'median_pairwise_iou': round(med_pair_iou, 4),
            'mean_base_vs_d2_iou': round(mean_base_d2_iou, 4),
            'median_base_vs_d2_iou': round(med_base_d2_iou, 4),
            'mean_base_vs_d2_dice': round(mean_base_d2_dice, 4),
            'mean_7way_strict_iou': round(mean_iou7, 4),
            'median_7way_strict_iou': round(med_iou7, 4),
            'mean_7way_intersection_px': round(mean_inter7, 1),
            'mean_7way_union_px': round(mean_union7, 1),
            'mean_ge4_models_share_ratio': round(mean_ge4_ratio, 4),
            'mean_ge5_models_share_ratio': round(mean_ge5_ratio, 4),
            'mean_ge6_models_share_ratio': round(mean_ge6_ratio, 4),
        })

    df_summary = pd.DataFrame(summary_rows)
    out_summary_path = os.path.join(out_dir, 'bridge_spatial_consensus_summary.csv')
    df_summary.to_csv(out_summary_path, index=False)
    print(f"Saved: {out_summary_path}")

    # Pairwise Model IoU Matrix across the 101 consensus cases
    sub_101 = df_per_image[df_per_image['is_7_consensus'] == 1]
    matrix_rows = []
    for m1 in models:
        row_m = {'model': m1}
        for m2 in models:
            if m1 == m2:
                row_m[m2] = 1.0
            else:
                col_name = f'iou_{m1}_vs_{m2}' if f'iou_{m1}_vs_{m2}' in sub_101.columns else f'iou_{m2}_vs_{m1}'
                row_m[m2] = round(float(sub_101[col_name].mean()), 4)
        matrix_rows.append(row_m)
    df_matrix = pd.DataFrame(matrix_rows)
    out_matrix_path = os.path.join(out_dir, 'bridge_pairwise_model_iou_matrix_101consensus.csv')
    df_matrix.to_csv(out_matrix_path, index=False)
    print(f"Saved: {out_matrix_path}")

    # Print summary to console
    print("\n" + "=" * 80)
    print("TASK A RESULTS SUMMARY:")
    print("=" * 80)
    for r in summary_rows:
        print(f"Stratum: {r['stratum']:24s} (N={r['N_cases']})")
        print(f"  - Mean Pairwise IoU:       {r['mean_pairwise_iou']:.4f} (Median: {r['median_pairwise_iou']:.4f})")
        print(f"  - Base vs D2 Spatial IoU:   {r['mean_base_vs_d2_iou']:.4f} (Dice: {r['mean_base_vs_d2_dice']:.4f})")
        print(f"  - 7-Way Strict IoU:         {r['mean_7way_strict_iou']:.4f}")
        print(f"  - >=4 Models Core Ratio:    {r['mean_ge4_models_share_ratio']:.4f}")
        print(f"  - >=6 Models Core Ratio:    {r['mean_ge6_models_share_ratio']:.4f}")
        print("-" * 50)

    print("\nPairwise Model Spatial IoU Matrix on 101 Consensus Cases:")
    print(df_matrix.to_string(index=False))

if __name__ == '__main__':
    main()
