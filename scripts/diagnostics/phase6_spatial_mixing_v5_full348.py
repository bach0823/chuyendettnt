#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_spatial_mixing_v5_full348.py

Phase 6 V5 Counterfactual Full-Validation Generalization Test (Both-Half: alpha1=0.5, alpha2=0.5):
Evaluates:
Does the promising V5 mechanistic spatial-mixing observation generalize to the full N=348 validation population?

Conditions:
- C0_Normal: alpha1=1.0, alpha2=1.0 (Candidate B baseline on all 348 images)
- BothHalf:  alpha1=0.5, alpha2=0.5 (Off-center spatial mixing in Decoder Block 1 Conv1 & Conv2 halved)

STRICT CONSTRAINTS:
- Diagnostic-Only: Zero training, zero gradient updates, zero model mutations saved to disk.
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
    abs_cc_error = abs(pred_cc - gt_cc)

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

    # Spurious Islands
    spurious_island_count = 0
    for p_id in range(1, pred_cc + 1):
        overlapping_gt = np.unique(gt_labels[pred_labels == p_id])
        overlapping_gt = overlapping_gt[overlapping_gt > 0]
        if len(overlapping_gt) == 0:
            spurious_island_count += 1

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
        'pred_cc': pred_cc,
        'gt_cc': gt_cc,
        'abs_cc_error': abs_cc_error,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'has_bridge': has_bridge,
        'bridge_events': bridge_events,
        'merged_gt_components': len(merged_gt_set),
        'has_breakage': has_breakage,
        'fragmented_gt_components': fragmented_gt_components,
        'spurious_island_count': spurious_island_count,
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


def make_attenuated_weight(orig_weight: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Computes W' = W_center + alpha * W_off.
    For a 3x3 conv:
    Center tap is [:, :, 1, 1].
    All 8 off-center taps [:, :, (i,j) != (1,1)] are multiplied by alpha.
    """
    w_new = orig_weight.clone()
    for i in range(3):
        for j in range(3):
            if (i, j) != (1, 1):
                w_new[:, :, i, j] *= alpha
    return w_new


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

    out_dir = 'results/diagnostics/phase6_spatial_mixing_v5_full348'
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

    # Load metadata mappings from 7-models master and consensus files
    master_csv = 'results/diagnostics/phase6_d2_topology/topology_7models_master_paired.csv'
    consensus_csv = 'results/diagnostics/phase6_bridge_localization/bridge_consensus_per_image.csv'

    df_master = pd.read_csv(master_csv)
    df_consensus = pd.read_csv(consensus_csv)

    # Build metadata lookup dicts
    thin66_set = set(df_master[df_master['is_thin_66'] == True]['case_name'])
    thin5_set = set(df_master[df_master['is_thin_low_area_5'] == True]['case_name'])
    cat_map = dict(zip(df_master['case_name'], df_master['candidate_b_category']))

    c7_consensus_set = set(df_consensus[df_consensus['n_models_with_bridge'] == 7]['image_id'])
    d2_cured_set = set(df_consensus[df_consensus['d2_transition_type'] == 'Cured_by_D2']['image_id'])
    d2_created_set = set(df_consensus[df_consensus['d2_transition_type'] == 'Created_by_D2']['image_id'])
    base_bridge_set = set(df_consensus[df_consensus['Base_bridge'] == 1]['image_id'])

    print(f"Metadata verified: Thin66 N={len(thin66_set)}, Thin5 N={len(thin5_set)}, C7 N={len(c7_consensus_set)}, D2-Cured N={len(d2_cured_set)}, D2-Created N={len(d2_created_set)}, BaseBridges N={len(base_bridge_set)}")

    # Load all 348 validation images and masks
    data_root = 'datasets/Crack500_ready'
    val_img_dir = os.path.join(data_root, 'val', 'images')
    val_mask_dir = os.path.join(data_root, 'val', 'masks')

    mask_files = sorted(glob.glob(os.path.join(val_mask_dir, '*')))
    mask_files = [f for f in mask_files if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg']]
    assert len(mask_files) == 348, f"Expected 348 masks, found {len(mask_files)}"

    val_samples = []
    for mf in mask_files:
        stem = os.path.splitext(os.path.basename(mf))[0]
        ip = os.path.join(val_img_dir, stem + '.png')
        if not os.path.exists(ip):
            ip = os.path.join(val_img_dir, stem + '.jpg')
        assert os.path.exists(ip), f"Missing image {ip}"

        img = cv2.imread(ip)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
        target_bin = (target > 127).astype(np.uint8)

        # Categorize sample
        if stem in c7_consensus_set:
            pheno_group = 'Consensus_7of7'
        elif stem in d2_cured_set:
            pheno_group = 'Cured_by_D2'
        elif stem in d2_created_set:
            pheno_group = 'Created_by_D2'
        elif stem in base_bridge_set:
            pheno_group = 'Low_Consensus_Bridge'
        else:
            pheno_group = 'Clean_Base'

        val_samples.append({
            'case_name': stem,
            'img': img,
            'target_bin': target_bin,
            'group': pheno_group,
            'is_thin_66': int(stem in thin66_set),
            'is_thin_low_area_5': int(stem in thin5_set),
            'category': cat_map.get(stem, 'unknown')
        })

    # =========================================================================
    # STEP 1: RUN C0 NORMAL CANDIDATE B INFERENCE ON ALL 348 SAMPLES
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: RUNNING C0 NORMAL CANDIDATE B INFERENCE (N=348)")
    print("=" * 80)

    c0_results = {}
    c0_corridors = {}

    for s in tqdm(val_samples, desc="C0 Full Validation (N=348)"):
        cid = s['case_name']
        t_bin = s['target_bin']
        pred_bin, pred_prob = run_setting_a_inference(model, s['img'], device, tile_size=448)
        metrics = compute_topology_and_global_metrics(pred_bin, t_bin)
        whole_bridge_fp, neck_mask = isolate_base_bridge_corridor(pred_bin, t_bin)

        if np.sum(neck_mask) > 0:
            c0_corridors[cid] = neck_mask.copy()
            neck_median = float(np.median(pred_prob[neck_mask > 0]))
            neck_mean = float(np.mean(pred_prob[neck_mask > 0]))
            neck_p25 = float(np.percentile(pred_prob[neck_mask > 0], 25))
            neck_p75 = float(np.percentile(pred_prob[neck_mask > 0], 75))
            pct_ge50 = float(np.mean(pred_prob[neck_mask > 0] >= 0.5) * 100.0)
            pct_ge75 = float(np.mean(pred_prob[neck_mask > 0] >= 0.75) * 100.0)
            pct_ge90 = float(np.mean(pred_prob[neck_mask > 0] >= 0.90) * 100.0)
        else:
            neck_median = neck_mean = neck_p25 = neck_p75 = np.nan
            pct_ge50 = pct_ge75 = pct_ge90 = np.nan

        c0_results[cid] = {
            'metrics': metrics,
            'pred_bin': pred_bin,
            'pred_prob': pred_prob,
            'neck_median': neck_median,
            'neck_mean': neck_mean,
            'neck_p25': neck_p25,
            'neck_p75': neck_p75,
            'pct_ge50': pct_ge50,
            'pct_ge75': pct_ge75,
            'pct_ge90': pct_ge90,
        }

    # SANITY CHECK: C0 REPRODUCTION
    c0_dice_list = [c0_results[s['case_name']]['metrics']['dice'] for s in val_samples]
    c0_prec_list = [c0_results[s['case_name']]['metrics']['precision'] for s in val_samples]
    c0_rec_list = [c0_results[s['case_name']]['metrics']['recall'] for s in val_samples]
    c0_bridges_total = sum(c0_results[s['case_name']]['metrics']['has_bridge'] for s in val_samples)
    c0_breakage_total = sum(c0_results[s['case_name']]['metrics']['has_breakage'] for s in val_samples)

    print("\n--- C0 SANITY CHECK VERIFICATION ---")
    print(f"C0 Global Mean Dice:     {np.mean(c0_dice_list):.4f} (Expected: ~0.7641)")
    print(f"C0 Global Mean Precision:{np.mean(c0_prec_list):.4f} (Expected: ~0.7338)")
    print(f"C0 Global Mean Recall:   {np.mean(c0_rec_list):.4f} (Expected: ~0.8477)")
    print(f"C0 FalseBridge Count:    {c0_bridges_total} / 348 (Expected: 110)")
    print(f"C0 Breakage Count:       {c0_breakage_total} / 348 ({c0_breakage_total/348*100:.2f}%) (Expected: 33 / 9.48%)")

    assert c0_bridges_total == 110, f"C0 Base bridge count mismatch: {c0_bridges_total} != 110"
    assert abs(np.mean(c0_dice_list) - 0.7641) < 0.005, f"C0 Dice mismatch: {np.mean(c0_dice_list)}"
    print("C0 Sanity Check Passed with 100% Fidelity!")

    # =========================================================================
    # STEP 2: RUN BOTH-HALF (alpha1=0.5, alpha2=0.5) ON ALL 348 SAMPLES
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: RUNNING BOTH-HALF COUNTERFACTUAL (alpha1=0.5, alpha2=0.5) (N=348)")
    print("=" * 80)

    w1_half = make_attenuated_weight(orig_w1, 0.5)
    w2_half = make_attenuated_weight(orig_w2, 0.5)

    # Check center weights bitwise unchanged
    assert torch.equal(w1_half[:, :, 1, 1], orig_w1[:, :, 1, 1]), "Conv1 center tap altered!"
    assert torch.equal(w2_half[:, :, 1, 1], orig_w2[:, :, 1, 1]), "Conv2 center tap altered!"

    bothhalf_results = {}

    try:
        conv1.weight.data.copy_(w1_half)
        conv2.weight.data.copy_(w2_half)

        for s in tqdm(val_samples, desc="Both-Half Full Validation (N=348)"):
            cid = s['case_name']
            t_bin = s['target_bin']
            pred_bin, pred_prob = run_setting_a_inference(model, s['img'], device, tile_size=448)
            metrics = compute_topology_and_global_metrics(pred_bin, t_bin)
            whole_bridge_fp, neck_mask = isolate_base_bridge_corridor(pred_bin, t_bin)

            # Evaluate probability over C0 corridor (constant baseline coordinates)
            c0_neck = c0_corridors.get(cid, None)
            if c0_neck is not None and np.sum(c0_neck) > 0:
                neck_median = float(np.median(pred_prob[c0_neck > 0]))
                neck_mean = float(np.mean(pred_prob[c0_neck > 0]))
                neck_p25 = float(np.percentile(pred_prob[c0_neck > 0], 25))
                neck_p75 = float(np.percentile(pred_prob[c0_neck > 0], 75))
                pct_ge50 = float(np.mean(pred_prob[c0_neck > 0] >= 0.5) * 100.0)
                pct_ge75 = float(np.mean(pred_prob[c0_neck > 0] >= 0.75) * 100.0)
                pct_ge90 = float(np.mean(pred_prob[c0_neck > 0] >= 0.90) * 100.0)
            else:
                neck_median = neck_mean = neck_p25 = neck_p75 = np.nan
                pct_ge50 = pct_ge75 = pct_ge90 = np.nan

            bothhalf_results[cid] = {
                'metrics': metrics,
                'pred_bin': pred_bin,
                'pred_prob': pred_prob,
                'neck_median': neck_median,
                'neck_mean': neck_mean,
                'neck_p25': neck_p25,
                'neck_p75': neck_p75,
                'pct_ge50': pct_ge50,
                'pct_ge75': pct_ge75,
                'pct_ge90': pct_ge90,
            }

    finally:
        conv1.weight.data.copy_(orig_w1)
        conv2.weight.data.copy_(orig_w2)

    assert torch.equal(conv1.weight.data, orig_w1), "Model Conv1 weight restoration failed!"
    assert torch.equal(conv2.weight.data, orig_w2), "Model Conv2 weight restoration failed!"
    print("\nSanity Check Passed: In-memory weights verified 100% restored to baseline.")

    # =========================================================================
    # STEP 3: ASSEMBLE PER-SAMPLE RESULTS CSV
    # =========================================================================
    per_sample_rows = []
    for s in val_samples:
        cid = s['case_name']
        m0 = c0_results[cid]['metrics']
        mh = bothhalf_results[cid]['metrics']

        base_br = m0['has_bridge']
        half_br = mh['has_bridge']

        if base_br == 1 and half_br == 0:
            trans = 'cured'
        elif base_br == 1 and half_br == 1:
            trans = 'persistent'
        elif base_br == 0 and half_br == 1:
            trans = 'created'
        else:
            trans = 'clean'

        base_brk = m0['has_breakage']
        half_brk = mh['has_breakage']

        base_neck_med = c0_results[cid]['neck_median']
        half_neck_med = bothhalf_results[cid]['neck_median']
        delta_neck_med = half_neck_med - base_neck_med if not np.isnan(base_neck_med) and not np.isnan(half_neck_med) else np.nan

        per_sample_rows.append({
            'image_id': cid,
            'group': s['group'],
            'is_thin_66': s['is_thin_66'],
            'is_thin_low_area_5': s['is_thin_low_area_5'],
            'category': s['category'],
            'base_dice': round(m0['dice'], 5),
            'bothhalf_dice': round(mh['dice'], 5),
            'delta_dice': round(mh['dice'] - m0['dice'], 5),
            'base_precision': round(m0['precision'], 5),
            'bothhalf_precision': round(mh['precision'], 5),
            'delta_precision': round(mh['precision'] - m0['precision'], 5),
            'base_recall': round(m0['recall'], 5),
            'bothhalf_recall': round(mh['recall'], 5),
            'delta_recall': round(mh['recall'] - m0['recall'], 5),
            'base_area_excess': round(m0['area_excess'], 2),
            'bothhalf_area_excess': round(mh['area_excess'], 2),
            'base_cldice': round(m0['cldice'], 5),
            'bothhalf_cldice': round(mh['cldice'], 5),
            'base_bridge': base_br,
            'bothhalf_bridge': half_br,
            'bridge_transition': trans,
            'base_breakage': base_brk,
            'bothhalf_breakage': half_brk,
            'base_connector_median': round(base_neck_med, 4) if not np.isnan(base_neck_med) else np.nan,
            'bothhalf_connector_median': round(half_neck_med, 4) if not np.isnan(half_neck_med) else np.nan,
            'delta_connector_median': round(delta_neck_med, 4) if not np.isnan(delta_neck_med) else np.nan,
        })

    df_per_sample = pd.DataFrame(per_sample_rows)
    per_sample_csv = os.path.join(out_dir, 'full348_bothhalf_per_sample.csv')
    df_per_sample.to_csv(per_sample_csv, index=False)
    print(f"Saved: {per_sample_csv} (N=348)")

    # =========================================================================
    # STEP 4: GLOBAL READOUTS & SUMMARY
    # =========================================================================
    bh_dice_list = [bothhalf_results[s['case_name']]['metrics']['dice'] for s in val_samples]
    bh_prec_list = [bothhalf_results[s['case_name']]['metrics']['precision'] for s in val_samples]
    bh_rec_list = [bothhalf_results[s['case_name']]['metrics']['recall'] for s in val_samples]
    bh_aexc_list = [bothhalf_results[s['case_name']]['metrics']['area_excess'] for s in val_samples]
    bh_cldice_list = [bothhalf_results[s['case_name']]['metrics']['cldice'] for s in val_samples]
    bh_tprec_list = [bothhalf_results[s['case_name']]['metrics']['tprec'] for s in val_samples]
    bh_tsens_list = [bothhalf_results[s['case_name']]['metrics']['tsens'] for s in val_samples]
    bh_abs_cc_err = [bothhalf_results[s['case_name']]['metrics']['abs_cc_error'] for s in val_samples]
    bh_islands = [bothhalf_results[s['case_name']]['metrics']['spurious_island_count'] for s in val_samples]

    c0_aexc_list = [c0_results[s['case_name']]['metrics']['area_excess'] for s in val_samples]
    c0_cldice_list = [c0_results[s['case_name']]['metrics']['cldice'] for s in val_samples]
    c0_tprec_list = [c0_results[s['case_name']]['metrics']['tprec'] for s in val_samples]
    c0_tsens_list = [c0_results[s['case_name']]['metrics']['tsens'] for s in val_samples]
    c0_abs_cc_err = [c0_results[s['case_name']]['metrics']['abs_cc_error'] for s in val_samples]
    c0_islands = [c0_results[s['case_name']]['metrics']['spurious_island_count'] for s in val_samples]

    wins = sum(df_per_sample['delta_dice'] > 0.001)
    ties = sum(abs(df_per_sample['delta_dice']) <= 0.001)
    losses = sum(df_per_sample['delta_dice'] < -0.001)

    bh_bridges_total = sum(df_per_sample['bothhalf_bridge'])
    bh_breakage_total = sum(df_per_sample['bothhalf_breakage'])

    summary_data = [
        {'metric': 'N_samples', 'Base_C0': 348, 'BothHalf': 348, 'Delta': 0},
        {'metric': 'Global_Mean_Dice', 'Base_C0': round(float(np.mean(c0_dice_list)), 4), 'BothHalf': round(float(np.mean(bh_dice_list)), 4), 'Delta': round(float(np.mean(bh_dice_list) - np.mean(c0_dice_list)), 4)},
        {'metric': 'Global_Median_Dice', 'Base_C0': round(float(np.median(c0_dice_list)), 4), 'BothHalf': round(float(np.median(bh_dice_list)), 4), 'Delta': round(float(np.median(bh_dice_list) - np.median(c0_dice_list)), 4)},
        {'metric': 'Win_Tie_Loss_Dice', 'Base_C0': '-', 'BothHalf': f"{wins}/{ties}/{losses}", 'Delta': '-'},
        {'metric': 'Precision', 'Base_C0': round(float(np.mean(c0_prec_list)), 4), 'BothHalf': round(float(np.mean(bh_prec_list)), 4), 'Delta': round(float(np.mean(bh_prec_list) - np.mean(c0_prec_list)), 4)},
        {'metric': 'Recall', 'Base_C0': round(float(np.mean(c0_rec_list)), 4), 'BothHalf': round(float(np.mean(bh_rec_list)), 4), 'Delta': round(float(np.mean(bh_rec_list) - np.mean(c0_rec_list)), 4)},
        {'metric': 'Area_Excess_Pct', 'Base_C0': round(float(np.mean(c0_aexc_list)), 2), 'BothHalf': round(float(np.mean(bh_aexc_list)), 2), 'Delta': round(float(np.mean(bh_aexc_list) - np.mean(c0_aexc_list)), 2)},
        {'metric': 'clDice', 'Base_C0': round(float(np.mean(c0_cldice_list)), 4), 'BothHalf': round(float(np.mean(bh_cldice_list)), 4), 'Delta': round(float(np.mean(bh_cldice_list) - np.mean(c0_cldice_list)), 4)},
        {'metric': 'Tprec', 'Base_C0': round(float(np.mean(c0_tprec_list)), 4), 'BothHalf': round(float(np.mean(bh_tprec_list)), 4), 'Delta': round(float(np.mean(bh_tprec_list) - np.mean(c0_tprec_list)), 4)},
        {'metric': 'Tsens', 'Base_C0': round(float(np.mean(c0_tsens_list)), 4), 'BothHalf': round(float(np.mean(bh_tsens_list)), 4), 'Delta': round(float(np.mean(bh_tsens_list) - np.mean(c0_tsens_list)), 4)},
        {'metric': 'FalseBridge_Count', 'Base_C0': c0_bridges_total, 'BothHalf': bh_bridges_total, 'Delta': bh_bridges_total - c0_bridges_total},
        {'metric': 'FalseBridge_Rate_Pct', 'Base_C0': round(c0_bridges_total / 348 * 100.0, 2), 'BothHalf': round(bh_bridges_total / 348 * 100.0, 2), 'Delta': round((bh_bridges_total - c0_bridges_total) / 348 * 100.0, 2)},
        {'metric': 'Breakage_Count', 'Base_C0': c0_breakage_total, 'BothHalf': bh_breakage_total, 'Delta': bh_breakage_total - c0_breakage_total},
        {'metric': 'Breakage_Rate_Pct', 'Base_C0': round(c0_breakage_total / 348 * 100.0, 2), 'BothHalf': round(bh_breakage_total / 348 * 100.0, 2), 'Delta': round((bh_breakage_total - c0_breakage_total) / 348 * 100.0, 2)},
        {'metric': 'Spurious_Islands_Total', 'Base_C0': sum(c0_islands), 'BothHalf': sum(bh_islands), 'Delta': sum(bh_islands) - sum(c0_islands)},
        {'metric': 'Mean_Abs_CC_Error', 'Base_C0': round(float(np.mean(c0_abs_cc_err)), 4), 'BothHalf': round(float(np.mean(bh_abs_cc_err)), 4), 'Delta': round(float(np.mean(bh_abs_cc_err) - np.mean(c0_abs_cc_err)), 4)},
    ]

    df_summary = pd.DataFrame(summary_data)
    summary_csv = os.path.join(out_dir, 'full348_bothhalf_summary.csv')
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")
    print("\n--- GLOBAL FULL-348 COMPARISON ---")
    print(df_summary.to_string(index=False))

    # =========================================================================
    # STEP 5: BRIDGE TRANSITION ANALYSIS
    # =========================================================================
    transition_groups = [
        ('All_348_Images', df_per_sample),
        ('Base_110_Bridges', df_per_sample[df_per_sample['base_bridge'] == 1]),
        ('Consensus_7of7_101', df_per_sample[df_per_sample['group'] == 'Consensus_7of7']),
        ('D2_Cured_7', df_per_sample[df_per_sample['group'] == 'Cured_by_D2']),
        ('D2_Created_6', df_per_sample[df_per_sample['group'] == 'Created_by_D2']),
        ('Low_Consensus_9', df_per_sample[df_per_sample['group'] == 'Low_Consensus_Bridge']),
        ('Clean_Base_228', df_per_sample[df_per_sample['base_bridge'] == 0]),
    ]

    trans_rows = []
    for gname, gdf in transition_groups:
        n_cases = len(gdf)
        if n_cases == 0:
            continue
        c0_br = int(gdf['base_bridge'].sum())
        bh_br = int(gdf['bothhalf_bridge'].sum())
        cured = int(sum(gdf['bridge_transition'] == 'cured'))
        persistent = int(sum(gdf['bridge_transition'] == 'persistent'))
        created = int(sum(gdf['bridge_transition'] == 'created'))
        net_change = bh_br - c0_br
        cure_fraction = round(cured / max(c0_br, 1) * 100.0, 2) if c0_br > 0 else np.nan

        trans_rows.append({
            'group': gname,
            'N': n_cases,
            'base_bridges': c0_br,
            'bothhalf_bridges': bh_br,
            'cured': cured,
            'persistent': persistent,
            'created': created,
            'net_change': net_change,
            'cure_fraction_pct': cure_fraction,
            'persistence_rate_pct': round(persistent / max(c0_br, 1) * 100.0, 2) if c0_br > 0 else np.nan,
        })

    df_trans = pd.DataFrame(trans_rows)
    trans_csv = os.path.join(out_dir, 'full348_bridge_transition.csv')
    df_trans.to_csv(trans_csv, index=False)
    print(f"\nSaved: {trans_csv}")
    print("\n--- BRIDGE TRANSITIONS BY SUBGROUP ---")
    print(df_trans.to_string(index=False))

    # =========================================================================
    # STEP 6: SUBGROUP PERFORMANCE & SAFETY GUARDRAILS
    # =========================================================================
    subgroup_evals = [
        ('Global_348', df_per_sample),
        ('Thin_Cracks_66', df_per_sample[df_per_sample['is_thin_66'] == 1]),
        ('Thin_Low_Area_5', df_per_sample[df_per_sample['is_thin_low_area_5'] == 1]),
        ('Consensus_7of7_101', df_per_sample[df_per_sample['group'] == 'Consensus_7of7']),
        ('D2_Cured_7', df_per_sample[df_per_sample['group'] == 'Cured_by_D2']),
        ('D2_Created_6', df_per_sample[df_per_sample['group'] == 'Created_by_D2']),
    ]

    subgroup_summary_rows = []
    for gname, gdf in subgroup_evals:
        n_cases = len(gdf)
        if n_cases == 0:
            continue
        subgroup_summary_rows.append({
            'subgroup': gname,
            'N': n_cases,
            'base_mean_dice': round(float(gdf['base_dice'].mean()), 4),
            'bothhalf_mean_dice': round(float(gdf['bothhalf_dice'].mean()), 4),
            'delta_dice': round(float(gdf['bothhalf_dice'].mean() - gdf['base_dice'].mean()), 4),
            'base_mean_recall': round(float(gdf['base_recall'].mean()), 4),
            'bothhalf_mean_recall': round(float(gdf['bothhalf_recall'].mean()), 4),
            'base_mean_precision': round(float(gdf['base_precision'].mean()), 4),
            'bothhalf_mean_precision': round(float(gdf['bothhalf_precision'].mean()), 4),
            'base_breakage_count': int(gdf['base_breakage'].sum()),
            'bothhalf_breakage_count': int(gdf['bothhalf_breakage'].sum()),
            'bothhalf_breakage_pct': round(float(gdf['bothhalf_breakage'].mean() * 100.0), 2),
            'base_bridges': int(gdf['base_bridge'].sum()),
            'bothhalf_bridges': int(gdf['bothhalf_bridge'].sum()),
            'bridge_net_change': int(gdf['bothhalf_bridge'].sum() - gdf['base_bridge'].sum()),
        })

    df_subgroups = pd.DataFrame(subgroup_summary_rows)
    subgroup_csv = os.path.join(out_dir, 'full348_subgroup_summary.csv')
    df_subgroups.to_csv(subgroup_csv, index=False)
    print(f"\nSaved: {subgroup_csv}")
    print("\n--- SUBGROUP PERFORMANCE & GUARDRAILS ---")
    print(df_subgroups.to_string(index=False))

    # =========================================================================
    # STEP 7: CONNECTOR PROBABILITY ANALYSIS
    # =========================================================================
    prob_groups = [
        ('Base_110_Bridges', df_per_sample[df_per_sample['base_bridge'] == 1]),
        ('Consensus_7of7_101', df_per_sample[df_per_sample['group'] == 'Consensus_7of7']),
        ('D2_Cured_7', df_per_sample[df_per_sample['group'] == 'Cured_by_D2']),
        ('Low_Consensus_9', df_per_sample[df_per_sample['group'] == 'Low_Consensus_Bridge']),
    ]

    prob_summary_rows = []
    for gname, gdf in prob_groups:
        b_probs = gdf['base_connector_median'].dropna()
        h_probs = gdf['bothhalf_connector_median'].dropna()
        d_probs = gdf['delta_connector_median'].dropna()

        if len(b_probs) == 0:
            continue

        prob_summary_rows.append({
            'group': gname,
            'N': len(b_probs),
            'base_prob_mean': round(float(b_probs.mean()), 4),
            'base_prob_median': round(float(b_probs.median()), 4),
            'bothhalf_prob_mean': round(float(h_probs.mean()), 4),
            'bothhalf_prob_median': round(float(h_probs.median()), 4),
            'delta_median_prob': round(float(h_probs.median() - b_probs.median()), 4),
            'mean_delta_prob': round(float(d_probs.mean()), 4),
            'base_pct_ge50': round(float(np.mean(b_probs >= 0.5) * 100.0), 2),
            'bothhalf_pct_ge50': round(float(np.mean(h_probs >= 0.5) * 100.0), 2),
            'bothhalf_pct_ge75': round(float(np.mean(h_probs >= 0.75) * 100.0), 2),
            'bothhalf_pct_ge90': round(float(np.mean(h_probs >= 0.90) * 100.0), 2),
        })

    df_prob = pd.DataFrame(prob_summary_rows)
    prob_csv = os.path.join(out_dir, 'full348_connector_probability.csv')
    df_prob.to_csv(prob_csv, index=False)
    print(f"\nSaved: {prob_csv}")
    print("\n--- CONNECTOR PROBABILITY SHIFT ---")
    print(df_prob.to_string(index=False))

    print("\n" + "=" * 80)
    print("PHASE 6 V5 FULL-VALIDATION GENERALIZATION TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    main()
