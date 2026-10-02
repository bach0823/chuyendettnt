#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_counterfactual_shallow_skip.py

Phase 6 Diagnostic-Only Counterfactual Shallow Skip Ablation:
Answers: Does removing shallow high-resolution skip information at S1 and/or S0
reduce the false-bridge signal, or are those features merely reflecting a bridge decision
that was already determined upstream?

Conditions evaluated:
- C0: Normal (Candidate B baseline, no intervention)
- C1_zero: S1-off (zeroing Stage 1 skip: 96 ch, 56x56)
- C2_zero: S0-off (zeroing Stage 0 skip: 48 ch, 112x112)
- C3_zero: S1+S0-off (zeroing both S1 and S0 skips)
- C1_mean: S1-mean (spatial mean reference ablation, preserves channel energy)
- C2_mean: S0-mean (spatial mean reference ablation)
- C3_mean: S1+S0-mean (spatial mean reference ablation)

Sample set (N=25 deterministic samples):
- 10 Consensus 7/7 cases
- 5 Base->D2 cured cases
- 5 Base-clean->D2 created cases
- 5 Low-consensus cases

Strict rules:
- No training, no optimizer, no checkpoint modification.
- Test set remains sealed.
- Setting A tiling (448x448, stride 448).
- Fixed threshold 0.5 for all binary readouts.
"""

import gc
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
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, 'SAGE_LITE')
from tools.run_phase6_c_topology_diagnostic import load_model_from_checkpoint

# -----------------------------------------------------------------------------
# Metric Computation Utilities
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
    has_bridge = bool(bridge_events > 0)

    # 2. Breakage / Fragmentation
    fragmented_gt_components = 0
    for g_id in range(1, gt_cc + 1):
        overlapping_pred = np.unique(pred_labels[gt_labels == g_id])
        overlapping_pred = overlapping_pred[overlapping_pred > 0]
        if len(overlapping_pred) >= 2:
            fragmented_gt_components += 1
    has_breakage = bool(fragmented_gt_components > 0)

    # 3. Spurious Islands
    spurious_island_count = 0
    for p_id in range(1, pred_cc + 1):
        overlapping_gt = np.unique(gt_labels[pred_labels == p_id])
        overlapping_gt = overlapping_gt[overlapping_gt > 0]
        if len(overlapping_gt) == 0:
            spurious_island_count += 1

    # 4. clDice
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
        'spurious_island_count': spurious_island_count,
        'tprec': tprec,
        'tsens': tsens,
        'cldice': cldice,
    }

def isolate_base_bridge_corridor(pred_bin: np.ndarray, target_bin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the reference minimal connector neck mask and whole bridge FP mask on Base prediction.
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

        cc_fp = p_mask & (target_bin == 0)
        bridge_fp_mask = bridge_fp_mask | cc_fp.astype(np.uint8)

        gt_list = list(overlapping_gt)
        for i_idx in range(len(gt_list)):
            gi = gt_list[i_idx]
            mask_i = (gt_labels == gi)
            dt_i = distance_transform_edt(1 - mask_i)

            for j_idx in range(i_idx + 1, len(gt_list)):
                gj = gt_list[j_idx]
                mask_j = (gt_labels == gj)
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

def compute_prob_stats(probs: np.ndarray) -> Dict[str, float]:
    if len(probs) == 0:
        return {
            'mean': np.nan, 'median': np.nan, 'P25': np.nan, 'P75': np.nan,
            'frac_ge50': np.nan, 'frac_ge75': np.nan, 'frac_ge90': np.nan, 'n_pixels': 0
        }
    return {
        'mean': float(np.mean(probs)),
        'median': float(np.median(probs)),
        'P25': float(np.percentile(probs, 25)),
        'P75': float(np.percentile(probs, 75)),
        'frac_ge50': float(np.sum(probs >= 0.50) / len(probs)),
        'frac_ge75': float(np.sum(probs >= 0.75) / len(probs)),
        'frac_ge90': float(np.sum(probs >= 0.90) / len(probs)),
        'n_pixels': int(len(probs)),
    }

# -----------------------------------------------------------------------------
# Setting A Inference with Counterfactual Skip Hooks
# -----------------------------------------------------------------------------

def run_counterfactual_inference_setting_a(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    condition: str = 'C0',
    tile_size: int = 448,
    batch_size: int = 4
) -> Dict[str, Any]:
    """
    Executes Setting A tiling inference with exact counterfactual intervention.
    Captures:
    - full_logits (H, W)
    - full_prob (H, W)
    - stage-wise activations (S3, S2, S1, S0, Head112, Final448)
    - tensor norms for S1/S0 skips and merged blocks (for distribution-shift check)
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

    act_s3 = np.zeros((pH // 32, pW // 32), dtype=np.float32)
    act_s2 = np.zeros((pH // 16, pW // 16), dtype=np.float32)
    act_s1 = np.zeros((pH // 8, pW // 8), dtype=np.float32)
    act_s0 = np.zeros((pH // 4, pW // 4), dtype=np.float32)
    logits_head112 = np.zeros((pH // 4, pW // 4), dtype=np.float32)
    logits_final448 = np.zeros((pH, pW), dtype=np.float32)

    # Tensor norm accumulators
    s1_skip_l2_list, s1_skip_abs_list = [], []
    s0_skip_l2_list, s0_skip_abs_list = [], []
    s1_out_l2_list, s1_out_abs_list = [], []
    s0_out_l2_list, s0_out_abs_list = [], []

    hook_handles = []
    cache = {}

    # Define Pre-hooks for Counterfactual Interventions
    def pre_hook_s1(module, args):
        x_in, skip = args
        # Record norms before intervention
        s1_skip_l2_list.append(float(skip.norm().item() / np.sqrt(skip.numel())))
        s1_skip_abs_list.append(float(skip.abs().mean().item()))

        if condition in ('C1_zero', 'C3_zero'):
            skip_mod = torch.zeros_like(skip)
        elif condition in ('C1_mean', 'C3_mean'):
            skip_mod = skip.mean(dim=(2, 3), keepdim=True).expand_as(skip)
        else:
            skip_mod = skip
        return (x_in, skip_mod)

    def pre_hook_s0(module, args):
        x_in, skip = args
        s0_skip_l2_list.append(float(skip.norm().item() / np.sqrt(skip.numel())))
        s0_skip_abs_list.append(float(skip.abs().mean().item()))

        if condition in ('C2_zero', 'C3_zero'):
            skip_mod = torch.zeros_like(skip)
        elif condition in ('C2_mean', 'C3_mean'):
            skip_mod = skip.mean(dim=(2, 3), keepdim=True).expand_as(skip)
        else:
            skip_mod = skip
        return (x_in, skip_mod)

    # Define Post-hooks for stage activation and merged norm extraction
    def post_hook_s3(module, inp, out):
        cache['s3'] = inp[0].detach()  # bottleneck feature map (B, 384, 14, 14)

    def post_hook_s2(module, inp, out):
        cache['s2'] = out.detach()      # (B, 192, 28, 28)

    def post_hook_s1(module, inp, out):
        cache['s1'] = out.detach()      # (B, 96, 56, 56)
        s1_out_l2_list.append(float(out.norm().item() / np.sqrt(out.numel())))
        s1_out_abs_list.append(float(out.abs().mean().item()))

    def post_hook_s0(module, inp, out):
        cache['s0'] = out.detach()      # (B, 48, 112, 112)
        s0_out_l2_list.append(float(out.norm().item() / np.sqrt(out.numel())))
        s0_out_abs_list.append(float(out.abs().mean().item()))

    def post_hook_head(module, inp, out):
        cache['head112'] = out.detach()  # (B, 1, 112, 112)

    # Register hooks
    h_pre_s1 = model.decoder.decoder_blocks[1].register_forward_pre_hook(pre_hook_s1)
    h_pre_s0 = model.decoder.decoder_blocks[2].register_forward_pre_hook(pre_hook_s0)
    h_s3 = model.decoder.decoder_blocks[0].register_forward_hook(post_hook_s3)
    h_s2 = model.decoder.decoder_blocks[0].register_forward_hook(post_hook_s2)
    h_s1 = model.decoder.decoder_blocks[1].register_forward_hook(post_hook_s1)
    h_s0 = model.decoder.decoder_blocks[2].register_forward_hook(post_hook_s0)
    h_hd = model.decoder.segmentation_head.register_forward_hook(post_hook_head)

    hook_handles.extend([h_pre_s1, h_pre_s0, h_s3, h_s2, h_s1, h_s0, h_hd])

    try:
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch = torch.stack(patches[i:i+batch_size]).to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    out_logits = model(batch)

                b_final = out_logits.squeeze(1).float().cpu().numpy()
                b_s3 = cache['s3'].abs().mean(dim=1).float().cpu().numpy()
                b_s2 = cache['s2'].abs().mean(dim=1).float().cpu().numpy()
                b_s1 = cache['s1'].abs().mean(dim=1).float().cpu().numpy()
                b_s0 = cache['s0'].abs().mean(dim=1).float().cpu().numpy()
                b_hd = cache['head112'].squeeze(1).float().cpu().numpy()

                for j in range(len(b_final)):
                    y, x = coords[i + j]
                    act_s3[y//32 : y//32 + 14, x//32 : x//32 + 14] = b_s3[j]
                    act_s2[y//16 : y//16 + 28, x//16 : x//16 + 28] = b_s2[j]
                    act_s1[y//8 : y//8 + 56, x//8 : x//8 + 56] = b_s1[j]
                    act_s0[y//4 : y//4 + 112, x//4 : x//4 + 112] = b_s0[j]
                    logits_head112[y//4 : y//4 + 112, x//4 : x//4 + 112] = b_hd[j]
                    logits_final448[y : y + 448, x : x + 448] = b_final[j]
    finally:
        for h in hook_handles:
            h.remove()

    h_s3_dim, w_s3_dim = H // 32, W // 32
    h_s2_dim, w_s2_dim = H // 16, W // 16
    h_s1_dim, w_s1_dim = H // 8, W // 8
    h_s0_dim, w_s0_dim = H // 4, W // 4

    logits_cropped = logits_final448[:H, :W]
    prob_cropped = 1.0 / (1.0 + np.exp(-logits_cropped.astype(np.float64)))

    return {
        'logits': logits_cropped,
        'prob': prob_cropped,
        'pred_bin': (logits_cropped > 0.0).astype(np.uint8),
        's3_act': act_s3[:h_s3_dim, :w_s3_dim],
        's2_act': act_s2[:h_s2_dim, :w_s2_dim],
        's1_act': act_s1[:h_s1_dim, :w_s1_dim],
        's0_act': act_s0[:h_s0_dim, :w_s0_dim],
        'head112_logits': logits_head112[:h_s0_dim, :w_s0_dim],
        'tensor_norms': {
            's1_skip_l2': float(np.mean(s1_skip_l2_list)) if s1_skip_l2_list else np.nan,
            's1_skip_abs': float(np.mean(s1_skip_abs_list)) if s1_skip_abs_list else np.nan,
            's0_skip_l2': float(np.mean(s0_skip_l2_list)) if s0_skip_l2_list else np.nan,
            's0_skip_abs': float(np.mean(s0_skip_abs_list)) if s0_skip_abs_list else np.nan,
            's1_out_l2': float(np.mean(s1_out_l2_list)) if s1_out_l2_list else np.nan,
            's1_out_abs': float(np.mean(s1_out_abs_list)) if s1_out_abs_list else np.nan,
            's0_out_l2': float(np.mean(s0_out_l2_list)) if s0_out_l2_list else np.nan,
            's0_out_abs': float(np.mean(s0_out_abs_list)) if s0_out_abs_list else np.nan,
        }
    }

# -----------------------------------------------------------------------------
# Main Diagnostic Runner
# -----------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    out_dir = 'results/diagnostics/phase6_bridge_localization_v3'
    os.makedirs(out_dir, exist_ok=True)

    config_path = 'results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml'
    ckpt_path = 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'
    master_csv = 'results/diagnostics/phase6_d2_topology/topology_7models_master_paired.csv'
    df_master = pd.read_csv(master_csv)

    # 25 Deterministic evaluation cases
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
    assert len(sample_cases_meta) == 25

    data_root = 'datasets/Crack500_ready'
    val_img_dir = os.path.join(data_root, 'val', 'images')
    val_mask_dir = os.path.join(data_root, 'val', 'masks')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = False

    print("=" * 80)
    print("PHASE 6-D COUNTERFACTUAL INFERENCE DIAGNOSTIC (v3)")
    print(f"Device: {device} | Checkpoint: {ckpt_path}")
    print("=" * 80)

    model = load_model_from_checkpoint(config_path, ckpt_path, device)
    model.eval()

    # Verify Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Sanity Check 1] Model parameter count: {total_params:,} (expected 10,118,955)")
    assert total_params == 10118955, f"Parameter count mismatch: {total_params}"

    conditions = [
        ('C0', 'Normal (No intervention)'),
        ('C1_zero', 'S1-off (Zero S1 skip: 96 ch, 56x56)'),
        ('C2_zero', 'S0-off (Zero S0 skip: 48 ch, 112x112)'),
        ('C3_zero', 'S1+S0-off (Zero both S1 & S0 skips)'),
        ('C1_mean', 'S1-mean (Spatial mean reference ablation)'),
        ('C2_mean', 'S0-mean (Spatial mean reference ablation)'),
        ('C3_mean', 'S1+S0-mean (Spatial mean reference ablation)'),
    ]

    # Preload images and GT targets
    samples = []
    for c_id, grp in sample_cases_meta:
        ip = os.path.join(val_img_dir, c_id + '.jpg')
        if not os.path.exists(ip):
            ip = os.path.join(val_img_dir, c_id + '.png')
        mp = os.path.join(val_mask_dir, c_id + '.png')
        if not os.path.exists(mp):
            mp = os.path.join(val_mask_dir, c_id + '.jpg')
        assert os.path.exists(ip), f"Missing {ip}"
        assert os.path.exists(mp), f"Missing {mp}"

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

    # STEP 1: Execute C0 first and establish FIXED Base reference masks
    print("\n" + "=" * 80)
    print("STEP 1: Executing Condition C0 (Normal Baseline) and establishing reference masks...")
    print("=" * 80)

    c0_results = {}
    base_ref_masks = {}

    for s in tqdm(samples, desc="C0 Inference"):
        c_id = s['case_name']
        res = run_counterfactual_inference_setting_a(model, s['img'], device, condition='C0', tile_size=448)
        metrics = compute_topology_and_global_metrics(res['pred_bin'], s['target_bin'])
        whole_bridge_fp, neck_mask = isolate_base_bridge_corridor(res['pred_bin'], s['target_bin'])

        c0_results[c_id] = {
            'res': res,
            'metrics': metrics,
            'whole_bridge_fp': whole_bridge_fp,
            'neck_mask': neck_mask,
            'has_base_bridge': metrics['has_bridge'],
        }

        # Store fixed reference masks
        base_ref_masks[c_id] = {
            'neck_mask': neck_mask,
            'whole_bridge_fp': whole_bridge_fp,
            'crack_mask': (res['pred_bin'] == 1) & (s['target_bin'] == 1),
            'bg_mask': (res['pred_bin'] == 0) & (s['target_bin'] == 0),
            'has_base_bridge': metrics['has_bridge'],
        }

    # Verify Sanity Check C0 reproduction against official master CSV
    c0_dices = [c0_results[s['case_name']]['metrics']['dice'] for s in samples]
    c0_bridges = [c0_results[s['case_name']]['metrics']['has_bridge'] for s in samples]
    c0_breakages = [c0_results[s['case_name']]['metrics']['has_breakage'] for s in samples]

    sub_master = df_master[df_master['case_name'].isin([s['case_name'] for s in samples])].sort_values('case_name')
    sorted_samples = sorted(samples, key=lambda x: x['case_name'])
    master_dices = [sub_master[sub_master['case_name'] == s['case_name']]['Base_dice'].values[0] for s in sorted_samples]
    master_bridges = [sub_master[sub_master['case_name'] == s['case_name']]['Base_bridge_events'].values[0] > 0 for s in sorted_samples]
    master_breakages = [sub_master[sub_master['case_name'] == s['case_name']]['Base_fragmented_gt_components'].values[0] > 0 for s in sorted_samples]

    max_dice_diff = max(abs(c0_results[s['case_name']]['metrics']['dice'] - m_d) for s, m_d in zip(sorted_samples, master_dices))
    bridge_match = all(c0_results[s['case_name']]['metrics']['has_bridge'] == m_b for s, m_b in zip(sorted_samples, master_bridges))
    breakage_match = all(c0_results[s['case_name']]['metrics']['has_breakage'] == m_br for s, m_br in zip(sorted_samples, master_breakages))

    print(f"\n[Sanity Check 2] C0 reproduction vs Official Base Master:")
    print(f"  - Max Dice discrepancy: {max_dice_diff:.8f} (tolerance 1e-4)")
    print(f"  - Bridge classifications match exactly: {bridge_match} (16/25 bridge positive)")
    print(f"  - Breakage classifications match exactly: {breakage_match} (1/25 breakage positive)")
    assert max_dice_diff < 1e-4, f"C0 Dice mismatch: {max_dice_diff}"
    assert bridge_match, "C0 Bridge mismatch!"
    assert breakage_match, "C0 Breakage mismatch!"
    print(">>> SANITY CHECK PASS: C0 exactly reproduces official Candidate B baseline! <<<\n")

    # STEP 2: Execute All Conditions and Record Measurements
    print("=" * 80)
    print("STEP 2: Executing Counterfactual Interventions (C0, C1, C2, C3)...")
    print("=" * 80)

    per_sample_rows = []
    bridge_prob_rows = []
    stage_act_rows = []

    for cond_key, cond_desc in conditions:
        print(f"\nRunning Condition: {cond_key} — {cond_desc}...")
        t0 = time.time()

        for s in tqdm(samples, desc=f"Eval {cond_key}"):
            c_id = s['case_name']
            grp = s['stratum']
            target_bin = s['target_bin']
            H, W = target_bin.shape

            ref = base_ref_masks[c_id]
            neck_ref = ref['neck_mask']
            has_base_bridge = ref['has_base_bridge']

            if cond_key == 'C0':
                res = c0_results[c_id]['res']
                metrics = c0_results[c_id]['metrics']
            else:
                res = run_counterfactual_inference_setting_a(
                    model, s['img'], device, condition=cond_key, tile_size=448
                )
                metrics = compute_topology_and_global_metrics(res['pred_bin'], target_bin)

            prob_map = res['prob']

            # Fixed-site bridge probability readout
            if np.sum(neck_ref) > 0:
                neck_probs = prob_map[neck_ref > 0]
                p_stats = compute_prob_stats(neck_probs)
            else:
                p_stats = compute_prob_stats(np.array([]))

            # Topology change vs Base
            new_has_bridge = metrics['has_bridge']
            bridge_cured_here = bool(has_base_bridge and not new_has_bridge)
            bridge_created_here = bool(not has_base_bridge and new_has_bridge)
            bridge_remains = bool(has_base_bridge and new_has_bridge)

            # Record per-sample row
            per_sample_rows.append({
                'case_name': c_id,
                'stratum': grp,
                'condition': cond_key,
                'has_base_bridge': int(has_base_bridge),
                'pred_has_bridge': int(new_has_bridge),
                'bridge_remains': int(bridge_remains),
                'bridge_cured': int(bridge_cured_here),
                'bridge_created': int(bridge_created_here),
                'bridge_events': metrics['bridge_events'],
                'has_breakage': int(metrics['has_breakage']),
                'fragmented_gt_cc': metrics['fragmented_gt_components'],
                'spurious_islands': metrics['spurious_island_count'],
                'dice': round(metrics['dice'], 4),
                'precision': round(metrics['precision'], 4),
                'recall': round(metrics['recall'], 4),
                'area_excess': round(metrics['area_excess'], 2),
                'cldice': round(metrics['cldice'], 4),
                'neck_prob_mean': round(p_stats['mean'], 4),
                'neck_prob_median': round(p_stats['median'], 4),
                'neck_prob_p25': round(p_stats['P25'], 4),
                'neck_prob_p75': round(p_stats['P75'], 4),
                'neck_frac_ge50': round(p_stats['frac_ge50'], 4),
                'neck_frac_ge75': round(p_stats['frac_ge75'], 4),
                'neck_frac_ge90': round(p_stats['frac_ge90'], 4),
                's1_skip_norm_l2': round(res['tensor_norms']['s1_skip_l2'], 4),
                's0_skip_norm_l2': round(res['tensor_norms']['s0_skip_l2'], 4),
                's1_out_norm_l2': round(res['tensor_norms']['s1_out_l2'], 4),
                's0_out_norm_l2': round(res['tensor_norms']['s0_out_l2'], 4),
            })

            # Record bridge probability row (for all 16 cases with Base bridge)
            if has_base_bridge:
                bridge_prob_rows.append({
                    'case_name': c_id,
                    'stratum': grp,
                    'condition': cond_key,
                    'neck_pixels': int(np.sum(neck_ref)),
                    'mean': round(p_stats['mean'], 4),
                    'median': round(p_stats['median'], 4),
                    'P25': round(p_stats['P25'], 4),
                    'P75': round(p_stats['P75'], 4),
                    'pct_ge50': round(p_stats['frac_ge50'] * 100.0, 2),
                    'pct_ge75': round(p_stats['frac_ge75'] * 100.0, 2),
                    'pct_ge90': round(p_stats['frac_ge90'] * 100.0, 2),
                    'bridge_remains': int(bridge_remains),
                })

            # Record Stage-Wise Activations on Fixed Reference Masks
            stages = [
                ('S3_Bottleneck_14x14', res['s3_act'], 32, False),
                ('S2_DecoderBlock0_28x28', res['s2_act'], 16, False),
                ('S1_DecoderBlock1_56x56', res['s1_act'], 8, False),
                ('S0_DecoderBlock2_112x112', res['s0_act'], 4, False),
                ('Head_PreUpsample_112x112', res['head112_logits'], 4, True),
                ('Final_Logits_448x448', res['logits'], 1, True),
            ]

            neck_m = ref['neck_mask']
            crack_m = ref['crack_mask']
            bg_m = ref['bg_mask']

            for s_name, f_map, scale, is_logit in stages:
                h_s, w_s = f_map.shape
                neck_s = (cv2.resize(neck_m.astype(np.float32), (w_s, h_s), interpolation=cv2.INTER_AREA) > 0.05)
                crack_s = (cv2.resize(crack_m.astype(np.float32), (w_s, h_s), interpolation=cv2.INTER_AREA) > 0.05)
                bg_s = (cv2.resize(bg_m.astype(np.float32), (w_s, h_s), interpolation=cv2.INTER_AREA) > 0.5)

                if not is_logit:
                    neck_act = float(np.mean(f_map[neck_s])) if np.sum(neck_s) > 0 else np.nan
                    crack_act = float(np.mean(f_map[crack_s])) if np.sum(crack_s) > 0 else np.nan
                    bg_act = float(np.mean(f_map[bg_s])) if np.sum(bg_s) > 0 else np.nan
                    contrast = float(neck_act / (bg_act + 1e-6)) if not np.isnan(neck_act) and not np.isnan(bg_act) else np.nan
                    stage_act_rows.append({
                        'case_name': c_id,
                        'stratum': grp,
                        'condition': cond_key,
                        'stage': s_name,
                        'scale': scale,
                        'neck_act': round(neck_act, 4),
                        'crack_act': round(crack_act, 4),
                        'bg_act': round(bg_act, 4),
                        'contrast_neck_over_bg': round(contrast, 4),
                    })
                else:
                    p_stage = 1.0 / (1.0 + np.exp(-f_map.astype(np.float64)))
                    p_neck = p_stage[neck_s]
                    med_p = float(np.median(p_neck)) if len(p_neck) > 0 else np.nan
                    mean_p = float(np.mean(p_neck)) if len(p_neck) > 0 else np.nan
                    stage_act_rows.append({
                        'case_name': c_id,
                        'stratum': grp,
                        'condition': cond_key,
                        'stage': s_name,
                        'scale': scale,
                        'neck_act': round(mean_p, 4),
                        'crack_act': np.nan,
                        'bg_act': np.nan,
                        'contrast_neck_over_bg': round(med_p, 4),  # used for median prob
                    })

        dt = time.time() - t0
        print(f"Condition {cond_key} evaluated in {dt:.1f}s.")

    # Save Output CSVs
    df_per_sample = pd.DataFrame(per_sample_rows)
    out_sample_csv = os.path.join(out_dir, 'counterfactual_conditions_per_sample.csv')
    df_per_sample.to_csv(out_sample_csv, index=False)
    print(f"\nSaved: {out_sample_csv} ({len(df_per_sample)} rows)")

    df_bridge_prob = pd.DataFrame(bridge_prob_rows)
    out_prob_csv = os.path.join(out_dir, 'counterfactual_bridge_probability.csv')
    df_bridge_prob.to_csv(out_prob_csv, index=False)
    print(f"Saved: {out_prob_csv} ({len(df_bridge_prob)} rows)")

    df_stage_act = pd.DataFrame(stage_act_rows)
    out_stage_csv = os.path.join(out_dir, 'counterfactual_stage_activation.csv')
    df_stage_act.to_csv(out_stage_csv, index=False)
    print(f"Saved: {out_stage_csv} ({len(df_stage_act)} rows)")

    # STEP 3: Summary Table Across Conditions & Strata
    print("\n" + "=" * 80)
    print("STEP 3: Aggregating counterfactual summary...")
    print("=" * 80)

    summary_rows = []
    target_conditions = ['C0', 'C1_zero', 'C2_zero', 'C3_zero', 'C1_mean', 'C2_mean', 'C3_mean']

    for cond_k in target_conditions:
        sub_c = df_per_sample[df_per_sample['condition'] == cond_k]
        sub_bridge16 = sub_c[sub_c['has_base_bridge'] == 1]
        sub_c7 = sub_c[sub_c['stratum'] == 'Consensus_7of7']
        sub_cured = sub_c[sub_c['stratum'] == 'Cured_by_D2']
        sub_created = sub_c[sub_c['stratum'] == 'Created_by_D2']

        # Total bridges remaining out of 16
        bridges_rem_16 = int(sub_bridge16['pred_has_bridge'].sum())
        bridges_rem_c7 = int(sub_c7['pred_has_bridge'].sum())
        bridges_created_5 = int(sub_created['pred_has_bridge'].sum())
        breakage_cases_25 = int(sub_c['has_breakage'].sum())

        mean_dice_25 = sub_c['dice'].mean()
        mean_cldice_25 = sub_c['cldice'].mean()
        mean_recall_25 = sub_c['recall'].mean()
        mean_prec_25 = sub_c['precision'].mean()

        # Neck probability on 10 Consensus 7/7 cases
        c7_probs = df_bridge_prob[(df_bridge_prob['condition'] == cond_k) & (df_bridge_prob['stratum'] == 'Consensus_7of7')]
        neck_med_c7 = c7_probs['median'].mean()
        neck_ge75_c7 = c7_probs['pct_ge75'].mean()
        neck_ge50_c7 = c7_probs['pct_ge50'].mean()

        summary_rows.append({
            'condition': cond_k,
            'bridges_remaining_out_of_16': f"{bridges_rem_16}/16",
            'bridges_remaining_c7_out_of_10': f"{bridges_rem_c7}/10",
            'bridges_created_in_cured_c_out_of_5': f"{bridges_created_5}/5",
            'breakage_cases_out_of_25': f"{breakage_cases_25}/25",
            'dice_25_mean': round(mean_dice_25, 4),
            'cldice_25_mean': round(mean_cldice_25, 4),
            'precision_25_mean': round(mean_prec_25, 4),
            'recall_25_mean': round(mean_recall_25, 4),
            'neck_median_prob_c7': round(neck_med_c7, 4),
            'neck_pct_ge50_c7': round(neck_ge50_c7, 2),
            'neck_pct_ge75_c7': round(neck_ge75_c7, 2),
            's1_skip_norm_l2': round(sub_c['s1_skip_norm_l2'].mean(), 4),
            's0_skip_norm_l2': round(sub_c['s0_skip_norm_l2'].mean(), 4),
            's1_out_norm_l2': round(sub_c['s1_out_norm_l2'].mean(), 4),
            's0_out_norm_l2': round(sub_c['s0_out_norm_l2'].mean(), 4),
        })

    df_summary = pd.DataFrame(summary_rows)
    out_summary_csv = os.path.join(out_dir, 'counterfactual_summary.csv')
    df_summary.to_csv(out_summary_csv, index=False)
    print(f"Saved: {out_summary_csv}")

    print("\n" + "=" * 80)
    print("PHASE 6 COUNTERFACTUAL RESULTS SUMMARY:")
    print("=" * 80)
    print(df_summary[['condition', 'bridges_remaining_out_of_16', 'bridges_remaining_c7_out_of_10',
                      'breakage_cases_out_of_25', 'dice_25_mean', 'neck_median_prob_c7', 'neck_pct_ge75_c7']].to_string(index=False))

if __name__ == '__main__':
    main()
