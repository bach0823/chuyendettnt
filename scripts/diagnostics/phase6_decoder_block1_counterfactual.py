#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_decoder_block1_counterfactual.py

Phase 6 Bridge Localization v4:
Causal & Intra-Block-1 Localization Diagnostic

Answers:
1. Is the false bridge already represented at S2 (28x28), or is it CREATED/AMPLIFIED
   inside Decoder Block 1 during the S2 -> S1 transition?
2. At which exact internal operation of Decoder Block 1 does bridge-related activation
   first become strongly amplified?
   - T0: S2 input to Block 1 (28x28, 192 ch)
   - T1: Post-upsample (56x56, 192 ch)
   - T2: Post-skip concat (56x56, 288 ch)
   - T3: Post-conv1 (56x56, 96 ch)
   - T4: Post-conv2 / Block 1 out (56x56, 96 ch)
3. Does destroying the spatial arrangement of S2 features (via an on-manifold spatial shuffle)
   materially change bridge behavior?
4. Do D2-cured cases exhibit a different S2/Block-1 signature from 7/7 persistent cases?

Strict Diagnostic-Only Rules:
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
# Metric Utilities
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
    has_bridge = bool(bridge_events > 0)

    # Breakage
    fragmented_gt_components = 0
    for g_id in range(1, gt_cc + 1):
        overlapping_pred = np.unique(pred_labels[gt_labels == g_id])
        overlapping_pred = overlapping_pred[overlapping_pred > 0]
        if len(overlapping_pred) >= 2:
            fragmented_gt_components += 1
    has_breakage = bool(fragmented_gt_components > 0)

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

def compute_distribution_stats(arr: np.ndarray) -> Dict[str, float]:
    if len(arr) == 0:
        return {
            'N': 0, 'mean': np.nan, 'std': np.nan, 'median': np.nan,
            'P25': np.nan, 'P75': np.nan, 'P90': np.nan, 'min': np.nan, 'max': np.nan
        }
    return {
        'N': int(len(arr)),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'median': float(np.median(arr)),
        'P25': float(np.percentile(arr, 25)),
        'P75': float(np.percentile(arr, 75)),
        'P90': float(np.percentile(arr, 90)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr))
    }

# -----------------------------------------------------------------------------
# Intra-Block-1 Hook & S2-Shuffle Inference Engine
# -----------------------------------------------------------------------------

def run_intra_block1_inference(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    shuffle_s2: bool = False,
    shuffle_seed: int = 42,
    tile_size: int = 448,
    batch_size: int = 4
) -> Dict[str, Any]:
    """
    Executes Setting A tiling inference while capturing:
    T0: S2 input entering Block 1 (28x28, 192 ch)
    T1: Upsample output inside Block 1 (56x56, 192 ch)
    T2: Skip concat inside Block 1 (56x56, 288 ch)
    T3: Post-conv1 inside Block 1 (56x56, 96 ch)
    T4: Post-conv2 inside Block 1 (56x56, 96 ch = Block 1 output)
    Final: Final logits (448x448)

    If shuffle_s2=True, applies a shared spatial permutation to T0 before entering Block 1.
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

    # Output feature arrays (channel-mean absolute magnitude)
    act_t0 = np.zeros((pH // 16, pW // 16), dtype=np.float32)  # S2 (28x28)
    act_t1 = np.zeros((pH // 8, pW // 8), dtype=np.float32)    # Upsample (56x56)
    act_t2 = np.zeros((pH // 8, pW // 8), dtype=np.float32)    # Concat (56x56)
    act_t3 = np.zeros((pH // 8, pW // 8), dtype=np.float32)    # Post-conv1 (56x56)
    act_t4 = np.zeros((pH // 8, pW // 8), dtype=np.float32)    # Post-conv2 / Out (56x56)
    final_logits = np.zeros((pH, pW), dtype=np.float32)

    cache = {}
    hook_handles = []

    # Permutation generator for S2 spatial shuffle
    rng = torch.Generator().manual_seed(shuffle_seed)
    perm_28x28 = torch.randperm(28 * 28, generator=rng)

    shuffle_sanity = {
        'mean_diff_max': 0.0,
        'std_diff_max': 0.0,
        'norm_diff_max': 0.0
    }

    def pre_hook_block1(module, args):
        x_in, skip = args
        # x_in is T0 (S2 input: B, 192, 28, 28)
        cache['t0'] = x_in.detach()

        if shuffle_s2:
            B, C, h, w = x_in.shape
            assert h == 28 and w == 28
            flat_x = x_in.view(B, C, h * w)
            shuffled_x = flat_x[:, :, perm_28x28.to(x_in.device)].view(B, C, h, w)

            # Sanity verification of marginal invariance
            m_diff = (x_in.mean(dim=(2, 3)) - shuffled_x.mean(dim=(2, 3))).abs().max().item()
            s_diff = (x_in.std(dim=(2, 3)) - shuffled_x.std(dim=(2, 3))).abs().max().item()
            n_diff = (x_in.norm(dim=(2, 3)) - shuffled_x.norm(dim=(2, 3))).abs().max().item()
            shuffle_sanity['mean_diff_max'] = max(shuffle_sanity['mean_diff_max'], m_diff)
            shuffle_sanity['std_diff_max'] = max(shuffle_sanity['std_diff_max'], s_diff)
            shuffle_sanity['norm_diff_max'] = max(shuffle_sanity['norm_diff_max'], n_diff)

            return (shuffled_x, skip)
        return (x_in, skip)

    def post_hook_upsample(module, inp, out):
        cache['t1'] = out.detach()  # T1: (B, 192, 56, 56)

    def pre_hook_conv1(module, args):
        cache['t2'] = args[0].detach()  # T2: (B, 288, 56, 56)

    def post_hook_conv1(module, inp, out):
        cache['t3'] = out.detach()  # T3: (B, 96, 56, 56)

    def post_hook_conv2(module, inp, out):
        cache['t4'] = out.detach()  # T4: (B, 96, 56, 56)

    b1 = model.decoder.decoder_blocks[1]
    h0 = b1.register_forward_pre_hook(pre_hook_block1)
    h1 = b1.upsample.register_forward_hook(post_hook_upsample)
    h2 = b1.conv1.register_forward_pre_hook(pre_hook_conv1)
    h3 = b1.conv1.register_forward_hook(post_hook_conv1)
    h4 = b1.conv2.register_forward_hook(post_hook_conv2)
    hook_handles.extend([h0, h1, h2, h3, h4])

    try:
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch = torch.stack(patches[i:i+batch_size]).to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    out_logits = model(batch)

                b_final = out_logits.squeeze(1).float().cpu().numpy()
                b_t0 = cache['t0'].abs().mean(dim=1).float().cpu().numpy()
                b_t1 = cache['t1'].abs().mean(dim=1).float().cpu().numpy()
                b_t2 = cache['t2'].abs().mean(dim=1).float().cpu().numpy()
                b_t3 = cache['t3'].abs().mean(dim=1).float().cpu().numpy()
                b_t4 = cache['t4'].abs().mean(dim=1).float().cpu().numpy()

                for j in range(len(b_final)):
                    y, x = coords[i + j]
                    # T0 (stride 16)
                    act_t0[y//16 : y//16 + 28, x//16 : x//16 + 28] = b_t0[j]
                    # T1, T2, T3, T4 (stride 8)
                    act_t1[y//8 : y//8 + 56, x//8 : x//8 + 56] = b_t1[j]
                    act_t2[y//8 : y//8 + 56, x//8 : x//8 + 56] = b_t2[j]
                    act_t3[y//8 : y//8 + 56, x//8 : x//8 + 56] = b_t3[j]
                    act_t4[y//8 : y//8 + 56, x//8 : x//8 + 56] = b_t4[j]
                    # Final (stride 1)
                    final_logits[y : y + 448, x : x + 448] = b_final[j]
    finally:
        for h in hook_handles:
            h.remove()

    h_s2_dim, w_s2_dim = H // 16, W // 16
    h_s1_dim, w_s1_dim = H // 8, W // 8

    logits_cropped = final_logits[:H, :W]
    prob_cropped = 1.0 / (1.0 + np.exp(-logits_cropped.astype(np.float64)))

    return {
        'logits': logits_cropped,
        'prob': prob_cropped,
        'pred_bin': (logits_cropped > 0.0).astype(np.uint8),
        't0_act': act_t0[:h_s2_dim, :w_s2_dim],
        't1_act': act_t1[:h_s1_dim, :w_s1_dim],
        't2_act': act_t2[:h_s1_dim, :w_s1_dim],
        't3_act': act_t3[:h_s1_dim, :w_s1_dim],
        't4_act': act_t4[:h_s1_dim, :w_s1_dim],
        'shuffle_sanity': shuffle_sanity
    }

# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    out_dir = 'results/diagnostics/phase6_bridge_localization_v4'
    os.makedirs(out_dir, exist_ok=True)

    config_path = 'results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml'
    ckpt_path = 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'
    master_csv = 'results/diagnostics/phase6_d2_topology/topology_7models_master_paired.csv'
    df_master = pd.read_csv(master_csv)

    # 25 Deterministic cases
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

    # Save selected_case_ids.csv
    df_cases = pd.DataFrame(sample_cases_meta, columns=['image_id', 'stratum'])
    case_ids_csv = os.path.join(out_dir, 'selected_case_ids.csv')
    df_cases.to_csv(case_ids_csv, index=False)
    print(f"Saved: {case_ids_csv}")

    data_root = 'datasets/Crack500_ready'
    val_img_dir = os.path.join(data_root, 'val', 'images')
    val_mask_dir = os.path.join(data_root, 'val', 'masks')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = False

    print("=" * 80)
    print("PHASE 6 BRIDGE LOCALIZATION v4: INTRA-BLOCK-1 & S2 CAUSAL TEST")
    print(f"Device: {device} | Checkpoint: {ckpt_path}")
    print("=" * 80)

    model = load_model_from_checkpoint(config_path, ckpt_path, device)
    model.eval()

    # Parameter check
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Sanity Check 1] Model parameter count: {total_params:,} (expected 10,118,955)")
    assert total_params == 10118955, f"Parameter count mismatch: {total_params}"

    # Load validation sample data
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

    # STEP 1: Execute C0 Normal Baseline with Intra-Block-1 Tracking
    print("\n" + "=" * 80)
    print("STEP 1: Executing C0 Normal Baseline & Extracting T0..T4 Intermediate Feature Maps...")
    print("=" * 80)

    c0_results = {}
    base_ref_masks = {}
    stage_stat_rows = []
    s2_dist_rows = []

    for s in tqdm(samples, desc="C0 Inference & Intra-Block-1 Tracking"):
        c_id = s['case_name']
        grp = s['stratum']
        target_bin = s['target_bin']
        H, W = target_bin.shape

        res = run_intra_block1_inference(model, s['img'], device, shuffle_s2=False, tile_size=448)
        metrics = compute_topology_and_global_metrics(res['pred_bin'], target_bin)
        whole_bridge_fp, neck_mask = isolate_base_bridge_corridor(res['pred_bin'], target_bin)

        crack_mask = (res['pred_bin'] == 1) & (target_bin == 1)
        bg_mask = (res['pred_bin'] == 0) & (target_bin == 0)

        # Nearby GT background mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dil_target = cv2.dilate(target_bin, kernel)
        nearby_bg_mask = (dil_target > 0) & (target_bin == 0) & (res['pred_bin'] == 0)
        if np.sum(nearby_bg_mask) == 0:
            nearby_bg_mask = bg_mask

        other_fp_mask = (res['pred_bin'] == 1) & (target_bin == 0) & (whole_bridge_fp == 0)

        c0_results[c_id] = {
            'res': res,
            'metrics': metrics,
            'has_base_bridge': metrics['has_bridge'],
        }

        base_ref_masks[c_id] = {
            'neck_mask': neck_mask,
            'whole_bridge_fp': whole_bridge_fp,
            'crack_mask': crack_mask,
            'bg_mask': bg_mask,
            'nearby_bg_mask': nearby_bg_mask,
            'other_fp_mask': other_fp_mask,
            'has_base_bridge': metrics['has_bridge'],
        }

        # TASK A: Intermediate Hook Tracking (T0, T1, T2, T3, T4)
        t_stages = [
            ('T0_S2_In_28x28', res['t0_act'], 16),
            ('T1_Upsample_56x56', res['t1_act'], 8),
            ('T2_Concat_56x56', res['t2_act'], 8),
            ('T3_PostConv1_56x56', res['t3_act'], 8),
            ('T4_PostConv2_56x56', res['t4_act'], 8),
        ]

        for s_name, f_map, scale in t_stages:
            h_s, w_s = f_map.shape
            neck_s = (cv2.resize(neck_mask.astype(np.float32), (w_s, h_s), interpolation=cv2.INTER_AREA) > 0.05)
            bg_s = (cv2.resize(nearby_bg_mask.astype(np.float32), (w_s, h_s), interpolation=cv2.INTER_AREA) > 0.3)
            crack_s = (cv2.resize(crack_mask.astype(np.float32), (w_s, h_s), interpolation=cv2.INTER_AREA) > 0.05)
            other_fp_s = (cv2.resize(other_fp_mask.astype(np.float32), (w_s, h_s), interpolation=cv2.INTER_AREA) > 0.05)

            neck_vals = f_map[neck_s] if np.sum(neck_s) > 0 else np.array([])
            bg_vals = f_map[bg_s] if np.sum(bg_s) > 0 else np.array([])
            crack_vals = f_map[crack_s] if np.sum(crack_s) > 0 else np.array([])
            other_fp_vals = f_map[other_fp_s] if np.sum(other_fp_s) > 0 else np.array([])

            st_neck = compute_distribution_stats(neck_vals)
            st_bg = compute_distribution_stats(bg_vals)
            st_crack = compute_distribution_stats(crack_vals)
            st_ofp = compute_distribution_stats(other_fp_vals)

            neck_mean = st_neck['mean']
            bg_mean = st_bg['mean']
            crack_mean = st_crack['mean']

            contrast_bg = float(neck_mean / (bg_mean + 1e-6)) if not np.isnan(neck_mean) and not np.isnan(bg_mean) else np.nan
            diff_bg = float(neck_mean - bg_mean) if not np.isnan(neck_mean) and not np.isnan(bg_mean) else np.nan
            ratio_crack = float(neck_mean / (crack_mean + 1e-6)) if not np.isnan(neck_mean) and not np.isnan(crack_mean) else np.nan
            diff_crack = float(neck_mean - crack_mean) if not np.isnan(neck_mean) and not np.isnan(crack_mean) else np.nan

            stage_stat_rows.append({
                'case_name': c_id,
                'stratum': grp,
                'stage': s_name,
                'scale': scale,
                'neck_pixels': st_neck['N'],
                'neck_mean': round(neck_mean, 4),
                'neck_median': round(st_neck['median'], 4),
                'neck_p25': round(st_neck['P25'], 4),
                'neck_p75': round(st_neck['P75'], 4),
                'neck_p90': round(st_neck['P90'], 4),
                'neck_std': round(st_neck['std'], 4),
                'bg_mean': round(bg_mean, 4),
                'bg_median': round(st_bg['median'], 4),
                'bg_std': round(st_bg['std'], 4),
                'crack_mean': round(crack_mean, 4),
                'crack_median': round(st_crack['median'], 4),
                'other_fp_mean': round(st_ofp['mean'], 4),
                'contrast_neck_over_bg': round(contrast_bg, 4),
                'diff_neck_minus_bg': round(diff_bg, 4),
                'ratio_neck_over_crack': round(ratio_crack, 4),
                'diff_neck_minus_crack': round(diff_crack, 4),
            })

            # TASK B: S2 Detailed Distribution (T0 stage)
            if s_name == 'T0_S2_In_28x28':
                s2_dist_rows.append({
                    'case_name': c_id,
                    'stratum': grp,
                    'has_base_bridge': int(metrics['has_bridge']),
                    'neck_mean': round(neck_mean, 4),
                    'neck_median': round(st_neck['median'], 4),
                    'neck_p25': round(st_neck['P25'], 4),
                    'neck_p75': round(st_neck['P75'], 4),
                    'neck_p90': round(st_neck['P90'], 4),
                    'neck_std': round(st_neck['std'], 4),
                    'bg_mean': round(bg_mean, 4),
                    'bg_median': round(st_bg['median'], 4),
                    'bg_p25': round(st_bg['P25'], 4),
                    'bg_p75': round(st_bg['P75'], 4),
                    'bg_p90': round(st_bg['P90'], 4),
                    'crack_mean': round(crack_mean, 4),
                    'crack_median': round(st_crack['median'], 4),
                    'contrast_s2_neck_over_bg': round(contrast_bg, 4),
                    'diff_s2_neck_minus_bg': round(diff_bg, 4),
                    'ratio_s2_neck_over_crack': round(ratio_crack, 4),
                })

    # Verify Sanity Check C0 reproduction
    sorted_samples = sorted(samples, key=lambda x: x['case_name'])
    sub_master = df_master[df_master['case_name'].isin([s['case_name'] for s in samples])].sort_values('case_name')
    master_dices = [sub_master[sub_master['case_name'] == s['case_name']]['Base_dice'].values[0] for s in sorted_samples]
    master_bridges = [sub_master[sub_master['case_name'] == s['case_name']]['Base_bridge_events'].values[0] > 0 for s in sorted_samples]

    max_dice_diff = max(abs(c0_results[s['case_name']]['metrics']['dice'] - m_d) for s, m_d in zip(sorted_samples, master_dices))
    bridge_match = all(c0_results[s['case_name']]['metrics']['has_bridge'] == m_b for s, m_b in zip(sorted_samples, master_bridges))

    print(f"\n[Sanity Check 2] C0 reproduction vs Official Base Master:")
    print(f"  - Max Dice discrepancy: {max_dice_diff:.8f} (tolerance 1e-4)")
    print(f"  - Bridge classifications match exactly: {bridge_match} (16/25 bridge positive)")
    assert max_dice_diff < 1e-4, f"C0 Dice mismatch: {max_dice_diff}"
    assert bridge_match, "C0 Bridge mismatch!"
    print(">>> SANITY CHECK PASS: C0 exactly reproduces official Base baseline! <<<\n")

    # Save Task A & Task B CSVs
    df_stage_stats = pd.DataFrame(stage_stat_rows)
    out_stage_csv = os.path.join(out_dir, 'decoder_block1_stage_statistics.csv')
    df_stage_stats.to_csv(out_stage_csv, index=False)
    print(f"Saved: {out_stage_csv} ({len(df_stage_stats)} rows)")

    df_s2_dist = pd.DataFrame(s2_dist_rows)
    out_s2_csv = os.path.join(out_dir, 'decoder_s2_distribution.csv')
    df_s2_dist.to_csv(out_s2_csv, index=False)
    print(f"Saved: {out_s2_csv} ({len(df_s2_dist)} rows)")

    # STEP 2: Execute Task C — S2 Spatial Shuffle Counterfactual
    print("\n" + "=" * 80)
    print("STEP 2: Executing Task C — S2 Spatial Shuffle Counterfactual...")
    print("=" * 80)

    s2_shuffle_rows = []

    for s in tqdm(samples, desc="S2 Spatial Shuffle Inference"):
        c_id = s['case_name']
        grp = s['stratum']
        target_bin = s['target_bin']
        ref = base_ref_masks[c_id]
        has_base_bridge = ref['has_base_bridge']
        neck_ref = ref['neck_mask']

        # Normal C0 metrics
        m_c0 = c0_results[c_id]['metrics']
        p_c0 = c0_results[c_id]['res']['prob']
        neck_p_c0 = p_c0[neck_ref > 0] if np.sum(neck_ref) > 0 else np.array([])
        c0_neck_med = float(np.median(neck_p_c0)) if len(neck_p_c0) > 0 else np.nan

        # Run S2 Spatial Shuffle Inference
        res_shuf = run_intra_block1_inference(model, s['img'], device, shuffle_s2=True, shuffle_seed=42, tile_size=448)
        m_shuf = compute_topology_and_global_metrics(res_shuf['pred_bin'], target_bin)
        p_shuf = res_shuf['prob']
        neck_p_shuf = p_shuf[neck_ref > 0] if np.sum(neck_ref) > 0 else np.array([])
        shuf_neck_med = float(np.median(neck_p_shuf)) if len(neck_p_shuf) > 0 else np.nan
        shuf_neck_ge50 = float(np.sum(neck_p_shuf >= 0.50) / len(neck_p_shuf)) if len(neck_p_shuf) > 0 else np.nan
        shuf_neck_ge75 = float(np.sum(neck_p_shuf >= 0.75) / len(neck_p_shuf)) if len(neck_p_shuf) > 0 else np.nan

        bridge_remains = bool(has_base_bridge and m_shuf['has_bridge'])

        s2_shuffle_rows.append({
            'case_name': c_id,
            'stratum': grp,
            'has_base_bridge': int(has_base_bridge),
            'c0_dice': round(m_c0['dice'], 4),
            'c0_precision': round(m_c0['precision'], 4),
            'c0_recall': round(m_c0['recall'], 4),
            'c0_cldice': round(m_c0['cldice'], 4),
            'c0_has_bridge': int(m_c0['has_bridge']),
            'c0_has_breakage': int(m_c0['has_breakage']),
            'c0_neck_median_prob': round(c0_neck_med, 4),
            'shuf_dice': round(m_shuf['dice'], 4),
            'shuf_precision': round(m_shuf['precision'], 4),
            'shuf_recall': round(m_shuf['recall'], 4),
            'shuf_cldice': round(m_shuf['cldice'], 4),
            'shuf_has_bridge': int(m_shuf['has_bridge']),
            'shuf_has_breakage': int(m_shuf['has_breakage']),
            'shuf_bridge_remains': int(bridge_remains),
            'shuf_neck_median_prob': round(shuf_neck_med, 4),
            'shuf_neck_frac_ge50': round(shuf_neck_ge50, 4),
            'shuf_neck_frac_ge75': round(shuf_neck_ge75, 4),
            's2_shuffle_norm_diff': round(res_shuf['shuffle_sanity']['norm_diff_max'], 8),
            's2_shuffle_mean_diff': round(res_shuf['shuffle_sanity']['mean_diff_max'], 8),
        })

    df_s2_shuf = pd.DataFrame(s2_shuffle_rows)
    out_shuf_csv = os.path.join(out_dir, 's2_shuffle_per_sample.csv')
    df_s2_shuf.to_csv(out_shuf_csv, index=False)
    print(f"Saved: {out_shuf_csv} ({len(df_s2_shuf)} rows)")

    # STEP 3: Aggregating Summary Table
    print("\n" + "=" * 80)
    print("STEP 3: Aggregating Summary Tables across Strata & Internal Operations...")
    print("=" * 80)

    summary_rows = []

    # 1. Intra-Block-1 Operation Contrast Progression on 10 Consensus 7/7 cases
    sub_c7_stages = df_stage_stats[df_stage_stats['stratum'] == 'Consensus_7of7']
    sub_cured_stages = df_stage_stats[df_stage_stats['stratum'] == 'Cured_by_D2']

    stage_order = ['T0_S2_In_28x28', 'T1_Upsample_56x56', 'T2_Concat_56x56', 'T3_PostConv1_56x56', 'T4_PostConv2_56x56']
    stage_prog_rows = []
    for stg in stage_order:
        c7_stg = sub_c7_stages[sub_c7_stages['stage'] == stg]
        cured_stg = sub_cured_stages[sub_cured_stages['stage'] == stg]
        stage_prog_rows.append({
            'operation': stg,
            'c7_mean_neck_act': round(c7_stg['neck_mean'].mean(), 4),
            'c7_mean_bg_act': round(c7_stg['bg_mean'].mean(), 4),
            'c7_contrast_neck_over_bg': round(c7_stg['contrast_neck_over_bg'].mean(), 4),
            'c7_diff_neck_minus_bg': round(c7_stg['diff_neck_minus_bg'].mean(), 4),
            'c7_ratio_neck_over_crack': round(c7_stg['ratio_neck_over_crack'].mean(), 4),
            'cured_contrast_neck_over_bg': round(cured_stg['contrast_neck_over_bg'].mean(), 4),
            'cured_diff_neck_minus_bg': round(cured_stg['diff_neck_minus_bg'].mean(), 4),
        })
    df_stage_prog = pd.DataFrame(stage_prog_rows)
    out_prog_csv = os.path.join(out_dir, 'decoder_block1_operation_progression.csv')
    df_stage_prog.to_csv(out_prog_csv, index=False)
    print(f"Saved: {out_prog_csv}")

    # 2. Counterfactual Summary (C0 vs S2_spatial_shuffle)
    shuf_c7 = df_s2_shuf[df_s2_shuf['stratum'] == 'Consensus_7of7']
    shuf_cured = df_s2_shuf[df_s2_shuf['stratum'] == 'Cured_by_D2']
    shuf_all = df_s2_shuf

    summary_rows.append({
        'stratum': 'Consensus_7of7_10cases',
        'N_cases': len(shuf_c7),
        'c0_bridges_remaining': f"{int(shuf_c7['c0_has_bridge'].sum())}/10",
        'shuf_bridges_remaining': f"{int(shuf_c7['shuf_has_bridge'].sum())}/10",
        'c0_mean_dice': round(shuf_c7['c0_dice'].mean(), 4),
        'shuf_mean_dice': round(shuf_c7['shuf_dice'].mean(), 4),
        'c0_mean_recall': round(shuf_c7['c0_recall'].mean(), 4),
        'shuf_mean_recall': round(shuf_c7['shuf_recall'].mean(), 4),
        'c0_breakage_cases': f"{int(shuf_c7['c0_has_breakage'].sum())}/10",
        'shuf_breakage_cases': f"{int(shuf_c7['shuf_has_breakage'].sum())}/10",
        'c0_neck_median_prob': round(shuf_c7['c0_neck_median_prob'].mean(), 4),
        'shuf_neck_median_prob': round(shuf_c7['shuf_neck_median_prob'].mean(), 4),
        'shuf_neck_pct_ge50': round(shuf_c7['shuf_neck_frac_ge50'].mean() * 100.0, 2),
    })

    summary_rows.append({
        'stratum': 'Cured_by_D2_5cases',
        'N_cases': len(shuf_cured),
        'c0_bridges_remaining': f"{int(shuf_cured['c0_has_bridge'].sum())}/5",
        'shuf_bridges_remaining': f"{int(shuf_cured['shuf_has_bridge'].sum())}/5",
        'c0_mean_dice': round(shuf_cured['c0_dice'].mean(), 4),
        'shuf_mean_dice': round(shuf_cured['shuf_dice'].mean(), 4),
        'c0_mean_recall': round(shuf_cured['c0_recall'].mean(), 4),
        'shuf_mean_recall': round(shuf_cured['shuf_recall'].mean(), 4),
        'c0_breakage_cases': f"{int(shuf_cured['c0_has_breakage'].sum())}/5",
        'shuf_breakage_cases': f"{int(shuf_cured['shuf_has_breakage'].sum())}/5",
        'c0_neck_median_prob': round(shuf_cured['c0_neck_median_prob'].mean(), 4),
        'shuf_neck_median_prob': round(shuf_cured['shuf_neck_median_prob'].mean(), 4),
        'shuf_neck_pct_ge50': round(shuf_cured['shuf_neck_frac_ge50'].mean() * 100.0, 2),
    })

    summary_rows.append({
        'stratum': 'All_25_Representative_Cases',
        'N_cases': len(shuf_all),
        'c0_bridges_remaining': f"{int(shuf_all['c0_has_bridge'].sum())}/16",
        'shuf_bridges_remaining': f"{int(shuf_all['shuf_has_bridge'].sum())}/16",
        'c0_mean_dice': round(shuf_all['c0_dice'].mean(), 4),
        'shuf_mean_dice': round(shuf_all['shuf_dice'].mean(), 4),
        'c0_mean_recall': round(shuf_all['c0_recall'].mean(), 4),
        'shuf_mean_recall': round(shuf_all['shuf_recall'].mean(), 4),
        'c0_breakage_cases': f"{int(shuf_all['c0_has_breakage'].sum())}/25",
        'shuf_breakage_cases': f"{int(shuf_all['shuf_has_breakage'].sum())}/25",
        'c0_neck_median_prob': round(shuf_all['c0_neck_median_prob'].dropna().mean(), 4),
        'shuf_neck_median_prob': round(shuf_all['shuf_neck_median_prob'].dropna().mean(), 4),
        'shuf_neck_pct_ge50': round(shuf_all['shuf_neck_frac_ge50'].dropna().mean() * 100.0, 2),
    })

    df_summary = pd.DataFrame(summary_rows)
    out_summary_csv = os.path.join(out_dir, 'counterfactual_v4_summary.csv')
    df_summary.to_csv(out_summary_csv, index=False)
    print(f"Saved: {out_summary_csv}")

    print("\n" + "=" * 80)
    print("TASK A: INTRA-BLOCK-1 OPERATION CONTRAST PROGRESSION (10 CONSENSUS CASES):")
    print("=" * 80)
    print(df_stage_prog.to_string(index=False))

    print("\n" + "=" * 80)
    print("TASK C: S2 SPATIAL SHUFFLE COUNTERFACTUAL SUMMARY:")
    print("=" * 80)
    print(df_summary.to_string(index=False))

if __name__ == '__main__':
    main()
