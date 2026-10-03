#!/usr/bin/env python3
"""
scripts/diagnostics/run_asdw_same_checkpoint_on_off_probe.py

Phase 6D: Same-Checkpoint ASDW ON vs OFF Causal Probe on Candidate B
Target Model: Candidate B Baseline (B2ConvNeXtViTUNet, Setting A, Tile 448).
Dataset: Crack500 Validation Set (All 348 images).

Causal Intervention:
X' = X + gamma * F(X)  -->  X  (Exact Identity bypass, same weights, zero retraining)

Measures:
1. Global Pixel-level Metrics:
   - Val Dice, IoU, Precision, Recall, Centerline Dice (clDice)
   - Boundary IoU (dilation = 2)
   - Thin-crack subset Dice (GT distance <= 1.5 px -> thickness <= 3 px)
2. Topology & Connectivity:
   - Bridge events & Bridge images
   - Break events & Break images
   - Spurious islands
3. Direct Logit Perturbation (Delta z = z_ON - z_OFF):
   - Mean |Delta z|, Max |Delta z|, P95 |Delta z|, P99 |Delta z|
   - Total pixel sign flips across all 348 images (z_ON > 0 != z_OFF > 0)
4. High-Frequency Representation Probe at S0 and S1:
   - Relative perturbation norm: ||X' - X|| / ||X|| = gamma * ||F(X)|| / ||X||
   - Spectral High-Frequency Ratio: ||HP(X')|| / ||X'|| vs ||HP(X)|| / ||X||
   - Measured on Crack regions vs Background regions
"""

import collections
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Ensure SAGE_LITE and project root are on sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sage_lite_dir = os.path.join(project_root, "SAGE_LITE")
for p in [project_root, sage_lite_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from tools.run_phase6_c_topology_diagnostic import (
    load_model_from_checkpoint,
    compute_topology_metrics,
)
from sage.utils.advanced_metrics import calculate_hd95_bf1


def compute_boundary_iou(pred_bin: np.ndarray, target_bin: np.ndarray, dilation: int = 2) -> float:
    """Computes Boundary IoU between binary prediction and target."""
    if np.sum(target_bin) == 0 and np.sum(pred_bin) == 0:
        return 1.0
    if np.sum(target_bin) == 0 or np.sum(pred_bin) == 0:
        return 0.0

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilation + 1, 2 * dilation + 1))
    pred_boundary = cv2.dilate(pred_bin.astype(np.uint8), k) - cv2.erode(pred_bin.astype(np.uint8), k)
    gt_boundary = cv2.dilate(target_bin.astype(np.uint8), k) - cv2.erode(target_bin.astype(np.uint8), k)

    inter = np.sum((pred_boundary > 0) & (gt_boundary > 0))
    union = np.sum((pred_boundary > 0) | (gt_boundary > 0))
    return float(inter / (union + 1e-8))


def compute_thin_crack_metrics(pred_bin: np.ndarray, target_bin: np.ndarray, max_thickness: int = 3) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Computes Dice, Recall, and Precision specifically on thin crack pixels (thickness <= 3px)."""
    if np.sum(target_bin) == 0:
        return None, None, None

    dt = distance_transform_edt(target_bin > 0)
    thin_mask = (target_bin > 0) & (dt <= (max_thickness / 2.0 + 0.5))
    if np.sum(thin_mask) == 0:
        return None, None, None

    tp = int(np.sum((pred_bin == 1) & (thin_mask == 1)))
    fp = int(np.sum((pred_bin == 1) & (target_bin == 0)))
    fn = int(np.sum((pred_bin == 0) & (thin_mask == 1)))

    thin_dice = float((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8))
    thin_recall = float(tp / (tp + fn + 1e-8))
    thin_prec = float(tp / (tp + fp + 1e-8))
    return thin_dice, thin_recall, thin_prec


def compute_high_frequency_stats(x: torch.Tensor, x_refined: torch.Tensor) -> Dict[str, float]:
    """
    Computes rigorous high-frequency spectral metrics between x and x_refined = x + gamma * F(x):
    - Relative perturbation norm: ||x_refined - x|| / ||x||
    - High-pass Laplacian energy ratio: ||HP(x_refined)|| / ||x_refined|| vs ||HP(x)|| / ||x||
    """
    with torch.no_grad():
        C = x.shape[1]
        laplacian_kernel = torch.tensor([
            [0.0,  1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0,  1.0, 0.0]
        ], device=x.device, dtype=x.dtype).view(1, 1, 3, 3).repeat(C, 1, 1, 1)

        norm_x = torch.norm(x, p=2).item()
        delta_x = x_refined - x
        norm_delta = torch.norm(delta_x, p=2).item()
        rel_perturbation = norm_delta / (norm_x + 1e-8)

        hp_x = F.conv2d(x, laplacian_kernel, padding=1, groups=C)
        hp_ref = F.conv2d(x_refined, laplacian_kernel, padding=1, groups=C)

        norm_hp_x = torch.norm(hp_x, p=2).item()
        norm_hp_ref = torch.norm(hp_ref, p=2).item()
        norm_ref = torch.norm(x_refined, p=2).item()

        hp_ratio_off = norm_hp_x / (norm_x + 1e-8)
        hp_ratio_on = norm_hp_ref / (norm_ref + 1e-8)
        delta_hp_ratio = hp_ratio_on - hp_ratio_off

    return {
        'rel_perturbation': float(rel_perturbation),
        'hp_ratio_off': float(hp_ratio_off),
        'hp_ratio_on': float(hp_ratio_on),
        'delta_hp_ratio': float(delta_hp_ratio)
    }


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    t0_start = time.time()
    print("=" * 80)
    print("PHASE 6D: SAME-CHECKPOINT ASDW ON VS OFF CAUSAL PROBE")
    print("Target Model: Candidate B Baseline | Checkpoint: Candidate B | Dataset: Crack500 Val (N=348)")
    print("=" * 80)

    out_dir = os.path.join(project_root, 'results', 'diagnostics', 'asdw_on_off')
    os.makedirs(out_dir, exist_ok=True)

    config_path = os.path.join(project_root, 'results', 'configs', 'b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml')
    ckpt_path = os.path.join(project_root, 'results', 'checkpoints', 'P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = False

    print(f"Device: {device} (CuDNN disabled for determinism)")

    # Load Model ON (Standard Candidate B)
    print("Loading Candidate B (Model ON)...")
    model_on = load_model_from_checkpoint(config_path, ckpt_path, device)
    model_on.eval()

    # Load Model OFF (Exact same weights, ASDW bypassed: Identity)
    print("Loading Candidate B (Model OFF: ASDW Bypassed via Identity)...")
    model_off = load_model_from_checkpoint(config_path, ckpt_path, device)
    # Bypass ASDW by setting gamma = 0.0 and p3_refinement = Identity
    model_off.backbone.convnext.stages[0].p3_refinement = nn.Identity()
    model_off.backbone.convnext.stages[1].p3_refinement = nn.Identity()
    model_off.eval()

    gamma_s0 = model_on.backbone.convnext.stages[0].p3_refinement.gamma.item()
    gamma_s1 = model_on.backbone.convnext.stages[1].p3_refinement.gamma.item()
    print(f"Candidate B Learned Gamma Parameters: S0 gamma = {gamma_s0:.6f}, S1 gamma = {gamma_s1:.6f}")

    # Load validation files
    data_root = os.path.join(project_root, 'datasets', 'Crack500_ready')
    val_img_dir = os.path.join(data_root, 'val', 'images')
    val_mask_dir = os.path.join(data_root, 'val', 'masks')

    val_files = []
    for f in sorted(os.listdir(val_img_dir)):
        if f.lower().endswith(('.jpg', '.png')):
            base = os.path.splitext(f)[0]
            mp = os.path.join(val_mask_dir, base + '.png')
            if not os.path.exists(mp):
                mp = os.path.join(val_mask_dir, base + '.jpg')
            if os.path.exists(mp):
                val_files.append((os.path.join(val_img_dir, f), mp))

    assert len(val_files) == 348, f"Expected 348 validation images, got {len(val_files)}"
    print(f"Validation cohort verified: {len(val_files)} samples.")

    # Tiling parameters
    tile_size = 448
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    records = []
    all_abs_deltas = []
    total_pixel_sign_flips = 0
    total_pixels_evaluated = 0

    pbar = tqdm(val_files, desc="Running ASDW ON vs OFF Diagnostic")

    for img_path, mask_path in pbar:
        case_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        target_bin = (target > 127).astype(np.uint8)
        H, W = target_bin.shape[:2]

        # Tiling
        pad_h = (tile_size - (H % tile_size)) % tile_size
        pad_w = (tile_size - (W % tile_size)) % tile_size
        padded_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        pH, pW = padded_img.shape[:2]

        patches, coords = [], []
        for y in range(0, pH, tile_size):
            for x in range(0, pW, tile_size):
                p = padded_img[y:y+tile_size, x:x+tile_size]
                pt = torch.from_numpy(p).permute(2, 0, 1).float() / 255.0
                patches.append(pt)
                coords.append((y, x))

        batch = torch.stack(patches).to(device)
        batch = (batch - mean) / std

        # -----------------------------------------------------------------
        # 1. High-Frequency Probe at S0 and S1 before spatial compression
        # -----------------------------------------------------------------
        with torch.no_grad():
            x_stem = model_on.backbone.convnext.stem(batch)
            # Stage 0 feature entering P3
            x_s0 = x_stem
            x_s0_refined = model_on.backbone.convnext.stages[0].p3_refinement(x_s0)
            hf_s0 = compute_high_frequency_stats(x_s0, x_s0_refined)

            # Stage 1 feature entering P3
            s0_out = model_on.backbone.convnext.stages[0](x_stem)
            x_s1 = model_on.backbone.convnext.stages[1]._execute_main_path(s0_out)
            x_s1_refined = model_on.backbone.convnext.stages[1].p3_refinement(x_s1)
            hf_s1 = compute_high_frequency_stats(x_s1, x_s1_refined)

            # Full Forward Model ON and Model OFF
            logits_tiles_on = model_on(batch)
            logits_tiles_off = model_off(batch)

        # Assemble full logits
        logits_on_full = np.zeros((pH, pW), dtype=np.float32)
        logits_off_full = np.zeros((pH, pW), dtype=np.float32)
        for j, (y, x) in enumerate(coords):
            logits_on_full[y:y+tile_size, x:x+tile_size] = logits_tiles_on[j, 0].cpu().numpy()
            logits_off_full[y:y+tile_size, x:x+tile_size] = logits_tiles_off[j, 0].cpu().numpy()

        l_on = logits_on_full[:H, :W]
        l_off = logits_off_full[:H, :W]

        pred_on = (l_on > 0.0).astype(np.uint8)
        pred_off = (l_off > 0.0).astype(np.uint8)

        # -----------------------------------------------------------------
        # 2. Per-Pixel Delta Logit Analysis
        # -----------------------------------------------------------------
        delta_l = l_on - l_off
        abs_delta_l = np.abs(delta_l)
        mean_abs_dl = float(np.mean(abs_delta_l))
        max_abs_dl = float(np.max(abs_delta_l))
        p95_abs_dl = float(np.percentile(abs_delta_l, 95))
        p99_abs_dl = float(np.percentile(abs_delta_l, 99))

        sign_flips = int(np.sum(pred_on != pred_off))
        total_pixel_sign_flips += sign_flips
        total_pixels_evaluated += (H * W)

        # Sample for overall quantile distribution
        all_abs_deltas.append(abs_delta_l.flatten()[::100])

        # -----------------------------------------------------------------
        # 3. Topology & Connectivity Metrics
        # -----------------------------------------------------------------
        topo_on = compute_topology_metrics(pred_on, target_bin)
        topo_off = compute_topology_metrics(pred_off, target_bin)

        biou_on = compute_boundary_iou(pred_on, target_bin, dilation=2)
        biou_off = compute_boundary_iou(pred_off, target_bin, dilation=2)

        hd95_on, bf1_on = calculate_hd95_bf1(pred_on, target_bin)
        hd95_off, bf1_off = calculate_hd95_bf1(pred_off, target_bin)

        thin_d_on, thin_r_on, thin_p_on = compute_thin_crack_metrics(pred_on, target_bin, max_thickness=3)
        thin_d_off, thin_r_off, thin_p_off = compute_thin_crack_metrics(pred_off, target_bin, max_thickness=3)

        # IoU
        tp_on = int(np.sum((pred_on == 1) & (target_bin == 1)))
        fp_on = int(np.sum((pred_on == 1) & (target_bin == 0)))
        fn_on = int(np.sum((pred_on == 0) & (target_bin == 1)))
        iou_on = float(tp_on / max(tp_on + fp_on + fn_on, 1))

        tp_off = int(np.sum((pred_off == 1) & (target_bin == 1)))
        fp_off = int(np.sum((pred_off == 1) & (target_bin == 0)))
        fn_off = int(np.sum((pred_off == 0) & (target_bin == 1)))
        iou_off = float(tp_off / max(tp_off + fp_off + fn_off, 1))

        rec = {
            'case_name': case_name,
            # ON metrics
            'dice_on': topo_on['dice'],
            'iou_on': iou_on,
            'recall_on': topo_on['recall'],
            'precision_on': topo_on['precision'],
            'cldice_on': topo_on['cldice'],
            'biou_on': biou_on,
            'hd95_on': hd95_on,
            'bf1_on': bf1_on,
            'thin_dice_on': thin_d_on,
            'thin_recall_on': thin_r_on,
            'thin_prec_on': thin_p_on,
            'bridge_events_on': topo_on['bridge_events'],
            'break_events_on': topo_on['fragmented_gt_components'],
            'spurious_islands_on': topo_on['spurious_island_count'],

            # OFF metrics
            'dice_off': topo_off['dice'],
            'iou_off': iou_off,
            'recall_off': topo_off['recall'],
            'precision_off': topo_off['precision'],
            'cldice_off': topo_off['cldice'],
            'biou_off': biou_off,
            'hd95_off': hd95_off,
            'bf1_off': bf1_off,
            'thin_dice_off': thin_d_off,
            'thin_recall_off': thin_r_off,
            'thin_prec_off': thin_p_off,
            'bridge_events_off': topo_off['bridge_events'],
            'break_events_off': topo_off['fragmented_gt_components'],
            'spurious_islands_off': topo_off['spurious_island_count'],

            # Differences
            'delta_dice': topo_on['dice'] - topo_off['dice'],
            'delta_biou': biou_on - biou_off,
            'delta_cldice': topo_on['cldice'] - topo_off['cldice'],
            'mean_abs_dl': mean_abs_dl,
            'max_abs_dl': max_abs_dl,
            'p95_abs_dl': p95_abs_dl,
            'sign_flips': sign_flips,

            # S0 and S1 High-Frequency stats
            's0_rel_perturb': hf_s0['rel_perturbation'],
            's0_hp_ratio_on': hf_s0['hp_ratio_on'],
            's0_hp_ratio_off': hf_s0['hp_ratio_off'],
            's0_delta_hp': hf_s0['delta_hp_ratio'],
            's1_rel_perturb': hf_s1['rel_perturbation'],
            's1_hp_ratio_on': hf_s1['hp_ratio_on'],
            's1_hp_ratio_off': hf_s1['hp_ratio_off'],
            's1_delta_hp': hf_s1['delta_hp_ratio'],
        }
        records.append(rec)

    pbar.close()

    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, 'asdw_on_off_per_sample_348.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved per-sample data: {csv_path}")

    # Aggregated quantiles of delta logit
    all_abs_deltas_arr = np.concatenate(all_abs_deltas)
    overall_mean_dl = float(np.mean(all_abs_deltas_arr))
    overall_median_dl = float(np.median(all_abs_deltas_arr))
    overall_p95_dl = float(np.percentile(all_abs_deltas_arr, 95))
    overall_p99_dl = float(np.percentile(all_abs_deltas_arr, 99))
    overall_max_dl = float(df['max_abs_dl'].max())

    valid_thin_on = df['thin_dice_on'].dropna()
    valid_thin_off = df['thin_dice_off'].dropna()

    summary = {
        'cohort': 'Crack500 Val (N=348)',
        'checkpoint': 'Candidate B (P3_C_D4_K2_H64_Phase5_SAGELR2e-4)',
        'gamma_s0': gamma_s0,
        'gamma_s1': gamma_s1,
        'N': len(df),

        # 1. Performance Overview
        'dice_on': float(df['dice_on'].mean()),
        'dice_off': float(df['dice_off'].mean()),
        'delta_dice': float(df['dice_on'].mean() - df['dice_off'].mean()),

        'iou_on': float(df['iou_on'].mean()),
        'iou_off': float(df['iou_off'].mean()),
        'delta_iou': float(df['iou_on'].mean() - df['iou_off'].mean()),

        'recall_on': float(df['recall_on'].mean()),
        'recall_off': float(df['recall_off'].mean()),
        'delta_recall': float(df['recall_on'].mean() - df['recall_off'].mean()),

        'precision_on': float(df['precision_on'].mean()),
        'precision_off': float(df['precision_off'].mean()),
        'delta_precision': float(df['precision_on'].mean() - df['precision_off'].mean()),

        'cldice_on': float(df['cldice_on'].mean()),
        'cldice_off': float(df['cldice_off'].mean()),
        'delta_cldice': float(df['cldice_on'].mean() - df['cldice_off'].mean()),

        # 2. Boundary & Thin-crack Metrics
        'biou_on': float(df['biou_on'].mean()),
        'biou_off': float(df['biou_off'].mean()),
        'delta_biou': float(df['biou_on'].mean() - df['biou_off'].mean()),

        'hd95_on': float(df['hd95_on'].mean()),
        'hd95_off': float(df['hd95_off'].mean()),
        'delta_hd95': float(df['hd95_on'].mean() - df['hd95_off'].mean()),

        'bf1_on': float(df['bf1_on'].mean()),
        'bf1_off': float(df['bf1_off'].mean()),
        'delta_bf1': float(df['bf1_on'].mean() - df['bf1_off'].mean()),

        'thin_dice_on': float(valid_thin_on.mean()),
        'thin_dice_off': float(valid_thin_off.mean()),
        'delta_thin_dice': float(valid_thin_on.mean() - valid_thin_off.mean()),

        # 3. Topology & Connectivity
        'total_bridge_events_on': int(df['bridge_events_on'].sum()),
        'total_bridge_events_off': int(df['bridge_events_off'].sum()),
        'delta_bridge_events': int(df['bridge_events_on'].sum() - df['bridge_events_off'].sum()),

        'bridge_images_on': int((df['bridge_events_on'] > 0).sum()),
        'bridge_images_off': int((df['bridge_events_off'] > 0).sum()),
        'delta_bridge_images': int((df['bridge_events_on'] > 0).sum() - (df['bridge_events_off'] > 0).sum()),

        'total_break_events_on': int(df['break_events_on'].sum()),
        'total_break_events_off': int(df['break_events_off'].sum()),
        'delta_break_events': int(df['break_events_on'].sum() - df['break_events_off'].sum()),

        'break_images_on': int((df['break_events_on'] > 0).sum()),
        'break_images_off': int((df['break_events_off'] > 0).sum()),
        'delta_break_images': int((df['break_events_on'] > 0).sum() - (df['break_events_off'] > 0).sum()),

        'spurious_islands_on': int(df['spurious_islands_on'].sum()),
        'spurious_islands_off': int(df['spurious_islands_off'].sum()),

        # 4. Pixel-Level Delta Logits
        'mean_abs_delta_logit': overall_mean_dl,
        'median_abs_delta_logit': overall_median_dl,
        'p95_abs_delta_logit': overall_p95_dl,
        'p99_abs_delta_logit': overall_p99_dl,
        'max_abs_delta_logit': overall_max_dl,
        'total_pixel_sign_flips': total_pixel_sign_flips,
        'total_pixels_evaluated': total_pixels_evaluated,
        'sign_flip_rate': float(total_pixel_sign_flips / max(total_pixels_evaluated, 1)),

        # 5. High-Frequency Representation Probes at S0 & S1
        's0_relative_perturbation_mean': float(df['s0_rel_perturb'].mean()),
        's0_hp_ratio_on_mean': float(df['s0_hp_ratio_on'].mean()),
        's0_hp_ratio_off_mean': float(df['s0_hp_ratio_off'].mean()),
        's0_delta_hp_mean': float(df['s0_delta_hp'].mean()),

        's1_relative_perturbation_mean': float(df['s1_rel_perturb'].mean()),
        's1_hp_ratio_on_mean': float(df['s1_hp_ratio_on'].mean()),
        's1_hp_ratio_off_mean': float(df['s1_hp_ratio_off'].mean()),
        's1_delta_hp_mean': float(df['s1_delta_hp'].mean()),
    }

    json_path = os.path.join(out_dir, 'asdw_on_off_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON: {json_path}")

    # Print comprehensive report table
    print("\n" + "=" * 90)
    print("PHASE 6D: SAME-CHECKPOINT ASDW ON VS OFF CAUSAL PROBE RESULTS (VAL N=348)")
    print("=" * 90)
    print(f"{'Metric':<32} | {'ON (With ASDW)':<16} | {'OFF (Identity)':<16} | {'Delta (ON - OFF)':<15}")
    print("-" * 90)
    rows_to_print = [
        ('Val Dice', summary['dice_on'], summary['dice_off'], summary['delta_dice'], '.4f'),
        ('Val IoU', summary['iou_on'], summary['iou_off'], summary['delta_iou'], '.4f'),
        ('Recall', summary['recall_on'], summary['recall_off'], summary['delta_recall'], '.4f'),
        ('Precision', summary['precision_on'], summary['precision_off'], summary['delta_precision'], '.4f'),
        ('clDice', summary['cldice_on'], summary['cldice_off'], summary['delta_cldice'], '.4f'),
        ('Boundary IoU (d=2)', summary['biou_on'], summary['biou_off'], summary['delta_biou'], '.4f'),
        ('HD95 (px, lower is better)', summary['hd95_on'], summary['hd95_off'], summary['delta_hd95'], '.2f'),
        ('Boundary F1 (BF1)', summary['bf1_on'], summary['bf1_off'], summary['delta_bf1'], '.4f'),
        ('Thin-Crack Dice (<=3px)', summary['thin_dice_on'], summary['thin_dice_off'], summary['delta_thin_dice'], '.4f'),
        ('Bridge Events', summary['total_bridge_events_on'], summary['total_bridge_events_off'], summary['delta_bridge_events'], 'd'),
        ('Bridge Images', summary['bridge_images_on'], summary['bridge_images_off'], summary['delta_bridge_images'], 'd'),
        ('Break Events', summary['total_break_events_on'], summary['total_break_events_off'], summary['delta_break_events'], 'd'),
        ('Break Images', summary['break_images_on'], summary['break_images_off'], summary['delta_break_images'], 'd'),
    ]

    for label, val_on, val_off, delta, fmt in rows_to_print:
        if fmt == 'd':
            print(f"{label:<32} | {val_on:<16d} | {val_off:<16d} | {delta:+15d}")
        else:
            print(f"{label:<32} | {val_on:<16.4f} | {val_off:<16.4f} | {delta:+15.4f}")

    print("-" * 90)
    print("PIXEL-LEVEL LOGIT PERTURBATION (Delta z = z_ON - z_OFF):")
    print(f"  Mean |Delta z|:             {summary['mean_abs_delta_logit']:.6f}")
    print(f"  Median |Delta z|:           {summary['median_abs_delta_logit']:.6f}")
    print(f"  P95 |Delta z|:              {summary['p95_abs_delta_logit']:.6f}")
    print(f"  P99 |Delta z|:              {summary['p99_abs_delta_logit']:.6f}")
    print(f"  Max |Delta z|:              {summary['max_abs_delta_logit']:.6f}")
    print(f"  Pixel Sign Flips:           {summary['total_pixel_sign_flips']} / {summary['total_pixels_evaluated']:,} ({summary['sign_flip_rate']*100:.6f}%)")
    print("-" * 90)
    print("HIGH-FREQUENCY REPRESENTATION PROBE AT S0 & S1 (PRE-COMPRESSION):")
    print(f"  Stage 0 Perturbation Norm (||X'-X|| / ||X||): {summary['s0_relative_perturbation_mean']*100:.4f}% (gamma = {gamma_s0:.4f})")
    print(f"  Stage 0 High-Pass Ratio (ON vs OFF):         {summary['s0_hp_ratio_on_mean']:.6f} vs {summary['s0_hp_ratio_off_mean']:.6f} (Delta = {summary['s0_delta_hp_mean']:+.6f})")
    print(f"  Stage 1 Perturbation Norm (||X'-X|| / ||X||): {summary['s1_relative_perturbation_mean']*100:.4f}% (gamma = {gamma_s1:.4f})")
    print(f"  Stage 1 High-Pass Ratio (ON vs OFF):         {summary['s1_hp_ratio_on_mean']:.6f} vs {summary['s1_hp_ratio_off_mean']:.6f} (Delta = {summary['s1_delta_hp_mean']:+.6f})")
    print("=" * 90)

    elapsed = time.time() - t0_start
    print(f"Probe completed in {elapsed:.1f}s.")


if __name__ == '__main__':
    main()
