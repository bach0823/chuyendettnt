#!/usr/bin/env python3
"""
scripts/diagnostics/run_p3_asdw_ablation_evaluation.py

Phase 6D: P3 Feature/Detail Enhancement Ablation (With vs Without ASDW)
Target Model: Candidate B Baseline (B2ConvNeXtViTUNet, Setting A, Tile 448).
Dataset: Crack500 Validation Set (All 348 images).

Conditions Evaluated:
1. Model C: P3-ASDW (Candidate B Baseline with ASDW Refinement active)
2. Model A: Baseline Without ASDW (P3-Identity: CNN S0/S1 -> AdaptiveAvgPool2d(28,28) -> ViT)

Metrics Computed:
- Val Dice, Precision, Recall
- Centerline Dice (clDice)
- Boundary IoU (dilation = 2)
- Hausdorff Distance 95 (HD95) & Boundary F1 (BF1)
- Thin-Crack Preservation (Dice on crack pixels with local thickness <= 3 px)
- Topology: Bridge Events, Bridge Images, Break Events, Break Images, Spurious Islands
- Computational Cost: Runtime (ms/image), Peak VRAM (MB)
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


def compute_thin_crack_dice(pred_bin: np.ndarray, target_bin: np.ndarray, max_thickness: int = 3) -> Optional[float]:
    """Computes Dice specifically on thin cracks (distance from boundary <= max_thickness / 2)."""
    if np.sum(target_bin) == 0:
        return None

    dt = distance_transform_edt(target_bin > 0)
    thin_mask = (target_bin > 0) & (dt <= (max_thickness / 2.0 + 0.5))
    if np.sum(thin_mask) == 0:
        return None

    tp = np.sum((pred_bin == 1) & (thin_mask == 1))
    fp = np.sum((pred_bin == 1) & (target_bin == 0))
    fn = np.sum((pred_bin == 0) & (thin_mask == 1))
    return float((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8))


def run_inference_setting_a(
    model: nn.Module,
    img_rgb: np.ndarray,
    device: torch.device,
    tile_size: int = 448
) -> np.ndarray:
    """Standard Setting A deterministic tiling inference."""
    H, W = img_rgb.shape[:2]
    pad_h = (tile_size - (H % tile_size)) % tile_size
    pad_w = (tile_size - (W % tile_size)) % tile_size
    padded_img = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    pH, pW = padded_img.shape[:2]

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    patches, coords = [], []
    for y in range(0, pH, tile_size):
        for x in range(0, pW, tile_size):
            p = padded_img[y:y+tile_size, x:x+tile_size]
            pt = torch.from_numpy(p).permute(2, 0, 1).float() / 255.0
            patches.append(pt)
            coords.append((y, x))

    batch = torch.stack(patches).to(device)
    batch = (batch - mean) / std

    with torch.no_grad():
        logits_tiles = model(batch)

    logits_full = np.zeros((pH, pW), dtype=np.float32)
    for j, (y, x) in enumerate(coords):
        logits_full[y:y+tile_size, x:x+tile_size] = logits_tiles[j, 0].cpu().numpy()

    return logits_full[:H, :W]


def evaluate_dataset(
    model: nn.Module,
    val_files: List[Tuple[str, str]],
    device: torch.device,
    desc: str
) -> Dict[str, Any]:
    """Runs full evaluation on all validation samples."""
    model.eval()
    records = []
    runtimes = []

    # Reset CUDA memory stats
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    for img_path, mask_path in tqdm(val_files, desc=desc):
        case_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        target_bin = (target > 127).astype(np.uint8)

        t0 = time.time()
        logits = run_inference_setting_a(model, img, device)
        t_elapsed = time.time() - t0
        runtimes.append(t_elapsed * 1000.0)  # ms

        pred_bin = (logits > 0.0).astype(np.uint8)

        # 1. Topology & connectivity
        topo = compute_topology_metrics(pred_bin, target_bin)

        # 2. Boundary IoU & HD95
        b_iou = compute_boundary_iou(pred_bin, target_bin, dilation=2)
        hd95, bf1 = calculate_hd95_bf1(pred_bin, target_bin)

        # 3. Thin crack Dice
        thin_dice = compute_thin_crack_dice(pred_bin, target_bin, max_thickness=3)

        rec = {
            'case_name': case_name,
            'dice': topo['dice'],
            'recall': topo['recall'],
            'precision': topo['precision'],
            'cldice': topo['cldice'],
            'boundary_iou': b_iou,
            'hd95': hd95,
            'bf1': bf1,
            'thin_dice': thin_dice,
            'bridge_events': topo['bridge_events'],
            'has_bridge': 1 if topo['bridge_events'] > 0 else 0,
            'break_events': topo['fragmented_gt_components'],
            'has_break': 1 if topo['fragmented_gt_components'] > 0 else 0,
            'spurious_islands': topo['spurious_island_count'],
            'runtime_ms': t_elapsed * 1000.0
        }
        records.append(rec)

    df = pd.DataFrame(records)
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == 'cuda'
        else 0.0
    )

    valid_thin = df['thin_dice'].dropna()

    summary = {
        'N': len(df),
        'dice': float(df['dice'].mean()),
        'recall': float(df['recall'].mean()),
        'precision': float(df['precision'].mean()),
        'cldice': float(df['cldice'].mean()),
        'boundary_iou': float(df['boundary_iou'].mean()),
        'hd95': float(df['hd95'].mean()),
        'bf1': float(df['bf1'].mean()),
        'thin_dice': float(valid_thin.mean()) if len(valid_thin) > 0 else 0.0,
        'total_bridge_events': int(df['bridge_events'].sum()),
        'bridge_images': int(df['has_bridge'].sum()),
        'total_break_events': int(df['break_events'].sum()),
        'break_images': int(df['has_break'].sum()),
        'spurious_islands': int(df['spurious_islands'].sum()),
        'mean_runtime_ms': float(np.mean(runtimes)),
        'peak_vram_mb': float(peak_vram_mb),
    }

    return summary, df


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    t0_start = time.time()
    print("=" * 80)
    print("PHASE 6D: P3-ASDW ABLATION EVALUATION (WITH VS WITHOUT ASDW)")
    print("Target Model: Candidate B Baseline | Dataset: Crack500 Val (N=348)")
    print("=" * 80)

    out_dir = os.path.join(project_root, 'results', 'diagnostics', 'phase6d_p3_ablation')
    os.makedirs(out_dir, exist_ok=True)

    config_path = os.path.join(project_root, 'results', 'configs', 'b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml')
    ckpt_path = os.path.join(project_root, 'results', 'checkpoints', 'P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = False

    print(f"Device: {device} (CuDNN disabled for numerical determinism)")

    # Collect validation files
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

    print(f"Loaded {len(val_files)} validation pairs.")
    assert len(val_files) == 348, f"Expected exactly 348 validation pairs, got {len(val_files)}"

    # =========================================================================
    # 1. EVALUATION 1: Candidate B with ASDW (Current Baseline)
    # =========================================================================
    print("\n" + "-" * 80)
    print("1. EVALUATING CANDIDATE B WITH P3-ASDW (BASELINE)...")
    print("-" * 80)
    model_c = load_model_from_checkpoint(config_path, ckpt_path, device)
    model_c.eval()

    sum_c, df_c = evaluate_dataset(model_c, val_files, device, "Evaluating P3-C (With ASDW)")
    df_c.to_csv(os.path.join(out_dir, 'metrics_p3_with_asdw_per_sample.csv'), index=False)

    # =========================================================================
    # 2. EVALUATION 2: Candidate B Without ASDW (P3-Identity Counterfactual)
    # =========================================================================
    print("\n" + "-" * 80)
    print("2. EVALUATING CANDIDATE B WITHOUT ASDW (IDENTITY COUNTERFACTUAL)...")
    print("-" * 80)
    # Load fresh model and bypass ASDW refinement on Stage 0 and Stage 1
    model_a = load_model_from_checkpoint(config_path, ckpt_path, device)
    model_a.backbone.convnext.stages[0].p3_refinement = nn.Identity()
    model_a.backbone.convnext.stages[1].p3_refinement = nn.Identity()
    model_a.eval()

    sum_a, df_a = evaluate_dataset(model_a, val_files, device, "Evaluating P3-A (Without ASDW)")
    df_a.to_csv(os.path.join(out_dir, 'metrics_p3_without_asdw_per_sample.csv'), index=False)

    # Save summary JSON
    summary_all = {
        'P3_C_With_ASDW': sum_c,
        'P3_A_Without_ASDW': sum_a,
        'Delta_C_minus_A': {
            'delta_dice': sum_c['dice'] - sum_a['dice'],
            'delta_recall': sum_c['recall'] - sum_a['recall'],
            'delta_precision': sum_c['precision'] - sum_a['precision'],
            'delta_cldice': sum_c['cldice'] - sum_a['cldice'],
            'delta_boundary_iou': sum_c['boundary_iou'] - sum_a['boundary_iou'],
            'delta_hd95': sum_c['hd95'] - sum_a['hd95'],
            'delta_bf1': sum_c['bf1'] - sum_a['bf1'],
            'delta_thin_dice': sum_c['thin_dice'] - sum_a['thin_dice'],
            'delta_bridge_events': sum_c['total_bridge_events'] - sum_a['total_bridge_events'],
            'delta_bridge_images': sum_c['bridge_images'] - sum_a['bridge_images'],
            'delta_break_events': sum_c['total_break_events'] - sum_a['total_break_events'],
            'delta_break_images': sum_c['break_images'] - sum_a['break_images'],
            'delta_runtime_ms': sum_c['mean_runtime_ms'] - sum_a['mean_runtime_ms'],
            'delta_vram_mb': sum_c['peak_vram_mb'] - sum_a['peak_vram_mb'],
        }
    }

    sum_json = os.path.join(out_dir, 'p3_asdw_ablation_summary.json')
    with open(sum_json, 'w', encoding='utf-8') as f:
        json.dump(summary_all, f, indent=2)
    print(f"\nSaved summary JSON to {sum_json}")

    # Print comparison table
    print("\n" + "=" * 90)
    print("PHASE 6D: P3-ASDW ABLATION COMPARISON TABLE (VAL N=348)")
    print("=" * 90)
    print(f"{'Metric':<28} | {'With ASDW (Run C)':<18} | {'Without ASDW (Run A)':<20} | {'Delta (C - A)':<15}")
    print("-" * 90)
    metrics_to_print = [
        ('Val Dice', 'dice', '.4f'),
        ('Recall', 'recall', '.4f'),
        ('Precision', 'precision', '.4f'),
        ('clDice', 'cldice', '.4f'),
        ('Boundary IoU', 'boundary_iou', '.4f'),
        ('HD95 (px, lower is better)', 'hd95', '.2f'),
        ('Boundary F1 (BF1)', 'bf1', '.4f'),
        ('Thin-Crack Dice (<=3px)', 'thin_dice', '.4f'),
        ('Bridge Events', 'total_bridge_events', 'd'),
        ('Bridge Images', 'bridge_images', 'd'),
        ('Break Events', 'total_break_events', 'd'),
        ('Break Images', 'break_images', 'd'),
        ('Spurious Islands', 'spurious_islands', 'd'),
        ('Mean Runtime (ms/img)', 'mean_runtime_ms', '.1f'),
        ('Peak VRAM (MB)', 'peak_vram_mb', '.1f'),
    ]

    for label, key, fmt in metrics_to_print:
        val_c = sum_c[key]
        val_a = sum_a[key]
        delta = val_c - val_a
        val_c_str = f"{val_c:{fmt}}"
        val_a_str = f"{val_a:{fmt}}"
        delta_str = f"{delta:+{fmt}}"
        print(f"{label:<28} | {val_c_str:<18} | {val_a_str:<20} | {delta_str:<15}")
    print("=" * 90)

    elapsed = time.time() - t0_start
    print(f"Ablation evaluation completed in {elapsed:.1f}s.")


if __name__ == '__main__':
    main()
