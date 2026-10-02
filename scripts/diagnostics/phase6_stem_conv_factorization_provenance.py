#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_stem_conv_factorization_provenance.py

Phase 6: Stem Conv2d(3, 48, 4, 4) Mathematical Factorization Diagnostic
Scientific Goal:
Determine whether representation ambiguity at raw_conv arises primarily from:
1. Spatial aggregation / receptive-field mixing of the stride-4 projection (DC spatial pooling vs AC spatial filter),
2. Learned channel projection (color/channel weights vs spatial patterns via SVD Rank-1),
3. Physical receptive-field boundary bleed (crack pixels sharing the 4x4 token) vs pure background gap activation.

Factorization Modes Evaluated:
1. raw_conv_full: Full learned Conv2d(3, 48, 4, 4, s=4)
2. raw_conv_dc:   DC component = AvgPool2d(4, 4) + Learned 1x1 Channel Projection (3.74% weight energy)
3. raw_conv_ac:   AC component = Pure Zero-Mean Spatial Filter Bank (96.26% weight energy)
4. raw_conv_rank1: SVD Rank-1 Separable Component (86.59% weight energy: Channel_Proj (x) Spatial_Pattern)

Cohorts Evaluated:
- Consensus 7/7: N = 101
- Cured by D2: N = 7
- Clean Control: N = 56 (gt_cc >= 2, 0 bridges across all 7 models)

HARD LOCKS:
- DIAGNOSTIC-ONLY: Zero training, zero gradients, zero checkpoint mutations.
- Setting A evaluation path preserved.
- Sealed test set strictly untouched.
"""

import hashlib
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
# Checkpoint Hash Verification
# -----------------------------------------------------------------------------

def compute_model_param_hash(model: nn.Module) -> str:
    """Computes SHA256 of all model parameters to ensure zero bitwise weight drift."""
    hasher = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        hasher.update(name.encode('utf-8'))
        hasher.update(p.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def compute_file_hash(path: str) -> str:
    """Computes SHA256 of file on disk."""
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# -----------------------------------------------------------------------------
# ROI Isolation & Pairwise Separation Utilities
# -----------------------------------------------------------------------------

def isolate_bridged_pairs_and_rois(
    pred_bin: np.ndarray,
    target_bin: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Isolates:
    - neck_mask (corridor of FP bridging GT CCs)
    - crack_mask (merged GT CCs)
    - bg_mask (local background outside corridor)
    - pair_records: List of bridged GT component pairs
    """
    H, W = target_bin.shape
    num_gt_cc, gt_labels = cv2.connectedComponents(target_bin.astype(np.uint8), connectivity=8)
    num_pred_cc, pred_labels = cv2.connectedComponents(pred_bin.astype(np.uint8), connectivity=8)
    pred_cc = int(max(num_pred_cc - 1, 0))
    gt_cc = int(max(num_gt_cc - 1, 0))

    neck_mask = np.zeros((H, W), dtype=np.uint8)
    crack_mask = np.zeros((H, W), dtype=np.uint8)
    corridor_union = np.zeros((H, W), dtype=np.uint8)
    pair_records = []

    if gt_cc < 2 or pred_cc < 1:
        return neck_mask, crack_mask, neck_mask, pair_records

    for p_id in range(1, pred_cc + 1):
        cc_pred = (pred_labels == p_id)
        overlapping_gt = np.unique(gt_labels[cc_pred])
        overlapping_gt = overlapping_gt[overlapping_gt > 0]

        if len(overlapping_gt) >= 2:
            cc_fp = cc_pred & (target_bin == 0)

            for g_id in overlapping_gt:
                crack_mask = crack_mask | (gt_labels == g_id).astype(np.uint8)

            for i in range(len(overlapping_gt)):
                for j in range(i + 1, len(overlapping_gt)):
                    gi = overlapping_gt[i]
                    gj = overlapping_gt[j]
                    mask_i = (gt_labels == gi)
                    mask_j = (gt_labels == gj)

                    dt_i = distance_transform_edt(~mask_i)
                    gap_dist = float(np.min(dt_i[mask_j]))
                    radius = max(int(np.ceil(gap_dist / 2.0)) + 2, 3)
                    ksize = 2 * radius + 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

                    dil_i = cv2.dilate(mask_i.astype(np.uint8), kernel)
                    dil_j = cv2.dilate(mask_j.astype(np.uint8), kernel)

                    corridor = (dil_i > 0) & (dil_j > 0)
                    corridor_union = corridor_union | corridor.astype(np.uint8)
                    neck = cc_fp & corridor
                    neck_mask = neck_mask | neck.astype(np.uint8)

                    # 4x downsampling grid cell collision test (112x112 grid)
                    mi_t = torch.from_numpy(mask_i).float().unsqueeze(0).unsqueeze(0)
                    mj_t = torch.from_numpy(mask_j).float().unsqueeze(0).unsqueeze(0)
                    mi_112 = F.max_pool2d(mi_t, 4, 4).squeeze() > 0
                    mj_112 = F.max_pool2d(mj_t, 4, 4).squeeze() > 0
                    grid_cell_collision = bool((mi_112 & mj_112).any().item())

                    pair_records.append({
                        'gt_i': int(gi),
                        'gt_j': int(gj),
                        'gap_distance_px': gap_dist,
                        'grid_cell_collision': grid_cell_collision,
                        'mask_i_448': mask_i,
                        'mask_j_448': mask_j,
                        'corridor_448': corridor,
                        'neck_448': neck
                    })

    if len(pair_records) > 0 and np.sum(neck_mask) == 0:
        for p_id in range(1, pred_cc + 1):
            cc_pred = (pred_labels == p_id)
            overlapping_gt = np.unique(gt_labels[cc_pred])
            overlapping_gt = overlapping_gt[overlapping_gt > 0]
            if len(overlapping_gt) >= 2:
                neck_mask = neck_mask | (cc_pred & (target_bin == 0)).astype(np.uint8)

    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    dilated_corridor = cv2.dilate(corridor_union, kernel_bg)
    nearby_bg_mask = (dilated_corridor > 0) & (target_bin == 0) & (pred_bin == 0)

    if np.sum(nearby_bg_mask) < 50:
        kernel_bg_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        dilated_corridor_large = cv2.dilate(corridor_union, kernel_bg_large)
        nearby_bg_mask = (dilated_corridor_large > 0) & (target_bin == 0) & (pred_bin == 0)

    return neck_mask, crack_mask, nearby_bg_mask.astype(np.uint8), pair_records


def isolate_clean_pairs_and_rois(
    target_bin: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Isolates the closest separated pair of GT CCs and their gap corridor for Clean Control cases."""
    H, W = target_bin.shape
    num_gt_cc, gt_labels = cv2.connectedComponents(target_bin.astype(np.uint8), connectivity=8)

    min_gap = 1e9
    best_pair = None
    for i in range(1, num_gt_cc):
        m_i = (gt_labels == i)
        dt_i = distance_transform_edt(~m_i)
        for j in range(i + 1, num_gt_cc):
            m_j = (gt_labels == j)
            d = float(np.min(dt_i[m_j]))
            if d < min_gap:
                min_gap = d
                best_pair = (i, j)

    if best_pair is None:
        return np.zeros((H, W), dtype=np.uint8), target_bin.astype(np.uint8), np.ones((H, W), dtype=np.uint8), []

    i, j = best_pair
    m_i = (gt_labels == i)
    m_j = (gt_labels == j)
    radius = max(int(np.ceil(min_gap / 2.0)) + 2, 3)
    ksize = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dil_i = cv2.dilate(m_i.astype(np.uint8), kernel)
    dil_j = cv2.dilate(m_j.astype(np.uint8), kernel)
    corridor = (dil_i > 0) & (dil_j > 0)

    gap_neck = corridor & (target_bin == 0)
    crack_mask = (m_i | m_j).astype(np.uint8)

    k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    dil_corridor = cv2.dilate(corridor.astype(np.uint8), k_bg)
    nearby_bg = (dil_corridor > 0) & (target_bin == 0) & (~gap_neck)

    if np.sum(nearby_bg) < 50:
        k_bg_l = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        dil_corridor_l = cv2.dilate(corridor.astype(np.uint8), k_bg_l)
        nearby_bg = (dil_corridor_l > 0) & (target_bin == 0) & (~gap_neck)

    mi_t = torch.from_numpy(m_i).float().unsqueeze(0).unsqueeze(0)
    mj_t = torch.from_numpy(m_j).float().unsqueeze(0).unsqueeze(0)
    mi_112 = F.max_pool2d(mi_t, 4, 4).squeeze() > 0
    mj_112 = F.max_pool2d(mj_t, 4, 4).squeeze() > 0
    grid_cell_collision = bool((mi_112 & mj_112).any().item())

    pair_records = [{
        'gt_i': int(i),
        'gt_j': int(j),
        'gap_distance_px': min_gap,
        'grid_cell_collision': grid_cell_collision,
        'mask_i_448': m_i,
        'mask_j_448': m_j,
        'corridor_448': corridor,
        'neck_448': gap_neck
    }]

    return gap_neck.astype(np.uint8), crack_mask, nearby_bg.astype(np.uint8), pair_records


# -----------------------------------------------------------------------------
# Factorization Engine
# -----------------------------------------------------------------------------

FACTORIZATION_MODES = [
    'raw_conv_full',
    'raw_conv_dc',
    'raw_conv_ac',
    'raw_conv_rank1'
]

class StemConvFactorizationExtractor:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

        conv = self.model.backbone.convnext.stem[0]
        self.w = conv.weight.data.clone() # (48, 3, 4, 4)
        self.b = conv.bias.data.clone()   # (48,)

        # 1. DC (Spatial Mean) Component: W_dc = mean over (4, 4) spatial kernel
        self.w_dc = self.w.mean(dim=(2, 3), keepdim=True).expand_as(self.w)

        # 2. AC (Spatial Zero-Mean Detail) Component: W_ac = W - W_dc
        self.w_ac = self.w - self.w_dc

        # 3. SVD Rank-1 Component: W_rank1 = leading singular vector (channel x space)
        w_flat = self.w.view(48, 3, 16)
        w_r1 = torch.zeros_like(w_flat)
        for k in range(48):
            U, S, V = torch.svd(w_flat[k])
            w_r1[k] = S[0] * torch.outer(U[:, 0], V[:, 0])
        self.w_rank1 = w_r1.view(48, 3, 4, 4)

        # Compute energy ratios of components
        e_total = (self.w ** 2).sum().item()
        e_dc = (self.w_dc ** 2).sum().item()
        e_ac = (self.w_ac ** 2).sum().item()
        e_r1 = (self.w_rank1 ** 2).sum().item()

        self.energy_stats = {
            'e_total': e_total,
            'e_dc': e_dc,
            'pct_dc': (e_dc / e_total) * 100,
            'e_ac': e_ac,
            'pct_ac': (e_ac / e_total) * 100,
            'e_rank1': e_r1,
            'pct_rank1': (e_r1 / e_total) * 100
        }

    def run_image(
        self,
        image: np.ndarray,
        tile_size: int = 448
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, Dict[str, torch.Tensor]]]:
        """Setting A tiling inference with mathematical factorization of stem conv."""
        H, W = image.shape[:2]
        pad_h = (tile_size - (H % tile_size)) % tile_size
        pad_w = (tile_size - (W % tile_size)) % tile_size
        padded_img = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        pH, pW = padded_img.shape[:2]

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        patches = []
        coords = []
        for y in range(0, pH, tile_size):
            for x in range(0, pW, tile_size):
                patch = padded_img[y:y+tile_size, x:x+tile_size]
                patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
                patch_tensor = (patch_tensor - mean) / std
                patches.append(patch_tensor)
                coords.append((y, x))

        final_logits = np.zeros((pH, pW), dtype=np.float32)
        factorized_maps = {m: np.zeros((pH, pW), dtype=np.float32) for m in FACTORIZATION_MODES}
        patch_acts = {m: [] for m in FACTORIZATION_MODES}

        self.model.eval()
        with torch.no_grad():
            for pt, (py, px) in zip(patches, coords):
                # Official model output (Setting A baseline)
                logits = self.model(pt)
                if logits.shape[2:] != (tile_size, tile_size):
                    logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
                final_logits[py:py+tile_size, px:px+tile_size] = logits.squeeze().cpu().numpy()

                # Mathematical Factorizations of Stem Conv:
                out_full = F.conv2d(pt, self.w, self.b, stride=4)
                out_dc = F.conv2d(pt, self.w_dc, torch.zeros_like(self.b), stride=4) # DC has zero bias for pure linear decomposition
                out_ac = F.conv2d(pt, self.w_ac, self.b, stride=4)                  # AC retains bias
                out_rank1 = F.conv2d(pt, self.w_rank1, self.b, stride=4)

                outs = {
                    'raw_conv_full': out_full,
                    'raw_conv_dc': out_dc,
                    'raw_conv_ac': out_ac,
                    'raw_conv_rank1': out_rank1
                }

                for m_name in FACTORIZATION_MODES:
                    act = outs[m_name]
                    energy = torch.norm(act, p=2, dim=1).squeeze(0)  # (112, 112)
                    energy_448 = F.interpolate(
                        energy.unsqueeze(0).unsqueeze(0),
                        size=(tile_size, tile_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze().cpu().numpy()

                    factorized_maps[m_name][py:py+tile_size, px:px+tile_size] = energy_448
                    patch_acts[m_name].append({
                        'coords': (py, px),
                        'act': act.cpu()
                    })

        logits_cropped = final_logits[:H, :W]
        prob_cropped = 1.0 / (1.0 + np.exp(-logits_cropped))
        pred_bin = (logits_cropped > 0.0).astype(np.uint8)
        cropped_maps = {m: factorized_maps[m][:H, :W] for m in FACTORIZATION_MODES}

        return pred_bin, prob_cropped, cropped_maps, patch_acts


# -----------------------------------------------------------------------------
# Visualization Generator (8 Panels: 2 rows x 4 cols)
# -----------------------------------------------------------------------------

def plot_stem_conv_factorization_case(
    stem: str,
    cohort: str,
    image_rgb: np.ndarray,
    target_bin: np.ndarray,
    pred_bin: np.ndarray,
    neck_mask: np.ndarray,
    crack_mask: np.ndarray,
    bg_mask: np.ndarray,
    factorized_maps: Dict[str, np.ndarray],
    factorized_metrics: Dict[str, Dict[str, float]],
    bleed_stats: Dict[str, float],
    save_path: str
):
    """Renders 8 panels: Overlay, ROI Masks, Full Conv, DC Component, AC Component, Rank-1 SVD, Receptive Bleed Map, Trajectory."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    fig.suptitle(f"Stem Conv Mathematical Factorization: {stem} (Cohort: {cohort})", fontsize=16, fontweight='bold', y=0.98)

    # 1. Overlay
    overlay = image_rgb.copy()
    gt_contours, _ = cv2.findContours(target_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(overlay, pred_contours, -1, (255, 0, 0), 1)
    axes[0, 0].imshow(overlay)
    axes[0, 0].set_title("Input RGB + GT (Grn) / Pred (Red)", fontsize=11, fontweight='semibold')
    axes[0, 0].axis('off')

    # 2. ROIs
    roi_rgb = np.zeros_like(image_rgb)
    roi_rgb[crack_mask == 1] = [0, 200, 0]
    roi_rgb[bg_mask == 1] = [50, 50, 180]
    roi_rgb[neck_mask == 1] = [255, 40, 40]
    axes[0, 1].imshow(roi_rgb)
    axes[0, 1].set_title("ROIs: Crack (Grn) / Neck (Red) / BG (Blue)", fontsize=11, fontweight='semibold')
    axes[0, 1].axis('off')

    # Panels 2, 3, 4, 5
    coords_modes = [
        ('raw_conv_full', 0, 2, "Full Conv2d(3,48,4,4) [100% W]"),
        ('raw_conv_dc', 0, 3, "DC Component (AvgPool + 1x1) [3.7% W]"),
        ('raw_conv_ac', 1, 0, "AC Component (Zero-Mean Filter) [96.3% W]"),
        ('raw_conv_rank1', 1, 1, "SVD Rank-1 (Channel (x) Space) [86.6% W]"),
    ]

    neck_cnt, _ = cv2.findContours(neck_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for m_name, r, c, title in coords_modes:
        ax = axes[r, c]
        emap = factorized_maps[m_name]
        m = factorized_metrics[m_name]
        im = ax.imshow(emap, cmap='viridis')

        for cnt in neck_cnt:
            ax.plot(cnt[:, 0, 0], cnt[:, 0, 1], color='red', linewidth=1.2)

        r_norm = m['R_norm']
        sep_m = m['sep_margin']
        cos_nc = m.get('cos_neck_crack', float('nan'))
        stat_str = f"R_norm={r_norm:.2f} | SepM={sep_m:.2f} | cos(N,C)={cos_nc:.2f}"
        ax.set_title(f"{title}\n{stat_str}", fontsize=9.5)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel (1, 2): Physical Receptive Field Bleed Map
    ax_bleed = axes[1, 2]
    # Highlight pure background neck cells vs bleed neck cells
    bleed_vis = np.zeros_like(image_rgb)
    if 'pure_bg_mask' in bleed_stats:
        pure_bg_m = bleed_stats['pure_bg_mask']
        bleed_m = bleed_stats['bleed_mask']
        bleed_vis[crack_mask == 1] = [0, 180, 0]
        bleed_vis[pure_bg_m == 1] = [255, 50, 50]   # Pure background neck (red)
        bleed_vis[bleed_m == 1] = [255, 200, 0]      # Bleed neck (yellow)
    ax_bleed.imshow(bleed_vis)
    pct_pure = bleed_stats.get('pct_pure_bg', 0.0)
    pct_bleed = bleed_stats.get('pct_bleed', 0.0)
    ax_bleed.set_title(f"Receptive Field Bleed Map\nPure BG Neck={pct_pure:.1f}% | Bleed Neck={pct_bleed:.1f}%", fontsize=9.5, fontweight='bold')
    ax_bleed.axis('off')

    # Panel (1, 3): Factorization Trajectory Comparison
    ax_traj = axes[1, 3]
    x_labels = ['Full Conv', 'DC (Pool)', 'AC (Edge)', 'Rank-1 SVD']
    x_indices = list(range(len(x_labels)))
    r_norms = [factorized_metrics[m]['R_norm'] for m in FACTORIZATION_MODES]
    sep_margins = [factorized_metrics[m]['sep_margin'] for m in FACTORIZATION_MODES]

    ax_traj.plot(x_indices, r_norms, marker='o', color='#d95f02', linewidth=2.0, label='R_norm')
    ax_traj.plot(x_indices, sep_margins, marker='s', color='#1b9e77', linewidth=2.0, label='SepMargin')
    ax_traj.axhline(0.50, color='gray', linestyle='--', alpha=0.7, label='50% Emergence')
    ax_traj.axhline(0.20, color='blue', linestyle=':', alpha=0.7, label='Sep Valley (0.20)')
    ax_traj.axhline(0.00, color='black', linestyle='-', alpha=0.4)
    ax_traj.set_xticks(x_indices)
    ax_traj.set_xticklabels(x_labels, rotation=20, fontsize=9)
    ax_traj.set_title(f"Component Separation Profile\nSepMargin: DC={sep_margins[1]:+.2f} vs AC={sep_margins[2]:+.2f}", fontsize=10, fontweight='bold')
    ax_traj.set_ylabel("Metric Value")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(fontsize=8, loc='best')

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
# Main Diagnostic Routine
# -----------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Phase 6: Stem Conv2d(3, 48, 4, 4) Mathematical Factorization Diagnostic")
    print("=" * 80)

    config_path = 'results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml'
    ckpt_path = 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'
    output_dir = 'results/diagnostics/phase6_stem_conv_factorization'
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load Model and compute initial bitwise state hashes
    print(f"Loading Candidate B model from: {ckpt_path}...")
    init_file_hash = compute_file_hash(ckpt_path)
    model = load_model_from_checkpoint(config_path, ckpt_path, device=device)
    init_param_hash = compute_model_param_hash(model)
    print(f"Initial Model Parameters SHA256: {init_param_hash[:16]}...")
    print(f"Initial Disk Checkpoint SHA256: {init_file_hash[:16]}...")

    extractor = StemConvFactorizationExtractor(model, device)
    print("\nStem Conv Weight Energy Decomposition:")
    print(f"  - Total Weight Energy:     {extractor.energy_stats['e_total']:.4f}")
    print(f"  - DC (Spatial Mean):       {extractor.energy_stats['e_dc']:.4f} ({extractor.energy_stats['pct_dc']:.2f}%)")
    print(f"  - AC (Spatial Variance):   {extractor.energy_stats['e_ac']:.4f} ({extractor.energy_stats['pct_ac']:.2f}%)")
    print(f"  - Rank-1 SVD Separable:    {extractor.energy_stats['e_rank1']:.4f} ({extractor.energy_stats['pct_rank1']:.2f}%)")

    # Establish Cohorts
    consensus_path = 'results/diagnostics/phase6_bridge_localization/bridge_consensus_per_image.csv'
    df_meta = pd.read_csv(consensus_path)

    master_path = 'results/diagnostics/phase6_d2_topology/topology_7models_master_paired.csv'
    df_master = pd.read_csv(master_path)
    models = ['Base', 'A1', 'A2', 'B1', 'C1', 'D1', 'D2']
    bridge_cols = [f'{m}_bridge_events' for m in models]
    clean_mask = (df_master[bridge_cols] == 0).all(axis=1) & (df_master['gt_cc'] >= 2)
    clean_stems = set(df_master[clean_mask]['case_name'].tolist())

    c7_stems = set(df_meta[df_meta['n_models_with_bridge'] == 7]['image_id'].tolist())
    d2_cured_stems = set(df_meta[df_meta['d2_transition_type'] == 'Cured_by_D2']['image_id'].tolist())

    print(f"\nCohorts Identified:")
    print(f"  - Consensus 7/7:         N = {len(c7_stems)}")
    print(f"  - Cured by D2:           N = {len(d2_cured_stems)}")
    print(f"  - Clean Control (gt>=2): N = {len(clean_stems)}")

    cohort_list = []
    for s in sorted(list(c7_stems)):
        cohort_list.append((s, 'Consensus_7of7'))
    for s in sorted(list(d2_cured_stems)):
        cohort_list.append((s, 'Cured_by_D2'))
    for s in sorted(list(clean_stems)):
        cohort_list.append((s, 'Clean_Control'))

    vis_c7 = sorted(list(c7_stems))[:6]
    vis_d2 = sorted(list(d2_cured_stems))[:4]
    vis_clean = sorted(list(clean_stems))[:3]
    vis_set = set(vis_c7 + vis_d2 + vis_clean)

    per_sample_records = []
    pairwise_separation_records = []
    bleed_audit_records = []
    t0 = time.time()

    tile_size = 448

    for stem, cohort in tqdm(cohort_list, desc="Evaluating Stem Conv Factorization"):
        img_path = f"datasets/Crack500_ready/val/images/{stem}.jpg"
        mask_path = f"datasets/Crack500_ready/val/masks/{stem}.png"

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        target_bin = (mask > 0).astype(np.uint8)

        # Run Setting A inference with mathematical factorization
        pred_bin, prob_map, factorized_maps, patch_acts = extractor.run_image(img_rgb)

        # ROI & Pairwise isolation
        if cohort in ('Consensus_7of7', 'Cured_by_D2'):
            neck_mask, crack_mask, bg_mask, pair_records = isolate_bridged_pairs_and_rois(pred_bin, target_bin)
        else:
            neck_mask, crack_mask, bg_mask, pair_records = isolate_clean_pairs_and_rois(target_bin)

        n_neck_px = int(np.sum(neck_mask))
        n_crack_px = int(np.sum(crack_mask))
        n_bg_px = int(np.sum(bg_mask))

        # Receptive Field Bleed Audit on 112x112 grid
        t_t = torch.from_numpy(target_bin).float().unsqueeze(0).unsqueeze(0)
        n_t = torch.from_numpy(neck_mask).float().unsqueeze(0).unsqueeze(0)
        t_112_max = (F.max_pool2d(t_t, 4, 4).squeeze() > 0).numpy()
        n_112_max = (F.max_pool2d(n_t, 4, 4).squeeze() > 0).numpy()

        n_neck_cells = int(np.sum(n_112_max))
        if n_neck_cells > 0:
            bleed_cells_112 = n_112_max & t_112_max
            pure_bg_cells_112 = n_112_max & (~t_112_max)
            n_bleed_cells = int(np.sum(bleed_cells_112))
            n_pure_bg_cells = int(np.sum(pure_bg_cells_112))
            pct_bleed = float(n_bleed_cells / n_neck_cells * 100)
            pct_pure_bg = float(n_pure_bg_cells / n_neck_cells * 100)

            # Map back to 448 for visualization
            bleed_mask_up = cv2.resize(bleed_cells_112.astype(np.uint8), (target_bin.shape[1], target_bin.shape[0]), interpolation=cv2.INTER_NEAREST) & neck_mask
            pure_bg_mask_up = cv2.resize(pure_bg_cells_112.astype(np.uint8), (target_bin.shape[1], target_bin.shape[0]), interpolation=cv2.INTER_NEAREST) & neck_mask
        else:
            n_bleed_cells = 0
            n_pure_bg_cells = 0
            pct_bleed = 0.0
            pct_pure_bg = 100.0
            bleed_mask_up = np.zeros_like(neck_mask)
            pure_bg_mask_up = np.zeros_like(neck_mask)

        bleed_stats = {
            'n_neck_cells': n_neck_cells,
            'n_bleed_cells': n_bleed_cells,
            'n_pure_bg_cells': n_pure_bg_cells,
            'pct_bleed': pct_bleed,
            'pct_pure_bg': pct_pure_bg,
            'bleed_mask': bleed_mask_up,
            'pure_bg_mask': pure_bg_mask_up
        }

        bleed_audit_records.append({
            'image_id': stem,
            'cohort': cohort,
            'n_neck_cells_112': n_neck_cells,
            'n_bleed_cells_112': n_bleed_cells,
            'n_pure_bg_cells_112': n_pure_bg_cells,
            'pct_receptive_field_bleed': pct_bleed,
            'pct_pure_background_gap': pct_pure_bg
        })

        sample_row = {
            'image_id': stem,
            'cohort': cohort,
            'n_pairs': len(pair_records),
            'n_neck_pixels': n_neck_px,
            'n_crack_pixels': n_crack_px,
            'n_bg_pixels': n_bg_px,
            'pct_pure_bg_neck_cells': pct_pure_bg,
            'pct_bleed_neck_cells': pct_bleed,
        }

        factorized_metrics = {}

        # Compute Metrics across factorized modes
        for m_name in FACTORIZATION_MODES:
            # Cosine feature similarity between neck, crack, bg
            f_neck_list, f_crack_list, f_bg_list = [], [], []

            for item in patch_acts[m_name]:
                py, px = item['coords']
                act = item['act']  # (1, 48, 112, 112)

                patch_neck = neck_mask[py:py+tile_size, px:px+tile_size]
                patch_crack = crack_mask[py:py+tile_size, px:px+tile_size]
                patch_bg = bg_mask[py:py+tile_size, px:px+tile_size]

                if np.sum(patch_neck) > 0:
                    m_n = (F.interpolate(torch.from_numpy(patch_neck).float().unsqueeze(0).unsqueeze(0), size=(112, 112), mode='nearest').squeeze() > 0)
                    if m_n.any():
                        f_neck_list.append(act[0, :, m_n].mean(dim=1))

                if np.sum(patch_crack) > 0:
                    m_c = (F.interpolate(torch.from_numpy(patch_crack).float().unsqueeze(0).unsqueeze(0), size=(112, 112), mode='nearest').squeeze() > 0)
                    if m_c.any():
                        f_crack_list.append(act[0, :, m_c].mean(dim=1))

                if np.sum(patch_bg) > 0:
                    m_b = (F.interpolate(torch.from_numpy(patch_bg).float().unsqueeze(0).unsqueeze(0), size=(112, 112), mode='nearest').squeeze() > 0)
                    if m_b.any():
                        f_bg_list.append(act[0, :, m_b].mean(dim=1))

            if len(f_neck_list) > 0 and len(f_crack_list) > 0:
                vec_neck = torch.stack(f_neck_list).mean(dim=0)
                vec_crack = torch.stack(f_crack_list).mean(dim=0)
                cos_nc = float(F.cosine_similarity(vec_neck.unsqueeze(0), vec_crack.unsqueeze(0)).item())
            else:
                cos_nc = 0.0

            if len(f_neck_list) > 0 and len(f_bg_list) > 0:
                vec_neck = torch.stack(f_neck_list).mean(dim=0)
                vec_bg = torch.stack(f_bg_list).mean(dim=0)
                cos_nb = float(F.cosine_similarity(vec_neck.unsqueeze(0), vec_bg.unsqueeze(0)).item())
            else:
                cos_nb = 0.0

            emap = factorized_maps[m_name]
            e_neck = float(np.mean(emap[neck_mask == 1])) if n_neck_px > 0 else 0.0
            e_crack = float(np.mean(emap[crack_mask == 1])) if n_crack_px > 0 else 0.0
            e_bg = float(np.mean(emap[bg_mask == 1])) if n_bg_px > 0 else 0.0

            c_neck_crack = float(e_neck / (e_crack + 1e-6))
            c_neck_bg = float(e_neck / (e_bg + 1e-6))
            sep_margin = float(1.0 - c_neck_crack)

            denom = (e_crack - e_bg)
            r_norm = float((e_neck - e_bg) / denom) if abs(denom) > 1e-5 else 0.0

            factorized_metrics[m_name] = {
                'E_neck': e_neck,
                'E_crack': e_crack,
                'E_bg': e_bg,
                'C_neck_crack': c_neck_crack,
                'C_neck_bg': c_neck_bg,
                'sep_margin': sep_margin,
                'R_norm': r_norm,
                'cos_neck_crack': cos_nc,
                'cos_neck_bg': cos_nb
            }

            sample_row[f'{m_name}_R_norm'] = r_norm
            sample_row[f'{m_name}_sep_margin'] = sep_margin
            sample_row[f'{m_name}_C_neck_crack'] = c_neck_crack
            sample_row[f'{m_name}_E_neck'] = e_neck
            sample_row[f'{m_name}_E_crack'] = e_crack
            sample_row[f'{m_name}_E_bg'] = e_bg
            sample_row[f'{m_name}_cos_neck_crack'] = cos_nc
            sample_row[f'{m_name}_cos_neck_bg'] = cos_nb

        # Pairwise Spatial Separation across factorized modes
        for pair in pair_records:
            p_rec = {
                'image_id': stem,
                'cohort': cohort,
                'gt_i': pair['gt_i'],
                'gt_j': pair['gt_j'],
                'gap_distance_px': pair['gap_distance_px'],
                'grid_cell_collision_112': pair['grid_cell_collision'],
            }
            neck_pair = pair['neck_448']
            crack_pair = pair['mask_i_448'] | pair['mask_j_448']

            for m_name in FACTORIZATION_MODES:
                emap = factorized_maps[m_name]
                if np.sum(neck_pair) > 0:
                    e_n = float(np.mean(emap[neck_pair == 1]))
                else:
                    e_n = 0.0
                e_c = float(np.mean(emap[crack_pair == 1]))
                ratio = float(e_n / (e_c + 1e-6))
                sep_m = float(1.0 - ratio)

                is_separable = bool(sep_m >= 0.20 and not pair['grid_cell_collision'])

                p_rec[f'{m_name}_neck_energy'] = e_n
                p_rec[f'{m_name}_crack_energy'] = e_c
                p_rec[f'{m_name}_contrast'] = ratio
                p_rec[f'{m_name}_sep_margin'] = sep_m
                p_rec[f'{m_name}_is_separable'] = is_separable

            pairwise_separation_records.append(p_rec)

        per_sample_records.append(sample_row)

        # Plot cases
        if stem in vis_set:
            fig_path = os.path.join(figures_dir, f"{cohort.lower()}_{stem}_stem_conv_factorization.png")
            plot_stem_conv_factorization_case(
                stem=stem,
                cohort=cohort,
                image_rgb=img_rgb,
                target_bin=target_bin,
                pred_bin=pred_bin,
                neck_mask=neck_mask,
                crack_mask=crack_mask,
                bg_mask=bg_mask,
                factorized_maps=factorized_maps,
                factorized_metrics=factorized_metrics,
                bleed_stats=bleed_stats,
                save_path=fig_path
            )

    t1 = time.time()
    print(f"\nInference complete in {t1 - t0:.1f}s.")

    # Integrity verification
    final_param_hash = compute_model_param_hash(model)
    final_file_hash = compute_file_hash(ckpt_path)
    assert init_param_hash == final_param_hash, f"FATAL: Model weights modified! ({init_param_hash} vs {final_param_hash})"
    assert init_file_hash == final_file_hash, f"FATAL: Checkpoint file on disk modified! ({init_file_hash} vs {final_file_hash})"
    print(f"Model Parameters State Verified: Bitwise Identical ({final_param_hash[:16]}...)")
    print(f"Disk Checkpoint File Verified: Bitwise Identical ({final_file_hash[:16]}...)")

    # DataFrames
    df_samples = pd.DataFrame(per_sample_records)
    df_pairs = pd.DataFrame(pairwise_separation_records)
    df_bleed = pd.DataFrame(bleed_audit_records)

    sample_csv = os.path.join(output_dir, 'stem_conv_factorization_per_sample.csv')
    df_samples.to_csv(sample_csv, index=False)
    print(f"Saved: {sample_csv}")

    bleed_csv = os.path.join(output_dir, 'receptive_field_bleed_summary.csv')
    df_bleed.to_csv(bleed_csv, index=False)
    print(f"Saved: {bleed_csv}")

    # Build Stem Conv Factorization Summary Table
    summary_rows = []
    for cohort_name in ['Consensus_7of7', 'Cured_by_D2', 'Clean_Control']:
        sub_df = df_samples[df_samples['cohort'] == cohort_name]
        n_sub = len(sub_df)

        for m_name in FACTORIZATION_MODES:
            r_col = f'{m_name}_R_norm'
            sep_col = f'{m_name}_sep_margin'
            c_col = f'{m_name}_C_neck_crack'
            cos_nc_col = f'{m_name}_cos_neck_crack'
            cos_nb_col = f'{m_name}_cos_neck_bg'

            summary_rows.append({
                'cohort': cohort_name,
                'factorization_mode': m_name,
                'n_samples': n_sub,
                'median_R_norm': float(sub_df[r_col].median()),
                'mean_R_norm': float(sub_df[r_col].mean()),
                'p10_R_norm': float(np.percentile(sub_df[r_col], 10)),
                'p25_R_norm': float(np.percentile(sub_df[r_col], 25)),
                'p75_R_norm': float(np.percentile(sub_df[r_col], 75)),
                'p90_R_norm': float(np.percentile(sub_df[r_col], 90)),
                'pct_ge_50': float((sub_df[r_col] >= 0.50).mean() * 100),
                'pct_ge_70': float((sub_df[r_col] >= 0.70).mean() * 100),
                'median_sep_margin': float(sub_df[sep_col].median()),
                'mean_sep_margin': float(sub_df[sep_col].mean()),
                'median_neck_crack_ratio': float(sub_df[c_col].median()),
                'median_cos_neck_crack': float(sub_df[cos_nc_col].median()),
                'median_cos_neck_bg': float(sub_df[cos_nb_col].median()),
            })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, 'stem_conv_factorization_summary.csv')
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    # Build Pairwise Separation Factorization Summary Table
    pair_summary_rows = []
    for cohort_name in ['Consensus_7of7', 'Cured_by_D2', 'Clean_Control']:
        sub_pairs = df_pairs[df_pairs['cohort'] == cohort_name]
        n_pairs = len(sub_pairs)
        if n_pairs == 0:
            continue

        pct_grid_collision = float(sub_pairs['grid_cell_collision_112'].mean() * 100)
        median_gap = float(sub_pairs['gap_distance_px'].median())

        for m_name in FACTORIZATION_MODES:
            pct_sep = float(sub_pairs[f'{m_name}_is_separable'].mean() * 100)
            median_sep_m = float(sub_pairs[f'{m_name}_sep_margin'].median())
            median_contrast = float(sub_pairs[f'{m_name}_contrast'].median())

            pair_summary_rows.append({
                'cohort': cohort_name,
                'factorization_mode': m_name,
                'n_pairs_evaluated': n_pairs,
                'median_gap_px': median_gap,
                'pct_grid_cell_collision_112': pct_grid_collision,
                'pct_pairs_spatially_separable': pct_sep,
                'pct_pairs_representation_collapsed': float(100.0 - pct_sep),
                'median_separation_margin': median_sep_m,
                'median_contrast_neck_crack': median_contrast,
            })

    df_pair_summary = pd.DataFrame(pair_summary_rows)
    pair_sum_csv = os.path.join(output_dir, 'pairwise_separation_factorization.csv')
    df_pair_summary.to_csv(pair_sum_csv, index=False)
    print(f"Saved: {pair_sum_csv}")

    # Plot Multi-Cohort Trajectory across Factorized Modes
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    mode_x = ['Full Conv', 'DC (Pool+1x1)', 'AC (Spatial Filter)', 'Rank-1 SVD']

    colors = {'Consensus_7of7': '#d95f02', 'Cured_by_D2': '#7570b3', 'Clean_Control': '#1b9e77'}
    markers = {'Consensus_7of7': 'o', 'Cured_by_D2': 's', 'Clean_Control': '^'}

    for c_name in ['Consensus_7of7', 'Cured_by_D2', 'Clean_Control']:
        sub_s = df_summary[df_summary['cohort'] == c_name]
        axes[0].plot(mode_x, sub_s['median_R_norm'], marker=markers[c_name], linewidth=2.5, label=c_name, color=colors[c_name])
        axes[1].plot(mode_x, sub_s['median_sep_margin'], marker=markers[c_name], linewidth=2.5, label=c_name, color=colors[c_name])
        axes[2].plot(mode_x, sub_s['median_cos_neck_crack'], marker=markers[c_name], linewidth=2.5, label=c_name, color=colors[c_name])

    axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.6, label='50% Emergence Threshold')
    axes[0].set_title("Median Normalized Emergence (R_norm)\nAcross Stem Conv Mathematical Components", fontsize=11.5, fontweight='bold')
    axes[0].set_ylabel("Median R_norm")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[0].tick_params(axis='x', rotation=15)

    axes[1].axhline(0.20, color='blue', linestyle='--', alpha=0.6, label='Separation Valley (20%)')
    axes[1].axhline(0.0, color='black', linestyle=':', alpha=0.5)
    axes[1].set_title("Median Separation Margin (1 - E_neck / E_crack)\nDC Separates (+0.31) vs AC Collapses (-0.05)", fontsize=11.5, fontweight='bold')
    axes[1].set_ylabel("Separation Margin")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)
    axes[1].tick_params(axis='x', rotation=15)

    axes[2].set_title("Median Feature Cosine Similarity cos(f_neck, f_crack)\nSemantic Alignment Across Components", fontsize=11.5, fontweight='bold')
    axes[2].set_ylabel("Cosine Similarity")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9)
    axes[2].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    fig_summary_path = os.path.join(figures_dir, 'stem_conv_factorization_trajectory_comparison.png')
    plt.savefig(fig_summary_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_summary_path}")

    # Print Receptive Field Bleed Summary
    sub_c7_bleed = df_bleed[df_bleed['cohort'] == 'Consensus_7of7']
    print("\n" + "=" * 80)
    print("Receptive Field Bleed Audit (Consensus 7/7 Cohort):")
    print(f"  - Median % Pure Background Gap Cells: {sub_c7_bleed['pct_pure_background_gap'].median():.2f}%")
    print(f"  - Mean % Pure Background Gap Cells:   {sub_c7_bleed['pct_pure_background_gap'].mean():.2f}%")
    print(f"  - Median % Receptive Field Bleed:     {sub_c7_bleed['pct_receptive_field_bleed'].median():.2f}%")
    print("=" * 80)
    print("Stem Conv Mathematical Factorization Diagnostic Completed Successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
