#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_stem_stage0_provenance.py

Phase 6: Stem -> Stage-0 Block-Level Provenance Diagnostic
Scientific Goal:
Determine precisely where the bridge representation begins to degrade spatial separation:
1. Stem / stride-4 projection (Conv2d 3->48, 4x4, s=4 + LayerNorm2d)
2. Stage 0 Block 0 (7x7 depthwise conv + LayerNorm2d + GRN MLP + residual)
3. Stage 0 Block 1 (7x7 depthwise conv + LayerNorm2d + GRN MLP + residual = stage0_pre_sage)
4. Stage 0 Post-SAGE (SageLayer fusion)

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
    - pair_records: List of bridged GT component pairs with gap distances and 4x grid cell collision status
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

    # 4x grid cell collision test
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
# Hook & Inference Engine for Stem and Stage 0 Blocks
# -----------------------------------------------------------------------------

STAGE0_SUBSTAGES = [
    'stem',
    'stage0_block0',
    'stage0_block1',
    'stage0_post_sage'
]

class StemStage0ProvenanceExtractor:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.raw_activations: Dict[str, torch.Tensor] = {}
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        # 1. Stem: Conv2d(3, 48, 4, 4) + LayerNorm2d((48,))
        def stem_hook(m, inp, out):
            if out.dim() == 4 and out.shape[1] == 48 and out.shape[2] == 112 and out.shape[3] == 112:
                self.raw_activations['stem'] = out.detach()
        self.handles.append(self.model.backbone.convnext.stem.register_forward_hook(stem_hook))

        # 2. Stage 0 Block 0 (First learned 7x7 ConvNeXt block)
        blk0 = self.model.backbone.convnext.stages[0].main_block.module.blocks[0]
        def blk0_hook(m, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if t.dim() == 4 and t.shape[1] == 48 and t.shape[2] == 112 and t.shape[3] == 112:
                self.raw_activations['stage0_block0'] = t.detach()
        self.handles.append(blk0.register_forward_hook(blk0_hook))

        # 3. Stage 0 Block 1 (Second learned 7x7 ConvNeXt block = Stage 0 final pre-SAGE)
        blk1 = self.model.backbone.convnext.stages[0].main_block.module.blocks[1]
        def blk1_hook(m, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if t.dim() == 4 and t.shape[1] == 48 and t.shape[2] == 112 and t.shape[3] == 112:
                self.raw_activations['stage0_block1'] = t.detach()
        self.handles.append(blk1.register_forward_hook(blk1_hook))

        # 4. Stage 0 Post-SAGE (SageLayer output)
        stg0 = self.model.backbone.convnext.stages[0]
        def post_sage_hook(m, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if t.dim() == 4 and t.shape[1] == 48 and t.shape[2] == 112 and t.shape[3] == 112:
                self.raw_activations['stage0_post_sage'] = t.detach()
        self.handles.append(stg0.register_forward_hook(post_sage_hook))

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def run_image(self, image: np.ndarray, tile_size: int = 448) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Setting A tiling inference with extraction of stem & Stage 0 sub-blocks."""
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
        stage_full_maps = {s: np.zeros((pH, pW), dtype=np.float32) for s in STAGE0_SUBSTAGES}

        self.model.eval()
        with torch.no_grad():
            for patch_t, (py, px) in zip(patches, coords):
                inp = patch_t.unsqueeze(0).to(self.device)
                logits = self.model(inp)
                if logits.shape[2:] != (tile_size, tile_size):
                    logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)

                final_logits[py:py+tile_size, px:px+tile_size] = logits.squeeze().cpu().numpy()

                for s_name in STAGE0_SUBSTAGES:
                    act = self.raw_activations[s_name]  # (1, 48, 112, 112)
                    energy = torch.norm(act, p=2, dim=1).squeeze(0)  # (112, 112)

                    # Bilinear interpolate to tile size (448, 448)
                    energy_448 = F.interpolate(
                        energy.unsqueeze(0).unsqueeze(0),
                        size=(tile_size, tile_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze().cpu().numpy()

                    stage_full_maps[s_name][py:py+tile_size, px:px+tile_size] = energy_448

        logits_cropped = final_logits[:H, :W]
        prob_cropped = 1.0 / (1.0 + np.exp(-logits_cropped))
        pred_bin = (logits_cropped > 0.0).astype(np.uint8)
        stage_cropped_maps = {s: stage_full_maps[s][:H, :W] for s in STAGE0_SUBSTAGES}

        return pred_bin, prob_cropped, stage_cropped_maps


# -----------------------------------------------------------------------------
# Visualization Generator
# -----------------------------------------------------------------------------

def plot_stem_stage0_case(
    stem: str,
    cohort: str,
    image_rgb: np.ndarray,
    target_bin: np.ndarray,
    pred_bin: np.ndarray,
    neck_mask: np.ndarray,
    crack_mask: np.ndarray,
    bg_mask: np.ndarray,
    stage_maps: Dict[str, np.ndarray],
    stage_metrics: Dict[str, Dict[str, float]],
    save_path: str
):
    """Renders 6 panels: Overlay, ROI Masks, Stem, Block 0, Block 1, Post-SAGE."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"Stem -> Stage 0 Block-Level Provenance: {stem} (Cohort: {cohort})", fontsize=16, fontweight='bold', y=0.98)

    # 1. Overlay
    overlay = image_rgb.copy()
    gt_contours, _ = cv2.findContours(target_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(overlay, pred_contours, -1, (255, 0, 0), 1)
    axes[0, 0].imshow(overlay)
    axes[0, 0].set_title("Input RGB + GT (Green) / Pred (Red)", fontsize=11, fontweight='semibold')
    axes[0, 0].axis('off')

    # 2. ROIs
    roi_rgb = np.zeros_like(image_rgb)
    roi_rgb[crack_mask == 1] = [0, 200, 0]
    roi_rgb[bg_mask == 1] = [50, 50, 180]
    roi_rgb[neck_mask == 1] = [255, 40, 40]
    axes[0, 1].imshow(roi_rgb)
    axes[0, 1].set_title("ROIs: Crack (Grn) / Neck (Red) / BG (Blue)", fontsize=11, fontweight='semibold')
    axes[0, 1].axis('off')

    stages_coords = [
        ('stem', 0, 2, "1. Stem Stride-4 (4x4 Conv + LN)"),
        ('stage0_block0', 1, 0, "2. Stage 0 Block 0 (7x7 DW Conv)"),
        ('stage0_block1', 1, 1, "3. Stage 0 Block 1 (7x7 DW Conv = Pre-SAGE)"),
        ('stage0_post_sage', 1, 2, "4. Stage 0 Post-SAGE (Expert Fusion)"),
    ]

    for s_name, r, c, title in stages_coords:
        ax = axes[r, c]
        emap = stage_maps[s_name]
        m = stage_metrics[s_name]
        im = ax.imshow(emap, cmap='viridis')

        neck_cnt, _ = cv2.findContours(neck_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in neck_cnt:
            ax.plot(cnt[:, 0, 0], cnt[:, 0, 1], color='red', linewidth=1.2)

        r_norm = m['R_norm']
        sep_m = m['sep_margin']
        stat_str = f"R_norm={r_norm:.2f} | SepMargin={sep_m:.2f} | N/C={m['C_neck_crack']:.2f}"
        ax.set_title(f"{title}\n{stat_str}", fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
# Main Diagnostic Routine
# -----------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Phase 6: Stem -> Stage-0 Block-Level Provenance Diagnostic")
    print("=" * 80)

    config_path = 'results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml'
    ckpt_path = 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'
    output_dir = 'results/diagnostics/phase6_stem_stage0_provenance'
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

    extractor = StemStage0ProvenanceExtractor(model, device)

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

    print(f"Cohorts Identified:")
    print(f"  - Consensus 7/7:        N = {len(c7_stems)}")
    print(f"  - Cured by D2:          N = {len(d2_cured_stems)}")
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
    t0 = time.time()

    for stem, cohort in tqdm(cohort_list, desc="Evaluating Stem & Stage 0 Blocks"):
        img_path = f"datasets/Crack500_ready/val/images/{stem}.jpg"
        mask_path = f"datasets/Crack500_ready/val/masks/{stem}.png"

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        target_bin = (mask > 0).astype(np.uint8)

        # Run Setting A inference with feature extraction
        pred_bin, prob_map, stage_maps = extractor.run_image(img_rgb)

        # ROI & Pairwise isolation
        if cohort in ('Consensus_7of7', 'Cured_by_D2'):
            neck_mask, crack_mask, bg_mask, pair_records = isolate_bridged_pairs_and_rois(pred_bin, target_bin)
        else:
            neck_mask, crack_mask, bg_mask, pair_records = isolate_clean_pairs_and_rois(target_bin)

        n_neck_px = int(np.sum(neck_mask))
        n_crack_px = int(np.sum(crack_mask))
        n_bg_px = int(np.sum(bg_mask))

        stage_metrics = {}
        emergence_stage_50 = 'None'
        emergence_stage_70 = 'None'

        sample_row = {
            'image_id': stem,
            'cohort': cohort,
            'n_pairs': len(pair_records),
            'n_neck_pixels': n_neck_px,
            'n_crack_pixels': n_crack_px,
            'n_bg_pixels': n_bg_px,
        }

        for s_name in STAGE0_SUBSTAGES:
            emap = stage_maps[s_name]

            e_neck = float(np.mean(emap[neck_mask == 1])) if n_neck_px > 0 else 0.0
            e_crack = float(np.mean(emap[crack_mask == 1])) if n_crack_px > 0 else 0.0
            e_bg = float(np.mean(emap[bg_mask == 1])) if n_bg_px > 0 else 0.0

            c_neck_crack = float(e_neck / (e_crack + 1e-6))
            c_neck_bg = float(e_neck / (e_bg + 1e-6))
            sep_margin = float(1.0 - c_neck_crack)

            denom = (e_crack - e_bg)
            r_norm = float((e_neck - e_bg) / denom) if abs(denom) > 1e-5 else 0.0

            stage_metrics[s_name] = {
                'E_neck': e_neck,
                'E_crack': e_crack,
                'E_bg': e_bg,
                'C_neck_crack': c_neck_crack,
                'C_neck_bg': c_neck_bg,
                'sep_margin': sep_margin,
                'R_norm': r_norm
            }

            sample_row[f'{s_name}_R_norm'] = r_norm
            sample_row[f'{s_name}_sep_margin'] = sep_margin
            sample_row[f'{s_name}_C_neck_crack'] = c_neck_crack
            sample_row[f'{s_name}_E_neck'] = e_neck
            sample_row[f'{s_name}_E_crack'] = e_crack
            sample_row[f'{s_name}_E_bg'] = e_bg

            if emergence_stage_50 == 'None' and r_norm >= 0.50:
                emergence_stage_50 = s_name
            if emergence_stage_70 == 'None' and r_norm >= 0.70:
                emergence_stage_70 = s_name

        # Compute Deltas across Stage 0 transitions
        sample_row['delta_R_block0'] = stage_metrics['stage0_block0']['R_norm'] - stage_metrics['stem']['R_norm']
        sample_row['delta_R_block1'] = stage_metrics['stage0_block1']['R_norm'] - stage_metrics['stage0_block0']['R_norm']
        sample_row['delta_R_SAGE0'] = stage_metrics['stage0_post_sage']['R_norm'] - stage_metrics['stage0_block1']['R_norm']

        sample_row['emergence_stage_50'] = emergence_stage_50
        sample_row['emergence_stage_70'] = emergence_stage_70
        per_sample_records.append(sample_row)

        # Process Pairwise Spatial Separation
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

            for s_name in STAGE0_SUBSTAGES:
                emap = stage_maps[s_name]
                if np.sum(neck_pair) > 0:
                    e_n = float(np.mean(emap[neck_pair == 1]))
                else:
                    e_n = 0.0
                e_c = float(np.mean(emap[crack_pair == 1]))
                ratio = float(e_n / (e_c + 1e-6))
                sep_m = float(1.0 - ratio)

                # A pair is considered feature-separable if neck energy has a distinct valley (sep_m >= 0.20)
                is_separable = bool(sep_m >= 0.20 and not pair['grid_cell_collision'])

                p_rec[f'{s_name}_neck_energy'] = e_n
                p_rec[f'{s_name}_crack_energy'] = e_c
                p_rec[f'{s_name}_contrast'] = ratio
                p_rec[f'{s_name}_sep_margin'] = sep_m
                p_rec[f'{s_name}_is_separable'] = is_separable

            pairwise_separation_records.append(p_rec)

        # Plot cases
        if stem in vis_set:
            fig_path = os.path.join(figures_dir, f"{cohort.lower()}_{stem}_stem_stage0.png")
            plot_stem_stage0_case(
                stem=stem,
                cohort=cohort,
                image_rgb=img_rgb,
                target_bin=target_bin,
                pred_bin=pred_bin,
                neck_mask=neck_mask,
                crack_mask=crack_mask,
                bg_mask=bg_mask,
                stage_maps=stage_maps,
                stage_metrics=stage_metrics,
                save_path=fig_path
            )

    extractor.remove_hooks()
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

    sample_csv = os.path.join(output_dir, 'stem_stage0_per_sample.csv')
    df_samples.to_csv(sample_csv, index=False)
    print(f"Saved: {sample_csv}")

    # Build Stage 0 Summary Table
    summary_rows = []
    for cohort_name in ['Consensus_7of7', 'Cured_by_D2', 'Clean_Control']:
        sub_df = df_samples[df_samples['cohort'] == cohort_name]
        n_sub = len(sub_df)

        for s_name in STAGE0_SUBSTAGES:
            r_col = f'{s_name}_R_norm'
            sep_col = f'{s_name}_sep_margin'
            c_col = f'{s_name}_C_neck_crack'

            summary_rows.append({
                'cohort': cohort_name,
                'stage': s_name,
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
            })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, 'stem_stage0_summary.csv')
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    # Build First Emergence Distribution Table
    emergence_rows = []
    for cohort_name in ['Consensus_7of7', 'Cured_by_D2', 'Clean_Control']:
        sub_df = df_samples[df_samples['cohort'] == cohort_name]
        n_sub = len(sub_df)
        counts_50 = sub_df['emergence_stage_50'].value_counts().to_dict()
        counts_70 = sub_df['emergence_stage_70'].value_counts().to_dict()

        for s_name in STAGE0_SUBSTAGES + ['None']:
            c50 = counts_50.get(s_name, 0)
            c70 = counts_70.get(s_name, 0)
            emergence_rows.append({
                'cohort': cohort_name,
                'stage': s_name,
                'n_samples': n_sub,
                'count_ge_50': c50,
                'pct_ge_50': float(c50 / n_sub * 100),
                'count_ge_70': c70,
                'pct_ge_70': float(c70 / n_sub * 100),
            })

    df_emergence = pd.DataFrame(emergence_rows)
    emergence_csv = os.path.join(output_dir, 'first_emergence.csv')
    df_emergence.to_csv(emergence_csv, index=False)
    print(f"Saved: {emergence_csv}")

    # Build Spatial Separation Summary Table (Pair-level)
    sep_summary_rows = []
    for cohort_name in ['Consensus_7of7', 'Cured_by_D2', 'Clean_Control']:
        sub_pairs = df_pairs[df_pairs['cohort'] == cohort_name]
        n_pairs = len(sub_pairs)
        if n_pairs == 0:
            continue

        pct_grid_collision = float(sub_pairs['grid_cell_collision_112'].mean() * 100)
        median_gap = float(sub_pairs['gap_distance_px'].median())

        for s_name in STAGE0_SUBSTAGES:
            pct_sep = float(sub_pairs[f'{s_name}_is_separable'].mean() * 100)
            median_sep_m = float(sub_pairs[f'{s_name}_sep_margin'].median())
            median_contrast = float(sub_pairs[f'{s_name}_contrast'].median())

            sep_summary_rows.append({
                'cohort': cohort_name,
                'stage': s_name,
                'n_pairs_evaluated': n_pairs,
                'median_gap_px': median_gap,
                'pct_grid_cell_collision_112': pct_grid_collision,
                'pct_pairs_spatially_separable': pct_sep,
                'pct_pairs_representation_collapsed': float(100.0 - pct_sep),
                'median_separation_margin': median_sep_m,
                'median_contrast_neck_crack': median_contrast,
            })

    df_sep_summary = pd.DataFrame(sep_summary_rows)
    sep_sum_csv = os.path.join(output_dir, 'spatial_separation_summary.csv')
    df_sep_summary.to_csv(sep_sum_csv, index=False)
    print(f"Saved: {sep_sum_csv}")

    # Plot Multi-Cohort Trajectory across Stem and Stage 0 Blocks
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    stage_x = ['Stem', 'Stage 0 Blk 0', 'Stage 0 Blk 1 (Pre-SAGE)', 'Stage 0 Post-SAGE']

    colors = {'Consensus_7of7': '#d95f02', 'Cured_by_D2': '#7570b3', 'Clean_Control': '#1b9e77'}
    markers = {'Consensus_7of7': 'o', 'Cured_by_D2': 's', 'Clean_Control': '^'}

    for c_name in ['Consensus_7of7', 'Cured_by_D2', 'Clean_Control']:
        sub_s = df_summary[df_summary['cohort'] == c_name]
        axes[0].plot(stage_x, sub_s['median_R_norm'], marker=markers[c_name], linewidth=2.5, label=c_name, color=colors[c_name])
        axes[1].plot(stage_x, sub_s['median_sep_margin'], marker=markers[c_name], linewidth=2.5, label=c_name, color=colors[c_name])

    axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.6, label='50% Emergence Threshold')
    axes[0].set_title("Median Normalized Emergence Index (R_norm)\nFrom Stem to Stage 0 Output", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Median R_norm")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    axes[0].tick_params(axis='x', rotation=15)

    axes[1].axhline(0.20, color='gray', linestyle='--', alpha=0.6, label='Separation Valley Threshold (20%)')
    axes[1].axhline(0.0, color='black', linestyle=':', alpha=0.5)
    axes[1].set_title("Median Separation Margin (1 - E_neck / E_crack)\nPositive = Valley / Distinct; Zero/Negative = Collapsed", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Separation Margin")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    fig_summary_path = os.path.join(figures_dir, 'stem_stage0_trajectory_comparison.png')
    plt.savefig(fig_summary_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_summary_path}")

    print("\n" + "=" * 80)
    print("Stem -> Stage 0 Diagnostic Completed Successfully!")
    print("=" * 80)

if __name__ == '__main__':
    main()
