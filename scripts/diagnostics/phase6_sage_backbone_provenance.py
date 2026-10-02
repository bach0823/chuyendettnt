#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_sage_backbone_provenance.py

Phase 6: SAGE / Backbone Bridge Provenance Across Scales Diagnostic
Scientific Goal:
Determine where the bridge signal in Consensus 7/7 forms in the Encoder / SAGE / ViT hierarchy:
- Before SAGE injection?
- After SAGE injection?
- Inside ViT / bottleneck?

Stages Probed:
1. ConvNeXt Stage 0: pre-SAGE (112x112, 48 ch) vs post-SAGE (112x112, 48 ch)
2. ConvNeXt Stage 1: pre-SAGE (56x56, 96 ch)   vs post-SAGE (56x56, 96 ch)
3. ConvNeXt Stage 2: pre-SAGE (28x28, 192 ch)  vs post-SAGE (28x28, 192 ch)
4. ConvNeXt Stage 3: pre-SAGE (14x14, 384 ch)  vs post-SAGE (14x14, 384 ch)
5. Pre-ViT: post-linear + pos_embed + LayerNorm (14x14, 192 ch)
6. ViT Block 0: output of ViT Block 0 (14x14, 192 ch)
7. ViT Block 1: output of ViT Block 1 (14x14, 192 ch)
8. ViT Block 2: output of ViT Block 2 (14x14, 192 ch)
9. ViT Block 3: output of ViT Block 3 (14x14, 192 ch)
10. Bottleneck: post-reprojection to 384 ch (14x14, 384 ch)
11. S2: output of Decoder Block 0 (28x28, 192 ch)

Cohorts:
A. Consensus 7/7 (N=101)
B. D2-Cured (N=7)
C. Clean / No-Bridge Control (N=56 samples with gt_cc >= 2 and 0 bridges across all 7 models)

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

def compute_model_hash(model: nn.Module) -> str:
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
# ROI Isolation Utilities
# -----------------------------------------------------------------------------

def isolate_bridge_regions_bridged(pred_bin: np.ndarray, target_bin: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Isolates Neck, Crack, and local BG for bridged cases."""
    H, W = target_bin.shape
    num_gt_cc, gt_labels = cv2.connectedComponents(target_bin.astype(np.uint8), connectivity=8)
    num_pred_cc, pred_labels = cv2.connectedComponents(pred_bin.astype(np.uint8), connectivity=8)
    pred_cc = int(max(num_pred_cc - 1, 0))
    gt_cc = int(max(num_gt_cc - 1, 0))

    neck_mask = np.zeros((H, W), dtype=np.uint8)
    crack_mask = np.zeros((H, W), dtype=np.uint8)
    corridor_union = np.zeros((H, W), dtype=np.uint8)
    bridge_events = 0

    if gt_cc < 2 or pred_cc < 1:
        return neck_mask, crack_mask, neck_mask, 0

    for p_id in range(1, pred_cc + 1):
        cc_pred = (pred_labels == p_id)
        overlapping_gt = np.unique(gt_labels[cc_pred])
        overlapping_gt = overlapping_gt[overlapping_gt > 0]

        if len(overlapping_gt) >= 2:
            bridge_events += 1
            cc_fp = cc_pred & (target_bin == 0)

            for g_id in overlapping_gt:
                crack_mask = crack_mask | (gt_labels == g_id).astype(np.uint8)

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
                    corridor_union = corridor_union | corridor.astype(np.uint8)
                    neck = cc_fp & corridor
                    neck_mask = neck_mask | neck.astype(np.uint8)

    if bridge_events > 0 and np.sum(neck_mask) == 0:
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

    return neck_mask, crack_mask, nearby_bg_mask.astype(np.uint8), bridge_events


def isolate_bridge_regions_clean(target_bin: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Isolates the inter-component gap (moat) for Clean / No-Bridge control cases."""
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
        return np.zeros((H, W), dtype=np.uint8), target_bin.astype(np.uint8), np.ones((H, W), dtype=np.uint8)

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

    return gap_neck.astype(np.uint8), crack_mask, nearby_bg.astype(np.uint8)


# -----------------------------------------------------------------------------
# Multi-Scale SAGE / Backbone Hook Engine
# -----------------------------------------------------------------------------

STAGE_ORDER = [
    'stage0_pre_sage',
    'stage0_post_sage',
    'stage1_pre_sage',
    'stage1_post_sage',
    'stage2_pre_sage',
    'stage2_post_sage',
    'stage3_pre_sage',
    'stage3_post_sage',
    'pre_vit',
    'vit_block_0',
    'vit_block_1',
    'vit_block_2',
    'vit_block_3',
    'bottleneck',
    'S2'
]

STAGE_SHAPES = {
    0: (48, 112, 112),
    1: (96, 56, 56),
    2: (192, 28, 28),
    3: (384, 14, 14),
}

class SageBackboneProvenanceExtractor:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.raw_activations: Dict[str, torch.Tensor] = {}
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        # 1. ConvNeXt stages pre-SAGE and post-SAGE
        for i in range(4):
            stg = self.model.backbone.convnext.stages[i]
            exp_c, exp_h, exp_w = STAGE_SHAPES[i]

            def make_pre_hook(idx, c, h, w):
                def hook(m, inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    if t.dim() == 4 and t.shape[1] == c and t.shape[2] == h and t.shape[3] == w:
                        self.raw_activations[f'stage{idx}_pre_sage'] = t.detach()
                return hook
            self.handles.append(stg.main_block.register_forward_hook(make_pre_hook(i, exp_c, exp_h, exp_w)))

            def make_post_hook(idx, c, h, w):
                def hook(m, inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    if t.dim() == 4 and t.shape[1] == c and t.shape[2] == h and t.shape[3] == w:
                        self.raw_activations[f'stage{idx}_post_sage'] = t.detach()
                return hook
            self.handles.append(stg.register_forward_hook(make_post_hook(i, exp_c, exp_h, exp_w)))

        # 2. Pre-ViT hook
        self.handles.append(
            self.model.backbone.pre_transformer_norm.register_forward_hook(
                lambda m, inp, out: self.raw_activations.update({'pre_vit': out.detach()})
            )
        )

        # 3. ViT blocks hooks
        for i in range(4):
            blk = self.model.backbone.transformer_blocks[i]
            def make_vit_hook(idx):
                def hook(m, inp, out):
                    self.raw_activations[f'vit_block_{idx}'] = out.detach()
                return hook
            self.handles.append(blk.register_forward_hook(make_vit_hook(i)))

        # 4. Decoder hooks
        self.handles.append(
            self.model.decoder.decoder_blocks[0].register_forward_pre_hook(
                lambda m, args: self.raw_activations.update({'bottleneck': args[0].detach()})
            )
        )
        self.handles.append(
            self.model.decoder.decoder_blocks[0].register_forward_hook(
                lambda m, inp, out: self.raw_activations.update({'S2': out.detach()})
            )
        )

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def run_image(self, image: np.ndarray, tile_size: int = 448) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Setting A tiling inference with feature extraction."""
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
        stage_full_maps = {s: np.zeros((pH, pW), dtype=np.float32) for s in STAGE_ORDER}

        self.model.eval()
        with torch.no_grad():
            for patch_t, (py, px) in zip(patches, coords):
                inp = patch_t.unsqueeze(0).to(self.device)
                logits = self.model(inp)
                if logits.shape[2:] != (tile_size, tile_size):
                    logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)

                final_logits[py:py+tile_size, px:px+tile_size] = logits.squeeze().cpu().numpy()

                # Process spatial energy for each stage
                for s_name in STAGE_ORDER:
                    act = self.raw_activations[s_name]

                    # Token sequence reshape for ViT blocks
                    if act.dim() == 3 and act.shape[1] == 196:
                        # (1, 196, 192) -> (1, 192, 14, 14)
                        act = act.transpose(1, 2).reshape(1, act.shape[2], 14, 14)

                    # Compute L2 norm across channels
                    energy = torch.norm(act, p=2, dim=1).squeeze(0)  # (h, w)

                    # Bilinear interpolate to (448, 448)
                    if energy.shape != (tile_size, tile_size):
                        energy_448 = F.interpolate(
                            energy.unsqueeze(0).unsqueeze(0),
                            size=(tile_size, tile_size),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0).squeeze(0)
                    else:
                        energy_448 = energy

                    stage_full_maps[s_name][py:py+tile_size, px:px+tile_size] = energy_448.cpu().numpy()

        logits_cropped = final_logits[:H, :W]
        prob_cropped = 1.0 / (1.0 + np.exp(-logits_cropped))
        pred_bin = (logits_cropped > 0.0).astype(np.uint8)
        stage_cropped_maps = {s: stage_full_maps[s][:H, :W] for s in STAGE_ORDER}

        return pred_bin, prob_cropped, stage_cropped_maps


# -----------------------------------------------------------------------------
# Visualization Generator
# -----------------------------------------------------------------------------

def plot_backbone_provenance(
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
    """
    Renders 16 panels:
    Row 0: Overlay, ROI Masks, Stage 0 pre/post, Stage 1 pre/post
    Row 1: Stage 2 pre/post, Stage 3 pre/post, pre-ViT, ViT blk 0, ViT blk 3, Bottleneck
    """
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    fig.suptitle(f"SAGE / Backbone Bridge Provenance: {stem} (Cohort: {cohort})", fontsize=18, fontweight='bold', y=0.98)

    # 1. Overlay
    overlay = image_rgb.copy()
    gt_contours, _ = cv2.findContours(target_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(overlay, pred_contours, -1, (255, 0, 0), 1)
    axes[0, 0].imshow(overlay)
    axes[0, 0].set_title("Input RGB + GT (Grn) / Pred (Red)", fontsize=10, fontweight='semibold')
    axes[0, 0].axis('off')

    # 2. ROIs
    roi_rgb = np.zeros_like(image_rgb)
    roi_rgb[crack_mask == 1] = [0, 200, 0]
    roi_rgb[bg_mask == 1] = [50, 50, 180]
    roi_rgb[neck_mask == 1] = [255, 40, 40]
    axes[0, 1].imshow(roi_rgb)
    axes[0, 1].set_title("ROIs: Crack (Grn) / Gap-Neck (Red) / BG (Blue)", fontsize=10, fontweight='semibold')
    axes[0, 1].axis('off')

    # 13 Selected Stages to Plot
    stages_to_plot = [
        ('stage0_pre_sage', 0, 2, "Stage 0 Pre-SAGE (112)"),
        ('stage0_post_sage', 0, 3, "Stage 0 Post-SAGE (112)"),
        ('stage1_pre_sage', 0, 4, "Stage 1 Pre-SAGE (56)"),
        ('stage1_post_sage', 1, 0, "Stage 1 Post-SAGE (56)"),
        ('stage2_pre_sage', 1, 1, "Stage 2 Pre-SAGE (28)"),
        ('stage2_post_sage', 1, 2, "Stage 2 Post-SAGE (28)"),
        ('stage3_pre_sage', 1, 3, "Stage 3 Pre-SAGE (14)"),
        ('stage3_post_sage', 1, 4, "Stage 3 Post-SAGE (14)"),
        ('pre_vit', 2, 0, "Pre-ViT (14x14)"),
        ('vit_block_0', 2, 1, "ViT Block 0 (14x14)"),
        ('vit_block_3', 2, 2, "ViT Block 3 Final (14x14)"),
        ('bottleneck', 2, 3, "Bottleneck Reproj (14x14)"),
        ('S2', 2, 4, "S2 Decoder Out (28x28)"),
    ]

    for s_name, r, c, title in stages_to_plot:
        ax = axes[r, c]
        emap = stage_maps[s_name]
        m = stage_metrics[s_name]
        im = ax.imshow(emap, cmap='viridis')

        # Draw neck contour
        neck_cnt, _ = cv2.findContours(neck_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in neck_cnt:
            ax.plot(cnt[:, 0, 0], cnt[:, 0, 1], color='red', linewidth=1.2)

        r_norm = m['R_norm']
        c_nc = m['C_neck_crack']
        stat_str = f"R_norm={r_norm:.2f} | N/C={c_nc:.2f}"
        ax.set_title(f"{title}\n{stat_str}", fontsize=9)
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
    print("SAGE-Lite Phase 6: SAGE / Backbone Bridge Provenance Across Scales Diagnostic")
    print("=" * 80)

    config_path = 'results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml'
    ckpt_path = 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'
    output_dir = 'results/diagnostics/phase6_sage_backbone_provenance'
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load Model and compute initial bitwise state hash
    print(f"Loading Candidate B model from: {ckpt_path}...")
    init_file_hash = compute_file_hash(ckpt_path)
    model = load_model_from_checkpoint(config_path, ckpt_path, device=device)
    init_hash = compute_model_hash(model)
    print(f"Initial Model State SHA256: {init_hash[:16]}...")
    print(f"Initial Disk Checkpoint SHA256: {init_file_hash[:16]}...")

    extractor = SageBackboneProvenanceExtractor(model, device)

    # Load Metadata & Establish Cohorts
    consensus_path = 'results/diagnostics/phase6_bridge_localization/bridge_consensus_per_image.csv'
    df_meta = pd.read_csv(consensus_path)

    # Master paired for clean controls
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

    # Combine into unified execution list
    cohort_list = []
    for s in c7_stems:
        cohort_list.append((s, 'Consensus_7of7'))
    for s in d2_cured_stems:
        cohort_list.append((s, 'Cured_by_D2'))
    for s in sorted(list(clean_stems)):
        cohort_list.append((s, 'Clean_Control'))

    print(f"Total Cohort Size to Evaluate: N = {len(cohort_list)}")

    # Pre-select representative cases for visualization
    vis_c7 = sorted(list(c7_stems))[:6]
    vis_d2 = sorted(list(d2_cured_stems))[:5]
    vis_clean = sorted(list(clean_stems))[:3]
    vis_set = set(vis_c7 + vis_d2 + vis_clean)

    sage_records = []
    vit_records = []
    t0 = time.time()

    for stem, cohort in tqdm(cohort_list, desc="Extracting Backbone & SAGE Provenance"):
        img_path = f"datasets/Crack500_ready/val/images/{stem}.jpg"
        mask_path = f"datasets/Crack500_ready/val/masks/{stem}.png"

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        target_bin = (mask > 0).astype(np.uint8)

        # Run multi-scale Setting A inference
        pred_bin, prob_map, stage_maps = extractor.run_image(img_rgb)

        # Isolate ROIs
        if cohort in ('Consensus_7of7', 'Cured_by_D2'):
            neck_mask, crack_mask, bg_mask, b_events = isolate_bridge_regions_bridged(pred_bin, target_bin)
        else:
            neck_mask, crack_mask, bg_mask = isolate_bridge_regions_clean(target_bin)
            b_events = 0

        n_neck_px = int(np.sum(neck_mask))
        n_crack_px = int(np.sum(crack_mask))
        n_bg_px = int(np.sum(bg_mask))

        # Compute metrics across all 15 stages
        stage_metrics = {}
        emergence_stage_50 = 'None'
        emergence_stage_70 = 'None'

        for s_idx, s_name in enumerate(STAGE_ORDER):
            emap = stage_maps[s_name]

            e_neck = float(np.mean(emap[neck_mask == 1])) if n_neck_px > 0 else 0.0
            e_crack = float(np.mean(emap[crack_mask == 1])) if n_crack_px > 0 else 0.0
            e_bg = float(np.mean(emap[bg_mask == 1])) if n_bg_px > 0 else 0.0

            c_neck_bg = float(e_neck / (e_bg + 1e-6))
            c_neck_crack = float(e_neck / (e_crack + 1e-6))
            diff_neck_bg = float(e_neck - e_bg)

            denom = (e_crack - e_bg)
            r_norm = float((e_neck - e_bg) / denom) if abs(denom) > 1e-5 else 0.0

            stage_metrics[s_name] = {
                'E_neck': e_neck,
                'E_crack': e_crack,
                'E_bg': e_bg,
                'C_neck_bg': c_neck_bg,
                'C_neck_crack': c_neck_crack,
                'diff_neck_bg': diff_neck_bg,
                'R_norm': r_norm
            }

            if emergence_stage_50 == 'None' and r_norm >= 0.50:
                emergence_stage_50 = s_name
            if emergence_stage_70 == 'None' and r_norm >= 0.70:
                emergence_stage_70 = s_name

        # 1. Build SAGE record
        sage_row = {
            'image_id': stem,
            'cohort': cohort,
            'bridge_events': b_events,
            'n_neck_pixels': n_neck_px,
            'n_crack_pixels': n_crack_px,
            'n_bg_pixels': n_bg_px,
            'emergence_stage_50': emergence_stage_50,
            'emergence_stage_70': emergence_stage_70,
        }
        for i in range(4):
            pre_k = f'stage{i}_pre_sage'
            post_k = f'stage{i}_post_sage'
            r_pre = stage_metrics[pre_k]['R_norm']
            r_post = stage_metrics[post_k]['R_norm']
            delta_r = r_post - r_pre

            sage_row[f'{pre_k}_R_norm'] = r_pre
            sage_row[f'{post_k}_R_norm'] = r_post
            sage_row[f'stage{i}_Delta_R_SAGE'] = delta_r
            sage_row[f'{pre_k}_C_neck_crack'] = stage_metrics[pre_k]['C_neck_crack']
            sage_row[f'{post_k}_C_neck_crack'] = stage_metrics[post_k]['C_neck_crack']

        sage_records.append(sage_row)

        # 2. Build ViT record
        vit_row = {
            'image_id': stem,
            'cohort': cohort,
            'pre_vit_R_norm': stage_metrics['pre_vit']['R_norm'],
            'vit_block_0_R_norm': stage_metrics['vit_block_0']['R_norm'],
            'vit_block_1_R_norm': stage_metrics['vit_block_1']['R_norm'],
            'vit_block_2_R_norm': stage_metrics['vit_block_2']['R_norm'],
            'vit_block_3_R_norm': stage_metrics['vit_block_3']['R_norm'],
            'bottleneck_R_norm': stage_metrics['bottleneck']['R_norm'],
            'S2_R_norm': stage_metrics['S2']['R_norm'],
            'delta_R_vit_total': stage_metrics['vit_block_3']['R_norm'] - stage_metrics['pre_vit']['R_norm'],
            'delta_R_vit_blk0': stage_metrics['vit_block_0']['R_norm'] - stage_metrics['pre_vit']['R_norm'],
            'delta_R_vit_blk1': stage_metrics['vit_block_1']['R_norm'] - stage_metrics['vit_block_0']['R_norm'],
            'delta_R_vit_blk2': stage_metrics['vit_block_2']['R_norm'] - stage_metrics['vit_block_1']['R_norm'],
            'delta_R_vit_blk3': stage_metrics['vit_block_3']['R_norm'] - stage_metrics['vit_block_2']['R_norm'],
            'delta_R_reproj': stage_metrics['bottleneck']['R_norm'] - stage_metrics['vit_block_3']['R_norm'],
        }
        vit_records.append(vit_row)

        # Render visualizations
        if stem in vis_set:
            fig_path = os.path.join(figures_dir, f"{cohort.lower()}_{stem}_backbone_provenance.png")
            plot_backbone_provenance(
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
    final_hash = compute_model_hash(model)
    final_file_hash = compute_file_hash(ckpt_path)
    assert init_hash == final_hash, f"FATAL: Model weights modified during diagnostic! ({init_hash} vs {final_hash})"
    assert init_file_hash == final_file_hash, f"FATAL: Checkpoint file on disk modified! ({init_file_hash} vs {final_file_hash})"
    print(f"Model Parameters State Verified: Bitwise Identical ({final_hash[:16]}...)")
    print(f"Disk Checkpoint File Verified: Bitwise Identical ({final_file_hash[:16]}...)")

    # DataFrames
    df_sage = pd.DataFrame(sage_records)
    df_vit = pd.DataFrame(vit_records)

    sage_per_sample_csv = os.path.join(output_dir, 'sage_provenance_per_sample.csv')
    vit_per_sample_csv = os.path.join(output_dir, 'vit_provenance_per_sample.csv')
    df_sage.to_csv(sage_per_sample_csv, index=False)
    df_vit.to_csv(vit_per_sample_csv, index=False)
    print(f"Saved: {sage_per_sample_csv}")
    print(f"Saved: {vit_per_sample_csv}")

    # Build SAGE Summary Table
    sage_summary_rows = []
    for cohort_name in ['Consensus_7of7', 'Cured_by_D2', 'Clean_Control']:
        sub_df = df_sage[df_sage['cohort'] == cohort_name]
        n_sub = len(sub_df)

        for i in range(4):
            pre_col = f'stage{i}_pre_sage_R_norm'
            post_col = f'stage{i}_post_sage_R_norm'
            delta_col = f'stage{i}_Delta_R_SAGE'

            sage_summary_rows.append({
                'cohort': cohort_name,
                'stage': f'Stage_{i}',
                'n_samples': n_sub,
                'median_R_pre_sage': float(sub_df[pre_col].median()),
                'mean_R_pre_sage': float(sub_df[pre_col].mean()),
                'median_R_post_sage': float(sub_df[post_col].median()),
                'mean_R_post_sage': float(sub_df[post_col].mean()),
                'median_Delta_R': float(sub_df[delta_col].median()),
                'mean_Delta_R': float(sub_df[delta_col].mean()),
                'pct_samples_Delta_gt_0': float((sub_df[delta_col] > 0).mean() * 100),
                'pct_pre_ge_50': float((sub_df[pre_col] >= 0.50).mean() * 100),
                'pct_post_ge_50': float((sub_df[post_col] >= 0.50).mean() * 100),
                'pct_pre_ge_70': float((sub_df[pre_col] >= 0.70).mean() * 100),
                'pct_post_ge_70': float((sub_df[post_col] >= 0.70).mean() * 100),
            })

    df_sage_summary = pd.DataFrame(sage_summary_rows)
    sage_sum_csv = os.path.join(output_dir, 'sage_provenance_summary.csv')
    df_sage_summary.to_csv(sage_sum_csv, index=False)
    print(f"Saved: {sage_sum_csv}")

    # Build First Emergence Distribution Table
    emergence_rows = []
    all_stages = STAGE_ORDER + ['None']

    # Categorize stage types
    def categorize_stage(stg: str) -> str:
        if 'pre_sage' in stg:
            return 'CNN_Before_SAGE'
        elif 'post_sage' in stg:
            return 'CNN_After_SAGE'
        elif 'vit' in stg:
            return 'Inside_ViT'
        elif stg in ('bottleneck', 'S2'):
            return 'Decoder_Anchor'
        return 'None'

    for cohort_name in ['Consensus_7of7', 'Cured_by_D2', 'Clean_Control']:
        sub_df = df_sage[df_sage['cohort'] == cohort_name]
        n_sub = len(sub_df)
        counts_50 = sub_df['emergence_stage_50'].value_counts().to_dict()
        counts_70 = sub_df['emergence_stage_70'].value_counts().to_dict()

        for stg in all_stages:
            c50 = counts_50.get(stg, 0)
            c70 = counts_70.get(stg, 0)
            emergence_rows.append({
                'cohort': cohort_name,
                'stage': stg,
                'category': categorize_stage(stg),
                'n_samples': n_sub,
                'count_ge_50': c50,
                'pct_ge_50': float(c50 / n_sub * 100),
                'count_ge_70': c70,
                'pct_ge_70': float(c70 / n_sub * 100),
            })

    df_emergence = pd.DataFrame(emergence_rows)
    emergence_csv = os.path.join(output_dir, 'sage_first_emergence.csv')
    df_emergence.to_csv(emergence_csv, index=False)
    print(f"Saved: {emergence_csv}")

    # Build ViT Summary Table
    vit_summary_rows = []
    for cohort_name in ['Consensus_7of7', 'Cured_by_D2', 'Clean_Control']:
        sub_df = df_vit[df_vit['cohort'] == cohort_name]
        n_sub = len(sub_df)
        vit_summary_rows.append({
            'cohort': cohort_name,
            'n_samples': n_sub,
            'median_pre_vit': float(sub_df['pre_vit_R_norm'].median()),
            'median_vit_blk0': float(sub_df['vit_block_0_R_norm'].median()),
            'median_vit_blk1': float(sub_df['vit_block_1_R_norm'].median()),
            'median_vit_blk2': float(sub_df['vit_block_2_R_norm'].median()),
            'median_vit_blk3': float(sub_df['vit_block_3_R_norm'].median()),
            'median_bottleneck': float(sub_df['bottleneck_R_norm'].median()),
            'median_S2': float(sub_df['S2_R_norm'].median()),
            'mean_delta_vit_total': float(sub_df['delta_R_vit_total'].mean()),
            'median_delta_vit_total': float(sub_df['delta_R_vit_total'].median()),
            'pct_vit_amplification_gt_0': float((sub_df['delta_R_vit_total'] > 0).mean() * 100),
        })
    df_vit_sum = pd.DataFrame(vit_summary_rows)
    vit_sum_csv = os.path.join(output_dir, 'vit_provenance_summary.csv')
    df_vit_sum.to_csv(vit_sum_csv, index=False)
    print(f"Saved: {vit_sum_csv}")

    # Plot Multi-Cohort SAGE Delta Comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    stages_x = ['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3']
    for c_name, col, marker in [('Consensus_7of7', '#d95f02', 'o'), ('Cured_by_D2', '#7570b3', 's'), ('Clean_Control', '#1b9e77', '^')]:
        sub_s = df_sage_summary[df_sage_summary['cohort'] == c_name]
        axes[0].plot(stages_x, sub_s['median_Delta_R'], marker=marker, linewidth=2.5, label=f"{c_name} (N={len(df_sage[df_sage['cohort'] == c_name])})", color=col)
        axes[1].plot(stages_x, sub_s['pct_samples_Delta_gt_0'], marker=marker, linewidth=2.5, label=f"{c_name}", color=col)

    axes[0].axhline(0.0, color='black', linestyle='--', alpha=0.5)
    axes[0].set_title("Median Delta_R Across SAGE Injections\n(Delta_R = Post-SAGE - Pre-SAGE)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Median Delta_R_norm")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    axes[1].axhline(50.0, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_title("% Samples With Positive SAGE Amplification\n(Delta_R > 0)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("% Samples")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    sage_plot_path = os.path.join(figures_dir, 'sage_delta_comparison.png')
    plt.savefig(sage_plot_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved Plot: {sage_plot_path}")

    # Plot ViT Trajectory Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    vit_x = ['Pre-ViT', 'ViT Blk 0', 'ViT Blk 1', 'ViT Blk 2', 'ViT Blk 3', 'Bottleneck', 'S2']
    for c_name, col, marker in [('Consensus_7of7', '#d95f02', 'o'), ('Cured_by_D2', '#7570b3', 's'), ('Clean_Control', '#1b9e77', '^')]:
        sub_v = df_vit_sum[df_vit_sum['cohort'] == c_name].iloc[0]
        y_vals = [
            sub_v['median_pre_vit'],
            sub_v['median_vit_blk0'],
            sub_v['median_vit_blk1'],
            sub_v['median_vit_blk2'],
            sub_v['median_vit_blk3'],
            sub_v['median_bottleneck'],
            sub_v['median_S2']
        ]
        ax.plot(vit_x, y_vals, marker=marker, linewidth=2.5, label=f"{c_name}", color=col)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.6, label='50% Emergence Threshold')
    ax.set_title("Bridge Signal Trajectory Through ViT & Bottleneck Reprojection", fontsize=13, fontweight='bold')
    ax.set_ylabel("Median Normalized Emergence Index (R_norm)")
    ax.set_ylim(-0.1, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    vit_plot_path = os.path.join(figures_dir, 'vit_trajectory_comparison.png')
    plt.savefig(vit_plot_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved Plot: {vit_plot_path}")

    print("\n" + "=" * 80)
    print("SAGE / Backbone Bridge Provenance Diagnostic Completed Successfully!")
    print("=" * 80)

if __name__ == '__main__':
    main()
