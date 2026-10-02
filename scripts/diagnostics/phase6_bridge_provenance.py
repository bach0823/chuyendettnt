#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_bridge_provenance.py

Phase 6 Bridge Provenance Across Scales Diagnostic:
Tracks the emergence, representation, and amplification of false bridges
across the full 8-stage multi-scale architectural hierarchy of Candidate B:
1. bottleneck  (14x14, 384 ch)
2. S2          (28x28, 192 ch)
3. T1          (56x56, 192 ch - Block 1 post-upsample)
4. T2          (56x56, 288 ch - Block 1 post-skip-concat)
5. T3          (56x56, 96 ch  - Block 1 post-Conv1)
6. T4          (56x56, 96 ch  - Block 1 post-Conv2 / S1)
7. S0          (112x112, 48 ch- Block 2 output)
8. Final_Head  (448x448, 1 ch - Logits)

Cohort:
All N=110 Base False Bridge cases:
- 101 Consensus 7/7 cases
- 7 D2-cured cases
- 2 Remaining Base bridge cases

STRICT CONSTRAINTS:
- Diagnostic-Only: Zero training, zero gradient updates, zero checkpoint mutations.
- Sealed test set untouched.
- Setting A tiling inference preserved.
"""

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
# Mask Isolation Utilities
# -----------------------------------------------------------------------------

def isolate_bridge_regions(pred_bin: np.ndarray, target_bin: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Isolates:
    - connector_neck_mask: the narrow FP corridor bridging distinct GT CCs
    - matched_crack_mask: the ground-truth crack CCs bridged by the prediction
    - nearby_bg_mask: local background in the immediate perimeter of the bridge corridor
    """
    assert pred_bin.shape == target_bin.shape
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

            # Mark connected crack components
            for g_id in overlapping_gt:
                crack_mask = crack_mask | (gt_labels == g_id).astype(np.uint8)

            # Mark connector neck between pairs of components
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

    # Fallback if neck is somehow empty but bridge exists
    if bridge_events > 0 and np.sum(neck_mask) == 0:
        for p_id in range(1, pred_cc + 1):
            cc_pred = (pred_labels == p_id)
            overlapping_gt = np.unique(gt_labels[cc_pred])
            overlapping_gt = overlapping_gt[overlapping_gt > 0]
            if len(overlapping_gt) >= 2:
                neck_mask = neck_mask | (cc_pred & (target_bin == 0)).astype(np.uint8)

    # Nearby background: Dilate corridor by 12px, exclude any pred or target pixels
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    dilated_corridor = cv2.dilate(corridor_union, kernel_bg)
    nearby_bg_mask = (dilated_corridor > 0) & (target_bin == 0) & (pred_bin == 0)

    # If nearby bg is too sparse (< 50 px), expand further
    if np.sum(nearby_bg_mask) < 50:
        kernel_bg_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        dilated_corridor_large = cv2.dilate(corridor_union, kernel_bg_large)
        nearby_bg_mask = (dilated_corridor_large > 0) & (target_bin == 0) & (pred_bin == 0)

    return neck_mask, crack_mask, nearby_bg_mask.astype(np.uint8), bridge_events


# -----------------------------------------------------------------------------
# Multi-Scale Hook & Inference Engine
# -----------------------------------------------------------------------------

STAGE_NAMES = [
    'bottleneck',
    'S2',
    'T1',
    'T2',
    'T3',
    'T4',
    'S0',
    'Final_Head'
]

class MultiScaleProvenanceExtractor:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.raw_activations: Dict[str, torch.Tensor] = {}
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def pre_hook(name):
            def hook(module, args):
                self.raw_activations[name] = args[0].detach()
            return hook

        def post_hook(name):
            def hook(module, inp, out):
                self.raw_activations[name] = out.detach()
            return hook

        # 1. Bottleneck: input entering decoder_blocks[0] (14x14, 384 ch)
        self.handles.append(self.model.decoder.decoder_blocks[0].register_forward_pre_hook(pre_hook('bottleneck')))
        # 2. S2: output of decoder_blocks[0] (28x28, 192 ch)
        self.handles.append(self.model.decoder.decoder_blocks[0].register_forward_hook(post_hook('S2')))
        # 3. T1: upsample output inside Block 1 (56x56, 192 ch)
        self.handles.append(self.model.decoder.decoder_blocks[1].upsample.register_forward_hook(post_hook('T1')))
        # 4. T2: input to conv1 of Block 1 (post-skip concat, 56x56, 288 ch)
        self.handles.append(self.model.decoder.decoder_blocks[1].conv1.register_forward_pre_hook(pre_hook('T2')))
        # 5. T3: output of conv1 of Block 1 (56x56, 96 ch)
        self.handles.append(self.model.decoder.decoder_blocks[1].conv1.register_forward_hook(post_hook('T3')))
        # 6. T4: output of Block 1 / post-conv2 (56x56, 96 ch = S1)
        self.handles.append(self.model.decoder.decoder_blocks[1].register_forward_hook(post_hook('T4')))
        # 7. S0: output of Block 2 (112x112, 48 ch)
        self.handles.append(self.model.decoder.decoder_blocks[2].register_forward_hook(post_hook('S0')))

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def run_image(self, image: np.ndarray, tile_size: int = 448) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Executes Setting A tiling inference:
        Returns:
        - pred_bin (H, W)
        - prob_map (H, W)
        - stage_energy_maps: Dict[str, np.ndarray of shape (H, W)]
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
        stage_full_maps = {s: np.zeros((pH, pW), dtype=np.float32) for s in STAGE_NAMES}

        self.model.eval()
        with torch.no_grad():
            for patch_t, (py, px) in zip(patches, coords):
                inp = patch_t.unsqueeze(0).to(self.device)
                logits = self.model(inp)
                if logits.shape[2:] != (tile_size, tile_size):
                    logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
                
                self.raw_activations['Final_Head'] = logits.detach()
                final_logits[py:py+tile_size, px:px+tile_size] = logits.squeeze().cpu().numpy()

                # Process spatial energy for each stage
                for s_name in STAGE_NAMES:
                    act = self.raw_activations[s_name]  # (1, C, h, w)
                    # L2 norm across channels for each spatial coordinate
                    if s_name == 'Final_Head':
                        # For logits, spatial energy is sigmoid probability or absolute logit
                        # We use sigmoid probability for normalized response [0, 1]
                        prob_patch = torch.sigmoid(act)
                        energy = prob_patch.squeeze(0).squeeze(0)  # (448, 448)
                    else:
                        energy = torch.norm(act, p=2, dim=1).squeeze(0)  # (h, w)

                    # Bilinear interpolate continuous energy map to (448, 448)
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

        # Crop back to original (H, W)
        logits_cropped = final_logits[:H, :W]
        prob_cropped = 1.0 / (1.0 + np.exp(-logits_cropped))
        pred_bin = (logits_cropped > 0.0).astype(np.uint8)

        stage_cropped_maps = {s: stage_full_maps[s][:H, :W] for s in STAGE_NAMES}

        return pred_bin, prob_cropped, stage_cropped_maps


# -----------------------------------------------------------------------------
# Visualization Generator
# -----------------------------------------------------------------------------

def plot_case_provenance(
    stem: str,
    group: str,
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
    Renders a comprehensive 10-panel provenance dashboard for a single sample.
    """
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle(f"Phase 6 Bridge Provenance: {stem} (Cohort: {group})", fontsize=18, fontweight='bold', y=0.98)

    # 1. Overlay (Image + Pred + GT)
    overlay = image_rgb.copy()
    # Green contours for GT, Red contours for Pred, Yellow for Neck
    gt_contours, _ = cv2.findContours(target_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(overlay, pred_contours, -1, (255, 0, 0), 1)

    axes[0, 0].imshow(overlay)
    axes[0, 0].set_title("Input RGB + GT (Green) / Pred (Red)", fontsize=11, fontweight='semibold')
    axes[0, 0].axis('off')

    # 2. ROI Masks: Neck (Red), Crack (Green), Local BG (Blue)
    roi_rgb = np.zeros_like(image_rgb)
    roi_rgb[crack_mask == 1] = [0, 200, 0]
    roi_rgb[bg_mask == 1] = [50, 50, 180]
    roi_rgb[neck_mask == 1] = [255, 40, 40]
    axes[0, 1].imshow(roi_rgb)
    axes[0, 1].set_title("ROIs: Crack (Grn) / Neck (Red) / BG (Blue)", fontsize=11, fontweight='semibold')
    axes[0, 1].axis('off')

    # 3-10: 8 Stage Energy Maps
    # Layout:
    # Row 0: Overlay, ROIs, Bottleneck (14), S2 (28), T1 (56 up)
    # Row 1: T2 (56 cat), T3 (56 conv1), T4 (56 conv2/S1), S0 (112), Final Head (448)
    stage_coords = [
        ('bottleneck', 0, 2, "1. Bottleneck (14x14)"),
        ('S2', 0, 3, "2. S2 (28x28)"),
        ('T1', 0, 4, "3. T1 Post-Upsample (56x56)"),
        ('T2', 1, 0, "4. T2 Post-Skip (56x56)"),
        ('T3', 1, 1, "5. T3 Post-Conv1 (56x56)"),
        ('T4', 1, 2, "6. T4 Post-Conv2/S1 (56x56)"),
        ('S0', 1, 3, "7. S0 (112x112)"),
        ('Final_Head', 1, 4, "8. Final Head Prob (448x448)"),
    ]

    for s_name, r, c, title in stage_coords:
        ax = axes[r, c]
        emap = stage_maps[s_name]
        m = stage_metrics[s_name]

        im = ax.imshow(emap, cmap='viridis')
        # Overlay neck boundary contour in red
        neck_cnt, _ = cv2.findContours(neck_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in neck_cnt:
            ax.plot(cnt[:, 0, 0], cnt[:, 0, 1], color='red', linewidth=1.2)

        r_norm = m['R_norm']
        c_nc = m['C_neck_crack']
        c_nb = m['C_neck_bg']
        stat_str = f"R_norm={r_norm:.2f} | N/C={c_nc:.2f} | N/B={c_nb:.2f}"
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
    print("SAGE-Lite Phase 6: Bridge Provenance Across Scales Diagnostic")
    print("=" * 80)

    config_path = 'results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml'
    ckpt_path = 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'
    output_dir = 'results/diagnostics/phase6_bridge_provenance'
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load Model
    print(f"Loading Candidate B model from: {ckpt_path}...")
    model = load_model_from_checkpoint(config_path, ckpt_path, device=device)
    extractor = MultiScaleProvenanceExtractor(model, device)

    # Load Cohort Metadata
    consensus_path = 'results/diagnostics/phase6_bridge_localization/bridge_consensus_per_image.csv'
    assert os.path.exists(consensus_path), f"Missing {consensus_path}"
    df_meta = pd.read_csv(consensus_path)

    base_bridges_df = df_meta[df_meta['Base_bridge'] == 1].copy()
    print(f"Total Base False Bridge Cases: {len(base_bridges_df)}")

    # Classify into cohorts:
    # 1. Consensus_7of7 (101 cases)
    # 2. Cured_by_D2 (7 cases)
    # 3. Other_Base_Bridge (2 cases)
    def assign_cohort(row):
        if row['d2_transition_type'] == 'Cured_by_D2':
            return 'Cured_by_D2'
        elif row['n_models_with_bridge'] == 7:
            return 'Consensus_7of7'
        else:
            return 'Other_Base_Bridge'

    base_bridges_df['cohort'] = base_bridges_df.apply(assign_cohort, axis=1)
    print(base_bridges_df['cohort'].value_counts())

    # Pre-select representative cases for detailed visualization
    # 4 top Consensus cases + 4 D2-cured cases
    c7_cases = base_bridges_df[base_bridges_df['cohort'] == 'Consensus_7of7']['image_id'].tolist()
    d2_cases = base_bridges_df[base_bridges_df['cohort'] == 'Cured_by_D2']['image_id'].tolist()
    vis_cases = set(c7_cases[:4] + d2_cases[:4])

    records = []
    t0 = time.time()

    for idx, row in tqdm(base_bridges_df.iterrows(), total=len(base_bridges_df), desc="Extracting Provenance Across Scales"):
        stem = row['image_id']
        cohort = row['cohort']
        category = row['category']

        img_path = f"datasets/Crack500_ready/val/images/{stem}.jpg"
        mask_path = f"datasets/Crack500_ready/val/masks/{stem}.png"

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        target_bin = (mask > 0).astype(np.uint8)

        # Run multi-scale Setting A inference
        pred_bin, prob_map, stage_maps = extractor.run_image(img_rgb)

        # Isolate ROIs
        neck_mask, crack_mask, bg_mask, bridge_events = isolate_bridge_regions(pred_bin, target_bin)
        n_neck_px = int(np.sum(neck_mask))
        n_crack_px = int(np.sum(crack_mask))
        n_bg_px = int(np.sum(bg_mask))

        # Evaluate metrics for each stage
        row_dict = {
            'image_id': stem,
            'cohort': cohort,
            'category': category,
            'bridge_events': bridge_events,
            'n_neck_pixels': n_neck_px,
            'n_crack_pixels': n_crack_px,
            'n_bg_pixels': n_bg_px,
        }

        stage_metrics = {}
        emergence_stage_50 = 'None'
        emergence_stage_70 = 'None'

        for s_idx, s_name in enumerate(STAGE_NAMES):
            emap = stage_maps[s_name]

            if n_neck_px > 0:
                e_neck = float(np.mean(emap[neck_mask == 1]))
            else:
                e_neck = 0.0

            if n_crack_px > 0:
                e_crack = float(np.mean(emap[crack_mask == 1]))
            else:
                e_crack = 0.0

            if n_bg_px > 0:
                e_bg = float(np.mean(emap[bg_mask == 1]))
            else:
                e_bg = 0.0

            c_neck_bg = float(e_neck / (e_bg + 1e-6))
            c_neck_crack = float(e_neck / (e_crack + 1e-6))
            c_crack_bg = float(e_crack / (e_bg + 1e-6))
            diff_neck_bg = float(e_neck - e_bg)

            # Normalized Emergence Index: (E_neck - E_bg) / (E_crack - E_bg + 1e-6)
            denom = (e_crack - e_bg)
            if abs(denom) > 1e-5:
                r_norm = float((e_neck - e_bg) / denom)
            else:
                r_norm = 0.0

            stage_metrics[s_name] = {
                'E_neck': e_neck,
                'E_crack': e_crack,
                'E_bg': e_bg,
                'C_neck_bg': c_neck_bg,
                'C_neck_crack': c_neck_crack,
                'C_crack_bg': c_crack_bg,
                'diff_neck_bg': diff_neck_bg,
                'R_norm': r_norm
            }

            row_dict[f'{s_name}_E_neck'] = e_neck
            row_dict[f'{s_name}_E_crack'] = e_crack
            row_dict[f'{s_name}_E_bg'] = e_bg
            row_dict[f'{s_name}_C_neck_bg'] = c_neck_bg
            row_dict[f'{s_name}_C_neck_crack'] = c_neck_crack
            row_dict[f'{s_name}_C_crack_bg'] = c_crack_bg
            row_dict[f'{s_name}_diff_neck_bg'] = diff_neck_bg
            row_dict[f'{s_name}_R_norm'] = r_norm

            # Check emergence threshold crossings
            if emergence_stage_50 == 'None' and r_norm >= 0.50:
                emergence_stage_50 = s_name
            if emergence_stage_70 == 'None' and r_norm >= 0.70:
                emergence_stage_70 = s_name

        row_dict['emergence_stage_50'] = emergence_stage_50
        row_dict['emergence_stage_70'] = emergence_stage_70
        records.append(row_dict)

        # Plot representative cases
        if stem in vis_cases:
            save_path = os.path.join(figures_dir, f"{cohort.lower()}_{stem}_provenance.png")
            plot_case_provenance(
                stem=stem,
                group=cohort,
                image_rgb=img_rgb,
                target_bin=target_bin,
                pred_bin=pred_bin,
                neck_mask=neck_mask,
                crack_mask=crack_mask,
                bg_mask=bg_mask,
                stage_maps=stage_maps,
                stage_metrics=stage_metrics,
                save_path=save_path
            )

    extractor.remove_hooks()
    t1 = time.time()
    print(f"\nInference and multi-scale feature extraction complete in {t1 - t0:.1f}s.")

    # Convert to DataFrame
    df_results = pd.DataFrame(records)
    per_sample_csv = os.path.join(output_dir, 'provenance_per_sample.csv')
    df_results.to_csv(per_sample_csv, index=False)
    print(f"Saved: {per_sample_csv}")

    # Build Scale Summary DataFrame
    summary_rows = []
    for cohort_name in ['ALL_BASE_BRIDGES', 'Consensus_7of7', 'Cured_by_D2']:
        if cohort_name == 'ALL_BASE_BRIDGES':
            sub_df = df_results
        else:
            sub_df = df_results[df_results['cohort'] == cohort_name]

        n_sub = len(sub_df)
        if n_sub == 0:
            continue

        for s_name in STAGE_NAMES:
            summary_rows.append({
                'cohort': cohort_name,
                'stage': s_name,
                'n_samples': n_sub,
                'mean_E_neck': float(sub_df[f'{s_name}_E_neck'].mean()),
                'median_E_neck': float(sub_df[f'{s_name}_E_neck'].median()),
                'mean_E_crack': float(sub_df[f'{s_name}_E_crack'].mean()),
                'median_E_crack': float(sub_df[f'{s_name}_E_crack'].median()),
                'mean_E_bg': float(sub_df[f'{s_name}_E_bg'].mean()),
                'median_E_bg': float(sub_df[f'{s_name}_E_bg'].median()),
                'mean_C_neck_bg': float(sub_df[f'{s_name}_C_neck_bg'].mean()),
                'median_C_neck_bg': float(sub_df[f'{s_name}_C_neck_bg'].median()),
                'mean_C_neck_crack': float(sub_df[f'{s_name}_C_neck_crack'].mean()),
                'median_C_neck_crack': float(sub_df[f'{s_name}_C_neck_crack'].median()),
                'mean_R_norm': float(sub_df[f'{s_name}_R_norm'].mean()),
                'median_R_norm': float(sub_df[f'{s_name}_R_norm'].median()),
                'pct_samples_R_norm_ge_50': float((sub_df[f'{s_name}_R_norm'] >= 0.50).mean() * 100),
                'pct_samples_R_norm_ge_70': float((sub_df[f'{s_name}_R_norm'] >= 0.70).mean() * 100),
            })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, 'provenance_scale_summary.csv')
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    # Build Emergence Distribution DataFrame
    dist_rows = []
    all_stages_with_none = STAGE_NAMES + ['None']
    for cohort_name in ['Consensus_7of7', 'Cured_by_D2', 'ALL_BASE_BRIDGES']:
        if cohort_name == 'ALL_BASE_BRIDGES':
            sub_df = df_results
        else:
            sub_df = df_results[df_results['cohort'] == cohort_name]
        n_sub = len(sub_df)

        counts_50 = sub_df['emergence_stage_50'].value_counts().to_dict()
        counts_70 = sub_df['emergence_stage_70'].value_counts().to_dict()

        for stg in all_stages_with_none:
            c50 = counts_50.get(stg, 0)
            c70 = counts_70.get(stg, 0)
            dist_rows.append({
                'cohort': cohort_name,
                'stage': stg,
                'n_samples': n_sub,
                'count_emergence_50': c50,
                'pct_emergence_50': float(c50 / n_sub * 100),
                'count_emergence_70': c70,
                'pct_emergence_70': float(c70 / n_sub * 100),
            })

    df_dist = pd.DataFrame(dist_rows)
    dist_csv = os.path.join(output_dir, 'provenance_emergence_distribution.csv')
    df_dist.to_csv(dist_csv, index=False)
    print(f"Saved: {dist_csv}")

    # Plot Multi-Scale Trajectory Comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    c7_sum = df_summary[df_summary['cohort'] == 'Consensus_7of7']
    d2_sum = df_summary[df_summary['cohort'] == 'Cured_by_D2']

    # Subplot 1: Normalized Emergence Ratio R_norm across stages
    axes[0].plot(c7_sum['stage'], c7_sum['median_R_norm'], marker='o', linewidth=2.5, color='#d95f02', label='Consensus 7/7 (N=101)')
    axes[0].plot(d2_sum['stage'], d2_sum['median_R_norm'], marker='s', linewidth=2.5, color='#7570b3', label='Cured by D2 (N=7)')
    axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='50% Emergence Threshold')
    axes[0].axhline(0.7, color='black', linestyle=':', alpha=0.7, label='70% Emergence Threshold')
    axes[0].set_title("Normalized Emergence Index (R_norm) Across Scales\n(Median over Cohort)", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("R_norm = (Neck - BG) / (Crack - BG)")
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    axes[0].tick_params(axis='x', rotation=30)

    # Subplot 2: Neck / Crack Contrast Ratio
    axes[1].plot(c7_sum['stage'], c7_sum['median_C_neck_crack'], marker='o', linewidth=2.5, color='#d95f02', label='Consensus 7/7 (N=101)')
    axes[1].plot(d2_sum['stage'], d2_sum['median_C_neck_crack'], marker='s', linewidth=2.5, color='#7570b3', label='Cured by D2 (N=7)')
    axes[1].axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Equal to Crack (1.0x)')
    axes[1].set_title("Bridge Neck / Crack Contrast Ratio (C_neck_crack)\n(Median over Cohort)", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("Contrast = E_neck / E_crack")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    curve_fig_path = os.path.join(figures_dir, 'provenance_trajectories_consensus_vs_d2.png')
    plt.savefig(curve_fig_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved Trajectory Curve Figure: {curve_fig_path}")

    print("\n" + "=" * 80)
    print("Bridge Provenance Diagnostic Completed Successfully!")
    print("=" * 80)

if __name__ == '__main__':
    main()
