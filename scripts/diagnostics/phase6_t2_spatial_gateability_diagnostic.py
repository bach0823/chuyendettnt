#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_t2_spatial_gateability_diagnostic.py

Phase 6: T2 Spatial Discriminability / Gateability Diagnostic
Target Locus: Decoder Block 1 entry (T2: B = 192 ch, S = 96 ch at 56x56 resolution)

Scientific Objective:
Determine whether there exists a spatial statistic or feature relation between Skip (S)
and Bottleneck (B) at T2 that discriminates:
1. True Crack region (CRACK)
2. False-Bridge Neck corridor (NECK)
3. Ordinary Background (BACKGROUND)
without requiring ground-truth at inference time.

Candidate Spatial Statistics Evaluated (strictly derived from B and S):
- Energy:
  * E_B(x, y) = ||B(:, x, y)||_2
  * E_S(x, y) = ||S(:, x, y)||_2
  * log_ratio = log((E_S + eps) / (E_B + eps))
- Normalized Energy / Contrast:
  * z_B = z-score(E_B), z_S = z-score(E_S)
  * norm_E_B = min_max(E_B), norm_E_S = min_max(E_S)
  * norm_ratio = (norm_E_S + 0.05) / (norm_E_B + 0.05)
- Stream Agreement:
  * cosine_BS = cosine similarity between pooled B (96 ch) and S (96 ch)
- Local Spatial Gradient / Variance:
  * grad_E_B = ||nabla E_B||_2, grad_E_S = ||nabla E_S||_2 (Sobel)
  * var_E_B = local 3x3 variance of E_B, var_E_S = local 3x3 variance of E_S
- Cross-Stream Disagreement & Candidate Gates:
  * disagree_diff = z_S - z_B (Relative Skip elevation over Bottleneck)
  * disagree_abs = |z_S - z_B|
  * candidate_gate_B = norm_E_S * (1.0 - norm_E_B) (High Skip + Low Bottleneck)
  * candidate_gate_C = (norm_E_S + 0.05) / (norm_E_B + 0.05)

Cohorts Evaluated:
- Consensus Resistant (N = 82)
- Consensus Sensitive (N = 19)
- Cured by D2 (N = 7)
- Clean Control (N = 56)
Total N = 164. Sealed test set (N = 1124) strictly unaccessed.

HARD LOCKS:
- DIAGNOSTIC-ONLY: Zero training passes, zero backward passes, zero optimizer steps.
- Bitwise verification of checkpoint file and parameter SHA256 before and after.
- Setting A canonical evaluation path preserved (448x448, stride 448 non-overlapping, reflect padding, tau=0.5).
- In-memory pre-hook feature capture; zero modifications to model weights.
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
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, 'SAGE_LITE')
from tools.run_phase6_c_topology_diagnostic import (
    load_model_from_checkpoint,
    compute_topology_metrics
)
from scripts.diagnostics.phase6_stem_factorization_provenance import (
    isolate_bridged_pairs_and_rois,
    isolate_clean_pairs_and_rois
)


# -----------------------------------------------------------------------------
# Integrity Verification Utilities
# -----------------------------------------------------------------------------

def compute_file_hash(path: str) -> str:
    """Computes SHA256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_model_param_hash(model: nn.Module) -> str:
    """Computes SHA256 hash across all named parameters of a model."""
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode('utf-8'))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


# -----------------------------------------------------------------------------
# T2 Feature Capture Engine
# -----------------------------------------------------------------------------

class T2FeatureCaptureEngine:
    """
    Captures un-modified T2 tensor (B = 192 ch, S = 96 ch) at Decoder Block 1 Conv1 entry
    during standard Setting A baseline forward pass.
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

        self.dec_b1 = model.decoder.decoder_blocks[1]
        self.conv1_seq = self.dec_b1.conv1

        self.current_patch_t2: torch.Tensor = None
        self.pre_hook_handle = None
        self._register_hook()

    def _register_hook(self):
        def conv1_pre_hook(module, args):
            # args[0] is X of shape (B, 288, 56, 56)
            self.current_patch_t2 = args[0].detach().clone().cpu()
            return args

        self.pre_hook_handle = self.conv1_seq.register_forward_pre_hook(conv1_pre_hook)

    def cleanup(self):
        if self.pre_hook_handle:
            self.pre_hook_handle.remove()
            self.pre_hook_handle = None

    def run_image(
        self,
        image_rgb: np.ndarray,
        tile_size: int = 448,
        batch_size: int = 4
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Runs Setting A tiling inference and returns binary prediction, probability map, and captured T2 patch tensors."""
        H, W = image_rgb.shape[:2]
        pad_h = (tile_size - (H % tile_size)) % tile_size
        pad_w = (tile_size - (W % tile_size)) % tile_size
        padded_img = np.pad(image_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        pH, pW = padded_img.shape[:2]

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        final_logits = np.zeros((pH, pW), dtype=np.float32)

        patches = []
        coords = []
        for y in range(0, pH, tile_size):
            for x in range(0, pW, tile_size):
                patch = padded_img[y:y+tile_size, x:x+tile_size]
                pt = torch.from_numpy(patch).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
                pt = (pt - mean) / std
                patches.append(pt)
                coords.append((y, x))

        patch_records = []
        self.model.eval()

        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch_pt = torch.cat(patches[i:i+batch_size], dim=0)
                batch_coords = coords[i:i+batch_size]

                self.current_patch_t2 = None
                logits = self.model(batch_pt)
                if logits.shape[2:] != (tile_size, tile_size):
                    logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)

                captured_t2 = self.current_patch_t2  # (B, 288, 56, 56)

                for b_idx, (py, px) in enumerate(batch_coords):
                    final_logits[py:py+tile_size, px:px+tile_size] = logits[b_idx].squeeze().cpu().numpy()
                    patch_records.append({
                        'coords': (py, px),
                        't2': captured_t2[b_idx:b_idx+1],  # (1, 288, 56, 56)
                    })

        logits_cropped = final_logits[:H, :W]
        prob_cropped = 1.0 / (1.0 + np.exp(-logits_cropped))
        pred_bin = (logits_cropped > 0.0).astype(np.uint8)

        return pred_bin, prob_cropped, patch_records


# -----------------------------------------------------------------------------
# Spatial Statistics Computation Engine
# -----------------------------------------------------------------------------

def assemble_full_t2_maps(
    patch_records: List[Dict[str, Any]],
    pH: int,
    pW: int,
    tile_size: int = 448
) -> Tuple[np.ndarray, np.ndarray]:
    """Assembles full spatial feature grids for Bottleneck B (192 ch) and Skip S (96 ch)."""
    fH, fW = pH // 8, pW // 8
    B_grid = np.zeros((192, fH, fW), dtype=np.float32)
    S_grid = np.zeros((96, fH, fW), dtype=np.float32)

    for p in patch_records:
        py, px = p['coords']
        fy, fx = py // 8, px // 8
        t2_patch = p['t2'][0].numpy()  # (288, 56, 56)
        B_grid[:, fy:fy+56, fx:fx+56] = t2_patch[:192]
        S_grid[:, fy:fy+56, fx:fx+56] = t2_patch[192:]

    return B_grid, S_grid


def compute_spatial_statistics(
    B: np.ndarray,
    S: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Computes all candidate spatial statistics purely from B and S.
    B shape: (192, fH, fW), S shape: (96, fH, fW).
    Returns dict of 2D spatial maps of shape (fH, fW).
    """
    eps = 1e-4

    # 1. Channel L2 Energies
    E_B = np.sqrt(np.sum(B ** 2, axis=0))  # (fH, fW)
    E_S = np.sqrt(np.sum(S ** 2, axis=0))  # (fH, fW)
    log_ratio = np.log((E_S + eps) / (E_B + eps))

    # 2. Normalized Energies (min-max and z-score)
    min_b, max_b = np.min(E_B), np.max(E_B)
    min_s, max_s = np.min(E_S), np.max(E_S)
    norm_E_B = (E_B - min_b) / (max_b - min_b + 1e-6)
    norm_E_S = (E_S - min_s) / (max_s - min_s + 1e-6)

    mean_b, std_b = np.mean(E_B), np.std(E_B)
    mean_s, std_s = np.mean(E_S), np.std(E_S)
    z_B = (E_B - mean_b) / (std_b + 1e-6)
    z_S = (E_S - mean_s) / (std_s + 1e-6)

    # 3. Disagreement Metrics
    disagree_diff = z_S - z_B  # High when Skip is elevated relative to Bottleneck
    disagree_abs = np.abs(z_S - z_B)
    norm_ratio = (norm_E_S + 0.05) / (norm_E_B + 0.05)

    # 4. Stream Agreement (Cosine between 96-channel pooled B and S)
    # Pool B from 192 -> 96 channels by averaging pairs
    B_96 = 0.5 * (B[0::2] + B[1::2])  # (96, fH, fW)
    dot_BS = np.sum(B_96 * S, axis=0)
    norm_B96 = np.sqrt(np.sum(B_96 ** 2, axis=0))
    norm_S96 = np.sqrt(np.sum(S ** 2, axis=0))
    cosine_BS = dot_BS / (norm_B96 * norm_S96 + 1e-6)

    # 5. Local Spatial Gradient Magnitude (Sobel)
    gx_B = cv2.Sobel(E_B.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy_B = cv2.Sobel(E_B.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_E_B = np.sqrt(gx_B ** 2 + gy_B ** 2)

    gx_S = cv2.Sobel(E_S.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy_S = cv2.Sobel(E_S.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_E_S = np.sqrt(gx_S ** 2 + gy_S ** 2)

    # 6. Local 3x3 Spatial Variance
    mean_EB_3x3 = cv2.blur(E_B.astype(np.float32), (3, 3))
    mean_EB2_3x3 = cv2.blur((E_B ** 2).astype(np.float32), (3, 3))
    var_E_B = np.maximum(0.0, mean_EB2_3x3 - mean_EB_3x3 ** 2)

    mean_ES_3x3 = cv2.blur(E_S.astype(np.float32), (3, 3))
    mean_ES2_3x3 = cv2.blur((E_S ** 2).astype(np.float32), (3, 3))
    var_E_S = np.maximum(0.0, mean_ES2_3x3 - mean_ES_3x3 ** 2)

    # 7. Candidate Gating Signals
    # Candidate B: High Skip Energy + Low Bottleneck Energy
    candidate_gate_B = norm_E_S * (1.0 - norm_E_B)
    # Candidate C: Normalized Ratio
    candidate_gate_C = norm_ratio

    return {
        'E_B': E_B,
        'E_S': E_S,
        'log_ratio': log_ratio,
        'z_B': z_B,
        'z_S': z_S,
        'norm_E_B': norm_E_B,
        'norm_E_S': norm_E_S,
        'disagree_diff': disagree_diff,
        'disagree_abs': disagree_abs,
        'norm_ratio': norm_ratio,
        'cosine_BS': cosine_BS,
        'grad_E_B': grad_E_B,
        'grad_E_S': grad_E_S,
        'var_E_B': var_E_B,
        'var_E_S': var_E_S,
        'candidate_gate_B': candidate_gate_B,
        'candidate_gate_C': candidate_gate_C,
    }


def compute_roc_auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes ROC-AUC safely, returning 0.5 if single class or degenerate."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def compute_cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Cliff's delta non-parametric effect size between two groups."""
    if len(x) == 0 or len(y) == 0:
        return 0.0
    # Use fast Mann-Whitney U relation
    n_x, n_y = len(x), len(y)
    # Subsample if large to prevent memory overflow
    if n_x > 5000:
        x = np.random.choice(x, 5000, replace=False)
        n_x = len(x)
    if n_y > 5000:
        y = np.random.choice(y, 5000, replace=False)
        n_y = len(y)
    # Compute comparisons
    diff = x[:, None] - y[None, :]
    delta = (np.sum(diff > 0) - np.sum(diff < 0)) / (n_x * n_y)
    return float(delta)


# -----------------------------------------------------------------------------
# Main Diagnostic Routine
# -----------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Phase 6: T2 Spatial Discriminability / Gateability Diagnostic")
    print("Analysis of Skip vs. Bottleneck Spatial Relations & Gate Feasibility")
    print("=" * 80)

    config_path = 'results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml'
    ckpt_path = 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'
    output_dir = 'results/diagnostics/phase6_t2_spatial_gateability'
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load Model and compute initial bitwise state hashes
    print(f"\nLoading Candidate B model from: {ckpt_path}...")
    init_file_hash = compute_file_hash(ckpt_path)
    model = load_model_from_checkpoint(config_path, ckpt_path, device=device)
    init_param_hash = compute_model_param_hash(model)
    print(f"Initial Model Parameters SHA256: {init_param_hash[:16]}...")
    print(f"Initial Disk Checkpoint SHA256: {init_file_hash[:16]}...")

    engine = T2FeatureCaptureEngine(model, device)

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

    prev_run_csv = 'results/diagnostics/phase6_stem_ac_attenuation/ac_attenuation_per_sample_alpha.csv'
    df_prev = pd.read_csv(prev_run_csv)
    c7_alpha0 = df_prev[(df_prev['cohort'] == 'Consensus_7of7') & (df_prev['alpha'] == 0.0)]
    stem_sensitive_stems = set(c7_alpha0[c7_alpha0['has_bridge'] == False]['image_id'].tolist())
    downstream_resistant_stems = set(c7_alpha0[c7_alpha0['has_bridge'] == True]['image_id'].tolist())

    print(f"\nCohorts Identified:")
    print(f"  - Consensus 7/7:                 N = {len(c7_stems)}")
    print(f"    * Stem-Sensitive (Cured@a=0):   N = {len(stem_sensitive_stems)} (18.8%)")
    print(f"    * Downstream-Resistant (Bridge): N = {len(downstream_resistant_stems)} (81.2%)")
    print(f"  - Cured by D2:                   N = {len(d2_cured_stems)}")
    print(f"  - Clean Control (gt>=2):         N = {len(clean_stems)}")

    unique_cohort_samples = []
    for s in sorted(list(c7_stems)):
        sub = 'Consensus_Resistant' if s in downstream_resistant_stems else 'Consensus_Sensitive'
        unique_cohort_samples.append({'stem': s, 'cohort': 'Consensus_7of7', 'subgroup': sub})
    for s in sorted(list(d2_cured_stems)):
        unique_cohort_samples.append({'stem': s, 'cohort': 'Cured_by_D2', 'subgroup': 'Cured_by_D2'})
    for s in sorted(list(clean_stems)):
        unique_cohort_samples.append({'stem': s, 'cohort': 'Clean_Control', 'subgroup': 'Clean_Control'})

    print(f"Total Unique Samples: {len(unique_cohort_samples)}")

    # Step 1: Run Setting A Baseline and Capture T2 Features
    print("\nRunning Setting A Baseline Inference and Capturing T2 Spatial Grids...")
    per_sample_records = []
    cohort_pixel_data = {
        'Consensus_Resistant': {k: {'CRACK': [], 'NECK': [], 'BG': []} for k in ['E_B', 'E_S', 'log_ratio', 'disagree_diff', 'candidate_gate_B', 'cosine_BS']},
        'Consensus_Sensitive': {k: {'CRACK': [], 'NECK': [], 'BG': []} for k in ['E_B', 'E_S', 'log_ratio', 'disagree_diff', 'candidate_gate_B', 'cosine_BS']},
        'Cured_by_D2': {k: {'CRACK': [], 'NECK': [], 'BG': []} for k in ['E_B', 'E_S', 'log_ratio', 'disagree_diff', 'candidate_gate_B', 'cosine_BS']},
        'Clean_Control': {k: {'CRACK': [], 'NECK': [], 'BG': []} for k in ['E_B', 'E_S', 'log_ratio', 'disagree_diff', 'candidate_gate_B', 'cosine_BS']},
    }

    t0 = time.time()
    representative_samples = {}

    for item in tqdm(unique_cohort_samples, desc="T2 Spatial Evaluation"):
        stem = item['stem']
        cohort = item['cohort']
        subgroup = item['subgroup']

        img_path = f"datasets/Crack500_ready/val/images/{stem}.jpg"
        mask_path = f"datasets/Crack500_ready/val/masks/{stem}.png"

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        target_bin = (gt_mask > 0).astype(np.uint8)

        H, W = img_rgb.shape[:2]
        tile_size = 448
        pad_h = (tile_size - (H % tile_size)) % tile_size
        pad_w = (tile_size - (W % tile_size)) % tile_size
        pH, pW = H + pad_h, W + pad_w

        pred_bin, prob_map, patch_records = engine.run_image(img_rgb, tile_size=tile_size, batch_size=4)
        topo = compute_topology_metrics(pred_bin, target_bin)

        if cohort in ['Consensus_7of7', 'Cured_by_D2']:
            neck_mask, crack_mask, bg_mask, _ = isolate_bridged_pairs_and_rois(pred_bin, target_bin)
        else:
            neck_mask, crack_mask, bg_mask, _ = isolate_clean_pairs_and_rois(target_bin)

        # Assemble full 2D grids for B and S
        B_grid, S_grid = assemble_full_t2_maps(patch_records, pH, pW, tile_size=tile_size)

        # Crop to unpadded region in feature space
        fH, fW = H // 8, W // 8
        B_crop = B_grid[:, :fH, :fW]
        S_crop = S_grid[:, :fH, :fW]

        # Downsample reference masks to (fH, fW)
        m_neck = (cv2.resize(neck_mask[:H, :W], (fW, fH), interpolation=cv2.INTER_NEAREST) > 0)
        m_crack = (cv2.resize(crack_mask[:H, :W], (fW, fH), interpolation=cv2.INTER_NEAREST) > 0)
        m_crack = m_crack & (~m_neck)
        m_bg = (cv2.resize(bg_mask[:H, :W], (fW, fH), interpolation=cv2.INTER_NEAREST) > 0)
        m_bg = m_bg & (~m_neck) & (~m_crack)

        # Compute candidate spatial statistics purely from B_crop and S_crop
        stat_maps = compute_spatial_statistics(B_crop, S_crop)

        # Cache representative samples for visualization
        if subgroup == 'Consensus_Resistant' and 'Resistant' not in representative_samples and np.sum(m_neck) > 10:
            representative_samples['Resistant'] = {'stem': stem, 'img': img_rgb, 'neck': neck_mask, 'crack': crack_mask, 'maps': stat_maps, 'topo': topo}
        elif subgroup == 'Consensus_Sensitive' and 'Sensitive' not in representative_samples and np.sum(m_neck) > 10:
            representative_samples['Sensitive'] = {'stem': stem, 'img': img_rgb, 'neck': neck_mask, 'crack': crack_mask, 'maps': stat_maps, 'topo': topo}
        elif cohort == 'Cured_by_D2' and 'D2_Cured' not in representative_samples and np.sum(m_neck) > 5:
            representative_samples['D2_Cured'] = {'stem': stem, 'img': img_rgb, 'neck': neck_mask, 'crack': crack_mask, 'maps': stat_maps, 'topo': topo}

        # Analyze statistics by region for this sample
        sample_row = {
            'image_id': stem,
            'cohort': cohort,
            'subgroup': subgroup,
            'has_bridge': bool(topo['bridge_events'] > 0),
            'n_neck_pixels': int(np.sum(m_neck)),
            'n_crack_pixels': int(np.sum(m_crack)),
            'n_bg_pixels': int(np.sum(m_bg)),
        }

        # Subgroup key for pooled pixels
        c_key = subgroup if cohort == 'Consensus_7of7' else cohort

        for stat_name, smap in stat_maps.items():
            vals_neck = smap[m_neck] if np.sum(m_neck) > 0 else np.array([], dtype=np.float32)
            vals_crack = smap[m_crack] if np.sum(m_crack) > 0 else np.array([], dtype=np.float32)
            vals_bg = smap[m_bg] if np.sum(m_bg) > 0 else np.array([], dtype=np.float32)

            med_n = float(np.median(vals_neck)) if len(vals_neck) > 0 else np.nan
            iqr_n = float(np.percentile(vals_neck, 75) - np.percentile(vals_neck, 25)) if len(vals_neck) > 0 else np.nan
            med_c = float(np.median(vals_crack)) if len(vals_crack) > 0 else np.nan
            iqr_c = float(np.percentile(vals_crack, 75) - np.percentile(vals_crack, 25)) if len(vals_crack) > 0 else np.nan
            med_b = float(np.median(vals_bg)) if len(vals_bg) > 0 else np.nan
            iqr_b = float(np.percentile(vals_bg, 75) - np.percentile(vals_bg, 25)) if len(vals_bg) > 0 else np.nan

            sample_row[f'{stat_name}_neck_median'] = med_n
            sample_row[f'{stat_name}_neck_iqr'] = iqr_n
            sample_row[f'{stat_name}_crack_median'] = med_c
            sample_row[f'{stat_name}_crack_iqr'] = iqr_c
            sample_row[f'{stat_name}_bg_median'] = med_b
            sample_row[f'{stat_name}_bg_iqr'] = iqr_b

            # Sample-level AUC: NECK vs CRACK
            if len(vals_neck) > 0 and len(vals_crack) > 0:
                y_nc = np.concatenate([np.ones_like(vals_neck), np.zeros_like(vals_crack)])
                s_nc = np.concatenate([vals_neck, vals_crack])
                sample_row[f'{stat_name}_auc_neck_vs_crack'] = compute_roc_auc_safe(y_nc, s_nc)
            else:
                sample_row[f'{stat_name}_auc_neck_vs_crack'] = np.nan

            # Sample-level AUC: NECK vs BACKGROUND
            if len(vals_neck) > 0 and len(vals_bg) > 0:
                y_nb = np.concatenate([np.ones_like(vals_neck), np.zeros_like(vals_bg)])
                s_nb = np.concatenate([vals_neck, vals_bg])
                sample_row[f'{stat_name}_auc_neck_vs_bg'] = compute_roc_auc_safe(y_nb, s_nb)
            else:
                sample_row[f'{stat_name}_auc_neck_vs_bg'] = np.nan

            # Pool selected key statistics for overall distribution analysis
            if stat_name in cohort_pixel_data[c_key]:
                if len(vals_crack) > 0:
                    cohort_pixel_data[c_key][stat_name]['CRACK'].extend(np.random.choice(vals_crack, min(len(vals_crack), 100), replace=False).tolist())
                if len(vals_neck) > 0:
                    cohort_pixel_data[c_key][stat_name]['NECK'].extend(vals_neck.tolist())
                if len(vals_bg) > 0:
                    cohort_pixel_data[c_key][stat_name]['BG'].extend(np.random.choice(vals_bg, min(len(vals_bg), 100), replace=False).tolist())

        per_sample_records.append(sample_row)

    engine.cleanup()
    t1 = time.time()
    print(f"\nSetting A Baseline & T2 feature extraction completed in {t1 - t0:.1f}s.")

    # Integrity verification
    final_param_hash = compute_model_param_hash(model)
    final_file_hash = compute_file_hash(ckpt_path)

    print(f"\nVerifying Bitwise State Invariance:")
    print(f"  Initial Param Hash: {init_param_hash[:16]}... | Final: {final_param_hash[:16]}...")
    print(f"  Initial File Hash:  {init_file_hash[:16]}... | Final: {final_file_hash[:16]}...")
    assert init_param_hash == final_param_hash, "HARD LOCK VIOLATION: Model parameters modified in memory!"
    assert init_file_hash == final_file_hash, "HARD LOCK VIOLATION: Checkpoint file modified on disk!"
    print("Bitwise Invariance Check: 100% PASSED.")

    df_samples = pd.DataFrame(per_sample_records)
    sample_csv = os.path.join(output_dir, 't2_spatial_statistics_per_sample.csv')
    df_samples.to_csv(sample_csv, index=False)
    print(f"Saved: {sample_csv}")

    # Step 2: Compute Gateability & Discriminability Summary by Statistic & Cohort
    print("\nComputing Discriminability Summary across Spatial Statistics...")
    stat_list = [
        'E_B', 'E_S', 'log_ratio', 'z_B', 'z_S',
        'norm_E_B', 'norm_E_S', 'disagree_diff', 'disagree_abs',
        'norm_ratio', 'cosine_BS', 'grad_E_B', 'grad_E_S',
        'var_E_B', 'var_E_S', 'candidate_gate_B', 'candidate_gate_C'
    ]

    cohort_keys = ['Consensus_Resistant', 'Consensus_Sensitive', 'Cured_by_D2', 'Clean_Control']
    gateability_records = []
    region_summary_records = []

    for c_key in cohort_keys:
        if c_key in ['Consensus_Resistant', 'Consensus_Sensitive']:
            c_df = df_samples[df_samples['subgroup'] == c_key]
        else:
            c_df = df_samples[df_samples['cohort'] == c_key]

        for stat in stat_list:
            med_neck_col = f'{stat}_neck_median'
            med_crack_col = f'{stat}_crack_median'
            med_bg_col = f'{stat}_bg_median'

            auc_nc_col = f'{stat}_auc_neck_vs_crack'
            auc_nb_col = f'{stat}_auc_neck_vs_bg'

            valid_nc = c_df[c_df['n_neck_pixels'] > 0]
            auc_nc_vals = valid_nc[auc_nc_col].dropna().values
            auc_nb_vals = valid_nc[auc_nb_col].dropna().values

            med_auc_nc = float(np.median(auc_nc_vals)) if len(auc_nc_vals) > 0 else np.nan
            med_auc_nb = float(np.median(auc_nb_vals)) if len(auc_nb_vals) > 0 else np.nan

            # Two-sided discriminability (distance from 0.5)
            discrim_nc = float(np.median(np.maximum(auc_nc_vals, 1.0 - auc_nc_vals))) if len(auc_nc_vals) > 0 else np.nan
            discrim_nb = float(np.median(np.maximum(auc_nb_vals, 1.0 - auc_nb_vals))) if len(auc_nb_vals) > 0 else np.nan

            # Overall pooled pixel statistics
            if stat in cohort_pixel_data[c_key]:
                p_neck = np.array(cohort_pixel_data[c_key][stat]['NECK'])
                p_crack = np.array(cohort_pixel_data[c_key][stat]['CRACK'])
                p_bg = np.array(cohort_pixel_data[c_key][stat]['BG'])

                delta_cliff_nc = compute_cliffs_delta(p_neck, p_crack) if len(p_neck) > 0 and len(p_crack) > 0 else np.nan
                delta_cliff_nb = compute_cliffs_delta(p_neck, p_bg) if len(p_neck) > 0 and len(p_bg) > 0 else np.nan
            else:
                delta_cliff_nc, delta_cliff_nb = np.nan, np.nan

            gateability_records.append({
                'cohort': c_key,
                'statistic': stat,
                'median_neck': float(c_df[med_neck_col].median()),
                'iqr_neck': float(c_df[med_neck_col].quantile(0.75) - c_df[med_neck_col].quantile(0.25)),
                'median_crack': float(c_df[med_crack_col].median()),
                'iqr_crack': float(c_df[med_crack_col].quantile(0.75) - c_df[med_crack_col].quantile(0.25)),
                'median_bg': float(c_df[med_bg_col].median()),
                'iqr_bg': float(c_df[med_bg_col].quantile(0.75) - c_df[med_bg_col].quantile(0.25)),
                'directional_auc_neck_vs_crack': med_auc_nc,
                'discriminability_neck_vs_crack': discrim_nc,
                'cliffs_delta_neck_vs_crack': delta_cliff_nc,
                'directional_auc_neck_vs_bg': med_auc_nb,
                'discriminability_neck_vs_bg': discrim_nb,
                'cliffs_delta_neck_vs_bg': delta_cliff_nb,
            })

            # Record region summary
            region_summary_records.append({
                'cohort': c_key,
                'statistic': stat,
                'NECK_median': float(c_df[med_neck_col].median()),
                'CRACK_median': float(c_df[med_crack_col].median()),
                'BG_median': float(c_df[med_bg_col].median()),
                'NECK_minus_CRACK': float(c_df[med_neck_col].median() - c_df[med_crack_col].median()),
                'NECK_minus_BG': float(c_df[med_neck_col].median() - c_df[med_bg_col].median()),
            })

    df_gateability = pd.DataFrame(gateability_records)
    gate_csv = os.path.join(output_dir, 't2_gateability_summary.csv')
    df_gateability.to_csv(gate_csv, index=False)
    print(f"Saved: {gate_csv}")

    df_region = pd.DataFrame(region_summary_records)
    region_csv = os.path.join(output_dir, 't2_spatial_statistics_by_region.csv')
    df_region.to_csv(region_csv, index=False)
    print(f"Saved: {region_csv}")

    print("\nTop Candidate Gate Statistics for Consensus Resistant (Ranked by Discriminability):")
    res_stats = df_gateability[df_gateability['cohort'] == 'Consensus_Resistant'].sort_values(by='discriminability_neck_vs_crack', ascending=False)
    print(res_stats[['statistic', 'median_neck', 'median_crack', 'median_bg', 'directional_auc_neck_vs_crack', 'discriminability_neck_vs_crack', 'discriminability_neck_vs_bg']].head(10).to_string(index=False))

    # Step 3: Visualizations
    print("\nGenerating Diagnostic Figures...")

    # Figure 1: Spatial Statistics Distribution (NECK vs CRACK vs BG)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    plot_stats = ['E_S', 'E_B', 'log_ratio', 'disagree_diff', 'candidate_gate_B', 'cosine_BS']
    stat_titles = [
        '1. Skip Energy E_S (||S||_2)',
        '2. Bottleneck Energy E_B (||B||_2)',
        '3. Log Ratio log(E_S / E_B)',
        '4. Cross-Stream Disagree (z_S - z_B)',
        '5. Candidate Gate B: norm_E_S * (1 - norm_E_B)',
        '6. Stream Cosine Agreement cos(B_pool, S)'
    ]

    res_df = df_samples[df_samples['subgroup'] == 'Consensus_Resistant']

    for idx, (st, title) in enumerate(zip(plot_stats, stat_titles)):
        ax = axes[idx // 3, idx % 3]

        data_crack = res_df[f'{st}_crack_median'].dropna().values
        data_neck = res_df[f'{st}_neck_median'].dropna().values
        data_bg = res_df[f'{st}_bg_median'].dropna().values

        box_data = [data_crack, data_neck, data_bg]
        positions = [1, 2, 3]
        labels = ['CRACK', 'NECK', 'BG']
        colors = ['#1f77b4', '#d62728', '#7f7f7f']

        bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.55,
                        medianprops=dict(color='black', linewidth=1.5))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Annotate median values
        for pos, dat in zip(positions, box_data):
            if len(dat) > 0:
                m_val = np.median(dat)
                ax.text(pos, m_val, f' {m_val:.2f}', verticalalignment='bottom', fontsize=8.5, fontweight='bold')

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(title, fontsize=10.5)
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, 't2_spatial_statistics_distribution.png')
    plt.savefig(fig1_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {fig1_path}")

    # Figure 2: Spatial Heatmaps on Representative Cases
    if representative_samples:
        n_cases = len(representative_samples)
        fig, axes = plt.subplots(n_cases, 6, figsize=(18, 3.2 * n_cases))
        if n_cases == 1:
            axes = np.expand_dims(axes, 0)

        case_keys = list(representative_samples.keys())
        for row_idx, ck in enumerate(case_keys):
            case = representative_samples[ck]
            img = case['img']
            neck_m = case['neck']
            crack_m = case['crack']
            smaps = case['maps']

            # Col 0: Image + Overlay
            vis = img.copy()
            vis[crack_m > 0] = [0, 220, 0]   # Green crack
            vis[neck_m > 0] = [255, 30, 30]   # Red neck
            axes[row_idx, 0].imshow(vis)
            axes[row_idx, 0].set_title(f"{ck} ({case['stem']})\nGT & Corridor", fontsize=9.5)
            axes[row_idx, 0].axis('off')

            # Col 1: E_S map
            im1 = axes[row_idx, 1].imshow(smaps['E_S'], cmap='viridis')
            axes[row_idx, 1].set_title("Skip Energy E_S", fontsize=9.5)
            axes[row_idx, 1].axis('off')
            plt.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

            # Col 2: E_B map
            im2 = axes[row_idx, 2].imshow(smaps['E_B'], cmap='viridis')
            axes[row_idx, 2].set_title("Bottleneck Energy E_B", fontsize=9.5)
            axes[row_idx, 2].axis('off')
            plt.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)

            # Col 3: Disagree z_S - z_B
            im3 = axes[row_idx, 3].imshow(smaps['disagree_diff'], cmap='coolwarm', vmin=-2, vmax=2)
            axes[row_idx, 3].set_title("Disagree (z_S - z_B)", fontsize=9.5)
            axes[row_idx, 3].axis('off')
            plt.colorbar(im3, ax=axes[row_idx, 3], fraction=0.046, pad=0.04)

            # Col 4: Candidate Gate B
            im4 = axes[row_idx, 4].imshow(smaps['candidate_gate_B'], cmap='magma')
            axes[row_idx, 4].set_title("Gate B: nE_S*(1-nE_B)", fontsize=9.5)
            axes[row_idx, 4].axis('off')
            plt.colorbar(im4, ax=axes[row_idx, 4], fraction=0.046, pad=0.04)

            # Col 5: Candidate Gate C (norm_ratio)
            im5 = axes[row_idx, 5].imshow(smaps['norm_ratio'], cmap='plasma', vmax=3)
            axes[row_idx, 5].set_title("Gate C: Norm Ratio", fontsize=9.5)
            axes[row_idx, 5].axis('off')
            plt.colorbar(im5, ax=axes[row_idx, 5], fraction=0.046, pad=0.04)

        plt.tight_layout()
        fig2_path = os.path.join(figures_dir, 't2_spatial_gate_heatmaps.png')
        plt.savefig(fig2_path, dpi=160, bbox_inches='tight')
        plt.close()
        print(f"Saved figure: {fig2_path}")

    # Figure 3: Scatter Plot E_S vs E_B (CRACK vs NECK vs BG)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    target_cohorts = [('Consensus_Resistant', 'Consensus Resistant (N=82)', axes[0]),
                       ('Consensus_Sensitive', 'Consensus Sensitive (N=19)', axes[1]),
                       ('Cured_by_D2', 'Cured by D2 (N=7)', axes[2])]

    for c_key, c_title, ax in target_cohorts:
        p_c_es = np.array(cohort_pixel_data[c_key]['E_S']['CRACK'])
        p_c_eb = np.array(cohort_pixel_data[c_key]['E_B']['CRACK'])

        p_n_es = np.array(cohort_pixel_data[c_key]['E_S']['NECK'])
        p_n_eb = np.array(cohort_pixel_data[c_key]['E_B']['NECK'])

        p_b_es = np.array(cohort_pixel_data[c_key]['E_S']['BG'])
        p_b_eb = np.array(cohort_pixel_data[c_key]['E_B']['BG'])

        # Subsample for clear scatter plot
        if len(p_b_es) > 1500:
            idx_b = np.random.choice(len(p_b_es), 1500, replace=False)
            p_b_es, p_b_eb = p_b_es[idx_b], p_b_eb[idx_b]
        if len(p_c_es) > 1500:
            idx_c = np.random.choice(len(p_c_es), 1500, replace=False)
            p_c_es, p_c_eb = p_c_es[idx_c], p_c_eb[idx_c]

        ax.scatter(p_b_eb, p_b_es, c='#7f7f7f', alpha=0.3, s=15, label='BACKGROUND')
        ax.scatter(p_c_eb, p_c_es, c='#1f77b4', alpha=0.5, s=25, label='CRACK')
        ax.scatter(p_n_eb, p_n_es, c='#d62728', alpha=0.8, s=40, edgecolors='black', label=f'NECK ({len(p_n_es)} px)')

        ax.set_xlabel('Bottleneck Energy E_B (||B||_2)', fontsize=9.5)
        ax.set_ylabel('Skip Energy E_S (||S||_2)', fontsize=9.5)
        ax.set_title(c_title, fontsize=10.5)
        ax.legend(fontsize=8.5)
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, 't2_es_vs_eb_scatter.png')
    plt.savefig(fig3_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {fig3_path}")

    print("\n" + "=" * 80)
    print("Phase 6 T2 Spatial Gateability Diagnostic Completed Successfully.")
    print("=" * 80)


if __name__ == '__main__':
    main()
