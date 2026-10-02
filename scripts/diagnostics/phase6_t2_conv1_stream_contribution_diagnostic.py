#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_t2_conv1_stream_contribution_diagnostic.py

Phase 6: T2 Conv1 Stream-Contribution / Fusion Diagnostic
Target Locus: Decoder Block 1 Conv1 (Conv2d: 288 -> 96, k=3, s=1, p=1, bias=True)

Decomposition of Conv1 weights and inputs:
- W_B = W[:, 0:192, :, :] (Bottleneck contribution from deep ViT/Bottleneck)
- W_S = W[:, 192:288, :, :] (Skip contribution from CNN Stage 1)
- bias is preserved identically.

Linear Decomposition Identity:
  Y    = Conv1(B, S) = W_B * B + W_S * S + bias
  Y_B  = W_B * B + bias (Skip = 0)
  Y_S  = W_S * S + bias (Bottleneck = 0)
  Y_BS = Y_B + Y_S - bias == Y (Verified to float32 numerical tolerance)

Counterfactual Stream Attenuation Conditions:
1. Baseline: (alpha_B = 1.0, alpha_S = 1.0)
2. Bottleneck_Only: (alpha_B = 1.0, alpha_S = 0.0)
3. Skip_Only: (alpha_B = 0.0, alpha_S = 1.0)
4. S_x0.75: (alpha_B = 1.0, alpha_S = 0.75)
5. S_x0.50: (alpha_B = 1.0, alpha_S = 0.50)
6. S_x0.25: (alpha_B = 1.0, alpha_S = 0.25)
7. B_x0.75: (alpha_B = 0.75, alpha_S = 1.0)
8. B_x0.50: (alpha_B = 0.50, alpha_S = 1.0)
9. B_x0.25: (alpha_B = 0.25, alpha_S = 1.0)

Cohorts Evaluated:
- Consensus Resistant (N = 82)
- Consensus Sensitive (N = 19)
- Cured by D2 (N = 7)
- Clean Control (N = 56)
Total N = 164. Sealed test set (N = 1124) strictly unaccessed.

HARD LOCKS:
- DIAGNOSTIC-ONLY: Zero training passes, zero backward passes, zero optimizer steps.
- Bitwise verification of checkpoint file and parameter SHA256 before and after.
- Setting A evaluation path preserved (448x448, stride 448 non-overlapping, reflect padding, tau=0.5).
- In-memory pre-hook tensor scaling and instant exact restoration.
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
from skimage.morphology import skeletonize
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
# T2 Conv1 Factorization Engine
# -----------------------------------------------------------------------------

class T2Conv1FactorizationEngine:
    """
    Manages in-memory pre-hook surgery at Decoder Block 1 Conv1.
    Performs tensor scaling and extracts Y, Y_B, Y_S, and Y_BS representation metrics.
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

        self.dec_b1 = model.decoder.decoder_blocks[1]
        self.conv_seq = self.dec_b1.conv1
        self.conv1_layer = self.conv_seq[0]  # nn.Conv2d(288, 96, 3, p=1, bias=True)

        self.alpha_B: float = 1.0
        self.alpha_S: float = 1.0
        self.mode: str = 'baseline'

        self.last_patch_diagnostics: List[Dict[str, Any]] = []
        self.pre_hook_handle = None
        self._register_pre_hook()

    def _register_pre_hook(self):
        def conv1_pre_hook(module, args):
            # args[0] is X of shape (B, 288, 56, 56)
            x_in = args[0]

            # In baseline mode, compute Y, Y_B, Y_S, Y_BS for representation profiling
            if self.mode == 'baseline':
                with torch.no_grad():
                    w = self.conv1_layer.weight  # (96, 288, 3, 3)
                    b = self.conv1_layer.bias    # (96,)

                    w_B = w[:, :192, :, :]
                    w_S = w[:, 192:, :, :]

                    B_slice = x_in[:, :192, :, :]
                    S_slice = x_in[:, 192:, :, :]

                    # Linear contributions at Conv1 output
                    Y_full = F.conv2d(x_in, w, bias=b, padding=1)
                    Y_B = F.conv2d(B_slice, w_B, bias=b, padding=1)
                    Y_S = F.conv2d(S_slice, w_S, bias=b, padding=1)
                    Y_BS = Y_B + Y_S - b.view(1, -1, 1, 1)

                    # Verify linearity tolerance
                    max_diff = (Y_full - Y_BS).abs().max().item()

                    self.last_patch_diagnostics.append({
                        'max_diff': max_diff,
                        'Y_full': Y_full.detach().cpu(),
                        'Y_B': Y_B.detach().cpu(),
                        'Y_S': Y_S.detach().cpu(),
                        'Y_BS': Y_BS.detach().cpu(),
                        'X_B': B_slice.detach().cpu(),
                        'X_S': S_slice.detach().cpu(),
                    })
                return args

            # In counterfactual mode, scale slices in-place on a clone
            x_mod = x_in.clone()
            if self.alpha_B != 1.0:
                x_mod[:, :192, :, :] *= float(self.alpha_B)
            if self.alpha_S != 1.0:
                x_mod[:, 192:, :, :] *= float(self.alpha_S)
            return (x_mod,)

        self.pre_hook_handle = self.conv_seq.register_forward_pre_hook(conv1_pre_hook)

    def set_surgery(self, mode: str, alpha_B: float = 1.0, alpha_S: float = 1.0):
        self.mode = mode
        self.alpha_B = alpha_B
        self.alpha_S = alpha_S
        self.last_patch_diagnostics = []

    def cleanup(self):
        if self.pre_hook_handle:
            self.pre_hook_handle.remove()
            self.pre_hook_handle = None
        self.set_surgery('baseline', 1.0, 1.0)

    def run_image(
        self,
        image_rgb: np.ndarray,
        tile_size: int = 448,
        batch_size: int = 4
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Runs Setting A tiling inference and returns binary prediction, probability map, and patch diagnostics."""
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

        self.last_patch_diagnostics = []
        self.model.eval()

        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch_pt = torch.cat(patches[i:i+batch_size], dim=0)
                batch_coords = coords[i:i+batch_size]

                logits = self.model(batch_pt)
                if logits.shape[2:] != (tile_size, tile_size):
                    logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)

                for b_idx, (py, px) in enumerate(batch_coords):
                    final_logits[py:py+tile_size, px:px+tile_size] = logits[b_idx].squeeze().cpu().numpy()

        logits_cropped = final_logits[:H, :W]
        prob_cropped = 1.0 / (1.0 + np.exp(-logits_cropped))
        pred_bin = (logits_cropped > 0.0).astype(np.uint8)

        # Pair patch diagnostics with coordinates if available
        collected_patches = []
        if self.last_patch_diagnostics:
            diag_idx = 0
            for batch_item in self.last_patch_diagnostics:
                b_size = batch_item['Y_full'].shape[0]
                for b_i in range(b_size):
                    if diag_idx < len(coords):
                        py, px = coords[diag_idx]
                        collected_patches.append({
                            'coords': (py, px),
                            'max_diff': batch_item['max_diff'],
                            'Y_full': batch_item['Y_full'][b_i:b_i+1],
                            'Y_B': batch_item['Y_B'][b_i:b_i+1],
                            'Y_S': batch_item['Y_S'][b_i:b_i+1],
                            'Y_BS': batch_item['Y_BS'][b_i:b_i+1],
                            'X_B': batch_item['X_B'][b_i:b_i+1],
                            'X_S': batch_item['X_S'][b_i:b_i+1],
                        })
                        diag_idx += 1

        return pred_bin, prob_cropped, collected_patches


# -----------------------------------------------------------------------------
# Metric Computation Helper on Spatial Maps
# -----------------------------------------------------------------------------

def extract_stream_metrics_from_patches(
    patch_list: List[Dict[str, Any]],
    stream_key: str,
    neck_mask: np.ndarray,
    crack_mask: np.ndarray,
    bg_mask: np.ndarray,
    tile_size: int = 448
) -> Dict[str, float]:
    """
    Extracts energy (L2 norm), SepMargin, R_norm, Contrast, and Cosine similarity
    from the patch tensor list for a specific stream output (e.g. Y_full, Y_B, Y_S, Y_BS).
    """
    neck_vecs = []
    crack_vecs = []
    bg_vecs = []

    neck_energies = []
    crack_energies = []
    bg_energies = []

    for item in patch_list:
        py, px = item['coords']
        tensor_act = item[stream_key]  # (1, 96, 56, 56) or (1, C, 56, 56)
        C, H_feat, W_feat = tensor_act.shape[1], tensor_act.shape[2], tensor_act.shape[3]

        p_neck = neck_mask[py:py+tile_size, px:px+tile_size]
        p_crack = crack_mask[py:py+tile_size, px:px+tile_size]
        p_bg = bg_mask[py:py+tile_size, px:px+tile_size]

        # Downsample masks to feature resolution (56x56)
        if np.sum(p_neck) > 0:
            m_n = (F.interpolate(torch.from_numpy(p_neck).float().unsqueeze(0).unsqueeze(0),
                                 size=(H_feat, W_feat), mode='nearest').squeeze() > 0)
            if m_n.any():
                feats = tensor_act[0, :, m_n]  # (C, N_pix)
                neck_vecs.append(feats.mean(dim=1))
                neck_energies.extend(torch.norm(feats, p=2, dim=0).tolist())

        if np.sum(p_crack) > 0:
            m_c = (F.interpolate(torch.from_numpy(p_crack).float().unsqueeze(0).unsqueeze(0),
                                 size=(H_feat, W_feat), mode='nearest').squeeze() > 0)
            if m_c.any():
                feats = tensor_act[0, :, m_c]
                crack_vecs.append(feats.mean(dim=1))
                crack_energies.extend(torch.norm(feats, p=2, dim=0).tolist())

        if np.sum(p_bg) > 0:
            m_b = (F.interpolate(torch.from_numpy(p_bg).float().unsqueeze(0).unsqueeze(0),
                                 size=(H_feat, W_feat), mode='nearest').squeeze() > 0)
            if m_b.any():
                feats = tensor_act[0, :, m_b]
                bg_vecs.append(feats.mean(dim=1))
                bg_energies.extend(torch.norm(feats, p=2, dim=0).tolist())

    e_neck = float(np.mean(neck_energies)) if len(neck_energies) > 0 else 0.0
    e_crack = float(np.mean(crack_energies)) if len(crack_energies) > 0 else 0.0
    e_bg = float(np.mean(bg_energies)) if len(bg_energies) > 0 else 0.0

    contrast = float(e_neck / (e_crack + 1e-6))
    sep_margin = float(1.0 - contrast)
    denom = (e_crack - e_bg)
    r_norm = float((e_neck - e_bg) / denom) if abs(denom) > 1e-5 else 0.0

    if len(neck_vecs) > 0 and len(crack_vecs) > 0:
        v_neck = torch.stack(neck_vecs).mean(dim=0)
        v_crack = torch.stack(crack_vecs).mean(dim=0)
        cos_sim = float(F.cosine_similarity(v_neck.unsqueeze(0), v_crack.unsqueeze(0)).item())
    else:
        cos_sim = 0.0

    return {
        'E_neck': e_neck,
        'E_crack': e_crack,
        'E_bg': e_bg,
        'contrast': contrast,
        'sep_margin': sep_margin,
        'R_norm': r_norm,
        'cosine_neck_crack': cos_sim
    }


# -----------------------------------------------------------------------------
# Main Diagnostic Routine
# -----------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Phase 6: T2 Conv1 Stream-Contribution / Fusion Diagnostic")
    print("Direct Weight Slicing & Counterfactual Stream Attenuation")
    print("=" * 80)

    config_path = 'results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml'
    ckpt_path = 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'
    output_dir = 'results/diagnostics/phase6_t2_conv1_stream_contribution'
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

    engine = T2Conv1FactorizationEngine(model, device)

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

    print(f"\nCohort Breakdown:")
    df_cohort = pd.DataFrame(unique_cohort_samples)
    print(df_cohort.groupby(['cohort', 'subgroup']).size())
    print(f"Total Unique Samples: {len(unique_cohort_samples)}")

    # Step 1: Pre-load Images, Masks and compute Reference ROIs & Conv1 Linear Representation Profiling
    print("\nPre-loading images, GT masks, and computing reference Conv1 representations...")
    sample_data = {}
    conv1_representation_records = []
    linear_diffs = []

    for item in tqdm(unique_cohort_samples, desc="Preloading & Conv1 Profiling"):
        stem = item['stem']
        cohort = item['cohort']
        subgroup = item['subgroup']

        img_path = f"datasets/Crack500_ready/val/images/{stem}.jpg"
        mask_path = f"datasets/Crack500_ready/val/masks/{stem}.png"

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        target_bin = (gt_mask > 0).astype(np.uint8)

        # Baseline run (records patch diagnostics Y, Y_B, Y_S, Y_BS)
        engine.set_surgery('baseline', 1.0, 1.0)
        pred_bin, prob_map, patch_diags = engine.run_image(img_rgb)

        if cohort in ['Consensus_7of7', 'Cured_by_D2']:
            neck_mask, crack_mask, bg_mask, pair_records = isolate_bridged_pairs_and_rois(
                pred_bin, target_bin
            )
        else:
            neck_mask, crack_mask, bg_mask, pair_records = isolate_clean_pairs_and_rois(
                target_bin
            )

        sample_data[stem] = {
            'img_rgb': img_rgb,
            'target_bin': target_bin,
            'neck_mask': neck_mask,
            'crack_mask': crack_mask,
            'bg_mask': bg_mask,
            'pair_records': pair_records
        }

        # Track linearity verification difference |Y - Y_BS|
        for p in patch_diags:
            linear_diffs.append(p['max_diff'])

        # Extract metrics for Y_full, Y_B, Y_S, Y_BS
        m_Y_full = extract_stream_metrics_from_patches(patch_diags, 'Y_full', neck_mask, crack_mask, bg_mask)
        m_Y_B = extract_stream_metrics_from_patches(patch_diags, 'Y_B', neck_mask, crack_mask, bg_mask)
        m_Y_S = extract_stream_metrics_from_patches(patch_diags, 'Y_S', neck_mask, crack_mask, bg_mask)
        m_Y_BS = extract_stream_metrics_from_patches(patch_diags, 'Y_BS', neck_mask, crack_mask, bg_mask)

        # Linear Decomposition Interaction metric: I(m) = m(Y_BS) - m(Y_B) - m(Y_S)
        int_R_norm = m_Y_BS['R_norm'] - m_Y_B['R_norm'] - m_Y_S['R_norm']
        int_sep_margin = m_Y_BS['sep_margin'] - m_Y_B['sep_margin'] - m_Y_S['sep_margin']
        int_cosine = m_Y_BS['cosine_neck_crack'] - m_Y_B['cosine_neck_crack'] - m_Y_S['cosine_neck_crack']
        int_contrast = m_Y_BS['contrast'] - m_Y_B['contrast'] - m_Y_S['contrast']

        rep_row = {
            'image_id': stem,
            'cohort': cohort,
            'subgroup': subgroup,
            'max_diff_linearity': max([p['max_diff'] for p in patch_diags]) if patch_diags else 0.0,
            # Y (Baseline Full)
            'Y_R_norm': m_Y_full['R_norm'],
            'Y_sep_margin': m_Y_full['sep_margin'],
            'Y_contrast': m_Y_full['contrast'],
            'Y_cosine': m_Y_full['cosine_neck_crack'],
            'Y_E_neck': m_Y_full['E_neck'],
            'Y_E_crack': m_Y_full['E_crack'],
            'Y_E_bg': m_Y_full['E_bg'],
            # Y_B (Bottleneck only)
            'Y_B_R_norm': m_Y_B['R_norm'],
            'Y_B_sep_margin': m_Y_B['sep_margin'],
            'Y_B_contrast': m_Y_B['contrast'],
            'Y_B_cosine': m_Y_B['cosine_neck_crack'],
            'Y_B_E_neck': m_Y_B['E_neck'],
            'Y_B_E_crack': m_Y_B['E_crack'],
            'Y_B_E_bg': m_Y_B['E_bg'],
            # Y_S (Skip only)
            'Y_S_R_norm': m_Y_S['R_norm'],
            'Y_S_sep_margin': m_Y_S['sep_margin'],
            'Y_S_contrast': m_Y_S['contrast'],
            'Y_S_cosine': m_Y_S['cosine_neck_crack'],
            'Y_S_E_neck': m_Y_S['E_neck'],
            'Y_S_E_crack': m_Y_S['E_crack'],
            'Y_S_E_bg': m_Y_S['E_bg'],
            # Y_BS (Recomposed)
            'Y_BS_R_norm': m_Y_BS['R_norm'],
            'Y_BS_sep_margin': m_Y_BS['sep_margin'],
            'Y_BS_contrast': m_Y_BS['contrast'],
            'Y_BS_cosine': m_Y_BS['cosine_neck_crack'],
            # Interaction Decomposition Metrics
            'I_R_norm': int_R_norm,
            'I_sep_margin': int_sep_margin,
            'I_cosine': int_cosine,
            'I_contrast': int_contrast,
        }
        conv1_representation_records.append(rep_row)

    print(f"\nLinearity Check across all patches: Max diff = {max(linear_diffs):.2e}, Mean diff = {np.mean(linear_diffs):.2e}")
    assert max(linear_diffs) < 1e-4, f"Linearity assertion failed! Max diff: {max(linear_diffs)}"

    df_conv1_rep = pd.DataFrame(conv1_representation_records)
    conv1_rep_csv = os.path.join(output_dir, 't2_conv1_representation_per_sample.csv')
    df_conv1_rep.to_csv(conv1_rep_csv, index=False)
    print(f"Saved: {conv1_rep_csv}")

    # Generate Representation Summary Table
    rep_summary_rows = []
    cohort_groups = [
        ('Consensus_7of7', 'All', df_conv1_rep[df_conv1_rep['cohort'] == 'Consensus_7of7']),
        ('Consensus_7of7', 'Consensus_Resistant', df_conv1_rep[df_conv1_rep['subgroup'] == 'Consensus_Resistant']),
        ('Consensus_7of7', 'Consensus_Sensitive', df_conv1_rep[df_conv1_rep['subgroup'] == 'Consensus_Sensitive']),
        ('Cured_by_D2', 'All', df_conv1_rep[df_conv1_rep['cohort'] == 'Cured_by_D2']),
        ('Clean_Control', 'All', df_conv1_rep[df_conv1_rep['cohort'] == 'Clean_Control']),
    ]

    for c_name, sub_name, sub_df in cohort_groups:
        rep_summary_rows.append({
            'cohort': c_name,
            'subgroup': sub_name,
            'n_samples': len(sub_df),
            'median_Y_R_norm': sub_df['Y_R_norm'].median(),
            'median_Y_B_R_norm': sub_df['Y_B_R_norm'].median(),
            'median_Y_S_R_norm': sub_df['Y_S_R_norm'].median(),
            'median_Y_sep_margin': sub_df['Y_sep_margin'].median(),
            'median_Y_B_sep_margin': sub_df['Y_B_sep_margin'].median(),
            'median_Y_S_sep_margin': sub_df['Y_S_sep_margin'].median(),
            'median_Y_cosine': sub_df['Y_cosine'].median(),
            'median_Y_B_cosine': sub_df['Y_B_cosine'].median(),
            'median_Y_S_cosine': sub_df['Y_S_cosine'].median(),
            'median_I_R_norm': sub_df['I_R_norm'].median(),
            'median_I_sep_margin': sub_df['I_sep_margin'].median(),
            'median_I_cosine': sub_df['I_cosine'].median(),
        })

    df_rep_summary = pd.DataFrame(rep_summary_rows)
    rep_sum_csv = os.path.join(output_dir, 't2_conv1_representation_summary.csv')
    df_rep_summary.to_csv(rep_sum_csv, index=False)
    print(f"Saved: {rep_sum_csv}")

    # Step 2: In-Memory Counterfactual Surgery Conditions
    # 9 Distinct Conditions:
    SURGERY_CONDITIONS = [
        ('Baseline', 'baseline', 1.0, 1.0),
        ('Bottleneck_Only', 'bottleneck_only', 1.0, 0.0),
        ('Skip_Only', 'skip_only', 0.0, 1.0),
        ('S_x0.75', 'attenuate_stream', 1.0, 0.75),
        ('S_x0.50', 'attenuate_stream', 1.0, 0.50),
        ('S_x0.25', 'attenuate_stream', 1.0, 0.25),
        ('B_x0.75', 'attenuate_stream', 0.75, 1.0),
        ('B_x0.50', 'attenuate_stream', 0.50, 1.0),
        ('B_x0.25', 'attenuate_stream', 0.25, 1.0),
    ]

    per_sample_counterfactuals = []
    t0 = time.time()

    print(f"\nBeginning Counterfactual Surgery Evaluation across {len(SURGERY_CONDITIONS)} conditions...")

    for cond_name, mode, a_B, a_S in SURGERY_CONDITIONS:
        print(f"\n---> Running Condition: {cond_name} (alpha_B={a_B:.2f}, alpha_S={a_S:.2f})...")
        engine.set_surgery(mode, a_B, a_S)

        for item in tqdm(unique_cohort_samples, desc=f"{cond_name}"):
            stem = item['stem']
            cohort = item['cohort']
            subgroup = item['subgroup']

            s_info = sample_data[stem]
            img_rgb = s_info['img_rgb']
            target_bin = s_info['target_bin']
            neck_mask = s_info['neck_mask']
            crack_mask = s_info['crack_mask']
            bg_mask = s_info['bg_mask']

            pred_bin, prob_map, _ = engine.run_image(img_rgb, tile_size=448, batch_size=4)
            topo = compute_topology_metrics(pred_bin, target_bin)

            neck_area = int(np.sum(neck_mask))
            crack_area = int(np.sum(crack_mask))
            bg_area = int(np.sum(bg_mask))

            prob_neck = float(np.mean(prob_map[neck_mask == 1])) if neck_area > 0 else 0.0
            prob_crack = float(np.mean(prob_map[crack_mask == 1])) if crack_area > 0 else 0.0
            prob_bg = float(np.mean(prob_map[bg_mask == 1])) if bg_area > 0 else 0.0

            per_sample_counterfactuals.append({
                'image_id': stem,
                'cohort': cohort,
                'subgroup': subgroup,
                'condition': cond_name,
                'alpha_B': a_B,
                'alpha_S': a_S,
                'has_bridge': bool(topo['bridge_events'] > 0),
                'bridge_events': topo['bridge_events'],
                'merged_gt_components': topo['merged_gt_components'],
                'has_breakage': bool(topo['fragmented_gt_components'] > 0),
                'fragmented_gt_components': topo['fragmented_gt_components'],
                'spurious_islands': topo['spurious_island_count'],
                'dice': topo['dice'],
                'precision': topo['precision'],
                'recall': topo['recall'],
                'cldice': topo['cldice'],
                'connector_prob_neck': prob_neck,
                'connector_prob_crack': prob_crack,
                'connector_prob_bg': prob_bg,
            })

    engine.cleanup()
    t1 = time.time()
    print(f"\nAll counterfactual conditions evaluated in {t1 - t0:.1f}s.")

    # Integrity verification
    final_param_hash = compute_model_param_hash(model)
    final_file_hash = compute_file_hash(ckpt_path)

    print(f"\nVerifying Bitwise State Invariance:")
    print(f"  Initial Param Hash: {init_param_hash[:16]}... | Final: {final_param_hash[:16]}...")
    print(f"  Initial File Hash:  {init_file_hash[:16]}... | Final: {final_file_hash[:16]}...")
    assert init_param_hash == final_param_hash, "HARD LOCK VIOLATION: Model parameters were modified in memory!"
    assert init_file_hash == final_file_hash, "HARD LOCK VIOLATION: Checkpoint file on disk was modified!"
    print("Bitwise Invariance Check: 100% PASSED.")

    df_cf = pd.DataFrame(per_sample_counterfactuals)
    cf_csv = os.path.join(output_dir, 't2_conv1_counterfactual_per_sample.csv')
    df_cf.to_csv(cf_csv, index=False)
    print(f"Saved: {cf_csv}")

    # Generate Counterfactual Matrix across Subgroups
    base_df = df_cf[df_cf['condition'] == 'Baseline'].set_index('image_id')

    subgroup_matrix_rows = []
    for cond_name, mode, a_B, a_S in SURGERY_CONDITIONS:
        c_df = df_cf[df_cf['condition'] == cond_name].set_index('image_id')

        # Resistant (N=82)
        r_base = base_df[base_df['subgroup'] == 'Consensus_Resistant']
        r_cond = c_df[c_df['subgroup'] == 'Consensus_Resistant']
        r_bridges = int(r_cond['has_bridge'].sum())
        r_cured = int((r_base['has_bridge'] & (~r_cond['has_bridge'])).sum())
        r_breakage = int(r_cond['has_breakage'].sum())
        r_recall = float(r_cond['recall'].median())
        r_conn = float(r_cond['connector_prob_neck'].median())

        # Sensitive (N=19)
        s_base = base_df[base_df['subgroup'] == 'Consensus_Sensitive']
        s_cond = c_df[c_df['subgroup'] == 'Consensus_Sensitive']
        s_bridges = int(s_cond['has_bridge'].sum())
        s_cured = int((s_base['has_bridge'] & (~s_cond['has_bridge'])).sum())
        s_breakage = int(s_cond['has_breakage'].sum())

        # Clean Control (N=56)
        ctrl_base = base_df[base_df['cohort'] == 'Clean_Control']
        ctrl_cond = c_df[c_df['cohort'] == 'Clean_Control']
        ctrl_bridges_created = int(ctrl_cond['has_bridge'].sum())
        ctrl_breakage = int(ctrl_cond['has_breakage'].sum())

        # D2 (N=7)
        d2_cond = c_df[c_df['cohort'] == 'Cured_by_D2']
        d2_bridges = int(d2_cond['has_bridge'].sum())

        subgroup_matrix_rows.append({
            'condition': cond_name,
            'alpha_B': a_B,
            'alpha_S': a_S,
            'Resistant_Bridges': f"{r_bridges}/82",
            'Resistant_Cured': f"{r_cured}/82 ({r_cured/82*100:.1f}%)",
            'Resistant_Breakage': f"{r_breakage}/82",
            'Resistant_Recall': f"{r_recall:.4f}",
            'Resistant_Med_ConnProb': f"{r_conn:.4f}",
            'Sensitive_Bridges': f"{s_bridges}/19",
            'Sensitive_Cured': f"{s_cured}/19 ({s_cured/19*100:.1f}%)",
            'Sensitive_Breakage': f"{s_breakage}/19",
            'Control_Bridges_Created': f"{ctrl_bridges_created}/56",
            'Control_Breakage': f"{ctrl_breakage}/56",
            'D2_Bridges': f"{d2_bridges}/7",
        })

    df_subgroup_matrix = pd.DataFrame(subgroup_matrix_rows)
    matrix_csv = os.path.join(output_dir, 't2_conv1_counterfactual_subgroup_matrix.csv')
    df_subgroup_matrix.to_csv(matrix_csv, index=False)
    print(f"Saved: {matrix_csv}")
    print("\nCounterfactual Subgroup Matrix:")
    print(df_subgroup_matrix[['condition', 'alpha_B', 'alpha_S', 'Resistant_Bridges', 'Resistant_Cured', 'Resistant_Recall', 'D2_Bridges']])

    # Step 3: Visualization Figures
    print("\nGenerating Diagnostic Figures...")

    # Figure 1: Conv1 Linear Representation Comparison (Y vs Y_B vs Y_S)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sub_res = df_conv1_rep[df_conv1_rep['subgroup'] == 'Consensus_Resistant']
    sub_sen = df_conv1_rep[df_conv1_rep['subgroup'] == 'Consensus_Sensitive']
    sub_d2 = df_conv1_rep[df_conv1_rep['cohort'] == 'Cured_by_D2']

    labels = ['Resistant (N=82)', 'Sensitive (N=19)', 'Cured by D2 (N=7)']
    x = np.arange(len(labels))
    width = 0.25

    # Plot 1: R_norm
    ax = axes[0]
    y_vals = [sub_res['Y_R_norm'].median(), sub_sen['Y_R_norm'].median(), sub_d2['Y_R_norm'].median()]
    yb_vals = [sub_res['Y_B_R_norm'].median(), sub_sen['Y_B_R_norm'].median(), sub_d2['Y_B_R_norm'].median()]
    ys_vals = [sub_res['Y_S_R_norm'].median(), sub_sen['Y_S_R_norm'].median(), sub_d2['Y_S_R_norm'].median()]

    ax.bar(x - width, yb_vals, width, label='Y_B (Bottleneck)', color='#2ca02c', alpha=0.85)
    ax.bar(x, ys_vals, width, label='Y_S (Skip)', color='#1f77b4', alpha=0.85)
    ax.bar(x + width, y_vals, width, label='Y (Full Fusion)', color='#d62728', alpha=0.85)
    ax.axhline(0.50, color='gray', linestyle='--', linewidth=1, label='Ambiguity Threshold (0.50)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Median Normalized Ratio R_norm')
    ax.set_title('Conv1 Output R_norm by Contribution')
    ax.legend(fontsize=8.5)
    ax.grid(True, linestyle=':', alpha=0.5)

    # Plot 2: SepMargin
    ax = axes[1]
    y_sep = [sub_res['Y_sep_margin'].median(), sub_sen['Y_sep_margin'].median(), sub_d2['Y_sep_margin'].median()]
    yb_sep = [sub_res['Y_B_sep_margin'].median(), sub_sen['Y_B_sep_margin'].median(), sub_d2['Y_B_sep_margin'].median()]
    ys_sep = [sub_res['Y_S_sep_margin'].median(), sub_sen['Y_S_sep_margin'].median(), sub_d2['Y_S_sep_margin'].median()]

    ax.bar(x - width, yb_sep, width, label='Y_B (Bottleneck)', color='#2ca02c', alpha=0.85)
    ax.bar(x, ys_sep, width, label='Y_S (Skip)', color='#1f77b4', alpha=0.85)
    ax.bar(x + width, y_sep, width, label='Y (Full Fusion)', color='#d62728', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Median Separation Margin (1 - E_neck/E_crack)')
    ax.set_title('Conv1 Output Separation Margin')
    ax.legend(fontsize=8.5)
    ax.grid(True, linestyle=':', alpha=0.5)

    # Plot 3: Cosine Similarity
    ax = axes[2]
    y_cos = [sub_res['Y_cosine'].median(), sub_sen['Y_cosine'].median(), sub_d2['Y_cosine'].median()]
    yb_cos = [sub_res['Y_B_cosine'].median(), sub_sen['Y_B_cosine'].median(), sub_d2['Y_B_cosine'].median()]
    ys_cos = [sub_res['Y_S_cosine'].median(), sub_sen['Y_S_cosine'].median(), sub_d2['Y_S_cosine'].median()]

    ax.bar(x - width, yb_cos, width, label='Y_B (Bottleneck)', color='#2ca02c', alpha=0.85)
    ax.bar(x, ys_cos, width, label='Y_S (Skip)', color='#1f77b4', alpha=0.85)
    ax.bar(x + width, y_cos, width, label='Y (Full Fusion)', color='#d62728', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Median Cosine Similarity (Neck vs Crack)')
    ax.set_title('Conv1 Neck vs Crack Cosine Similarity')
    ax.legend(fontsize=8.5)
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, 't2_conv1_stream_comparison.png')
    plt.savefig(fig1_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {fig1_path}")

    # Figure 2: Counterfactual Attenuation Sweep (Skip vs Bottleneck)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Parse bridge and breakage numbers for Resistant
    alphas = [0.0, 0.25, 0.50, 0.75, 1.0]

    # Skip attenuation curve (alpha_B = 1.0, alpha_S sweeping)
    skip_conds = ['Bottleneck_Only', 'S_x0.25', 'S_x0.50', 'S_x0.75', 'Baseline']
    skip_bridges = [int(df_subgroup_matrix[df_subgroup_matrix['condition'] == c]['Resistant_Bridges'].values[0].split('/')[0]) for c in skip_conds]
    skip_breakage = [int(df_subgroup_matrix[df_subgroup_matrix['condition'] == c]['Resistant_Breakage'].values[0].split('/')[0]) for c in skip_conds]
    skip_recall = [float(df_subgroup_matrix[df_subgroup_matrix['condition'] == c]['Resistant_Recall'].values[0]) for c in skip_conds]

    # Bottleneck attenuation curve (alpha_S = 1.0, alpha_B sweeping)
    btn_conds = ['Skip_Only', 'B_x0.25', 'B_x0.50', 'B_x0.75', 'Baseline']
    btn_bridges = [int(df_subgroup_matrix[df_subgroup_matrix['condition'] == c]['Resistant_Bridges'].values[0].split('/')[0]) for c in btn_conds]
    btn_breakage = [int(df_subgroup_matrix[df_subgroup_matrix['condition'] == c]['Resistant_Breakage'].values[0].split('/')[0]) for c in btn_conds]
    btn_recall = [float(df_subgroup_matrix[df_subgroup_matrix['condition'] == c]['Resistant_Recall'].values[0]) for c in btn_conds]

    ax = axes[0]
    ax.plot(alphas, skip_bridges, marker='o', linewidth=2, color='#1f77b4', label='Attenuate Skip (B=1.0)')
    ax.plot(alphas, btn_bridges, marker='s', linewidth=2, color='#2ca02c', label='Attenuate Bottleneck (S=1.0)')
    ax.set_xlabel('Stream Scaling Factor (alpha)')
    ax.set_ylabel('Resistant False Bridges (/82)')
    ax.set_title('False Bridge Reduction vs Stream Attenuation')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)

    ax = axes[1]
    ax.plot(alphas, skip_breakage, marker='o', linewidth=2, linestyle='--', color='#1f77b4', label='Breakage: Attenuate Skip')
    ax.plot(alphas, btn_breakage, marker='s', linewidth=2, linestyle='--', color='#2ca02c', label='Breakage: Attenuate Bottleneck')
    ax.plot(alphas, np.array(skip_recall) * 82, marker='^', linewidth=1.5, color='#1f77b4', alpha=0.5, label='Recall (scaled): Skip')
    ax.plot(alphas, np.array(btn_recall) * 82, marker='v', linewidth=1.5, color='#2ca02c', alpha=0.5, label='Recall (scaled): Bottleneck')
    ax.set_xlabel('Stream Scaling Factor (alpha)')
    ax.set_ylabel('Crack Breakage Events (/82)')
    ax.set_title('Crack Breakage & Continuity Cost')
    ax.legend(fontsize=8.5)
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, 't2_conv1_attenuation_sweep.png')
    plt.savefig(fig2_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {fig2_path}")

    print("\n" + "=" * 80)
    print("Phase 6 T2 Conv1 Stream-Contribution Diagnostic Completed Successfully.")
    print("=" * 80)


if __name__ == '__main__':
    main()
