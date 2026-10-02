#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_t2_channel_relu_interaction_diagnostic.py

Phase 6: Channel-wise ReLU Interaction & Spatial Selectivity Diagnostic
Target Locus: Decoder Block 1 Conv1 post-ReLU (96 channels at 56x56 resolution)

Scientific Objective:
Determine whether the non-linear interaction ignited at Conv1 post-ReLU:
1. Is driven by a sparse subset of bridge-selective channels (Sparse Mechanism),
2. Is diffusely distributed across many channels (Diffuse Mechanism),
3. Is spatially non-selective (amplifying neck and crack equally), or
4. Exhibits distinct channel signatures across cohorts (Cohort-Specific Mechanism).

Factorial 2x2 Decomposition at post-ReLU:
  For each channel c in 0..95 and spatial coordinate (x, y):
  I_c(x, y) = R_c(Full)(x, y) - R_c(B-only)(x, y) - R_c(S-only)(x, y) + R_c(Zero)(x, y)
  where R_c is activation strictly post-ReLU.

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
from scipy.stats import pearsonr, spearmanr
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
# Channel-wise ReLU Interaction Engine
# -----------------------------------------------------------------------------

class ChannelReLUInteractionEngine:
    """
    Manages in-memory pre-hook surgery at Decoder Block 1 Conv1 and captures
    exact channel-wise spatial activations at Conv1 post-ReLU.
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

        self.dec_b1 = model.decoder.decoder_blocks[1]
        self.conv1_seq = self.dec_b1.conv1
        self.conv1_2 = self.conv1_seq[2]  # ReLU(inplace=True)

        self.mode: str = 'full'
        self.pre_hook_handle = None
        self.post_relu_handle = None

        self.current_patch_post_relu: torch.Tensor = None
        self._register_hooks()

    def _register_hooks(self):
        # 1. Pre-hook for input condition
        def conv1_pre_hook(module, args):
            x = args[0]
            if self.mode == 'full':
                return args
            x_mod = x.clone()
            if self.mode == 'b_only':
                x_mod[:, 192:, :, :] = 0.0
            elif self.mode == 's_only':
                x_mod[:, :192, :, :] = 0.0
            elif self.mode == 'zero':
                x_mod[:, :, :, :] = 0.0
            return (x_mod,)

        self.pre_hook_handle = self.conv1_seq.register_forward_pre_hook(conv1_pre_hook)

        # 2. Hook to capture post-ReLU tensor of shape (B, 96, 56, 56)
        def hook_post_relu(module, inp, out):
            self.current_patch_post_relu = out.detach().clone().cpu()

        self.post_relu_handle = self.conv1_2.register_forward_hook(hook_post_relu)

    def set_mode(self, mode: str):
        assert mode in ['full', 'b_only', 's_only', 'zero']
        self.mode = mode
        self.current_patch_post_relu = None

    def cleanup(self):
        if self.pre_hook_handle:
            self.pre_hook_handle.remove()
            self.pre_hook_handle = None
        if self.post_relu_handle:
            self.post_relu_handle.remove()
            self.post_relu_handle = None
        self.set_mode('full')

    def run_image(
        self,
        image_rgb: np.ndarray,
        tile_size: int = 448,
        batch_size: int = 4
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Runs Setting A tiling inference and returns binary prediction, probability map, and patch post-ReLU tensors."""
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

                self.current_patch_post_relu = None
                logits = self.model(batch_pt)
                if logits.shape[2:] != (tile_size, tile_size):
                    logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)

                captured_acts = self.current_patch_post_relu  # (B, 96, 56, 56)

                for b_idx, (py, px) in enumerate(batch_coords):
                    final_logits[py:py+tile_size, px:px+tile_size] = logits[b_idx].squeeze().cpu().numpy()
                    patch_records.append({
                        'coords': (py, px),
                        'post_relu': captured_acts[b_idx:b_idx+1],  # (1, 96, 56, 56)
                    })

        logits_cropped = final_logits[:H, :W]
        prob_cropped = 1.0 / (1.0 + np.exp(-logits_cropped))
        pred_bin = (logits_cropped > 0.0).astype(np.uint8)

        return pred_bin, prob_cropped, patch_records


# -----------------------------------------------------------------------------
# Main Diagnostic Routine
# -----------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Phase 6: Channel-wise ReLU Interaction & Spatial Selectivity Diagnostic")
    print("Direct 96-Channel Spatial Interaction Profiling at Conv1 post-ReLU")
    print("=" * 80)

    config_path = 'results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml'
    ckpt_path = 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'
    output_dir = 'results/diagnostics/phase6_t2_channel_relu'
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

    engine = ChannelReLUInteractionEngine(model, device)

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

    # Pre-load Images, Masks and Reference ROIs
    print("\nPre-loading images, GT masks, and computing reference ROIs...")
    sample_data = {}
    for item in tqdm(unique_cohort_samples, desc="Preloading"):
        stem = item['stem']
        cohort = item['cohort']
        img_path = f"datasets/Crack500_ready/val/images/{stem}.jpg"
        mask_path = f"datasets/Crack500_ready/val/masks/{stem}.png"

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        target_bin = (gt_mask > 0).astype(np.uint8)

        engine.set_mode('full')
        pred_bin, _, _ = engine.run_image(img_rgb)

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

    # Step 1: Run 4 Factorial Conditions and Collect Post-ReLU Activations
    CONDITIONS = ['full', 'b_only', 's_only', 'zero']
    condition_patch_acts = {c: {} for c in CONDITIONS}
    condition_sample_outputs = {c: {} for c in CONDITIONS}

    t0 = time.time()
    print(f"\nEvaluating 4 Factorial Conditions across {len(unique_cohort_samples)} samples...")

    for cond in CONDITIONS:
        print(f"\n---> Condition: {cond}...")
        engine.set_mode(cond)

        for item in tqdm(unique_cohort_samples, desc=f"Evaluating {cond}"):
            stem = item['stem']
            s_info = sample_data[stem]
            img_rgb = s_info['img_rgb']
            target_bin = s_info['target_bin']
            neck_mask = s_info['neck_mask']

            pred_bin, prob_map, patch_records = engine.run_image(img_rgb, tile_size=448, batch_size=4)
            topo = compute_topology_metrics(pred_bin, target_bin)

            neck_area = int(np.sum(neck_mask))
            prob_neck = float(np.mean(prob_map[neck_mask == 1])) if neck_area > 0 else 0.0

            condition_patch_acts[cond][stem] = patch_records
            condition_sample_outputs[cond][stem] = {
                'has_bridge': bool(topo['bridge_events'] > 0),
                'bridge_events': topo['bridge_events'],
                'prob_neck': prob_neck,
                'recall': topo['recall'],
                'dice': topo['dice'],
            }

    engine.cleanup()
    t1 = time.time()
    print(f"\nAll 4 conditions evaluated in {t1 - t0:.1f}s.")

    # Integrity verification
    final_param_hash = compute_model_param_hash(model)
    final_file_hash = compute_file_hash(ckpt_path)

    print(f"\nVerifying Bitwise State Invariance:")
    print(f"  Initial Param Hash: {init_param_hash[:16]}... | Final: {final_param_hash[:16]}...")
    print(f"  Initial File Hash:  {init_file_hash[:16]}... | Final: {final_file_hash[:16]}...")
    assert init_param_hash == final_param_hash, "HARD LOCK VIOLATION: Model parameters modified in memory!"
    assert init_file_hash == final_file_hash, "HARD LOCK VIOLATION: Checkpoint file modified on disk!"
    print("Bitwise Invariance Check: 100% PASSED.")

    # Step 2: Compute Channel-Wise Spatial Interaction per Sample
    # I_c(x, y) = R_c(Full) - R_c(B-only) - R_c(S-only) + R_c(Zero)
    print("\nComputing Channel-wise Spatial Interaction across 96 channels for each sample...")
    tile_size = 448
    num_channels = 96

    channel_per_sample_records = []

    for item in tqdm(unique_cohort_samples, desc="Channel-wise Analysis"):
        stem = item['stem']
        cohort = item['cohort']
        subgroup = item['subgroup']

        s_info = sample_data[stem]
        neck_mask = s_info['neck_mask']
        crack_mask = s_info['crack_mask']

        patches_full = condition_patch_acts['full'][stem]
        patches_b = condition_patch_acts['b_only'][stem]
        patches_s = condition_patch_acts['s_only'][stem]
        patches_z = condition_patch_acts['zero'][stem]

        # Accumulators per channel for neck and crack pixels
        neck_pixels_by_c = [[] for _ in range(num_channels)]
        crack_pixels_by_c = [[] for _ in range(num_channels)]

        for p_idx in range(len(patches_full)):
            py, px = patches_full[p_idx]['coords']

            # Post-ReLU tensors: shape (1, 96, 56, 56)
            r_full = patches_full[p_idx]['post_relu'][0]
            r_b = patches_b[p_idx]['post_relu'][0]
            r_s = patches_s[p_idx]['post_relu'][0]
            r_z = patches_z[p_idx]['post_relu'][0]

            # Exact spatial pixel-wise interaction tensor: (96, 56, 56)
            i_patch = r_full - r_b - r_s + r_z

            p_neck = neck_mask[py:py+tile_size, px:px+tile_size]
            p_crack = crack_mask[py:py+tile_size, px:px+tile_size]

            # Downsample masks to 56x56
            m_n = (F.interpolate(torch.from_numpy(p_neck).float().unsqueeze(0).unsqueeze(0),
                                 size=(56, 56), mode='nearest').squeeze() > 0).numpy()
            m_c = (F.interpolate(torch.from_numpy(p_crack).float().unsqueeze(0).unsqueeze(0),
                                 size=(56, 56), mode='nearest').squeeze() > 0).numpy()
            m_c = m_c & (~m_n)  # Clean non-overlapping spatial partition

            if m_n.any():
                for c in range(num_channels):
                    neck_pixels_by_c[c].extend(i_patch[c, m_n].numpy().tolist())

            if m_c.any():
                for c in range(num_channels):
                    crack_pixels_by_c[c].extend(i_patch[c, m_c].numpy().tolist())

        # Sample bridge and connector status
        full_out = condition_sample_outputs['full'][stem]
        has_bridge = 1 if full_out['has_bridge'] else 0
        p_neck_full = full_out['prob_neck']

        for c in range(num_channels):
            n_pix = np.array(neck_pixels_by_c[c], dtype=np.float32)
            c_pix = np.array(crack_pixels_by_c[c], dtype=np.float32)

            med_neck = float(np.median(n_pix)) if len(n_pix) > 0 else 0.0
            mean_neck = float(np.mean(n_pix)) if len(n_pix) > 0 else 0.0
            frac_pos_neck = float(np.mean(n_pix > 0)) if len(n_pix) > 0 else 0.0

            med_crack = float(np.median(c_pix)) if len(c_pix) > 0 else 0.0
            mean_crack = float(np.mean(c_pix)) if len(c_pix) > 0 else 0.0
            frac_pos_crack = float(np.mean(c_pix > 0)) if len(c_pix) > 0 else 0.0

            delta_med = med_neck - med_crack
            delta_mean = mean_neck - mean_crack

            channel_per_sample_records.append({
                'image_id': stem,
                'cohort': cohort,
                'subgroup': subgroup,
                'channel': c,
                'med_I_neck': med_neck,
                'mean_I_neck': mean_neck,
                'med_I_crack': med_crack,
                'mean_I_crack': mean_crack,
                'delta_I_med': delta_med,
                'delta_I_mean': delta_mean,
                'frac_pos_neck': frac_pos_neck,
                'frac_pos_crack': frac_pos_crack,
                'has_bridge': has_bridge,
                'prob_neck_full': p_neck_full,
            })

    df_sample_channels = pd.DataFrame(channel_per_sample_records)
    sample_csv = os.path.join(output_dir, 'channel_relu_interaction_per_sample.csv')
    df_sample_channels.to_csv(sample_csv, index=False)
    print(f"Saved: {sample_csv}")

    # Step 3: Compute Summary Statistics per Channel and Cohort
    cohort_groups = [
        ('Consensus_7of7', 'Consensus_Resistant', df_sample_channels[df_sample_channels['subgroup'] == 'Consensus_Resistant']),
        ('Consensus_7of7', 'Consensus_Sensitive', df_sample_channels[df_sample_channels['subgroup'] == 'Consensus_Sensitive']),
        ('Cured_by_D2', 'All', df_sample_channels[df_sample_channels['cohort'] == 'Cured_by_D2']),
        ('Clean_Control', 'All', df_sample_channels[df_sample_channels['cohort'] == 'Clean_Control']),
    ]

    summary_records = []
    cohort_ranked_channels = {}

    for c_name, sub_name, sub_df in cohort_groups:
        key = f"{c_name}_{sub_name}"
        cohort_summary = []

        for c in range(num_channels):
            c_df = sub_df[sub_df['channel'] == c]

            med_I_neck = float(c_df['med_I_neck'].median())
            q25_I_neck = float(c_df['med_I_neck'].quantile(0.25))
            q75_I_neck = float(c_df['med_I_neck'].quantile(0.75))

            med_I_crack = float(c_df['med_I_crack'].median())
            q25_I_crack = float(c_df['med_I_crack'].quantile(0.25))
            q75_I_crack = float(c_df['med_I_crack'].quantile(0.75))

            med_delta = float(c_df['delta_I_med'].median())
            mean_delta = float(c_df['delta_I_med'].mean())
            q25_delta = float(c_df['delta_I_med'].quantile(0.25))
            q75_delta = float(c_df['delta_I_med'].quantile(0.75))

            med_frac_neck = float(c_df['frac_pos_neck'].median())
            med_frac_crack = float(c_df['frac_pos_crack'].median())

            # Fraction of samples where this channel has delta_I_med > 0 and med_I_neck > 0
            sample_coverage = float(((c_df['delta_I_med'] > 0) & (c_df['med_I_neck'] > 0)).mean())

            row = {
                'cohort': c_name,
                'subgroup': sub_name,
                'channel': c,
                'median_I_neck': med_I_neck,
                'q25_I_neck': q25_I_neck,
                'q75_I_neck': q75_I_neck,
                'median_I_crack': med_I_crack,
                'q25_I_crack': q25_I_crack,
                'q75_I_crack': q75_I_crack,
                'median_delta_I': med_delta,
                'mean_delta_I': mean_delta,
                'q25_delta_I': q25_delta,
                'q75_delta_I': q75_delta,
                'median_frac_pos_neck': med_frac_neck,
                'median_frac_pos_crack': med_frac_crack,
                'sample_coverage': sample_coverage,
            }
            summary_records.append(row)
            cohort_summary.append(row)

        cohort_df = pd.DataFrame(cohort_summary)
        cohort_ranked_channels[key] = cohort_df.sort_values(by='median_delta_I', ascending=False).reset_index(drop=True)

    df_summary = pd.DataFrame(summary_records)
    summary_csv = os.path.join(output_dir, 'channel_relu_interaction_summary.csv')
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    # Step 4: Cumulative Share Analysis (Sparse vs Diffuse Diagnostic)
    print("\nComputing Cumulative Share of Positive Interaction across Channels...")
    cum_share_records = []
    concentration_thresholds = {}

    for key, ranked_df in cohort_ranked_channels.items():
        # Positive delta_I values
        pos_values = ranked_df['median_delta_I'].clip(lower=0.0).values
        total_pos = np.sum(pos_values)

        if total_pos > 1e-6:
            cum_share = np.cumsum(pos_values) / total_pos
        else:
            cum_share = np.zeros_like(pos_values)

        k_50 = int(np.searchsorted(cum_share, 0.50) + 1) if total_pos > 1e-6 else 0
        k_80 = int(np.searchsorted(cum_share, 0.80) + 1) if total_pos > 1e-6 else 0
        k_90 = int(np.searchsorted(cum_share, 0.90) + 1) if total_pos > 1e-6 else 0

        concentration_thresholds[key] = {
            'k_50': k_50,
            'k_80': k_80,
            'k_90': k_90,
            'total_positive_channels': int(np.sum(pos_values > 0)),
            'total_pos_interaction': float(total_pos)
        }

        for rank_idx, row in ranked_df.iterrows():
            cum_share_records.append({
                'cohort_key': key,
                'rank': rank_idx + 1,
                'channel': int(row['channel']),
                'median_delta_I': float(row['median_delta_I']),
                'median_I_neck': float(row['median_I_neck']),
                'median_I_crack': float(row['median_I_crack']),
                'cumulative_share': float(cum_share[rank_idx]),
            })

    df_cum_share = pd.DataFrame(cum_share_records)
    cum_csv = os.path.join(output_dir, 'channel_cumulative_share_analysis.csv')
    df_cum_share.to_csv(cum_csv, index=False)
    print(f"Saved: {cum_csv}")

    print("\nConcentration Thresholds (Number of Channels for 50%, 80%, 90% Share):")
    for key, th in concentration_thresholds.items():
        print(f"  [{key}]: 50% = {th['k_50']} ch, 80% = {th['k_80']} ch, 90% = {th['k_90']} ch | Total Positive = {th['total_positive_channels']}/96")

    # Step 5: Cohort Channel Overlap Analysis
    print("\nAnalyzing Channel Overlap between Cohorts...")
    res_top10 = set(cohort_ranked_channels['Consensus_7of7_Consensus_Resistant']['channel'].iloc[:10].tolist())
    res_top20 = set(cohort_ranked_channels['Consensus_7of7_Consensus_Resistant']['channel'].iloc[:20].tolist())

    sen_top10 = set(cohort_ranked_channels['Consensus_7of7_Consensus_Sensitive']['channel'].iloc[:10].tolist())
    sen_top20 = set(cohort_ranked_channels['Consensus_7of7_Consensus_Sensitive']['channel'].iloc[:20].tolist())

    d2_top10 = set(cohort_ranked_channels['Cured_by_D2_All']['channel'].iloc[:10].tolist())
    d2_top20 = set(cohort_ranked_channels['Cured_by_D2_All']['channel'].iloc[:20].tolist())

    overlap_records = [
        {
            'comparison': 'Resistant vs Sensitive',
            'top10_intersection_count': len(res_top10 & sen_top10),
            'top10_jaccard': len(res_top10 & sen_top10) / len(res_top10 | sen_top10),
            'top10_shared_channels': sorted(list(res_top10 & sen_top10)),
            'top20_intersection_count': len(res_top20 & sen_top20),
            'top20_jaccard': len(res_top20 & sen_top20) / len(res_top20 | sen_top20),
            'top20_shared_channels': sorted(list(res_top20 & sen_top20)),
        },
        {
            'comparison': 'Resistant vs D2-Cured',
            'top10_intersection_count': len(res_top10 & d2_top10),
            'top10_jaccard': len(res_top10 & d2_top10) / len(res_top10 | d2_top10),
            'top10_shared_channels': sorted(list(res_top10 & d2_top10)),
            'top20_intersection_count': len(res_top20 & d2_top20),
            'top20_jaccard': len(res_top20 & d2_top20) / len(res_top20 | d2_top20),
            'top20_shared_channels': sorted(list(res_top20 & d2_top20)),
        },
        {
            'comparison': 'Sensitive vs D2-Cured',
            'top10_intersection_count': len(sen_top10 & d2_top10),
            'top10_jaccard': len(sen_top10 & d2_top10) / len(sen_top10 | d2_top10),
            'top10_shared_channels': sorted(list(sen_top10 & d2_top10)),
            'top20_intersection_count': len(sen_top20 & d2_top20),
            'top20_jaccard': len(sen_top20 & d2_top20) / len(sen_top20 | d2_top20),
            'top20_shared_channels': sorted(list(sen_top20 & d2_top20)),
        },
    ]

    df_overlap = pd.DataFrame(overlap_records)
    overlap_csv = os.path.join(output_dir, 'channel_cohort_overlap_analysis.csv')
    df_overlap.to_csv(overlap_csv, index=False)
    print(f"Saved: {overlap_csv}")
    print("\nCohort Overlap Summary:")
    print(df_overlap[['comparison', 'top10_intersection_count', 'top10_jaccard', 'top20_intersection_count', 'top20_jaccard']])

    # Step 6: Correlation with Final Connector Probability & Bridge State
    print("\nComputing Channel Correlation with Final Bridge Status...")
    df_c7 = df_sample_channels[df_sample_channels['cohort'] == 'Consensus_7of7']
    corr_records = []

    for c in range(num_channels):
        c_sub = df_c7[df_c7['channel'] == c]
        x_delta = c_sub['delta_I_med'].values
        y_prob = c_sub['prob_neck_full'].values
        y_bridge = c_sub['has_bridge'].values

        r_prob, p_prob = pearsonr(x_delta, y_prob) if np.std(x_delta) > 1e-6 and np.std(y_prob) > 1e-6 else (0.0, 1.0)
        s_prob, sp_p = spearmanr(x_delta, y_prob) if np.std(x_delta) > 1e-6 and np.std(y_prob) > 1e-6 else (0.0, 1.0)

        corr_records.append({
            'channel': c,
            'pearson_r_prob': float(r_prob),
            'pearson_p_prob': float(p_prob),
            'spearman_rho_prob': float(s_prob),
            'spearman_p_prob': float(sp_p),
        })

    df_corr = pd.DataFrame(corr_records)
    corr_csv = os.path.join(output_dir, 'channel_bridge_correlation.csv')
    df_corr.to_csv(corr_csv, index=False)
    print(f"Saved: {corr_csv}")

    # Step 7: Visualizations
    print("\nGenerating Diagnostic Figures...")

    # Figure 1: Channel ReLU Interaction Heatmap (96 channels x Cohorts)
    fig, ax = plt.subplots(figsize=(16, 4.5))
    heatmap_data = []
    cohort_keys = [
        'Consensus_7of7_Consensus_Resistant',
        'Consensus_7of7_Consensus_Sensitive',
        'Cured_by_D2_All',
        'Clean_Control_All'
    ]
    cohort_names = ['Resistant (N=82)', 'Sensitive (N=19)', 'Cured by D2 (N=7)', 'Clean Control (N=56)']

    for k in cohort_keys:
        sub_sum = df_summary[(df_summary['cohort'] + '_' + df_summary['subgroup']) == k].sort_values(by='channel')
        heatmap_data.append(sub_sum['median_delta_I'].values)

    heatmap_mat = np.array(heatmap_data)  # (4, 96)
    vmax = np.percentile(np.abs(heatmap_mat), 98)
    im = ax.imshow(heatmap_mat, aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)

    ax.set_yticks(np.arange(len(cohort_names)))
    ax.set_yticklabels(cohort_names, fontsize=10)
    ax.set_xlabel('Decoder Block 1 Conv1 Post-ReLU Channel Index (0..95)', fontsize=10.5)
    ax.set_title('Channel-wise Spatial Selectivity: Median Delta_I_c = I_c(neck) - I_c(crack)', fontsize=11.5)

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.22, shrink=0.6)
    cbar.set_label('Median Delta_I_c (Positive = Selectively Amplifies Bridge Corridor)', fontsize=9.5)

    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, 'channel_relu_interaction_heatmap.png')
    plt.savefig(fig1_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {fig1_path}")

    # Figure 2: Channel Interaction Rank Curve (Sparse vs Diffuse)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    colors = {'Consensus_7of7_Consensus_Resistant': '#d62728',
              'Consensus_7of7_Consensus_Sensitive': '#1f77b4',
              'Cured_by_D2_All': '#2ca02c',
              'Clean_Control_All': '#7f7f7f'}

    for k in cohort_keys:
        k_df = df_cum_share[df_cum_share['cohort_key'] == k].sort_values(by='rank')
        label_str = k.replace('Consensus_7of7_', '').replace('_All', '')
        ax.plot(k_df['rank'], k_df['cumulative_share'] * 100, label=label_str, color=colors[k], linewidth=2)

    ax.axhline(50, color='gray', linestyle=':', alpha=0.7)
    ax.axhline(80, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(90, color='gray', linestyle='-.', alpha=0.7)
    ax.text(2, 51, '50% Share', fontsize=8.5, color='gray')
    ax.text(2, 81, '80% Share', fontsize=8.5, color='gray')
    ax.text(2, 91, '90% Share', fontsize=8.5, color='gray')

    ax.set_xlabel('Ranked Channels (Sorted Descending by Delta_I_c)', fontsize=10)
    ax.set_ylabel('Cumulative Share of Total Positive Interaction (%)', fontsize=10)
    ax.set_title('Cumulative Interaction Share: Sparse vs. Diffuse Test', fontsize=11)
    ax.set_xlim(1, 96)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)

    # Plot 2: Absolute Delta_I values sorted by rank
    ax = axes[1]
    for k in cohort_keys:
        k_df = df_cum_share[df_cum_share['cohort_key'] == k].sort_values(by='rank')
        label_str = k.replace('Consensus_7of7_', '').replace('_All', '')
        ax.plot(k_df['rank'], k_df['median_delta_I'], label=label_str, color=colors[k], linewidth=2)

    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Channel Rank', fontsize=10)
    ax.set_ylabel('Median Delta_I_c', fontsize=10)
    ax.set_title('Rank-Ordered Interaction Magnitude Curve', fontsize=11)
    ax.set_xlim(1, 96)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, 'channel_interaction_rank_curve.png')
    plt.savefig(fig2_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {fig2_path}")

    # Figure 3: Neck vs Crack Scatter (Quadrant Analysis)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    scatter_targets = [
        ('Consensus_7of7_Consensus_Resistant', 'Resistant (N=82)', axes[0]),
        ('Consensus_7of7_Consensus_Sensitive', 'Sensitive (N=19)', axes[1]),
        ('Cured_by_D2_All', 'Cured by D2 (N=7)', axes[2]),
    ]

    for k, title_str, ax in scatter_targets:
        k_sum = df_summary[(df_summary['cohort'] + '_' + df_summary['subgroup']) == k]
        x_val = k_sum['median_I_crack'].values
        y_val = k_sum['median_I_neck'].values

        # Highlight Bridge-selective Quadrant (I_neck > 0 and I_crack <= 0)
        selective_mask = (y_val > 0) & (x_val <= 0)

        ax.scatter(x_val[~selective_mask], y_val[~selective_mask], color='#1f77b4', alpha=0.6, s=40, label='General/Other')
        ax.scatter(x_val[selective_mask], y_val[selective_mask], color='#d62728', alpha=0.9, s=60, edgecolors='black', label=f'Bridge-Selective ({int(np.sum(selective_mask))})')

        # Annotate top 3 selective channels
        top3_indices = np.argsort(k_sum['median_delta_I'].values)[::-1][:3]
        for idx in top3_indices:
            ch_num = int(k_sum['channel'].iloc[idx])
            ax.annotate(f"Ch {ch_num}", (x_val[idx], y_val[idx]),
                        xytext=(4, 4), textcoords='offset points', fontsize=8, fontweight='bold')

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Median I_c(crack)', fontsize=9.5)
        ax.set_ylabel('Median I_c(neck)', fontsize=9.5)
        ax.set_title(title_str, fontsize=10.5)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, 'channel_neck_vs_crack_scatter.png')
    plt.savefig(fig3_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {fig3_path}")

    print("\n" + "=" * 80)
    print("Phase 6 Channel-wise ReLU Interaction Diagnostic Completed Successfully.")
    print("=" * 80)


if __name__ == '__main__':
    main()
