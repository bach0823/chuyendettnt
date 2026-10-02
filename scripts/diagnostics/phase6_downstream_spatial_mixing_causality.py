#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_downstream_spatial_mixing_causality.py

Phase 6: Downstream Spatial-Mixing Causality Diagnostic
Tests whether off-center spatial mixing at:
  A. Stage 0 Block 0 (7x7 depthwise Conv)
  B. Decoder Block 1 (Conv1 and Conv2 3x3 standard Convs)
is causally necessary for false bridge formation in:
  - 82 Consensus Resistant cases
  - 19 Consensus Sensitive cases
  - 7 Cured by D2 controls
  - 56 Clean Controls
Total N = 164 validation samples.

Protocol:
- Strictly DIAGNOSTIC-ONLY (zero training, zero backward, zero optimizer).
- Bitwise verification of checkpoint file and parameter SHA256 before and after.
- Setting A evaluation path preserved (448x448, stride 448 non-overlapping, reflect padding, tau=0.5).
- In-memory weight attenuation and instant exact restoration.
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
# Kernel Attenuation Functions
# -----------------------------------------------------------------------------

def make_attenuated_dw_weight(orig_weight: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Computes W' = W_center + alpha * W_off for a 7x7 depthwise conv.
    Kernel shape: (C_out, 1, 7, 7). Center tap is (3, 3).
    All 48 off-center taps (i, j) != (3, 3) are multiplied by alpha.
    """
    w_new = orig_weight.clone()
    for i in range(7):
        for j in range(7):
            if (i, j) != (3, 3):
                w_new[:, :, i, j] *= alpha
    return w_new


def make_attenuated_3x3_weight(orig_weight: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Computes W' = W_center + alpha * W_off for a 3x3 conv.
    Kernel shape: (C_out, C_in, 3, 3). Center tap is (1, 1).
    All 8 off-center taps (i, j) != (1, 1) are multiplied by alpha.
    """
    w_new = orig_weight.clone()
    for i in range(3):
        for j in range(3):
            if (i, j) != (1, 1):
                w_new[:, :, i, j] *= alpha
    return w_new


# -----------------------------------------------------------------------------
# Counterfactual Causality Engine
# -----------------------------------------------------------------------------

STAGE_HOOKS = [
    'stem_ln',
    'stage0_block0',
    'bottleneck',
    'S2',
    'T2',
    'T3',
    'T4_S1',
    'S0'
]


class DownstreamCausalityEngine:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

        # Target modules
        # 1. Stage0 Block0 (7x7 depthwise Conv)
        self.b0 = model.backbone.convnext.stages[0].main_block.module.blocks[0]
        self.conv_dw = self.b0.conv_dw
        self.orig_w_dw = self.conv_dw.weight.data.clone()

        # 2. Decoder Block 1 (Conv1 and Conv2 3x3 Convs)
        self.dec_b1 = model.decoder.decoder_blocks[1]
        self.conv1 = self.dec_b1.conv1[0]
        self.conv2 = self.dec_b1.conv2[0]
        self.orig_w_conv1 = self.conv1.weight.data.clone()
        self.orig_w_conv2 = self.conv2.weight.data.clone()

        self.raw_activations: Dict[str, torch.Tensor] = {}
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        # 1. stem_ln (output of ConvNeXt stem)
        self.handles.append(self.model.backbone.convnext.stem[1].register_forward_hook(
            lambda m, inp, out: self.raw_activations.update({'stem_ln': (out[0] if isinstance(out, tuple) else out).detach()})
        ))

        # 2. stage0_block0 (output of Block 0)
        def blk0_hook(m, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if t.ndim == 4 and t.shape[1] == 48 and t.shape[2] == 112:
                self.raw_activations.update({'stage0_block0': t.detach()})
        self.handles.append(self.b0.register_forward_hook(blk0_hook))

        # 3. bottleneck (input to decoder_blocks[0])
        self.handles.append(self.model.decoder.decoder_blocks[0].register_forward_pre_hook(
            lambda m, args: self.raw_activations.update({'bottleneck': args[0].detach()})
        ))

        # 4. S2 (output of decoder Block 0 / 28x28)
        self.handles.append(self.model.decoder.decoder_blocks[0].register_forward_hook(
            lambda m, inp, out: self.raw_activations.update({'S2': (out[0] if isinstance(out, tuple) else out).detach()})
        ))

        # 5. T2 (input to conv1 of Block 1 post-skip concat)
        self.handles.append(self.dec_b1.conv1.register_forward_pre_hook(
            lambda m, args: self.raw_activations.update({'T2': args[0].detach()})
        ))

        # 6. T3 (output of Conv1 inside Block 1)
        self.handles.append(self.conv1.register_forward_hook(
            lambda m, inp, out: self.raw_activations.update({'T3': (out[0] if isinstance(out, tuple) else out).detach()})
        ))

        # 7. T4_S1 (output of Block 1 / post-conv2 / 56x56)
        self.handles.append(self.dec_b1.register_forward_hook(
            lambda m, inp, out: self.raw_activations.update({'T4_S1': (out[0] if isinstance(out, tuple) else out).detach()})
        ))

        # 8. S0 (output of Block 2 / 112x112)
        self.handles.append(self.model.decoder.decoder_blocks[2].register_forward_hook(
            lambda m, inp, out: self.raw_activations.update({'S0': (out[0] if isinstance(out, tuple) else out).detach()})
        ))

    def set_locus_a_alpha(self, alpha: float):
        """Attenuates off-center spatial taps of Stage 0 Block 0 depthwise conv."""
        w_att = make_attenuated_dw_weight(self.orig_w_dw, alpha)
        self.conv_dw.weight.data.copy_(w_att)

    def set_locus_b_alpha(self, alpha: float):
        """Attenuates off-center spatial taps of Decoder Block 1 Conv1 & Conv2."""
        w1_att = make_attenuated_3x3_weight(self.orig_w_conv1, alpha)
        w2_att = make_attenuated_3x3_weight(self.orig_w_conv2, alpha)
        self.conv1.weight.data.copy_(w1_att)
        self.conv2.weight.data.copy_(w2_att)

    def restore_all_weights(self):
        """Restores exact pristine weights in-memory."""
        self.conv_dw.weight.data.copy_(self.orig_w_dw)
        self.conv1.weight.data.copy_(self.orig_w_conv1)
        self.conv2.weight.data.copy_(self.orig_w_conv2)

    def cleanup(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.restore_all_weights()

    def run_image(
        self,
        image_rgb: np.ndarray,
        tile_size: int = 448,
        batch_size: int = 4
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Runs Setting A tiling inference and extracts spatial activation maps."""
        H, W = image_rgb.shape[:2]
        pad_h = (tile_size - (H % tile_size)) % tile_size
        pad_w = (tile_size - (W % tile_size)) % tile_size
        padded_img = np.pad(image_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        pH, pW = padded_img.shape[:2]

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        final_logits = np.zeros((pH, pW), dtype=np.float32)
        stage_full_energy = {s: np.zeros((pH, pW), dtype=np.float32) for s in STAGE_HOOKS}

        patches = []
        coords = []
        for y in range(0, pH, tile_size):
            for x in range(0, pW, tile_size):
                patch = padded_img[y:y+tile_size, x:x+tile_size]
                pt = torch.from_numpy(patch).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
                pt = (pt - mean) / std
                patches.append(pt)
                coords.append((y, x))

        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch_pt = torch.cat(patches[i:i+batch_size], dim=0)
                batch_coords = coords[i:i+batch_size]

                self.raw_activations.clear()
                logits = self.model(batch_pt)
                if logits.shape[2:] != (tile_size, tile_size):
                    logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)

                for b_idx, (py, px) in enumerate(batch_coords):
                    final_logits[py:py+tile_size, px:px+tile_size] = logits[b_idx].squeeze().cpu().numpy()

                for s_name in STAGE_HOOKS:
                    if s_name not in self.raw_activations:
                        continue
                    act = self.raw_activations[s_name]
                    energy = torch.norm(act, p=2, dim=1, keepdim=True)
                    energy_up = F.interpolate(energy, size=(tile_size, tile_size), mode='nearest')
                    for b_idx, (py, px) in enumerate(batch_coords):
                        stage_full_energy[s_name][py:py+tile_size, px:px+tile_size] = energy_up[b_idx].squeeze().cpu().numpy()

        logits_cropped = final_logits[:H, :W]
        prob_cropped = 1.0 / (1.0 + np.exp(-logits_cropped))
        pred_bin = (logits_cropped > 0.0).astype(np.uint8)
        stage_cropped_energy = {s: stage_full_energy[s][:H, :W] for s in STAGE_HOOKS}

        return pred_bin, prob_cropped, stage_cropped_energy


# -----------------------------------------------------------------------------
# Main Diagnostic Routine
# -----------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Phase 6: Downstream Spatial-Mixing Causality Diagnostic")
    print("Direct Counterfactual Testing of Stage0 Block0 vs Decoder Block 1")
    print("=" * 80)

    config_path = 'results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml'
    ckpt_path = 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'
    output_dir = 'results/diagnostics/phase6_downstream_spatial_mixing'
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

    engine = DownstreamCausalityEngine(model, device)

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

    # Pre-load Images, Masks and compute ROI corridors once
    print("\nPre-loading images, GT masks, and computing reference ROIs...")
    sample_data = {}
    for item in tqdm(unique_cohort_samples, desc="Preloading ROIs"):
        stem = item['stem']
        cohort = item['cohort']
        img_path = f"datasets/Crack500_ready/val/images/{stem}.jpg"
        mask_path = f"datasets/Crack500_ready/val/masks/{stem}.png"

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        target_bin = (gt_mask > 0).astype(np.uint8)

        pred_bin, prob_map, stage_maps = engine.run_image(img_rgb)
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

    # Define Counterfactual Interventions
    # Baseline: alpha_dw = 1.0, alpha_dec = 1.0
    # Series A: alpha_dw in [0.75, 0.50, 0.25, 0.00], alpha_dec = 1.0
    # Series B: alpha_dec in [0.75, 0.50, 0.25, 0.00], alpha_dw = 1.0
    INTERVENTIONS = [
        ('Baseline', 1.0, 1.0),
        # Series A: Stage 0 Block 0 DW Conv
        ('Stage0_DW_0.75', 0.75, 1.0),
        ('Stage0_DW_0.50', 0.50, 1.0),
        ('Stage0_DW_0.25', 0.25, 1.0),
        ('Stage0_DW_0.00', 0.00, 1.0),
        # Series B: Decoder Block 1 Conv1 & Conv2
        ('Decoder_B1_0.75', 1.0, 0.75),
        ('Decoder_B1_0.50', 1.0, 0.50),
        ('Decoder_B1_0.25', 1.0, 0.25),
        ('Decoder_B1_0.00', 1.0, 0.00),
    ]

    per_sample_records = []
    t0 = time.time()

    print(f"\nBeginning Evaluation across {len(INTERVENTIONS)} counterfactual conditions...")

    for cond_name, alpha_dw, alpha_dec in INTERVENTIONS:
        print(f"\n---> Running Condition: {cond_name} (alpha_dw={alpha_dw:.2f}, alpha_dec={alpha_dec:.2f})...")

        # Apply in-memory attenuation
        engine.restore_all_weights()
        if alpha_dw != 1.0:
            engine.set_locus_a_alpha(alpha_dw)
        if alpha_dec != 1.0:
            engine.set_locus_b_alpha(alpha_dec)

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

            pred_bin, prob_map, stage_maps = engine.run_image(img_rgb, tile_size=448, batch_size=4)
            topo = compute_topology_metrics(pred_bin, target_bin)

            # Connector neck probability and area
            neck_area = int(np.sum(neck_mask))
            connector_prob = float(np.mean(prob_map[neck_mask == 1])) if neck_area > 0 else 0.0
            bridge_fp_area = int(np.sum(pred_bin & (target_bin == 0)))

            sample_entry = {
                'image_id': stem,
                'cohort': cohort,
                'subgroup': subgroup,
                'condition': cond_name,
                'locus': 'Baseline' if cond_name == 'Baseline' else ('Stage0_DW' if 'Stage0' in cond_name else 'Decoder_B1'),
                'alpha_dw': alpha_dw,
                'alpha_dec': alpha_dec,
                'alpha': alpha_dw if 'Stage0' in cond_name else alpha_dec,
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
                'neck_area': neck_area,
                'bridge_fp_area': bridge_fp_area,
                'connector_prob': connector_prob,
            }

            # Representation metrics at key hook stages
            for s_name in STAGE_HOOKS:
                if s_name not in stage_maps:
                    continue
                emap = stage_maps[s_name]
                en = float(np.mean(emap[neck_mask == 1])) if neck_area > 0 else 0.0
                ec = float(np.mean(emap[crack_mask == 1])) if np.sum(crack_mask) > 0 else 0.0
                eb = float(np.mean(emap[bg_mask == 1])) if np.sum(bg_mask) > 0 else 0.0

                c_neck_crack = float(en / (ec + 1e-6))
                sep_margin = float(1.0 - c_neck_crack)
                denom = (ec - eb)
                r_norm = float((en - eb) / denom) if abs(denom) > 1e-5 else 0.0

                sample_entry[f'{s_name}_R_norm'] = r_norm
                sample_entry[f'{s_name}_sep_margin'] = sep_margin
                sample_entry[f'{s_name}_contrast'] = c_neck_crack

            per_sample_records.append(sample_entry)

    engine.cleanup()
    t1 = time.time()
    print(f"\nAll conditions evaluated in {t1 - t0:.1f}s.")

    # Integrity verification
    final_param_hash = compute_model_param_hash(model)
    final_file_hash = compute_file_hash(ckpt_path)
    assert init_param_hash == final_param_hash, f"FATAL: Model weights modified! ({init_param_hash} vs {final_param_hash})"
    assert init_file_hash == final_file_hash, f"FATAL: Checkpoint file on disk modified! ({init_file_hash} vs {final_file_hash})"
    print(f"Model Parameters State Verified: Bitwise Identical ({final_param_hash[:16]}...)")
    print(f"Disk Checkpoint File Verified: Bitwise Identical ({final_file_hash[:16]}...)")

    # DataFrames
    df_samples = pd.DataFrame(per_sample_records)
    sample_csv = os.path.join(output_dir, 'downstream_causality_per_sample.csv')
    df_samples.to_csv(sample_csv, index=False)
    print(f"Saved: {sample_csv}")

    # Establish baseline bridge states per sample to compute cured / persistent / created
    df_base = df_samples[df_samples['condition'] == 'Baseline'].set_index('image_id')
    base_bridge_map = df_base['has_bridge'].to_dict()

    df_samples['base_has_bridge'] = df_samples['image_id'].map(base_bridge_map)
    df_samples['is_cured'] = (df_samples['base_has_bridge'] == True) & (df_samples['has_bridge'] == False)
    df_samples['is_persistent'] = (df_samples['base_has_bridge'] == True) & (df_samples['has_bridge'] == True)
    df_samples['is_created'] = (df_samples['base_has_bridge'] == False) & (df_samples['has_bridge'] == True)

    # 1. Summary by Condition and Subgroup
    summary_rows = []
    groups_to_summarize = [
        ('Consensus_7of7', 'All'),
        ('Consensus_7of7', 'Consensus_Resistant'),
        ('Consensus_7of7', 'Consensus_Sensitive'),
        ('Cured_by_D2', 'All'),
        ('Clean_Control', 'All'),
    ]

    for cohort_name, grp_name in groups_to_summarize:
        if grp_name == 'All':
            sub_cohort = df_samples[df_samples['cohort'] == cohort_name]
        else:
            sub_cohort = df_samples[(df_samples['cohort'] == cohort_name) & (df_samples['subgroup'] == grp_name)]

        n_samples = len(sub_cohort['image_id'].unique())
        if n_samples == 0:
            continue

        for cond_name, _, _ in INTERVENTIONS:
            sub_c = sub_cohort[sub_cohort['condition'] == cond_name]
            if len(sub_c) == 0:
                continue

            n_bridge = int(sub_c['has_bridge'].sum())
            pct_bridge = (n_bridge / n_samples) * 100
            n_cured = int(sub_c['is_cured'].sum())
            pct_cured = (n_cured / n_samples) * 100
            n_breakage = int(sub_c['has_breakage'].sum())
            pct_breakage = (n_breakage / n_samples) * 100

            row = {
                'cohort': cohort_name,
                'subgroup': grp_name,
                'condition': cond_name,
                'locus': sub_c['locus'].iloc[0],
                'alpha': sub_c['alpha'].iloc[0],
                'n_samples': n_samples,
                'n_bridge': n_bridge,
                'pct_bridge': pct_bridge,
                'n_cured': n_cured,
                'pct_cured': pct_cured,
                'n_breakage': n_breakage,
                'pct_breakage': pct_breakage,
                'median_dice': float(sub_c['dice'].median()),
                'median_recall': float(sub_c['recall'].median()),
                'median_precision': float(sub_c['precision'].median()),
                'median_cldice': float(sub_c['cldice'].median()),
                'mean_connector_prob': float(sub_c['connector_prob'].mean()),
                'median_connector_prob': float(sub_c['connector_prob'].median()),
                'median_stage0_b0_R_norm': float(sub_c['stage0_block0_R_norm'].median()) if 'stage0_block0_R_norm' in sub_c else 0.0,
                'median_T4_S1_R_norm': float(sub_c['T4_S1_R_norm'].median()) if 'T4_S1_R_norm' in sub_c else 0.0,
                'median_final_head_sep_margin': float(sub_c['final_head_sep_margin'].median()) if 'final_head_sep_margin' in sub_c else 0.0,
            }
            summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, 'downstream_causality_summary.csv')
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    # 2. Subgroup Response Matrix (Resistant vs Sensitive across Loci)
    matrix_rows = []
    for cond_name, _, _ in INTERVENTIONS:
        sub_c = df_samples[df_samples['condition'] == cond_name]
        res_sub = sub_c[sub_c['subgroup'] == 'Consensus_Resistant']
        sens_sub = sub_c[sub_c['subgroup'] == 'Consensus_Sensitive']
        ctrl_sub = sub_c[sub_c['cohort'] == 'Clean_Control']
        d2_sub = sub_c[sub_c['cohort'] == 'Cured_by_D2']

        matrix_rows.append({
            'condition': cond_name,
            'locus': sub_c['locus'].iloc[0],
            'alpha': sub_c['alpha'].iloc[0],
            'Resistant_Bridges': f"{int(res_sub['has_bridge'].sum())}/82",
            'Resistant_Cured': f"{int(res_sub['is_cured'].sum())}/82 ({res_sub['is_cured'].mean()*100:.1f}%)",
            'Resistant_Breakage': f"{int(res_sub['has_breakage'].sum())}/82",
            'Resistant_Recall': f"{res_sub['recall'].median():.4f}",
            'Sensitive_Bridges': f"{int(sens_sub['has_bridge'].sum())}/19",
            'Sensitive_Cured': f"{int(sens_sub['is_cured'].sum())}/19 ({sens_sub['is_cured'].mean()*100:.1f}%)",
            'Sensitive_Breakage': f"{int(sens_sub['has_breakage'].sum())}/19",
            'Control_Bridges_Created': f"{int(ctrl_sub['has_bridge'].sum())}/56",
            'Control_Breakage': f"{int(ctrl_sub['has_breakage'].sum())}/56",
            'D2_Bridges': f"{int(d2_sub['has_bridge'].sum())}/7",
        })

    df_matrix = pd.DataFrame(matrix_rows)
    matrix_csv = os.path.join(output_dir, 'subgroup_response_matrix.csv')
    df_matrix.to_csv(matrix_csv, index=False)
    print(f"Saved: {matrix_csv}")
    print("\n--- Subgroup Response Matrix ---")
    print(df_matrix.to_string())

    # 3. Generate Visualizations
    # Figure 1: Bridge Count vs Breakage Trade-off by Locus
    plt.figure(figsize=(10, 6))
    alphas = [1.0, 0.75, 0.50, 0.25, 0.00]
    
    # Stage 0 DW
    s0_bridges = [int(df_samples[(df_samples['condition'] == f'Stage0_DW_{a:.2f}') & (df_samples['cohort'] == 'Consensus_7of7')]['has_bridge'].sum()) if a < 1.0 else int(df_samples[(df_samples['condition'] == 'Baseline') & (df_samples['cohort'] == 'Consensus_7of7')]['has_bridge'].sum()) for a in alphas]
    s0_breakages = [int(df_samples[(df_samples['condition'] == f'Stage0_DW_{a:.2f}') & (df_samples['cohort'] == 'Consensus_7of7')]['has_breakage'].sum()) if a < 1.0 else int(df_samples[(df_samples['condition'] == 'Baseline') & (df_samples['cohort'] == 'Consensus_7of7')]['has_breakage'].sum()) for a in alphas]

    # Decoder B1
    dec_bridges = [int(df_samples[(df_samples['condition'] == f'Decoder_B1_{a:.2f}') & (df_samples['cohort'] == 'Consensus_7of7')]['has_bridge'].sum()) if a < 1.0 else int(df_samples[(df_samples['condition'] == 'Baseline') & (df_samples['cohort'] == 'Consensus_7of7')]['has_bridge'].sum()) for a in alphas]
    dec_breakages = [int(df_samples[(df_samples['condition'] == f'Decoder_B1_{a:.2f}') & (df_samples['cohort'] == 'Consensus_7of7')]['has_breakage'].sum()) if a < 1.0 else int(df_samples[(df_samples['condition'] == 'Baseline') & (df_samples['cohort'] == 'Consensus_7of7')]['has_breakage'].sum()) for a in alphas]

    plt.plot(alphas, s0_bridges, 'o-', color='#e74c3c', label='Stage0 Block0 (7x7 DW) - False Bridges')
    plt.plot(alphas, s0_breakages, 's--', color='#e67e22', label='Stage0 Block0 (7x7 DW) - Crack Breakage')
    plt.plot(alphas, dec_bridges, 'o-', color='#2980b9', label='Decoder Block 1 (Conv1/2) - False Bridges')
    plt.plot(alphas, dec_breakages, 's--', color='#27ae60', label='Decoder Block 1 (Conv1/2) - Crack Breakage')

    plt.xlabel(r'Spatial Mixing Attenuation ($\alpha_{spatial}$)', fontsize=12)
    plt.ylabel('Count in Consensus 7/7 (N=101)', fontsize=12)
    plt.title('Causal Intervention: Stage0 Block0 vs Decoder Block 1 Spatial Mixing', fontsize=13, fontweight='bold')
    plt.gca().invert_xaxis()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='center left')
    fig1_path = os.path.join(figures_dir, 'bridge_vs_breakage_tradeoff_by_locus.png')
    plt.savefig(fig1_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig1_path}")

    # Figure 2: Resistant vs Sensitive Cured Count across Alphas
    plt.figure(figsize=(10, 5))
    s0_res_cured = [int(df_samples[(df_samples['condition'] == f'Stage0_DW_{a:.2f}') & (df_samples['subgroup'] == 'Consensus_Resistant')]['is_cured'].sum()) if a < 1.0 else 0 for a in alphas]
    s0_sens_cured = [int(df_samples[(df_samples['condition'] == f'Stage0_DW_{a:.2f}') & (df_samples['subgroup'] == 'Consensus_Sensitive')]['is_cured'].sum()) if a < 1.0 else 0 for a in alphas]
    dec_res_cured = [int(df_samples[(df_samples['condition'] == f'Decoder_B1_{a:.2f}') & (df_samples['subgroup'] == 'Consensus_Resistant')]['is_cured'].sum()) if a < 1.0 else 0 for a in alphas]
    dec_sens_cured = [int(df_samples[(df_samples['condition'] == f'Decoder_B1_{a:.2f}') & (df_samples['subgroup'] == 'Consensus_Sensitive')]['is_cured'].sum()) if a < 1.0 else 0 for a in alphas]

    plt.plot(alphas, s0_res_cured, 'o-', color='#c0392b', label='Stage0 DW: Resistant Cured (/82)')
    plt.plot(alphas, s0_sens_cured, '^--', color='#e67e22', label='Stage0 DW: Sensitive Cured (/19)')
    plt.plot(alphas, dec_res_cured, 's-', color='#2980b9', label='Decoder B1: Resistant Cured (/82)')
    plt.plot(alphas, dec_sens_cured, 'v--', color='#27ae60', label='Decoder B1: Sensitive Cured (/19)')

    plt.xlabel(r'Spatial Mixing Attenuation ($\alpha_{spatial}$)', fontsize=12)
    plt.ylabel('Cured False Bridge Cases', fontsize=12)
    plt.title('Subgroup Response: Resistant (N=82) vs Sensitive (N=19) to Spatial Attenuation', fontsize=13, fontweight='bold')
    plt.gca().invert_xaxis()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    fig2_path = os.path.join(figures_dir, 'resistant_cured_breakdown.png')
    plt.savefig(fig2_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig2_path}")

    print("\n" + "=" * 80)
    print("Phase 6 Downstream Spatial-Mixing Causality Diagnostic Completed Successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
