#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_t2_representation_factorization_diagnostic.py

Phase 6: T2 Representation / Skip-vs-Bottleneck Contribution & Fusion Factorization Diagnostic
Scientific Goal:
At T2 (Decoder Block 1 input = 192 ch upsampled bottleneck + 96 ch skip connection = 288 ch):
Determine whether false bridge ambiguity resides primarily in the bottleneck stream or
in the skip stream, whether feature concatenation improves or collapses component separation,
and how the 82 Resistant, 19 Sensitive, 7 D2-Cured, and 56 Clean Controls differ.

Streams at T2 (56x56 resolution):
1. Bottleneck Stream: T2[:, 0:192, :, :] (upsampled from S2 / bottleneck)
2. Skip Stream:       T2[:, 192:288, :, :] (from CNN Stage 1)
3. Fused Stream:      T2[:, 0:288, :, :] (full concatenated input to Conv1)

Counterfactual In-Memory Tensor Surgery Conditions:
1. Baseline (Full T2)
2. T2_Skip_Only (Bottleneck zeroed: T2[0:192] = 0)
3. T2_Bottleneck_Only (Skip zeroed: T2[192:288] = 0)
4. T2_Skip_Attenuated_0.50 (Skip scaled by 0.5)
5. T2_Bottleneck_Attenuated_0.50 (Bottleneck scaled by 0.5)
6. T2_Skip_Attenuated_0.25 (Skip scaled by 0.25)
7. T2_Bottleneck_Attenuated_0.25 (Bottleneck scaled by 0.25)

HARD LOCKS:
- DIAGNOSTIC-ONLY: Zero training, zero backward, zero optimizer, zero gradient updates.
- Bitwise verification of checkpoint file and parameter SHA256 before and after.
- Setting A evaluation path preserved (448x448, stride 448 non-overlapping, reflect padding, tau=0.5).
- In-memory pre-hook tensor surgery and instant exact restoration.
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
# T2 Factorization Engine
# -----------------------------------------------------------------------------

class T2FactorizationEngine:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

        self.dec_b1 = model.decoder.decoder_blocks[1]
        self.conv1 = self.dec_b1.conv1

        self.surgery_mode = 'baseline'
        self.surgery_alpha = 1.0

        self.last_t2_tensor: torch.Tensor = None
        self.pre_hook_handle = None
        self._register_surgery_hook()

    def _register_surgery_hook(self):
        def t2_pre_hook(module, args):
            # args[0] is T2 of shape (B, 288, 56, 56)
            t2 = args[0]
            # Store unmodified T2 for representation analysis
            self.last_t2_tensor = t2.detach().cpu()

            if self.surgery_mode == 'baseline':
                return args

            x = t2.clone()
            if self.surgery_mode == 'skip_only':
                x[:, :192, :, :] = 0.0
            elif self.surgery_mode == 'bottleneck_only':
                x[:, 192:, :, :] = 0.0
            elif self.surgery_mode == 'attenuate_skip':
                x[:, 192:, :, :] *= float(self.surgery_alpha)
            elif self.surgery_mode == 'attenuate_bottleneck':
                x[:, :192, :, :] *= float(self.surgery_alpha)
            return (x,)

        self.pre_hook_handle = self.conv1.register_forward_pre_hook(t2_pre_hook)

    def set_surgery(self, mode: str, alpha: float = 1.0):
        self.surgery_mode = mode
        self.surgery_alpha = alpha

    def cleanup(self):
        if self.pre_hook_handle:
            self.pre_hook_handle.remove()
            self.pre_hook_handle = None
        self.set_surgery('baseline', 1.0)

    def run_image(
        self,
        image_rgb: np.ndarray,
        tile_size: int = 448,
        batch_size: int = 4
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], List[Dict[str, Any]]]:
        """Runs Setting A tiling inference and extracts factorized T2 spatial maps."""
        H, W = image_rgb.shape[:2]
        pad_h = (tile_size - (H % tile_size)) % tile_size
        pad_w = (tile_size - (W % tile_size)) % tile_size
        padded_img = np.pad(image_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        pH, pW = padded_img.shape[:2]

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        final_logits = np.zeros((pH, pW), dtype=np.float32)

        # Full spatial maps for bottleneck, skip, fused
        stream_maps = {
            'bottleneck': np.zeros((pH, pW), dtype=np.float32),
            'skip': np.zeros((pH, pW), dtype=np.float32),
            'fused': np.zeros((pH, pW), dtype=np.float32)
        }
        patch_records = []

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

                logits = self.model(batch_pt)
                if logits.shape[2:] != (tile_size, tile_size):
                    logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)

                t2_batch = self.last_t2_tensor  # (B, 288, 56, 56)

                for b_idx, (py, px) in enumerate(batch_coords):
                    final_logits[py:py+tile_size, px:px+tile_size] = logits[b_idx].squeeze().cpu().numpy()

                    t2_sample = t2_batch[b_idx:b_idx+1]  # (1, 288, 56, 56)
                    t2_btn = t2_sample[:, :192, :, :]
                    t2_skp = t2_sample[:, 192:, :, :]
                    t2_fsd = t2_sample

                    e_btn = torch.norm(t2_btn, p=2, dim=1, keepdim=True)
                    e_skp = torch.norm(t2_skp, p=2, dim=1, keepdim=True)
                    e_fsd = torch.norm(t2_fsd, p=2, dim=1, keepdim=True)

                    up_btn = F.interpolate(e_btn, size=(tile_size, tile_size), mode='nearest').squeeze().numpy()
                    up_skp = F.interpolate(e_skp, size=(tile_size, tile_size), mode='nearest').squeeze().numpy()
                    up_fsd = F.interpolate(e_fsd, size=(tile_size, tile_size), mode='nearest').squeeze().numpy()

                    stream_maps['bottleneck'][py:py+tile_size, px:px+tile_size] = up_btn
                    stream_maps['skip'][py:py+tile_size, px:px+tile_size] = up_skp
                    stream_maps['fused'][py:py+tile_size, px:px+tile_size] = up_fsd

                    patch_records.append({
                        'coords': (py, px),
                        't2_btn': t2_btn[0],
                        't2_skp': t2_skp[0],
                        't2_fsd': t2_fsd[0],
                    })

        logits_cropped = final_logits[:H, :W]
        prob_cropped = 1.0 / (1.0 + np.exp(-logits_cropped))
        pred_bin = (logits_cropped > 0.0).astype(np.uint8)
        cropped_maps = {k: stream_maps[k][:H, :W] for k in stream_maps}

        return pred_bin, prob_cropped, cropped_maps, patch_records


# -----------------------------------------------------------------------------
# Main Diagnostic Routine
# -----------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Phase 6: T2 Representation / Skip-vs-Bottleneck Factorization Diagnostic")
    print("Direct Counterfactual Surgery and Cross-Component Relationship Analysis")
    print("=" * 80)

    config_path = 'results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml'
    ckpt_path = 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'
    output_dir = 'results/diagnostics/phase6_t2_factorization'
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

    engine = T2FactorizationEngine(model, device)

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
    baseline_t2_records = []
    pairwise_t2_records = []

    for item in tqdm(unique_cohort_samples, desc="Preloading & T2 Representation Profiling"):
        stem = item['stem']
        cohort = item['cohort']
        subgroup = item['subgroup']

        img_path = f"datasets/Crack500_ready/val/images/{stem}.jpg"
        mask_path = f"datasets/Crack500_ready/val/masks/{stem}.png"

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        target_bin = (gt_mask > 0).astype(np.uint8)

        # Baseline run
        engine.set_surgery('baseline', 1.0)
        pred_bin, prob_map, stream_maps, patch_acts = engine.run_image(img_rgb)

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

        # Detailed T2 Stream Factorization Analysis
        neck_area = int(np.sum(neck_mask))
        crack_area = int(np.sum(crack_mask))
        bg_area = int(np.sum(bg_mask))

        rep_row = {
            'image_id': stem,
            'cohort': cohort,
            'subgroup': subgroup,
            'neck_area': neck_area,
            'crack_area': crack_area,
            'bg_area': bg_area,
        }

        for stream_name in ['bottleneck', 'skip', 'fused']:
            s_map = stream_maps[stream_name]
            en = float(np.mean(s_map[neck_mask == 1])) if neck_area > 0 else 0.0
            ec = float(np.mean(s_map[crack_mask == 1])) if crack_area > 0 else 0.0
            eb = float(np.mean(s_map[bg_mask == 1])) if bg_area > 0 else 0.0

            denom = (ec - eb)
            r_norm = float((en - eb) / denom) if abs(denom) > 1e-5 else 0.0
            sep_margin = float(1.0 - (en / (ec + 1e-6)))

            rep_row[f'{stream_name}_E_neck'] = en
            rep_row[f'{stream_name}_E_crack'] = ec
            rep_row[f'{stream_name}_E_bg'] = eb
            rep_row[f'{stream_name}_R_norm'] = r_norm
            rep_row[f'{stream_name}_sep_margin'] = sep_margin
            rep_row[f'{stream_name}_contrast'] = float(en / (ec + 1e-6))

        # Energy shares
        e_sq_btn = rep_row['bottleneck_E_neck'] ** 2
        e_sq_skp = rep_row['skip_E_neck'] ** 2
        sum_sq = e_sq_btn + e_sq_skp + 1e-8
        rep_row['pct_neck_energy_bottleneck'] = (e_sq_btn / sum_sq) * 100.0
        rep_row['pct_neck_energy_skip'] = (e_sq_skp / sum_sq) * 100.0

        baseline_t2_records.append(rep_row)

        # Cross-Component Relationship Analysis for each disconnected pair
        for pair in pair_records:
            gt_i_mask = pair['mask_i_448']
            gt_j_mask = pair['mask_j_448']
            neck_pair = pair['neck_448']

            pair_row = {
                'image_id': stem,
                'cohort': cohort,
                'subgroup': subgroup,
                'gt_i': pair['gt_i'],
                'gt_j': pair['gt_j'],
                'gap_distance_px': pair['gap_distance_px'],
            }

            for stream_name in ['bottleneck', 'skip', 'fused']:
                s_map = stream_maps[stream_name]

                e_i = float(np.mean(s_map[gt_i_mask == 1])) if np.sum(gt_i_mask) > 0 else 0.0
                e_j = float(np.mean(s_map[gt_j_mask == 1])) if np.sum(gt_j_mask) > 0 else 0.0
                e_neck_p = float(np.mean(s_map[neck_pair == 1])) if np.sum(neck_pair) > 0 else 0.0

                min_e_crack = min(e_i, e_j) + 1e-6
                mean_e_crack = 0.5 * (e_i + e_j) + 1e-6

                # Valley ratio: neck energy / min(crack_i, crack_j). If > 1.0, there is NO valley!
                valley_ratio = float(e_neck_p / min_e_crack)
                pair_sep_margin = float(1.0 - (e_neck_p / mean_e_crack))

                pair_row[f'{stream_name}_E_i'] = e_i
                pair_row[f'{stream_name}_E_j'] = e_j
                pair_row[f'{stream_name}_E_neck'] = e_neck_p
                pair_row[f'{stream_name}_valley_ratio'] = valley_ratio
                pair_row[f'{stream_name}_sep_margin'] = pair_sep_margin
                pair_row[f'{stream_name}_has_valley'] = bool(valley_ratio < 0.85)

            pairwise_t2_records.append(pair_row)

    df_base_rep = pd.DataFrame(baseline_t2_records)
    rep_csv = os.path.join(output_dir, 't2_representation_factorization_per_sample.csv')
    df_base_rep.to_csv(rep_csv, index=False)
    print(f"Saved: {rep_csv}")

    df_pair_rep = pd.DataFrame(pairwise_t2_records)
    pair_csv = os.path.join(output_dir, 't2_pairwise_cross_component_per_pair.csv')
    df_pair_rep.to_csv(pair_csv, index=False)
    print(f"Saved: {pair_csv}")

    # Step 2: In-Memory Counterfactual Surgery Conditions
    SURGERY_CONDITIONS = [
        ('Baseline', 'baseline', 1.0),
        ('T2_Skip_Only', 'skip_only', 1.0),
        ('T2_Bottleneck_Only', 'bottleneck_only', 1.0),
        ('T2_Skip_Attenuated_0.50', 'attenuate_skip', 0.50),
        ('T2_Bottleneck_Attenuated_0.50', 'attenuate_bottleneck', 0.50),
        ('T2_Skip_Attenuated_0.25', 'attenuate_skip', 0.25),
        ('T2_Bottleneck_Attenuated_0.25', 'attenuate_bottleneck', 0.25),
    ]

    per_sample_counterfactuals = []
    t0 = time.time()

    print(f"\nBeginning Counterfactual Surgery Evaluation across {len(SURGERY_CONDITIONS)} conditions...")

    for cond_name, mode, alpha_val in SURGERY_CONDITIONS:
        print(f"\n---> Running Condition: {cond_name} (mode={mode}, alpha={alpha_val:.2f})...")
        engine.set_surgery(mode, alpha_val)

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

            pred_bin, prob_map, _, _ = engine.run_image(img_rgb, tile_size=448, batch_size=4)
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
                'mode': mode,
                'alpha': alpha_val,
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
    assert init_param_hash == final_param_hash, f"FATAL: Model weights modified! ({init_param_hash} vs {final_param_hash})"
    assert init_file_hash == final_file_hash, f"FATAL: Checkpoint file on disk modified! ({init_file_hash} vs {final_file_hash})"
    print(f"Model Parameters State Verified: Bitwise Identical ({final_param_hash[:16]}...)")
    print(f"Disk Checkpoint File Verified: Bitwise Identical ({final_file_hash[:16]}...)")

    df_cf = pd.DataFrame(per_sample_counterfactuals)
    cf_csv = os.path.join(output_dir, 't2_counterfactual_per_sample.csv')
    df_cf.to_csv(cf_csv, index=False)
    print(f"Saved: {cf_csv}")

    # Establish baseline bridge states per sample
    df_cf_base = df_cf[df_cf['condition'] == 'Baseline'].set_index('image_id')
    base_bridge_map = df_cf_base['has_bridge'].to_dict()

    df_cf['base_has_bridge'] = df_cf['image_id'].map(base_bridge_map)
    df_cf['is_cured'] = (df_cf['base_has_bridge'] == True) & (df_cf['has_bridge'] == False)
    df_cf['is_persistent'] = (df_cf['base_has_bridge'] == True) & (df_cf['has_bridge'] == True)
    df_cf['is_created'] = (df_cf['base_has_bridge'] == False) & (df_cf['has_bridge'] == True)

    # 1. Summary of T2 Factorization Baseline Representation
    summary_rep_rows = []
    groups_to_summarize = [
        ('Consensus_7of7', 'All'),
        ('Consensus_7of7', 'Consensus_Resistant'),
        ('Consensus_7of7', 'Consensus_Sensitive'),
        ('Cured_by_D2', 'All'),
        ('Clean_Control', 'All'),
    ]

    for cohort_name, grp_name in groups_to_summarize:
        if grp_name == 'All':
            sub_rep = df_base_rep[df_base_rep['cohort'] == cohort_name]
        else:
            sub_rep = df_base_rep[(df_base_rep['cohort'] == cohort_name) & (df_base_rep['subgroup'] == grp_name)]

        if len(sub_rep) == 0:
            continue

        summary_rep_rows.append({
            'cohort': cohort_name,
            'subgroup': grp_name,
            'n_samples': len(sub_rep),
            'median_btn_R_norm': float(sub_rep['bottleneck_R_norm'].median()),
            'median_skp_R_norm': float(sub_rep['skip_R_norm'].median()),
            'median_fsd_R_norm': float(sub_rep['fused_R_norm'].median()),
            'median_btn_sep_margin': float(sub_rep['bottleneck_sep_margin'].median()),
            'median_skp_sep_margin': float(sub_rep['skip_sep_margin'].median()),
            'median_fsd_sep_margin': float(sub_rep['fused_sep_margin'].median()),
            'median_pct_energy_bottleneck': float(sub_rep['pct_neck_energy_bottleneck'].median()),
            'median_pct_energy_skip': float(sub_rep['pct_neck_energy_skip'].median()),
        })

    df_sum_rep = pd.DataFrame(summary_rep_rows)
    sum_rep_csv = os.path.join(output_dir, 't2_representation_factorization_summary.csv')
    df_sum_rep.to_csv(sum_rep_csv, index=False)
    print(f"Saved: {sum_rep_csv}")
    print("\n--- T2 Representation Factorization Summary ---")
    print(df_sum_rep.to_string())

    # 2. Pairwise Cross-Component Valley Summary
    pair_summary_rows = []
    for cohort_name, grp_name in groups_to_summarize:
        if grp_name == 'All':
            sub_p = df_pair_rep[df_pair_rep['cohort'] == cohort_name]
        else:
            sub_p = df_pair_rep[(df_pair_rep['cohort'] == cohort_name) & (df_pair_rep['subgroup'] == grp_name)]

        if len(sub_p) == 0:
            continue

        pair_summary_rows.append({
            'cohort': cohort_name,
            'subgroup': grp_name,
            'n_pairs': len(sub_p),
            'median_btn_valley_ratio': float(sub_p['bottleneck_valley_ratio'].median()),
            'pct_btn_has_valley': float(sub_p['bottleneck_has_valley'].mean() * 100.0),
            'median_skp_valley_ratio': float(sub_p['skip_valley_ratio'].median()),
            'pct_skp_has_valley': float(sub_p['skip_has_valley'].mean() * 100.0),
            'median_fsd_valley_ratio': float(sub_p['fused_valley_ratio'].median()),
            'pct_fsd_has_valley': float(sub_p['fused_has_valley'].mean() * 100.0),
        })

    df_pair_sum = pd.DataFrame(pair_summary_rows)
    pair_sum_csv = os.path.join(output_dir, 't2_pairwise_cross_component_summary.csv')
    df_pair_sum.to_csv(pair_sum_csv, index=False)
    print(f"Saved: {pair_sum_csv}")
    print("\n--- T2 Pairwise Cross-Component Valley Summary ---")
    print(df_pair_sum.to_string())

    # 3. Subgroup Counterfactual Response Matrix
    matrix_rows = []
    for cond_name, mode, alpha_val in SURGERY_CONDITIONS:
        sub_c = df_cf[df_cf['condition'] == cond_name]
        res_sub = sub_c[sub_c['subgroup'] == 'Consensus_Resistant']
        sens_sub = sub_c[sub_c['subgroup'] == 'Consensus_Sensitive']
        ctrl_sub = sub_c[sub_c['cohort'] == 'Clean_Control']
        d2_sub = sub_c[sub_c['cohort'] == 'Cured_by_D2']

        matrix_rows.append({
            'condition': cond_name,
            'mode': mode,
            'alpha': alpha_val,
            'Resistant_Bridges': f"{int(res_sub['has_bridge'].sum())}/82",
            'Resistant_Cured': f"{int(res_sub['is_cured'].sum())}/82 ({res_sub['is_cured'].mean()*100:.1f}%)",
            'Resistant_Breakage': f"{int(res_sub['has_breakage'].sum())}/82",
            'Resistant_Recall': f"{res_sub['recall'].median():.4f}",
            'Resistant_Med_ConnProb': f"{res_sub['connector_prob_neck'].median():.4f}",
            'Sensitive_Bridges': f"{int(sens_sub['has_bridge'].sum())}/19",
            'Sensitive_Cured': f"{int(sens_sub['is_cured'].sum())}/19 ({sens_sub['is_cured'].mean()*100:.1f}%)",
            'Sensitive_Breakage': f"{int(sens_sub['has_breakage'].sum())}/19",
            'Control_Bridges_Created': f"{int(ctrl_sub['has_bridge'].sum())}/56",
            'Control_Breakage': f"{int(ctrl_sub['has_breakage'].sum())}/56",
            'D2_Bridges': f"{int(d2_sub['has_bridge'].sum())}/7",
        })

    df_matrix = pd.DataFrame(matrix_rows)
    matrix_csv = os.path.join(output_dir, 't2_counterfactual_subgroup_matrix.csv')
    df_matrix.to_csv(matrix_csv, index=False)
    print(f"Saved: {matrix_csv}")
    print("\n--- T2 Counterfactual Subgroup Matrix ---")
    print(df_matrix.to_string())

    # Visualizations
    # Figure 1: Energy & R_norm across Bottleneck vs Skip vs Fused at T2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cohort_labels = ['Resistant (82)', 'Sensitive (19)', 'D2-Cured (7)', 'Clean Control (56)']
    cohort_keys = [
        ('Consensus_7of7', 'Consensus_Resistant'),
        ('Consensus_7of7', 'Consensus_Sensitive'),
        ('Cured_by_D2', 'All'),
        ('Clean_Control', 'All'),
    ]

    btn_r = []
    skp_r = []
    fsd_r = []
    btn_share = []
    skp_share = []

    for c_name, s_name in cohort_keys:
        if s_name == 'All':
            sub = df_base_rep[df_base_rep['cohort'] == c_name]
        else:
            sub = df_base_rep[(df_base_rep['cohort'] == c_name) & (df_base_rep['subgroup'] == s_name)]
        btn_r.append(sub['bottleneck_R_norm'].median())
        skp_r.append(sub['skip_R_norm'].median())
        fsd_r.append(sub['fused_R_norm'].median())
        btn_share.append(sub['pct_neck_energy_bottleneck'].median())
        skp_share.append(sub['pct_neck_energy_skip'].median())

    x = np.arange(len(cohort_labels))
    w = 0.25

    # Left: R_norm
    axes[0].bar(x - w, btn_r, w, label='Bottleneck Stream (192ch)', color='#e67e22')
    axes[0].bar(x, skp_r, w, label='Skip Stream (96ch)', color='#3498db')
    axes[0].bar(x + w, fsd_r, w, label='Fused T2 (288ch)', color='#9b59b6')
    axes[0].axhline(y=0.50, color='r', linestyle='--', label='Ambiguity Threshold (0.50)')
    axes[0].set_ylabel(r'Median $R_{\mathrm{norm}}$', fontsize=11)
    axes[0].set_title(r'T2 Representation Ambiguity ($R_{\mathrm{norm}}$) by Stream', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cohort_labels, fontsize=10)
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right')

    # Right: Energy Share
    axes[1].bar(x - w/2, btn_share, w, label='Bottleneck Share (%)', color='#e67e22')
    axes[1].bar(x + w/2, skp_share, w, label='Skip Share (%)', color='#3498db')
    axes[1].set_ylabel('Corridor Energy Share (%)', fontsize=11)
    axes[1].set_title('Corridor Energy Share at T2 (Bottleneck vs Skip)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cohort_labels, fontsize=10)
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper right')

    fig1_path = os.path.join(figures_dir, 't2_stream_energy_and_rnorm_comparison.png')
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"Saved: {fig1_path}")

    # Figure 2: Counterfactual Bridges vs Breakage across Surgery Conditions
    fig, ax = plt.subplots(figsize=(10, 5))
    cond_names = [c[0] for c in SURGERY_CONDITIONS]
    df_cf_c7 = df_cf[df_cf['cohort'] == 'Consensus_7of7']
    c7_bridges = [int(df_cf_c7[df_cf_c7['condition'] == c]['has_bridge'].sum()) for c in cond_names]
    c7_breakages = [int(df_cf_c7[df_cf_c7['condition'] == c]['has_breakage'].sum()) for c in cond_names]

    x = np.arange(len(cond_names))
    w = 0.35
    ax.bar(x - w/2, c7_bridges, w, label='Consensus False Bridges (/101)', color='#e74c3c')
    ax.bar(x + w/2, c7_breakages, w, label='Consensus Breakages (/101)', color='#2980b9')

    ax.set_ylabel('Count (N=101)', fontsize=11)
    ax.set_title('T2 Stream Surgery: False Bridge Removal vs Crack Breakage', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('T2_', '').replace('_Attenuated_', ' ') for c in cond_names], rotation=30, ha='right', fontsize=9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend()

    fig2_path = os.path.join(figures_dir, 't2_counterfactual_bridge_vs_breakage.png')
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"Saved: {fig2_path}")

    print("\n" + "=" * 80)
    print("T2 Factorization Diagnostic Completed Successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
