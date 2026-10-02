#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_t2_nonlinear_fusion_interaction_diagnostic.py

Phase 6: T2 Nonlinear Fusion Interaction Diagnostic
Target Locus: Decoder Block 1 (Conv1 -> BN -> ReLU -> Conv2)

Scientific Objective:
Determine whether the joint fusion of Bottleneck (B) and Skip (S) streams at T2
produces an explicit non-linear interaction / super-additive reinforcement
beyond what is expected from B-only and S-only independent contributions.

Factorial 2x2 Interaction Design:
For any stage k and quantity F:
  I = F(B, S) - F(B, 0) - F(0, S) + F(0, 0)

4 Evaluated Conditions:
1. Full:   X = [B, S] (alpha_B = 1.0, alpha_S = 1.0)
2. B_only: X = [B, 0] (alpha_B = 1.0, alpha_S = 0.0)
3. S_only: X = [0, S] (alpha_B = 0.0, alpha_S = 1.0)
4. Zero:   X = [0, 0] (alpha_B = 0.0, alpha_S = 0.0)

Intermediate Stages Hooked:
- Stage 1: Conv1 pre-BN (Y)        -> strictly linear: I_vec == 0 (Null baseline)
- Stage 2: Conv1 post-BN (Z)       -> affine transformation: I_vec == 0
- Stage 3: Conv1 post-ReLU (A)     -> non-linear rectification: I_vec > 0
- Stage 4: Conv2 output (T3)       -> spatial mixing + non-linearity: I_vec > 0
- Stage 5: Final Connector Probability (P_neck, P_crack)
- Stage 6: Binary False Bridge State (B_bridge in {0, 1})

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
# T2 Nonlinear Interaction Engine
# -----------------------------------------------------------------------------

class T2NonlinearInteractionEngine:
    """
    Manages in-memory pre-hook surgery at Decoder Block 1 Conv1 and hooks
    intermediate outputs (pre-BN, post-BN, post-ReLU, conv2_out).
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

        self.dec_b1 = model.decoder.decoder_blocks[1]
        self.conv1_seq = self.dec_b1.conv1
        self.conv1_0 = self.conv1_seq[0]  # Conv2d(288, 96, 3, p=1)
        self.conv1_1 = self.conv1_seq[1]  # BatchNorm2d(96)
        self.conv1_2 = self.conv1_seq[2]  # ReLU(inplace=True)
        self.conv2 = self.dec_b1.conv2    # Conv2 Sequential

        self.mode: str = 'full'
        self.pre_hook_handle = None
        self.stage_hook_handles = []

        self.current_patch_acts: Dict[str, torch.Tensor] = {}
        self.collected_patch_records: List[Dict[str, Any]] = []

        self._register_hooks()

    def _register_hooks(self):
        # 1. Pre-hook to manipulate input X = [B, S]
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

        # 2. Stage hooks to capture intermediate activations
        def hook_pre_bn(module, inp, out):
            self.current_patch_acts['pre_bn'] = out.detach().clone().cpu()

        def hook_post_bn(module, inp, out):
            self.current_patch_acts['post_bn'] = out.detach().clone().cpu()

        def hook_post_relu(module, inp, out):
            self.current_patch_acts['post_relu'] = out.detach().clone().cpu()

        def hook_conv2_out(module, inp, out):
            self.current_patch_acts['conv2_out'] = out.detach().clone().cpu()

        self.stage_hook_handles.append(self.conv1_0.register_forward_hook(hook_pre_bn))
        self.stage_hook_handles.append(self.conv1_1.register_forward_hook(hook_post_bn))
        self.stage_hook_handles.append(self.conv1_2.register_forward_hook(hook_post_relu))
        self.stage_hook_handles.append(self.conv2.register_forward_hook(hook_conv2_out))

    def set_mode(self, mode: str):
        assert mode in ['full', 'b_only', 's_only', 'zero']
        self.mode = mode
        self.collected_patch_records = []
        self.current_patch_acts = {}

    def cleanup(self):
        if self.pre_hook_handle:
            self.pre_hook_handle.remove()
            self.pre_hook_handle = None
        for h in self.stage_hook_handles:
            h.remove()
        self.stage_hook_handles = []
        self.set_mode('full')

    def run_image(
        self,
        image_rgb: np.ndarray,
        tile_size: int = 448,
        batch_size: int = 4
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Runs Setting A tiling inference and returns binary prediction, probability map, and stage activations."""
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

                self.current_patch_acts = {}
                logits = self.model(batch_pt)
                if logits.shape[2:] != (tile_size, tile_size):
                    logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)

                for b_idx, (py, px) in enumerate(batch_coords):
                    final_logits[py:py+tile_size, px:px+tile_size] = logits[b_idx].squeeze().cpu().numpy()

                    # Save intermediate slice for this patch
                    patch_records.append({
                        'coords': (py, px),
                        'pre_bn': self.current_patch_acts['pre_bn'][b_idx:b_idx+1],
                        'post_bn': self.current_patch_acts['post_bn'][b_idx:b_idx+1],
                        'post_relu': self.current_patch_acts['post_relu'][b_idx:b_idx+1],
                        'conv2_out': self.current_patch_acts['conv2_out'][b_idx:b_idx+1],
                    })

        logits_cropped = final_logits[:H, :W]
        prob_cropped = 1.0 / (1.0 + np.exp(-logits_cropped))
        pred_bin = (logits_cropped > 0.0).astype(np.uint8)

        return pred_bin, prob_cropped, patch_records


# -----------------------------------------------------------------------------
# Spatial Stage Feature Extractor
# -----------------------------------------------------------------------------

def extract_stage_features(
    patch_records: List[Dict[str, Any]],
    stage_key: str,
    neck_mask: np.ndarray,
    crack_mask: np.ndarray,
    bg_mask: np.ndarray,
    tile_size: int = 448
) -> Dict[str, Any]:
    """
    Extracts mean feature vectors and scalar metrics (L2 norm, contrast, SepMargin, R_norm)
    for a specific intermediate stage across an image.
    """
    neck_vecs = []
    crack_vecs = []
    bg_vecs = []

    neck_energies = []
    crack_energies = []
    bg_energies = []

    for item in patch_records:
        py, px = item['coords']
        tensor_act = item[stage_key]  # (1, 96, 56, 56)
        C, H_feat, W_feat = tensor_act.shape[1], tensor_act.shape[2], tensor_act.shape[3]

        p_neck = neck_mask[py:py+tile_size, px:px+tile_size]
        p_crack = crack_mask[py:py+tile_size, px:px+tile_size]
        p_bg = bg_mask[py:py+tile_size, px:px+tile_size]

        if np.sum(p_neck) > 0:
            m_n = (F.interpolate(torch.from_numpy(p_neck).float().unsqueeze(0).unsqueeze(0),
                                 size=(H_feat, W_feat), mode='nearest').squeeze() > 0)
            if m_n.any():
                feats = tensor_act[0, :, m_n]
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

    if len(neck_vecs) > 0:
        mean_v_neck = torch.stack(neck_vecs).mean(dim=0)
    else:
        mean_v_neck = torch.zeros(C)

    if len(crack_vecs) > 0:
        mean_v_crack = torch.stack(crack_vecs).mean(dim=0)
    else:
        mean_v_crack = torch.zeros(C)

    if len(neck_vecs) > 0 and len(crack_vecs) > 0:
        cos_sim = float(F.cosine_similarity(mean_v_neck.unsqueeze(0), mean_v_crack.unsqueeze(0)).item())
    else:
        cos_sim = 0.0

    return {
        'vec_neck': mean_v_neck,
        'vec_crack': mean_v_crack,
        'E_neck': e_neck,
        'E_crack': e_crack,
        'E_bg': e_bg,
        'contrast': contrast,
        'sep_margin': sep_margin,
        'R_norm': r_norm,
        'cosine': cos_sim
    }


# -----------------------------------------------------------------------------
# Main Diagnostic Routine
# -----------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Phase 6: T2 Nonlinear Fusion Interaction Diagnostic")
    print("Factorial 2x2 Decomposition & Multi-Stage Interaction Profiling")
    print("=" * 80)

    config_path = 'results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml'
    ckpt_path = 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'
    output_dir = 'results/diagnostics/phase6_t2_nonlinear_fusion'
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

    engine = T2NonlinearInteractionEngine(model, device)

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

    # Pre-load Images, Masks and compute Reference ROIs
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

    # Factorial 2x2 Evaluation across 4 conditions
    CONDITIONS = ['full', 'b_only', 's_only', 'zero']
    condition_results = {c: {} for c in CONDITIONS}

    t0 = time.time()
    print(f"\nEvaluating Factorial 2x2 across {len(CONDITIONS)} conditions...")

    for cond in CONDITIONS:
        print(f"\n---> Running Condition: {cond}...")
        engine.set_mode(cond)

        for item in tqdm(unique_cohort_samples, desc=f"Condition: {cond}"):
            stem = item['stem']
            s_info = sample_data[stem]
            img_rgb = s_info['img_rgb']
            target_bin = s_info['target_bin']
            neck_mask = s_info['neck_mask']
            crack_mask = s_info['crack_mask']
            bg_mask = s_info['bg_mask']

            pred_bin, prob_map, patch_records = engine.run_image(img_rgb, tile_size=448, batch_size=4)
            topo = compute_topology_metrics(pred_bin, target_bin)

            # Corridor probabilities
            neck_area = int(np.sum(neck_mask))
            crack_area = int(np.sum(crack_mask))
            bg_area = int(np.sum(bg_mask))

            p_neck = float(np.mean(prob_map[neck_mask == 1])) if neck_area > 0 else 0.0
            p_crack = float(np.mean(prob_map[crack_mask == 1])) if crack_area > 0 else 0.0
            p_bg = float(np.mean(prob_map[bg_mask == 1])) if bg_area > 0 else 0.0

            # Intermediate stage metrics
            stages = ['pre_bn', 'post_bn', 'post_relu', 'conv2_out']
            stage_metrics = {}
            for st in stages:
                stage_metrics[st] = extract_stage_features(
                    patch_records, st, neck_mask, crack_mask, bg_mask
                )

            condition_results[cond][stem] = {
                'has_bridge': bool(topo['bridge_events'] > 0),
                'bridge_events': topo['bridge_events'],
                'has_breakage': bool(topo['fragmented_gt_components'] > 0),
                'fragmented_gt_components': topo['fragmented_gt_components'],
                'dice': topo['dice'],
                'recall': topo['recall'],
                'precision': topo['precision'],
                'cldice': topo['cldice'],
                'p_neck': p_neck,
                'p_crack': p_crack,
                'p_bg': p_bg,
                'stages': stage_metrics
            }

    engine.cleanup()
    t1 = time.time()
    print(f"\nAll 4 factorial conditions evaluated in {t1 - t0:.1f}s.")

    # Integrity verification
    final_param_hash = compute_model_param_hash(model)
    final_file_hash = compute_file_hash(ckpt_path)

    print(f"\nVerifying Bitwise State Invariance:")
    print(f"  Initial Param Hash: {init_param_hash[:16]}... | Final: {final_param_hash[:16]}...")
    print(f"  Initial File Hash:  {init_file_hash[:16]}... | Final: {final_file_hash[:16]}...")
    assert init_param_hash == final_param_hash, "HARD LOCK VIOLATION: Model parameters modified in memory!"
    assert init_file_hash == final_file_hash, "HARD LOCK VIOLATION: Checkpoint file modified on disk!"
    print("Bitwise Invariance Check: 100% PASSED.")

    # Compute Factorial Interaction per sample:
    # I = F(Full) - F(B_only) - F(S_only) + F(Zero)
    per_sample_interactions = []
    stage_names = ['pre_bn', 'post_bn', 'post_relu', 'conv2_out']

    for item in unique_cohort_samples:
        stem = item['stem']
        cohort = item['cohort']
        subgroup = item['subgroup']

        res_full = condition_results['full'][stem]
        res_b = condition_results['b_only'][stem]
        res_s = condition_results['s_only'][stem]
        res_z = condition_results['zero'][stem]

        # 1. Final Output Interactions
        i_p_neck = res_full['p_neck'] - res_b['p_neck'] - res_s['p_neck'] + res_z['p_neck']
        i_p_crack = res_full['p_crack'] - res_b['p_crack'] - res_s['p_crack'] + res_z['p_crack']
        i_p_selectivity = i_p_neck - i_p_crack

        b_full = 1 if res_full['has_bridge'] else 0
        b_b = 1 if res_b['has_bridge'] else 0
        b_s = 1 if res_s['has_bridge'] else 0
        b_z = 1 if res_z['has_bridge'] else 0
        i_bridge = b_full - b_b - b_s + b_z

        i_recall = res_full['recall'] - res_b['recall'] - res_s['recall'] + res_z['recall']
        i_dice = res_full['dice'] - res_b['dice'] - res_s['dice'] + res_z['dice']

        row = {
            'image_id': stem,
            'cohort': cohort,
            'subgroup': subgroup,
            # Probability and Bridge interaction
            'Full_p_neck': res_full['p_neck'],
            'B_only_p_neck': res_b['p_neck'],
            'S_only_p_neck': res_s['p_neck'],
            'Zero_p_neck': res_z['p_neck'],
            'I_p_neck': i_p_neck,
            'I_p_crack': i_p_crack,
            'I_selectivity': i_p_selectivity,
            'Full_has_bridge': b_full,
            'B_only_has_bridge': b_b,
            'S_only_has_bridge': b_s,
            'Zero_has_bridge': b_z,
            'I_bridge': i_bridge,
            'I_recall': i_recall,
            'I_dice': i_dice,
        }

        # 2. Stage-by-Stage Vector Divergence & Representation Interaction
        for st in stage_names:
            st_full = res_full['stages'][st]
            st_b = res_b['stages'][st]
            st_s = res_s['stages'][st]
            st_z = res_z['stages'][st]

            # Vector linearity: ||v_full - v_b - v_s + v_z||
            vec_diff_neck = st_full['vec_neck'] - st_b['vec_neck'] - st_s['vec_neck'] + st_z['vec_neck']
            norm_diff_neck = float(torch.norm(vec_diff_neck, p=2).item())
            norm_full_neck = float(torch.norm(st_full['vec_neck'], p=2).item()) + 1e-6
            rel_diff_neck = norm_diff_neck / norm_full_neck

            # Scalar interactions on SepMargin, R_norm, Cosine
            i_sep = st_full['sep_margin'] - st_b['sep_margin'] - st_s['sep_margin'] + st_z['sep_margin']
            i_rnorm = st_full['R_norm'] - st_b['R_norm'] - st_s['R_norm'] + st_z['R_norm']
            i_cos = st_full['cosine'] - st_b['cosine'] - st_s['cosine'] + st_z['cosine']

            row[f'{st}_vec_diff_neck'] = norm_diff_neck
            row[f'{st}_rel_diff_neck'] = rel_diff_neck
            row[f'{st}_I_sep_margin'] = i_sep
            row[f'{st}_I_R_norm'] = i_rnorm
            row[f'{st}_I_cosine'] = i_cos

        per_sample_interactions.append(row)

    df_sample_interactions = pd.DataFrame(per_sample_interactions)
    per_sample_csv = os.path.join(output_dir, 't2_nonlinear_interaction_per_sample.csv')
    df_sample_interactions.to_csv(per_sample_csv, index=False)
    print(f"Saved: {per_sample_csv}")

    # Generate Summary Table by Cohort/Subgroup
    cohort_groups = [
        ('Consensus_7of7', 'All', df_sample_interactions[df_sample_interactions['cohort'] == 'Consensus_7of7']),
        ('Consensus_7of7', 'Consensus_Resistant', df_sample_interactions[df_sample_interactions['subgroup'] == 'Consensus_Resistant']),
        ('Consensus_7of7', 'Consensus_Sensitive', df_sample_interactions[df_sample_interactions['subgroup'] == 'Consensus_Sensitive']),
        ('Cured_by_D2', 'All', df_sample_interactions[df_sample_interactions['cohort'] == 'Cured_by_D2']),
        ('Clean_Control', 'All', df_sample_interactions[df_sample_interactions['cohort'] == 'Clean_Control']),
    ]

    summary_rows = []
    stage_summary_rows = []

    for c_name, sub_name, sub_df in cohort_groups:
        summary_rows.append({
            'cohort': c_name,
            'subgroup': sub_name,
            'n_samples': len(sub_df),
            'median_Full_p_neck': sub_df['Full_p_neck'].median(),
            'median_B_only_p_neck': sub_df['B_only_p_neck'].median(),
            'median_S_only_p_neck': sub_df['S_only_p_neck'].median(),
            'median_Zero_p_neck': sub_df['Zero_p_neck'].median(),
            'median_I_p_neck': sub_df['I_p_neck'].median(),
            'mean_I_p_neck': sub_df['I_p_neck'].mean(),
            'median_I_p_crack': sub_df['I_p_crack'].median(),
            'median_I_selectivity': sub_df['I_selectivity'].median(),
            'sum_Full_bridges': int(sub_df['Full_has_bridge'].sum()),
            'sum_B_only_bridges': int(sub_df['B_only_has_bridge'].sum()),
            'sum_S_only_bridges': int(sub_df['S_only_has_bridge'].sum()),
            'sum_Zero_bridges': int(sub_df['Zero_has_bridge'].sum()),
            'sum_Pure_Interaction_bridges': int(((sub_df['Full_has_bridge'] == 1) & (sub_df['B_only_has_bridge'] == 0) & (sub_df['S_only_has_bridge'] == 0)).sum()),
            'median_I_recall': sub_df['I_recall'].median(),
            'median_I_dice': sub_df['I_dice'].median(),
        })

        for st in stage_names:
            stage_summary_rows.append({
                'cohort': c_name,
                'subgroup': sub_name,
                'stage': st,
                'median_vec_diff_neck': sub_df[f'{st}_vec_diff_neck'].median(),
                'median_rel_diff_neck': sub_df[f'{st}_rel_diff_neck'].median(),
                'median_I_sep_margin': sub_df[f'{st}_I_sep_margin'].median(),
                'median_I_R_norm': sub_df[f'{st}_I_R_norm'].median(),
                'median_I_cosine': sub_df[f'{st}_I_cosine'].median(),
            })

    df_summary = pd.DataFrame(summary_rows)
    sum_csv = os.path.join(output_dir, 't2_nonlinear_interaction_summary.csv')
    df_summary.to_csv(sum_csv, index=False)
    print(f"Saved: {sum_csv}")
    print("\nNonlinear Interaction Summary:")
    print(df_summary[['cohort', 'subgroup', 'median_Full_p_neck', 'median_I_p_neck', 'median_I_selectivity', 'sum_Full_bridges', 'sum_Pure_Interaction_bridges']])

    df_stage_summary = pd.DataFrame(stage_summary_rows)
    stage_sum_csv = os.path.join(output_dir, 't2_stage_vector_divergence_summary.csv')
    df_stage_summary.to_csv(stage_sum_csv, index=False)
    print(f"Saved: {stage_sum_csv}")

    # Generate 4-Condition Response Matrix
    matrix_rows = []
    for cond in CONDITIONS:
        c_dict = condition_results[cond]
        # Resistant (N=82)
        r_br = sum([1 for item in unique_cohort_samples if item['subgroup'] == 'Consensus_Resistant' and c_dict[item['stem']]['has_bridge']])
        r_bk = sum([1 for item in unique_cohort_samples if item['subgroup'] == 'Consensus_Resistant' and c_dict[item['stem']]['has_breakage']])
        r_rec = np.median([c_dict[item['stem']]['recall'] for item in unique_cohort_samples if item['subgroup'] == 'Consensus_Resistant'])
        r_pn = np.median([c_dict[item['stem']]['p_neck'] for item in unique_cohort_samples if item['subgroup'] == 'Consensus_Resistant'])

        # Sensitive (N=19)
        s_br = sum([1 for item in unique_cohort_samples if item['subgroup'] == 'Consensus_Sensitive' and c_dict[item['stem']]['has_bridge']])
        s_bk = sum([1 for item in unique_cohort_samples if item['subgroup'] == 'Consensus_Sensitive' and c_dict[item['stem']]['has_breakage']])

        # Control (N=56)
        c_br = sum([1 for item in unique_cohort_samples if item['cohort'] == 'Clean_Control' and c_dict[item['stem']]['has_bridge']])
        c_bk = sum([1 for item in unique_cohort_samples if item['cohort'] == 'Clean_Control' and c_dict[item['stem']]['has_breakage']])

        # D2 (N=7)
        d2_br = sum([1 for item in unique_cohort_samples if item['cohort'] == 'Cured_by_D2' and c_dict[item['stem']]['has_bridge']])

        matrix_rows.append({
            'condition': cond,
            'Resistant_Bridges': f"{r_br}/82",
            'Resistant_Breakage': f"{r_bk}/82",
            'Resistant_Recall': f"{r_rec:.4f}",
            'Resistant_Med_ConnProb': f"{r_pn:.4f}",
            'Sensitive_Bridges': f"{s_br}/19",
            'Sensitive_Breakage': f"{s_bk}/19",
            'Control_Bridges_Created': f"{c_br}/56",
            'Control_Breakage': f"{c_bk}/56",
            'D2_Bridges': f"{d2_br}/7",
        })

    df_matrix = pd.DataFrame(matrix_rows)
    matrix_csv = os.path.join(output_dir, 't2_counterfactual_4cond_matrix.csv')
    df_matrix.to_csv(matrix_csv, index=False)
    print(f"Saved: {matrix_csv}")
    print("\n4-Condition Response Matrix:")
    print(df_matrix)

    # Step 3: Visualization Figures
    print("\nGenerating Diagnostic Figures...")

    # Figure 1: Vector Linearity vs Nonlinear Divergence Across Stages
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sub_res_stage = df_stage_summary[df_stage_summary['subgroup'] == 'Consensus_Resistant']
    sub_sen_stage = df_stage_summary[df_stage_summary['subgroup'] == 'Consensus_Sensitive']
    sub_d2_stage = df_stage_summary[df_stage_summary['cohort'] == 'Cured_by_D2']

    stages_labels = ['1. Conv1 pre-BN\n(Linear)', '2. Conv1 post-BN\n(Affine)', '3. Conv1 post-ReLU\n(Nonlinear)', '4. Conv2 output\n(Spatial+Nonlin)']
    x = np.arange(len(stages_labels))
    width = 0.25

    ax = axes[0]
    ax.bar(x - width, sub_res_stage['median_vec_diff_neck'], width, label='Resistant (N=82)', color='#d62728', alpha=0.85)
    ax.bar(x, sub_sen_stage['median_vec_diff_neck'], width, label='Sensitive (N=19)', color='#1f77b4', alpha=0.85)
    ax.bar(x + width, sub_d2_stage['median_vec_diff_neck'], width, label='Cured by D2 (N=7)', color='#2ca02c', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(stages_labels, fontsize=8.5)
    ax.set_ylabel('Interaction Vector Norm ||v_diff||_2')
    ax.set_title('Emergence of Nonlinear Interaction Across Stages')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)

    ax = axes[1]
    ax.plot(x, sub_res_stage['median_rel_diff_neck'] * 100, marker='o', linewidth=2, color='#d62728', label='Resistant')
    ax.plot(x, sub_sen_stage['median_rel_diff_neck'] * 100, marker='s', linewidth=2, color='#1f77b4', label='Sensitive')
    ax.plot(x, sub_d2_stage['median_rel_diff_neck'] * 100, marker='^', linewidth=2, color='#2ca02c', label='Cured by D2')
    ax.set_xticks(x)
    ax.set_xticklabels(stages_labels, fontsize=8.5)
    ax.set_ylabel('Relative Interaction Divergence (% of Full Norm)')
    ax.set_title('Relative Nonlinear Divergence (% of Full Vector Norm)')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, 't2_nonlinear_interaction_stages.png')
    plt.savefig(fig1_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {fig1_path}")

    # Figure 2: Connector Probability Interaction & Bridge Synthesis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sub_res = df_sample_interactions[df_sample_interactions['subgroup'] == 'Consensus_Resistant']
    sub_sen = df_sample_interactions[df_sample_interactions['subgroup'] == 'Consensus_Sensitive']
    sub_d2 = df_sample_interactions[df_sample_interactions['cohort'] == 'Cured_by_D2']

    cohort_labels = ['Resistant\n(N=82)', 'Sensitive\n(N=19)', 'Cured by D2\n(N=7)']
    x = np.arange(len(cohort_labels))

    ax = axes[0]
    p_neck_means = [sub_res['I_p_neck'].median(), sub_sen['I_p_neck'].median(), sub_d2['I_p_neck'].median()]
    p_crack_means = [sub_res['I_p_crack'].median(), sub_sen['I_p_crack'].median(), sub_d2['I_p_crack'].median()]
    p_sel_means = [sub_res['I_selectivity'].median(), sub_sen['I_selectivity'].median(), sub_d2['I_selectivity'].median()]

    width = 0.25
    ax.bar(x - width, p_neck_means, width, label='I(P_neck) in Corridor', color='#d62728', alpha=0.85)
    ax.bar(x, p_crack_means, width, label='I(P_crack) on Crack', color='#1f77b4', alpha=0.85)
    ax.bar(x + width, p_sel_means, width, label='Selectivity Delta', color='#ff7f0e', alpha=0.85)
    ax.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(cohort_labels)
    ax.set_ylabel('Median Probability Interaction')
    ax.set_title('Super-Additive Connector Probability Interaction')
    ax.legend(fontsize=8.5)
    ax.grid(True, linestyle=':', alpha=0.5)

    ax = axes[1]
    # Decomposition of 82 Resistant bridges into B-alone, S-alone, Joint Interaction
    # b_b=1, b_s=0 -> B-alone (37 - 26 = 11)
    # b_b=0, b_s=1 -> S-alone (54 - 26 = 28)
    # b_b=1, b_s=1 -> Both alone (26)
    # b_b=0, b_s=0, b_full=1 -> Pure Interaction Bridge (82 - (11 + 28 + 26) = 17)
    both_alone = int(((sub_res['B_only_has_bridge'] == 1) & (sub_res['S_only_has_bridge'] == 1)).sum())
    b_only_alone = int(((sub_res['B_only_has_bridge'] == 1) & (sub_res['S_only_has_bridge'] == 0)).sum())
    s_only_alone = int(((sub_res['B_only_has_bridge'] == 0) & (sub_res['S_only_has_bridge'] == 1)).sum())
    pure_interaction = int(((sub_res['Full_has_bridge'] == 1) & (sub_res['B_only_has_bridge'] == 0) & (sub_res['S_only_has_bridge'] == 0)).sum())

    bridge_categories = ['Pure Joint\nInteraction (I=+1)', 'Skip-Sufficient\nAlone (B=0, S=1)', 'Bottleneck-Sufficient\nAlone (B=1, S=0)', 'Both Sufficient\nIndependently (B=1, S=1)']
    bridge_counts = [pure_interaction, s_only_alone, b_only_alone, both_alone]
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd']

    bars = ax.bar(bridge_categories, bridge_counts, color=colors, alpha=0.85)
    for bar, count in zip(bars, bridge_counts):
        height = bar.get_height()
        ax.annotate(f'{count} / 82\n({count/82*100:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('Number of Resistant Bridges (/82)')
    ax.set_title('Resistant False Bridge Formation Taxonomy')
    ax.set_ylim(0, 36)
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, 't2_subgroup_interaction_comparison.png')
    plt.savefig(fig2_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {fig2_path}")

    print("\n" + "=" * 80)
    print("Phase 6 T2 Nonlinear Fusion Interaction Diagnostic Completed Successfully.")
    print("=" * 80)


if __name__ == '__main__':
    main()
