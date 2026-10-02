#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_downstream_reconvergence_provenance.py

Phase 6: Downstream Re-convergence Provenance Diagnostic
Scientific Goal:
After representation separation is restored at the stem (under alpha=0.00, where stem SepMargin=+0.23),
determine WHERE in the downstream network and BY WHAT MECHANISM the bridge-like representation
re-emerges / re-converges to commit false bridges in the 82 downstream-resistant Consensus cases.

Pipeline Trajectory (21 Observation Points):
1. raw_conv          (112x112, 48 ch)
2. stem_ln           (112x112, 48 ch)
3. stage0_block0     (112x112, 48 ch)
4. stage0_block1     (112x112, 48 ch = pre-SAGE)
5. stage0_post_sage  (112x112, 48 ch)
6. stage1_post_sage  (56x56, 96 ch)
7. stage2_post_sage  (28x28, 192 ch)
8. stage3_post_sage  (14x14, 384 ch)
9. pre_vit           (14x14, 192 ch)
10. vit_block_0      (14x14, 192 ch)
11. vit_block_1      (14x14, 192 ch)
12. vit_block_2      (14x14, 192 ch)
13. vit_block_3      (14x14, 192 ch)
14. bottleneck       (14x14, 384 ch)
15. S2               (28x28, 192 ch)
16. T1               (56x56, 192 ch - upsample)
17. T2               (56x56, 288 ch - post-skip concat)
18. T3               (56x56, 96 ch  - post-conv1)
19. T4_S1            (56x56, 96 ch  - post-conv2 / S1)
20. S0               (112x112, 48 ch - decoder block 2)
21. final_head       (448x448, 1 ch - logits)

Cohorts Evaluated:
- Consensus 7/7: N = 101 (segmented into Stem-Sensitive N=19 vs Downstream-Resistant N=82)
- Cured by D2: N = 7
- Clean Control: N = 56 (gt_cc >= 2, 0 bridges across all 7 models)
Total N = 164 validation samples.

Counterfactual Conditions:
- alpha = 1.00 (Baseline full model)
- alpha = 0.00 (DC-only stem: restored stem separation)
- alpha = 0.25 (Intermediate contrast level)

HARD LOCKS:
- DIAGNOSTIC-ONLY: Zero training, zero gradients, zero permanent checkpoint mutations.
- Bitwise verification of checkpoint file and model named_parameters SHA256 before and after.
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
# Pipeline Hook Architecture (21 Stages)
# -----------------------------------------------------------------------------

STAGE_PIPELINE = [
    'raw_conv',
    'stem_ln',
    'stage0_block0',
    'stage0_block1',
    'stage0_post_sage',
    'stage1_post_sage',
    'stage2_post_sage',
    'stage3_post_sage',
    'pre_vit',
    'vit_block_0',
    'vit_block_1',
    'vit_block_2',
    'vit_block_3',
    'bottleneck',
    'S2',
    'T1',
    'T2',
    'T3',
    'T4_S1',
    'S0',
    'final_head'
]

STAGE_SHAPES_CHECK = {
    'raw_conv': (48, 112, 112),
    'stem_ln': (48, 112, 112),
    'stage0_block0': (48, 112, 112),
    'stage0_block1': (48, 112, 112),
    'stage0_post_sage': (48, 112, 112),
    'stage1_post_sage': (96, 56, 56),
    'stage2_post_sage': (192, 28, 28),
    'stage3_post_sage': (384, 14, 14),
    'pre_vit': (196, 192),
    'bottleneck': (384, 14, 14),
    'S2': (192, 28, 28),
    'T1': (192, 56, 56),
    'T2': (288, 56, 56),
    'T3': (96, 56, 56),
    'T4_S1': (96, 56, 56),
    'S0': (48, 112, 112),
}

class DownstreamProvenanceEngine:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.conv = model.backbone.convnext.stem[0]

        # Save bitwise original stem weights
        self.orig_w = self.conv.weight.data.clone()
        self.orig_b = self.conv.bias.data.clone()

        # Factorize into DC and AC
        self.w_dc = self.orig_w.mean(dim=(2, 3), keepdim=True).expand_as(self.orig_w)
        self.w_ac = self.orig_w - self.w_dc

        self.raw_activations: Dict[str, torch.Tensor] = {}
        self.handles = []
        self._register_all_hooks()

    def _register_all_hooks(self):
        # 1. raw_conv
        self.handles.append(self.conv.register_forward_hook(
            lambda m, inp, out: self.raw_activations.update({'raw_conv': (out[0] if isinstance(out, tuple) else out).detach()})
        ))
        # 2. stem_ln
        self.handles.append(self.model.backbone.convnext.stem[1].register_forward_hook(
            lambda m, inp, out: self.raw_activations.update({'stem_ln': (out[0] if isinstance(out, tuple) else out).detach()})
        ))
        # 3. stage0_block0
        blk0 = self.model.backbone.convnext.stages[0].main_block.module.blocks[0]
        def blk0_hook(m, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if t.dim() == 4 and t.shape[1] == 48 and t.shape[2] == 112:
                self.raw_activations['stage0_block0'] = t.detach()
        self.handles.append(blk0.register_forward_hook(blk0_hook))

        # 4. stage0_block1 (pre-SAGE)
        blk1 = self.model.backbone.convnext.stages[0].main_block.module.blocks[1]
        def blk1_hook(m, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if t.dim() == 4 and t.shape[1] == 48 and t.shape[2] == 112:
                self.raw_activations['stage0_block1'] = t.detach()
        self.handles.append(blk1.register_forward_hook(blk1_hook))

        # 5-8. CNN stages post-SAGE
        for i in range(4):
            stg = self.model.backbone.convnext.stages[i]
            exp_c = [48, 96, 192, 384][i]
            exp_h = [112, 56, 28, 14][i]
            def make_stg_hook(idx, c, h):
                def hook(m, inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    if t.dim() == 4 and t.shape[1] == c and t.shape[2] == h:
                        self.raw_activations[f'stage{idx}_post_sage'] = t.detach()
                return hook
            self.handles.append(stg.register_forward_hook(make_stg_hook(i, exp_c, exp_h)))

        # 9. pre_vit
        self.handles.append(self.model.backbone.pre_transformer_norm.register_forward_hook(
            lambda m, inp, out: self.raw_activations.update({'pre_vit': (out[0] if isinstance(out, tuple) else out).detach()})
        ))

        # 10-13. ViT blocks
        for i in range(4):
            tb = self.model.backbone.transformer_blocks[i]
            def make_vit_hook(idx):
                def hook(m, inp, out):
                    self.raw_activations[f'vit_block_{idx}'] = (out[0] if isinstance(out, tuple) else out).detach()
                return hook
            self.handles.append(tb.register_forward_hook(make_vit_hook(i)))

        # 14. bottleneck (input to decoder_blocks[0])
        self.handles.append(self.model.decoder.decoder_blocks[0].register_forward_pre_hook(
            lambda m, args: self.raw_activations.update({'bottleneck': args[0].detach()})
        ))
        # 15. S2 (output of decoder_blocks[0])
        self.handles.append(self.model.decoder.decoder_blocks[0].register_forward_hook(
            lambda m, inp, out: self.raw_activations.update({'S2': (out[0] if isinstance(out, tuple) else out).detach()})
        ))
        # 16. T1 (upsample output inside Block 1)
        self.handles.append(self.model.decoder.decoder_blocks[1].upsample.register_forward_hook(
            lambda m, inp, out: self.raw_activations.update({'T1': (out[0] if isinstance(out, tuple) else out).detach()})
        ))
        # 17. T2 (input to conv1 of Block 1 post-skip concat)
        self.handles.append(self.model.decoder.decoder_blocks[1].conv1.register_forward_pre_hook(
            lambda m, args: self.raw_activations.update({'T2': args[0].detach()})
        ))
        # 18. T3 (output of conv1 of Block 1)
        self.handles.append(self.model.decoder.decoder_blocks[1].conv1.register_forward_hook(
            lambda m, inp, out: self.raw_activations.update({'T3': (out[0] if isinstance(out, tuple) else out).detach()})
        ))
        # 19. T4_S1 (output of Block 1 / post-conv2)
        self.handles.append(self.model.decoder.decoder_blocks[1].register_forward_hook(
            lambda m, inp, out: self.raw_activations.update({'T4_S1': (out[0] if isinstance(out, tuple) else out).detach()})
        ))
        # 20. S0 (output of decoder Block 2)
        self.handles.append(self.model.decoder.decoder_blocks[2].register_forward_hook(
            lambda m, inp, out: self.raw_activations.update({'S0': (out[0] if isinstance(out, tuple) else out).detach()})
        ))

    def set_alpha(self, alpha: float):
        w_alpha = self.w_dc + float(alpha) * self.w_ac
        self.conv.weight.data.copy_(w_alpha)

    def restore_weights(self):
        self.conv.weight.data.copy_(self.orig_w)

    def cleanup(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.restore_weights()

    def run_image(
        self,
        image_rgb: np.ndarray,
        tile_size: int = 448,
        batch_size: int = 4
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, List[torch.Tensor]]]:
        """Runs Setting A tiling inference with extraction of all 21 pipeline stages."""
        H, W = image_rgb.shape[:2]
        pad_h = (tile_size - (H % tile_size)) % tile_size
        pad_w = (tile_size - (W % tile_size)) % tile_size
        padded_img = cv2.copyMakeBorder(image_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        pH, pW = padded_img.shape[:2]

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        patches = []
        coords = []
        for y in range(0, pH, tile_size):
            for x in range(0, pW, tile_size):
                patch = padded_img[y:y+tile_size, x:x+tile_size]
                p_t = torch.from_numpy(patch).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
                p_t = (p_t - mean) / std
                patches.append(p_t)
                coords.append((y, x))

        final_logits = np.zeros((pH, pW), dtype=np.float32)
        stage_full_energy = {s: np.zeros((pH, pW), dtype=np.float32) for s in STAGE_PIPELINE}
        stage_patch_acts = {s: [] for s in STAGE_PIPELINE}

        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch_p = torch.cat(patches[i:i+batch_size], dim=0)
                batch_coords = coords[i:i+batch_size]

                logits = self.model(batch_p)
                if logits.shape[2:] != (tile_size, tile_size):
                    logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)

                self.raw_activations['final_head'] = logits.detach()

                for b_idx, (py, px) in enumerate(batch_coords):
                    final_logits[py:py+tile_size, px:px+tile_size] = logits[b_idx].squeeze().cpu().numpy()

                # Process all 21 stages
                for s_name in STAGE_PIPELINE:
                    act = self.raw_activations[s_name]

                    # Token sequence reshape for ViT blocks
                    if act.dim() == 3 and act.shape[1] == 196:
                        # (B, 196, 192) -> (B, 192, 14, 14)
                        act = act.transpose(1, 2).reshape(act.shape[0], act.shape[2], 14, 14)

                    if s_name == 'final_head':
                        energy_b = torch.sigmoid(act) # (B, 1, 448, 448)
                    else:
                        energy_b = torch.norm(act, p=2, dim=1, keepdim=True) # (B, 1, h, w)

                    if energy_b.shape[2:] != (tile_size, tile_size):
                        energy_up = F.interpolate(energy_b, size=(tile_size, tile_size), mode='bilinear', align_corners=False)
                    else:
                        energy_up = energy_b

                    for b_idx, (py, px) in enumerate(batch_coords):
                        stage_full_energy[s_name][py:py+tile_size, px:px+tile_size] = energy_up[b_idx].squeeze().cpu().numpy()
                        stage_patch_acts[s_name].append({
                            'coords': (py, px),
                            'act': act[b_idx:b_idx+1].cpu()
                        })

        logits_cropped = final_logits[:H, :W]
        prob_cropped = 1.0 / (1.0 + np.exp(-logits_cropped))
        pred_bin = (logits_cropped > 0.0).astype(np.uint8)
        stage_cropped_energy = {s: stage_full_energy[s][:H, :W] for s in STAGE_PIPELINE}

        return pred_bin, prob_cropped, stage_cropped_energy, stage_patch_acts


# -----------------------------------------------------------------------------
# Main Diagnostic Routine
# -----------------------------------------------------------------------------

COUNTERFACTUAL_ALPHAS = [1.00, 0.25, 0.00]

def main():
    print("=" * 80)
    print("Phase 6: Downstream Re-convergence Provenance Diagnostic (21 Stages)")
    print("=" * 80)

    config_path = 'results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml'
    ckpt_path = 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'
    output_dir = 'results/diagnostics/phase6_downstream_reconvergence'
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

    engine = DownstreamProvenanceEngine(model, device)

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

    # Pre-identify Stem-Sensitive vs Downstream-Resistant cases from previous AC attenuation run
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

    cohort_list = []
    for s in sorted(list(c7_stems)):
        cohort_list.append((s, 'Consensus_7of7'))
    for s in sorted(list(d2_cured_stems)):
        cohort_list.append((s, 'Cured_by_D2'))
    for s in sorted(list(clean_stems)):
        cohort_list.append((s, 'Clean_Control'))

    # Preload sample data
    print("\nPreloading sample data and setting baseline reference corridors...")
    sample_data = {}
    for stem, cohort in tqdm(cohort_list, desc="Preloading"):
        img_path = f"datasets/Crack500_ready/val/images/{stem}.jpg"
        mask_path = f"datasets/Crack500_ready/val/masks/{stem}.png"
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        target_bin = (mask > 0).astype(np.uint8)

        subgroup = 'Consensus_Resistant' if stem in downstream_resistant_stems else ('Consensus_Sensitive' if stem in stem_sensitive_stems else cohort)

        sample_data[stem] = {
            'img_rgb': img_rgb,
            'target_bin': target_bin,
            'cohort': cohort,
            'subgroup': subgroup
        }

    # Step 1: Establish Baseline Reference ROIs at alpha=1.0
    engine.set_alpha(1.00)
    baseline_rois = {}
    for stem, cohort in tqdm(cohort_list, desc="Establishing Baseline Reference ROIs"):
        img_rgb = sample_data[stem]['img_rgb']
        target_bin = sample_data[stem]['target_bin']
        pred_bin, prob_map, stage_maps, _ = engine.run_image(img_rgb)

        if cohort in ('Consensus_7of7', 'Cured_by_D2'):
            neck_mask, crack_mask, bg_mask, pair_records = isolate_bridged_pairs_and_rois(pred_bin, target_bin)
        else:
            neck_mask, crack_mask, bg_mask, pair_records = isolate_clean_pairs_and_rois(target_bin)

        baseline_rois[stem] = {
            'neck_mask': neck_mask,
            'crack_mask': crack_mask,
            'bg_mask': bg_mask,
            'pair_records': pair_records
        }

    # Step 2: Run Counterfactual Conditions across all 21 Stages
    per_sample_records = []
    pairwise_records = []
    first_reemergence_records = []

    t0 = time.time()
    for alpha in COUNTERFACTUAL_ALPHAS:
        print(f"\nEvaluating Condition: alpha = {alpha:.2f} across all 21 stages...")
        engine.set_alpha(alpha)

        for stem, cohort in tqdm(cohort_list, desc=f"alpha={alpha:.2f}"):
            img_rgb = sample_data[stem]['img_rgb']
            target_bin = sample_data[stem]['target_bin']
            subgroup = sample_data[stem]['subgroup']

            neck_mask = baseline_rois[stem]['neck_mask']
            crack_mask = baseline_rois[stem]['crack_mask']
            bg_mask = baseline_rois[stem]['bg_mask']
            pair_records = baseline_rois[stem]['pair_records']

            pred_bin, prob_map, stage_maps, stage_acts = engine.run_image(img_rgb)
            topo = compute_topology_metrics(pred_bin, target_bin)

            n_neck_px = int(np.sum(neck_mask))
            n_crack_px = int(np.sum(crack_mask))
            n_bg_px = int(np.sum(bg_mask))

            sample_entry = {
                'image_id': stem,
                'cohort': cohort,
                'subgroup': subgroup,
                'alpha': alpha,
                'has_bridge': bool(topo['bridge_events'] > 0),
                'bridge_events': topo['bridge_events'],
                'merged_gt_components': topo['merged_gt_components'],
                'has_breakage': bool(topo['fragmented_gt_components'] > 0),
                'fragmented_gt_components': topo['fragmented_gt_components'],
                'dice': topo['dice'],
                'precision': topo['precision'],
                'recall': topo['recall'],
                'cldice': topo['cldice'],
            }

            first_reemerge_50 = 'None'
            first_reemerge_70 = 'None'

            for s_name in STAGE_PIPELINE:
                emap = stage_maps[s_name]

                e_neck = float(np.mean(emap[neck_mask == 1])) if n_neck_px > 0 else 0.0
                e_crack = float(np.mean(emap[crack_mask == 1])) if n_crack_px > 0 else 0.0
                e_bg = float(np.mean(emap[bg_mask == 1])) if n_bg_px > 0 else 0.0

                c_neck_crack = float(e_neck / (e_crack + 1e-6))
                sep_margin = float(1.0 - c_neck_crack)
                denom = (e_crack - e_bg)
                r_norm = float((e_neck - e_bg) / denom) if abs(denom) > 1e-5 else 0.0

                sample_entry[f'{s_name}_R_norm'] = r_norm
                sample_entry[f'{s_name}_sep_margin'] = sep_margin
                sample_entry[f'{s_name}_contrast'] = c_neck_crack
                sample_entry[f'{s_name}_E_neck'] = e_neck
                sample_entry[f'{s_name}_E_crack'] = e_crack
                sample_entry[f'{s_name}_E_bg'] = e_bg

                # Track earliest re-emergence under alpha=0.0
                if alpha == 0.00:
                    if first_reemerge_50 == 'None' and r_norm >= 0.50:
                        first_reemerge_50 = s_name
                    if first_reemerge_70 == 'None' and r_norm >= 0.70:
                        first_reemerge_70 = s_name

            if alpha == 0.00:
                sample_entry['first_reemerge_stage_50'] = first_reemerge_50
                sample_entry['first_reemerge_stage_70'] = first_reemerge_70
                first_reemergence_records.append({
                    'image_id': stem,
                    'cohort': cohort,
                    'subgroup': subgroup,
                    'first_reemerge_50': first_reemerge_50,
                    'first_reemerge_70': first_reemerge_70
                })

            per_sample_records.append(sample_entry)

            # Pairwise spatial separation
            for pair in pair_records:
                neck_p = pair['neck_448']
                crack_p = pair['mask_i_448'] | pair['mask_j_448']

                for s_name in STAGE_PIPELINE:
                    emap = stage_maps[s_name]
                    en = float(np.mean(emap[neck_p == 1])) if np.sum(neck_p) > 0 else 0.0
                    ec = float(np.mean(emap[crack_p == 1]))
                    ratio = float(en / (ec + 1e-6))
                    sep_m = float(1.0 - ratio)
                    is_sep = bool(sep_m >= 0.20 and not pair['grid_cell_collision'])

                    pairwise_records.append({
                        'image_id': stem,
                        'cohort': cohort,
                        'subgroup': subgroup,
                        'alpha': alpha,
                        'stage': s_name,
                        'gt_i': pair['gt_i'],
                        'gt_j': pair['gt_j'],
                        'is_separable': is_sep,
                        'sep_margin': sep_m,
                        'contrast': ratio
                    })

    engine.cleanup()
    t1 = time.time()
    print(f"\nAll 21 stages evaluated across conditions in {t1 - t0:.1f}s.")

    # Integrity verification
    final_param_hash = compute_model_param_hash(model)
    final_file_hash = compute_file_hash(ckpt_path)
    assert init_param_hash == final_param_hash, f"FATAL: Model weights modified! ({init_param_hash} vs {final_param_hash})"
    assert init_file_hash == final_file_hash, f"FATAL: Checkpoint file on disk modified! ({init_file_hash} vs {final_file_hash})"
    print(f"Model Parameters State Verified: Bitwise Identical ({final_param_hash[:16]}...)")
    print(f"Disk Checkpoint File Verified: Bitwise Identical ({final_file_hash[:16]}...)")

    # DataFrames
    df_samples = pd.DataFrame(per_sample_records)
    df_pairs = pd.DataFrame(pairwise_records)
    df_reemerge = pd.DataFrame(first_reemergence_records)

    sample_csv = os.path.join(output_dir, 'downstream_reconvergence_per_sample.csv')
    df_samples.to_csv(sample_csv, index=False)
    print(f"Saved: {sample_csv}")

    # Build Summary by Cohort & Subgroup & Alpha Table
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

        n_sub = len(sub_cohort['image_id'].unique())
        if n_sub == 0:
            continue

        for alpha in COUNTERFACTUAL_ALPHAS:
            sub_a = sub_cohort[sub_cohort['alpha'] == alpha]

            for s_name in STAGE_PIPELINE:
                r_col = f'{s_name}_R_norm'
                sep_col = f'{s_name}_sep_margin'
                c_col = f'{s_name}_contrast'

                summary_rows.append({
                    'cohort': cohort_name,
                    'subgroup': grp_name,
                    'alpha': alpha,
                    'stage': s_name,
                    'n_samples': n_sub,
                    'median_R_norm': float(sub_a[r_col].median()),
                    'mean_R_norm': float(sub_a[r_col].mean()),
                    'p25_R_norm': float(np.percentile(sub_a[r_col], 25)),
                    'p75_R_norm': float(np.percentile(sub_a[r_col], 75)),
                    'pct_ge_50': float((sub_a[r_col] >= 0.50).mean() * 100),
                    'pct_ge_70': float((sub_a[r_col] >= 0.70).mean() * 100),
                    'median_sep_margin': float(sub_a[sep_col].median()),
                    'mean_sep_margin': float(sub_a[sep_col].mean()),
                    'median_contrast': float(sub_a[c_col].median()),
                })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, 'downstream_reconvergence_summary.csv')
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    # Build Re-emergence Distribution Table for Alpha = 0.0
    reemerge_rows = []
    for grp_label in ['Consensus_Resistant', 'Consensus_Sensitive', 'Cured_by_D2', 'Clean_Control']:
        if grp_label in ('Consensus_Resistant', 'Consensus_Sensitive'):
            sub_re = df_reemerge[df_reemerge['subgroup'] == grp_label]
        else:
            sub_re = df_reemerge[df_reemerge['cohort'] == grp_label]

        n_g = len(sub_re)
        if n_g == 0:
            continue
        c50 = sub_re['first_reemerge_50'].value_counts().to_dict()
        c70 = sub_re['first_reemerge_70'].value_counts().to_dict()

        for s_name in STAGE_PIPELINE + ['None']:
            reemerge_rows.append({
                'subgroup': grp_label,
                'stage': s_name,
                'n_samples': n_g,
                'count_reemerge_50': c50.get(s_name, 0),
                'pct_reemerge_50': float(c50.get(s_name, 0) / n_g * 100),
                'count_reemerge_70': c70.get(s_name, 0),
                'pct_reemerge_70': float(c70.get(s_name, 0) / n_g * 100),
            })

    df_re_dist = pd.DataFrame(reemerge_rows)
    re_dist_csv = os.path.join(output_dir, 'first_reemergence_distribution.csv')
    df_re_dist.to_csv(re_dist_csv, index=False)
    print(f"Saved: {re_dist_csv}")

    # Build Pairwise Reconvergence Summary
    pair_summary_rows = []
    for cohort_name in ['Consensus_7of7', 'Cured_by_D2', 'Clean_Control']:
        sub_p = df_pairs[df_pairs['cohort'] == cohort_name]
        if len(sub_p) == 0:
            continue
        n_pairs = len(sub_p[(sub_p['alpha'] == 1.0) & (sub_p['stage'] == 'raw_conv')])

        for alpha in COUNTERFACTUAL_ALPHAS:
            for s_name in STAGE_PIPELINE:
                sub_pas = sub_p[(sub_p['alpha'] == alpha) & (sub_p['stage'] == s_name)]
                pct_sep = float(sub_pas['is_separable'].mean() * 100)
                pair_summary_rows.append({
                    'cohort': cohort_name,
                    'alpha': alpha,
                    'stage': s_name,
                    'n_pairs_evaluated': n_pairs,
                    'pct_pairs_spatially_separable': pct_sep,
                    'pct_pairs_representation_collapsed': float(100.0 - pct_sep),
                    'median_separation_margin': float(sub_pas['sep_margin'].median()),
                    'median_contrast': float(sub_pas['contrast'].median())
                })

    df_pair_sum = pd.DataFrame(pair_summary_rows)
    pair_sum_csv = os.path.join(output_dir, 'pairwise_reconvergence_summary.csv')
    df_pair_sum.to_csv(pair_sum_csv, index=False)
    print(f"Saved: {pair_sum_csv}")

    # -------------------------------------------------------------------------
    # Visualization Figures
    # -------------------------------------------------------------------------

    # Figure 1: Trajectory Comparison Across 21 Stages (Alpha=1.00 vs Alpha=0.00)
    fig, axes = plt.subplots(2, 2, figsize=(24, 12))
    fig.suptitle("Downstream Re-convergence Trajectory: Baseline (alpha=1.0) vs DC-Only (alpha=0.0)", fontsize=16, fontweight='bold', y=0.98)

    stage_x_indices = list(range(len(STAGE_PIPELINE)))

    # A. Median R_norm (Alpha=1.0)
    for c_name, col, mark in [('Consensus_7of7', '#d95f02', 'o'), ('Cured_by_D2', '#7570b3', 's'), ('Clean_Control', '#1b9e77', '^')]:
        sub = df_summary[(df_summary['cohort'] == c_name) & (df_summary['subgroup'] == 'All') & (df_summary['alpha'] == 1.00)]
        axes[0, 0].plot(stage_x_indices, sub['median_R_norm'], marker=mark, color=col, linewidth=2.0, label=c_name)
    axes[0, 0].axhline(0.50, color='gray', linestyle='--', alpha=0.7, label='50% Emergence')
    axes[0, 0].set_title("Baseline (alpha=1.00): Median R_norm Trajectory", fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(stage_x_indices)
    axes[0, 0].set_xticklabels(STAGE_PIPELINE, rotation=45, ha='right', fontsize=8.5)
    axes[0, 0].set_ylabel("Median R_norm")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=9)

    # B. Median R_norm (Alpha=0.0)
    for c_name, col, mark in [('Consensus_7of7', '#d95f02', 'o'), ('Cured_by_D2', '#7570b3', 's'), ('Clean_Control', '#1b9e77', '^')]:
        sub = df_summary[(df_summary['cohort'] == c_name) & (df_summary['subgroup'] == 'All') & (df_summary['alpha'] == 0.00)]
        axes[0, 1].plot(stage_x_indices, sub['median_R_norm'], marker=mark, color=col, linewidth=2.0, label=c_name)
    axes[0, 1].axhline(0.50, color='gray', linestyle='--', alpha=0.7, label='50% Emergence')
    axes[0, 1].set_title("DC-Only (alpha=0.00): Median R_norm Trajectory (Where Does It Re-emerge?)", fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(stage_x_indices)
    axes[0, 1].set_xticklabels(STAGE_PIPELINE, rotation=45, ha='right', fontsize=8.5)
    axes[0, 1].set_ylabel("Median R_norm")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=9)

    # C. Median SepMargin (Alpha=1.0)
    for c_name, col, mark in [('Consensus_7of7', '#d95f02', 'o'), ('Cured_by_D2', '#7570b3', 's'), ('Clean_Control', '#1b9e77', '^')]:
        sub = df_summary[(df_summary['cohort'] == c_name) & (df_summary['subgroup'] == 'All') & (df_summary['alpha'] == 1.00)]
        axes[1, 0].plot(stage_x_indices, sub['median_sep_margin'], marker=mark, color=col, linewidth=2.0, label=c_name)
    axes[1, 0].axhline(0.20, color='blue', linestyle=':', alpha=0.7, label='Separation Valley (0.20)')
    axes[1, 0].axhline(0.00, color='black', linestyle='-', alpha=0.4)
    axes[1, 0].set_title("Baseline (alpha=1.00): Median SepMargin Trajectory", fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(stage_x_indices)
    axes[1, 0].set_xticklabels(STAGE_PIPELINE, rotation=45, ha='right', fontsize=8.5)
    axes[1, 0].set_ylabel("SepMargin (1 - E_neck/E_crack)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=9)

    # D. Median SepMargin (Alpha=0.0)
    for c_name, col, mark in [('Consensus_7of7', '#d95f02', 'o'), ('Cured_by_D2', '#7570b3', 's'), ('Clean_Control', '#1b9e77', '^')]:
        sub = df_summary[(df_summary['cohort'] == c_name) & (df_summary['subgroup'] == 'All') & (df_summary['alpha'] == 0.00)]
        axes[1, 1].plot(stage_x_indices, sub['median_sep_margin'], marker=mark, color=col, linewidth=2.0, label=c_name)
    axes[1, 1].axhline(0.20, color='blue', linestyle=':', alpha=0.7, label='Separation Valley (0.20)')
    axes[1, 1].axhline(0.00, color='black', linestyle='-', alpha=0.4)
    axes[1, 1].set_title("DC-Only (alpha=0.00): Median SepMargin Trajectory (Separation Collapse Point)", fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(stage_x_indices)
    axes[1, 1].set_xticklabels(STAGE_PIPELINE, rotation=45, ha='right', fontsize=8.5)
    axes[1, 1].set_ylabel("SepMargin (1 - E_neck/E_crack)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=9)

    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, 'trajectory_comparison_alpha1_vs_alpha0.png')
    plt.savefig(fig1_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig1_path}")

    # Figure 2: Subgroup Divergence Trajectory (Stem-Sensitive N=19 vs Downstream-Resistant N=82 under Alpha=0.0)
    fig, axes = plt.subplots(1, 2, figsize=(22, 6))
    fig.suptitle("Subgroup Divergence Trajectory Under DC-Only (alpha=0.00): Stem-Sensitive vs Downstream-Resistant", fontsize=15, fontweight='bold')

    sub_res = df_summary[(df_summary['cohort'] == 'Consensus_7of7') & (df_summary['subgroup'] == 'Consensus_Resistant') & (df_summary['alpha'] == 0.00)]
    sub_sen = df_summary[(df_summary['cohort'] == 'Consensus_7of7') & (df_summary['subgroup'] == 'Consensus_Sensitive') & (df_summary['alpha'] == 0.00)]

    axes[0].plot(stage_x_indices, sub_res['median_R_norm'], marker='o', color='#e41a1c', linewidth=2.5, label='Downstream-Resistant (N=82, Bridge Persists)')
    axes[0].plot(stage_x_indices, sub_sen['median_R_norm'], marker='s', color='#377eb8', linewidth=2.5, label='Stem-Sensitive (N=19, Bridge Cured)')
    axes[0].axhline(0.50, color='gray', linestyle='--', alpha=0.7, label='50% Emergence')
    axes[0].set_title("Median R_norm Trajectory: Resistant vs Sensitive Cases (alpha=0.00)", fontsize=11, fontweight='bold')
    axes[0].set_xticks(stage_x_indices)
    axes[0].set_xticklabels(STAGE_PIPELINE, rotation=45, ha='right', fontsize=8.5)
    axes[0].set_ylabel("Median R_norm")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9.5)

    axes[1].plot(stage_x_indices, sub_res['median_sep_margin'], marker='o', color='#e41a1c', linewidth=2.5, label='Downstream-Resistant (N=82, Bridge Persists)')
    axes[1].plot(stage_x_indices, sub_sen['median_sep_margin'], marker='s', color='#377eb8', linewidth=2.5, label='Stem-Sensitive (N=19, Bridge Cured)')
    axes[1].axhline(0.20, color='blue', linestyle=':', alpha=0.7, label='Separation Valley (0.20)')
    axes[1].axhline(0.00, color='black', linestyle='-', alpha=0.4)
    axes[1].set_title("Median Separation Margin Trajectory: Divergence Point (alpha=0.00)", fontsize=11, fontweight='bold')
    axes[1].set_xticks(stage_x_indices)
    axes[1].set_xticklabels(STAGE_PIPELINE, rotation=45, ha='right', fontsize=8.5)
    axes[1].set_ylabel("SepMargin (1 - E_neck/E_crack)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9.5)

    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, 'subgroup_divergence_trajectory.png')
    plt.savefig(fig2_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig2_path}")

    print("\n" + "=" * 80)
    print("Downstream Re-convergence Provenance Diagnostic Completed Successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
