#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_stem_ac_attenuation_diagnostic.py

Phase 6: AC Attenuation Counterfactual Diagnostic at Stem Conv
Scientific Goal:
Evaluate the causal necessity and trade-offs of the AC (spatially contrast-sensitive)
component of the stride-4 stem projection Conv2d(3, 48, 4, 4, s=4).

Counterfactual Formulation:
y_alpha = b + y_DC + alpha * y_AC
where:
- W_alpha = W_DC + alpha * W_AC
- alpha in {0.0, 0.25, 0.50, 0.75, 1.00}
- alpha = 1.00 reproduces exact baseline (full Conv).
- alpha = 0.00 corresponds to DC-only (spatial pooling of local color + 1x1 projection).

Cohorts Evaluated:
- Consensus 7/7: N = 101
- Cured by D2: N = 7
- Clean Control: N = 56 (gt_cc >= 2, 0 bridges across all 7 models)
Total N = 164 validation samples.

Dual Readout:
1. Raw-stem feature metrics: R_norm, SepMargin, E_neck/E_crack, cos(f_neck, f_crack).
2. End-to-end downstream metrics: FalseBridge count, transitions (cured/persistent/created),
   Dice, Precision, Recall, clDice, Breakage rate.

HARD LOCKS:
- DIAGNOSTIC-ONLY: Zero training, zero gradients, zero permanent checkpoint mutations.
- Checkpoint file SHA256 and named_parameters SHA256 bitwise verified before and after.
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
# Counterfactual Engine
# -----------------------------------------------------------------------------

ALPHAS = [1.00, 0.75, 0.50, 0.25, 0.00]

class StemACAttenuationInference:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.conv = model.backbone.convnext.stem[0]

        # Save original bitwise tensor
        self.orig_w = self.conv.weight.data.clone()
        self.b = self.conv.bias.data.clone()

        # Factorize into DC and AC
        self.w_dc = self.orig_w.mean(dim=(2, 3), keepdim=True).expand_as(self.orig_w)
        self.w_ac = self.orig_w - self.w_dc

        self.raw_stem_act: Dict[str, torch.Tensor] = {}
        self.hook_handle = None
        self._register_hook()

    def _register_hook(self):
        def hook_fn(m, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if t.dim() == 4 and t.shape[1] == 48 and t.shape[2] == 112 and t.shape[3] == 112:
                self.raw_stem_act['stem'] = t.detach()
        self.hook_handle = self.conv.register_forward_hook(hook_fn)

    def set_alpha(self, alpha: float):
        """Linearly sets W_alpha = W_dc + alpha * W_ac."""
        w_alpha = self.w_dc + float(alpha) * self.w_ac
        self.conv.weight.data.copy_(w_alpha)

    def restore_original_weights(self):
        """Restores original weights bitwise."""
        self.conv.weight.data.copy_(self.orig_w)

    def cleanup(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
        self.restore_original_weights()

    def run_image(
        self,
        image_rgb: np.ndarray,
        tile_size: int = 448,
        batch_size: int = 4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Runs Setting A tiling inference with batched patches for speed.
        Returns: (pred_bin, prob_map, raw_stem_energy_map, patch_activations)
        """
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
        raw_energy_full = np.zeros((pH, pW), dtype=np.float32)
        patch_acts = []

        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch_p = torch.cat(patches[i:i+batch_size], dim=0)
                batch_coords = coords[i:i+batch_size]

                logits = self.model(batch_p)
                if logits.shape[2:] != (tile_size, tile_size):
                    logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)

                stem_out = self.raw_stem_act['stem'] # (B, 48, 112, 112)
                stem_energy = torch.norm(stem_out, p=2, dim=1, keepdim=True) # (B, 1, 112, 112)
                stem_energy_up = F.interpolate(stem_energy, size=(tile_size, tile_size), mode='bilinear', align_corners=False)

                for b_idx, (py, px) in enumerate(batch_coords):
                    final_logits[py:py+tile_size, px:px+tile_size] = logits[b_idx].squeeze().cpu().numpy()
                    raw_energy_full[py:py+tile_size, px:px+tile_size] = stem_energy_up[b_idx].squeeze().cpu().numpy()
                    patch_acts.append({
                        'coords': (py, px),
                        'act': stem_out[b_idx:b_idx+1].cpu()
                    })

        logits_cropped = final_logits[:H, :W]
        prob_cropped = 1.0 / (1.0 + np.exp(-logits_cropped))
        pred_bin = (logits_cropped > 0.0).astype(np.uint8)
        energy_cropped = raw_energy_full[:H, :W]

        return pred_bin, prob_cropped, energy_cropped, patch_acts


# -----------------------------------------------------------------------------
# Visualization Generator
# -----------------------------------------------------------------------------

def plot_case_across_alphas(
    stem: str,
    cohort: str,
    image_rgb: np.ndarray,
    target_bin: np.ndarray,
    baseline_pred: np.ndarray,
    preds_by_alpha: Dict[float, np.ndarray],
    neck_mask: np.ndarray,
    metrics_by_alpha: Dict[float, Dict[str, Any]],
    save_path: str
):
    """Renders 7 panels: Overlay, GT Mask, and Predictions for alpha = 1.0, 0.75, 0.50, 0.25, 0.0."""
    fig, axes = plt.subplots(1, 7, figsize=(28, 4.5))
    fig.suptitle(f"AC Attenuation Counterfactual: {stem} (Cohort: {cohort})", fontsize=15, fontweight='bold', y=1.02)

    # 1. Input RGB + Overlay
    overlay = image_rgb.copy()
    gt_cnt, _ = cv2.findContours(target_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    p_cnt, _ = cv2.findContours(baseline_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, gt_cnt, -1, (0, 255, 0), 2)
    cv2.drawContours(overlay, p_cnt, -1, (255, 0, 0), 1)
    axes[0].imshow(overlay)
    axes[0].set_title("Input RGB + GT/Base", fontsize=10.5, fontweight='semibold')
    axes[0].axis('off')

    # 2. Ground Truth Mask
    axes[1].imshow(target_bin, cmap='gray')
    axes[1].set_title("Ground Truth", fontsize=10.5, fontweight='semibold')
    axes[1].axis('off')

    for idx, a in enumerate(ALPHAS):
        ax = axes[idx + 2]
        pred = preds_by_alpha[a]
        m = metrics_by_alpha[a]

        vis = np.zeros_like(image_rgb)
        vis[target_bin == 1] = [0, 180, 0] # GT green
        vis[pred == 1] = [255, 40, 40]     # Pred red
        vis[(target_bin == 1) & (pred == 1)] = [255, 255, 0] # Overlap yellow

        ax.imshow(vis)
        n_bridges = m['bridge_events']
        dice = m['dice']
        cldice = m['cldice']
        r_norm = m.get('raw_R_norm', float('nan'))
        sep_m = m.get('raw_sep_margin', float('nan'))

        status_str = f"Bridges: {n_bridges} | Dice: {dice:.3f}\nclDice: {cldice:.3f} | R: {r_norm:.2f}"
        ax.set_title(f"alpha = {a:.2f}\n{status_str}", fontsize=9.5)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
# Main Diagnostic Routine
# -----------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Phase 6: AC Attenuation Counterfactual Diagnostic at Stem Conv")
    print("=" * 80)

    config_path = 'results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml'
    ckpt_path = 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'
    output_dir = 'results/diagnostics/phase6_stem_ac_attenuation'
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

    engine = StemACAttenuationInference(model, device)

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

    print(f"\nCohorts Identified:")
    print(f"  - Consensus 7/7:         N = {len(c7_stems)}")
    print(f"  - Cured by D2:           N = {len(d2_cured_stems)}")
    print(f"  - Clean Control (gt>=2): N = {len(clean_stems)}")

    cohort_list = []
    for s in sorted(list(c7_stems)):
        cohort_list.append((s, 'Consensus_7of7'))
    for s in sorted(list(d2_cured_stems)):
        cohort_list.append((s, 'Cured_by_D2'))
    for s in sorted(list(clean_stems)):
        cohort_list.append((s, 'Clean_Control'))

    vis_c7 = sorted(list(c7_stems))[:4]
    vis_d2 = sorted(list(d2_cured_stems))[:2]
    vis_clean = sorted(list(clean_stems))[:2]
    vis_set = set(vis_c7 + vis_d2 + vis_clean)

    # Preload and cache all target masks and baseline ROIs at alpha=1.0
    print("\nPreloading sample masks and establishing baseline reference corridors...")
    sample_data = {}
    for stem, cohort in tqdm(cohort_list, desc="Preloading"):
        img_path = f"datasets/Crack500_ready/val/images/{stem}.jpg"
        mask_path = f"datasets/Crack500_ready/val/masks/{stem}.png"
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        target_bin = (mask > 0).astype(np.uint8)

        sample_data[stem] = {
            'img_rgb': img_rgb,
            'target_bin': target_bin,
            'cohort': cohort
        }

    # Step 1: Run baseline alpha = 1.0 to establish reference ROIs and baseline predictions
    engine.set_alpha(1.00)
    baseline_records = {}
    for stem, cohort in tqdm(cohort_list, desc="Establishing alpha=1.0 Baseline"):
        img_rgb = sample_data[stem]['img_rgb']
        target_bin = sample_data[stem]['target_bin']
        pred_bin, prob_map, stem_energy, patch_acts = engine.run_image(img_rgb)

        if cohort in ('Consensus_7of7', 'Cured_by_D2'):
            neck_mask, crack_mask, bg_mask, pair_records = isolate_bridged_pairs_and_rois(pred_bin, target_bin)
        else:
            neck_mask, crack_mask, bg_mask, pair_records = isolate_clean_pairs_and_rois(target_bin)

        baseline_records[stem] = {
            'pred_bin': pred_bin,
            'neck_mask': neck_mask,
            'crack_mask': crack_mask,
            'bg_mask': bg_mask,
            'pair_records': pair_records
        }

    # Verify baseline bridge count
    c7_baseline_bridges = sum(1 for s in c7_stems if compute_topology_metrics(baseline_records[s]['pred_bin'], sample_data[s]['target_bin'])['bridge_events'] > 0)
    print(f"\nBaseline Verification at alpha=1.0: Consensus 7/7 Bridges = {c7_baseline_bridges}/{len(c7_stems)}")

    # Step 2: Run all alpha counterfactual conditions
    all_sample_records = []
    pairwise_records = []
    preds_cache_vis = {s: {} for s in vis_set}
    metrics_cache_vis = {s: {} for s in vis_set}

    t0 = time.time()
    for alpha in ALPHAS:
        print(f"\nEvaluating Condition: alpha = {alpha:.2f} ...")
        engine.set_alpha(alpha)

        for stem, cohort in tqdm(cohort_list, desc=f"alpha={alpha:.2f}"):
            img_rgb = sample_data[stem]['img_rgb']
            target_bin = sample_data[stem]['target_bin']
            base_rec = baseline_records[stem]
            neck_mask = base_rec['neck_mask']
            crack_mask = base_rec['crack_mask']
            bg_mask = base_rec['bg_mask']
            pair_records = base_rec['pair_records']

            pred_bin, prob_map, stem_energy, patch_acts = engine.run_image(img_rgb)

            # Compute Downstream Topology & Segmentation Metrics
            topo = compute_topology_metrics(pred_bin, target_bin)

            # Compute Raw Stem Metrics on reference ROIs
            n_neck_px = int(np.sum(neck_mask))
            n_crack_px = int(np.sum(crack_mask))
            n_bg_px = int(np.sum(bg_mask))

            e_neck = float(np.mean(stem_energy[neck_mask == 1])) if n_neck_px > 0 else 0.0
            e_crack = float(np.mean(stem_energy[crack_mask == 1])) if n_crack_px > 0 else 0.0
            e_bg = float(np.mean(stem_energy[bg_mask == 1])) if n_bg_px > 0 else 0.0

            c_neck_crack = float(e_neck / (e_crack + 1e-6))
            sep_margin = float(1.0 - c_neck_crack)
            denom = (e_crack - e_bg)
            r_norm = float((e_neck - e_bg) / denom) if abs(denom) > 1e-5 else 0.0

            # Feature Cosine Similarity
            tile_size = 448
            f_neck_list, f_crack_list, f_bg_list = [], [], []
            for item in patch_acts:
                py, px = item['coords']
                act = item['act'] # (1, 48, 112, 112)
                p_neck = neck_mask[py:py+tile_size, px:px+tile_size]
                p_crack = crack_mask[py:py+tile_size, px:px+tile_size]
                p_bg = bg_mask[py:py+tile_size, px:px+tile_size]

                if np.sum(p_neck) > 0:
                    m_n = (F.interpolate(torch.from_numpy(p_neck).float().unsqueeze(0).unsqueeze(0), size=(112, 112), mode='nearest').squeeze() > 0)
                    if m_n.any():
                        f_neck_list.append(act[0, :, m_n].mean(dim=1))
                if np.sum(p_crack) > 0:
                    m_c = (F.interpolate(torch.from_numpy(p_crack).float().unsqueeze(0).unsqueeze(0), size=(112, 112), mode='nearest').squeeze() > 0)
                    if m_c.any():
                        f_crack_list.append(act[0, :, m_c].mean(dim=1))
                if np.sum(p_bg) > 0:
                    m_b = (F.interpolate(torch.from_numpy(p_bg).float().unsqueeze(0).unsqueeze(0), size=(112, 112), mode='nearest').squeeze() > 0)
                    if m_b.any():
                        f_bg_list.append(act[0, :, m_b].mean(dim=1))

            if len(f_neck_list) > 0 and len(f_crack_list) > 0:
                vec_neck = torch.stack(f_neck_list).mean(dim=0)
                vec_crack = torch.stack(f_crack_list).mean(dim=0)
                cos_nc = float(F.cosine_similarity(vec_neck.unsqueeze(0), vec_crack.unsqueeze(0)).item())
            else:
                cos_nc = 0.0

            has_bridge = bool(topo['bridge_events'] > 0)
            has_breakage = bool(topo['fragmented_gt_components'] > 0)

            # Determine Transition relative to alpha=1.0
            base_has_bridge = bool(compute_topology_metrics(base_rec['pred_bin'], target_bin)['bridge_events'] > 0)
            if base_has_bridge and not has_bridge:
                transition = 'cured'
            elif base_has_bridge and has_bridge:
                transition = 'persistent'
            elif not base_has_bridge and has_bridge:
                transition = 'created'
            else:
                transition = 'clean'

            rec = {
                'image_id': stem,
                'cohort': cohort,
                'alpha': alpha,
                'has_bridge': has_bridge,
                'bridge_events': topo['bridge_events'],
                'merged_gt_components': topo['merged_gt_components'],
                'has_breakage': has_breakage,
                'fragmented_gt_components': topo['fragmented_gt_components'],
                'dice': topo['dice'],
                'precision': topo['precision'],
                'recall': topo['recall'],
                'cldice': topo['cldice'],
                'transition': transition,
                'raw_R_norm': r_norm,
                'raw_sep_margin': sep_margin,
                'raw_contrast_neck_crack': c_neck_crack,
                'raw_E_neck': e_neck,
                'raw_E_crack': e_crack,
                'raw_E_bg': e_bg,
                'raw_cos_neck_crack': cos_nc,
            }
            all_sample_records.append(rec)

            if stem in vis_set:
                preds_cache_vis[stem][alpha] = pred_bin
                metrics_cache_vis[stem][alpha] = rec

            # Pairwise spatial separation for bridged pairs
            for pair in pair_records:
                neck_p = pair['neck_448']
                crack_p = pair['mask_i_448'] | pair['mask_j_448']
                en = float(np.mean(stem_energy[neck_p == 1])) if np.sum(neck_p) > 0 else 0.0
                ec = float(np.mean(stem_energy[crack_p == 1]))
                ratio = float(en / (ec + 1e-6))
                sep_m = float(1.0 - ratio)
                is_sep = bool(sep_m >= 0.20 and not pair['grid_cell_collision'])

                pairwise_records.append({
                    'image_id': stem,
                    'cohort': cohort,
                    'alpha': alpha,
                    'gt_i': pair['gt_i'],
                    'gt_j': pair['gt_j'],
                    'gap_distance_px': pair['gap_distance_px'],
                    'grid_cell_collision': pair['grid_cell_collision'],
                    'neck_energy': en,
                    'crack_energy': ec,
                    'contrast': ratio,
                    'sep_margin': sep_m,
                    'is_separable': is_sep
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

    # DataFrames
    df_samples = pd.DataFrame(all_sample_records)
    df_pairs = pd.DataFrame(pairwise_records)

    sample_csv = os.path.join(output_dir, 'ac_attenuation_per_sample_alpha.csv')
    df_samples.to_csv(sample_csv, index=False)
    print(f"Saved: {sample_csv}")

    # Build Summary by Alpha Table
    summary_rows = []
    for cohort_name in ['Consensus_7of7', 'Cured_by_D2', 'Clean_Control']:
        sub_all = df_samples[df_samples['cohort'] == cohort_name]
        n_cohort = len(sub_all['image_id'].unique())

        for alpha in ALPHAS:
            sub = sub_all[sub_all['alpha'] == alpha]
            n_bridges = int(sub['has_bridge'].sum())
            pct_bridge = float(n_bridges / n_cohort * 100)
            n_broken = int(sub['has_breakage'].sum())
            pct_broken = float(n_broken / n_cohort * 100)

            # Transitions
            n_cured = int((sub['transition'] == 'cured').sum())
            n_pers = int((sub['transition'] == 'persistent').sum())
            n_created = int((sub['transition'] == 'created').sum())

            summary_rows.append({
                'cohort': cohort_name,
                'alpha': alpha,
                'n_samples': n_cohort,
                'n_bridge_cases': n_bridges,
                'pct_bridge_cases': pct_bridge,
                'n_bridge_events_total': int(sub['bridge_events'].sum()),
                'n_cured': n_cured,
                'n_persistent': n_pers,
                'n_created': n_created,
                'n_broken_cases': n_broken,
                'pct_broken_cases': pct_broken,
                'median_dice': float(sub['dice'].median()),
                'mean_dice': float(sub['dice'].mean()),
                'median_precision': float(sub['precision'].median()),
                'mean_precision': float(sub['precision'].mean()),
                'median_recall': float(sub['recall'].median()),
                'mean_recall': float(sub['recall'].mean()),
                'median_cldice': float(sub['cldice'].median()),
                'mean_cldice': float(sub['cldice'].mean()),
                'raw_median_R_norm': float(sub['raw_R_norm'].median()),
                'raw_mean_R_norm': float(sub['raw_R_norm'].mean()),
                'raw_pct_ge_50': float((sub['raw_R_norm'] >= 0.50).mean() * 100),
                'raw_pct_ge_70': float((sub['raw_R_norm'] >= 0.70).mean() * 100),
                'raw_median_sep_margin': float(sub['raw_sep_margin'].median()),
                'raw_median_contrast': float(sub['raw_contrast_neck_crack'].median()),
                'raw_median_cos_neck_crack': float(sub['raw_cos_neck_crack'].median()),
            })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, 'ac_attenuation_summary_by_alpha.csv')
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    # Build Transitions Table
    df_trans = df_summary[['cohort', 'alpha', 'n_samples', 'n_bridge_cases', 'pct_bridge_cases', 'n_cured', 'n_persistent', 'n_created', 'n_broken_cases', 'median_dice', 'median_recall', 'median_cldice']]
    trans_csv = os.path.join(output_dir, 'ac_attenuation_transitions.csv')
    df_trans.to_csv(trans_csv, index=False)
    print(f"Saved: {trans_csv}")

    # Pairwise Separation Summary by Alpha
    pair_summary_rows = []
    for cohort_name in ['Consensus_7of7', 'Cured_by_D2', 'Clean_Control']:
        sub_p_all = df_pairs[df_pairs['cohort'] == cohort_name]
        if len(sub_p_all) == 0:
            continue
        n_pairs = len(sub_p_all[sub_p_all['alpha'] == 1.0])

        for alpha in ALPHAS:
            sub_p = sub_p_all[sub_p_all['alpha'] == alpha]
            pct_sep = float(sub_p['is_separable'].mean() * 100)
            pair_summary_rows.append({
                'cohort': cohort_name,
                'alpha': alpha,
                'n_pairs_evaluated': n_pairs,
                'pct_pairs_spatially_separable': pct_sep,
                'pct_pairs_representation_collapsed': float(100.0 - pct_sep),
                'median_separation_margin': float(sub_p['sep_margin'].median()),
                'median_contrast_neck_crack': float(sub_p['contrast'].median())
            })

    df_pair_sum = pd.DataFrame(pair_summary_rows)
    pair_sum_csv = os.path.join(output_dir, 'pairwise_separation_by_alpha.csv')
    df_pair_sum.to_csv(pair_sum_csv, index=False)
    print(f"Saved: {pair_sum_csv}")

    # Generate Figures
    # 1. Downstream Performance & Trade-off figure (6 Panels)
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("AC Attenuation Counterfactual: Downstream Impact & Representation Trade-offs", fontsize=15, fontweight='bold', y=0.98)

    c7_sum = df_summary[df_summary['cohort'] == 'Consensus_7of7'].sort_values('alpha')
    alpha_x = c7_sum['alpha'].tolist()

    # Panel 0,0: False Bridge Cases
    axes[0, 0].plot(alpha_x, c7_sum['n_bridge_cases'], marker='o', color='#d95f02', linewidth=2.5, label='Consensus 7/7 (N=101)')
    d2_sum = df_summary[df_summary['cohort'] == 'Cured_by_D2'].sort_values('alpha')
    axes[0, 0].plot(alpha_x, d2_sum['n_bridge_cases'], marker='s', color='#7570b3', linewidth=2.0, label='Cured by D2 (N=7)')
    clean_sum = df_summary[df_summary['cohort'] == 'Clean_Control'].sort_values('alpha')
    axes[0, 0].plot(alpha_x, clean_sum['n_bridge_cases'], marker='^', color='#1b9e77', linewidth=2.0, label='Clean Control (N=56)')
    axes[0, 0].set_title("Downstream False Bridge Cases vs alpha", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel("alpha (AC Weight Scaling)")
    axes[0, 0].set_ylabel("Count of Cases with False Bridge")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=9)

    # Panel 0,1: Segmentation Quality Trade-off (Dice & clDice)
    axes[0, 1].plot(alpha_x, c7_sum['median_dice'], marker='o', color='#1f77b4', linewidth=2.0, label='Median Dice')
    axes[0, 1].plot(alpha_x, c7_sum['median_cldice'], marker='D', color='#2ca02c', linewidth=2.0, label='Median clDice')
    axes[0, 1].plot(alpha_x, c7_sum['median_recall'], marker='v', color='#ff7f0e', linewidth=2.0, label='Median Recall')
    axes[0, 1].plot(alpha_x, c7_sum['median_precision'], marker='x', color='#9467bd', linewidth=2.0, label='Median Precision')
    axes[0, 1].set_title("Consensus 7/7: Segmentation Quality vs alpha", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel("alpha")
    axes[0, 1].set_ylabel("Metric Value")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=9)

    # Panel 0,2: Breakage / Continuity Collapse
    axes[0, 2].plot(alpha_x, c7_sum['pct_broken_cases'], marker='o', color='#d62728', linewidth=2.5, label='Consensus 7/7')
    axes[0, 2].plot(alpha_x, clean_sum['pct_broken_cases'], marker='^', color='#1b9e77', linewidth=2.0, label='Clean Control')
    axes[0, 2].set_title("Crack Breakage Rate vs alpha (% Cases Fragmented)", fontsize=11, fontweight='bold')
    axes[0, 2].set_xlabel("alpha")
    axes[0, 2].set_ylabel("% Cases with Broken Components")
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend(fontsize=9)

    # Panel 1,0: Raw Stem R_norm Profile
    axes[1, 0].plot(alpha_x, c7_sum['raw_median_R_norm'], marker='o', color='#d95f02', linewidth=2.5, label='Consensus 7/7')
    axes[1, 0].plot(alpha_x, d2_sum['raw_median_R_norm'], marker='s', color='#7570b3', linewidth=2.0, label='Cured by D2')
    axes[1, 0].plot(alpha_x, clean_sum['raw_median_R_norm'], marker='^', color='#1b9e77', linewidth=2.0, label='Clean Control')
    axes[1, 0].axhline(0.50, color='gray', linestyle='--', alpha=0.7, label='50% Emergence')
    axes[1, 0].set_title("Raw Stem Median R_norm vs alpha", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel("alpha")
    axes[1, 0].set_ylabel("Median R_norm")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=9)

    # Panel 1,1: Raw Stem Separation Margin (SepMargin)
    axes[1, 1].plot(alpha_x, c7_sum['raw_median_sep_margin'], marker='o', color='#d95f02', linewidth=2.5, label='Consensus 7/7')
    axes[1, 1].plot(alpha_x, d2_sum['raw_median_sep_margin'], marker='s', color='#7570b3', linewidth=2.0, label='Cured by D2')
    axes[1, 1].plot(alpha_x, clean_sum['raw_median_sep_margin'], marker='^', color='#1b9e77', linewidth=2.0, label='Clean Control')
    axes[1, 1].axhline(0.20, color='blue', linestyle=':', alpha=0.7, label='Separation Valley (0.20)')
    axes[1, 1].axhline(0.00, color='black', linestyle='-', alpha=0.4)
    axes[1, 1].set_title("Raw Stem Median Separation Margin vs alpha", fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel("alpha")
    axes[1, 1].set_ylabel("SepMargin (1 - E_neck/E_crack)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=9)

    # Panel 1,2: Raw Stem Cosine Alignment
    axes[1, 2].plot(alpha_x, c7_sum['raw_median_cos_neck_crack'], marker='o', color='#d95f02', linewidth=2.5, label='Consensus 7/7')
    axes[1, 2].plot(alpha_x, d2_sum['raw_median_cos_neck_crack'], marker='s', color='#7570b3', linewidth=2.0, label='Cured by D2')
    axes[1, 2].plot(alpha_x, clean_sum['raw_median_cos_neck_crack'], marker='^', color='#1b9e77', linewidth=2.0, label='Clean Control')
    axes[1, 2].set_title("Raw Stem Feature Cosine Alignment cos(f_neck, f_crack)", fontsize=11, fontweight='bold')
    axes[1, 2].set_xlabel("alpha")
    axes[1, 2].set_ylabel("Cosine Similarity")
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend(fontsize=9)

    plt.tight_layout()
    fig_tradeoff_path = os.path.join(figures_dir, 'ac_attenuation_downstream_tradeoff.png')
    plt.savefig(fig_tradeoff_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_tradeoff_path}")

    # 2. Case visualizations across alpha
    for stem in vis_set:
        c_name = sample_data[stem]['cohort']
        fig_case_path = os.path.join(figures_dir, f"{c_name.lower()}_{stem}_ac_attenuation.png")
        plot_case_across_alphas(
            stem=stem,
            cohort=c_name,
            image_rgb=sample_data[stem]['img_rgb'],
            target_bin=sample_data[stem]['target_bin'],
            baseline_pred=baseline_records[stem]['pred_bin'],
            preds_by_alpha=preds_cache_vis[stem],
            neck_mask=baseline_records[stem]['neck_mask'],
            metrics_by_alpha=metrics_cache_vis[stem],
            save_path=fig_case_path
        )

    print("\n" + "=" * 80)
    print("AC Attenuation Counterfactual Diagnostic Completed Successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
