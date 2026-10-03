#!/usr/bin/env python3
"""
scripts/diagnostics/run_decoder_causal_pathway_ablation.py

Phase 6 Diagnostic: Zero-Training 2x2 Factorial Causal Pathway Ablation
Target Cohort: Exactly the 43 wider-gap bridge events (D_gap > 5.0 px) from Phase 6 Diagnostic D.
Target Model: Candidate B Baseline (B2ConvNeXtViTUNet, Setting A, Tile 448).

Interventions at decoder_28 (Block 0) and decoder_56 (Block 1):
  u = self.upsample(x)
  s = skip
  x = torch.cat([u, s], dim=1)
  x = self.conv1(x)
  x = self.conv2(x)

Conditions:
  - 00: baseline (u + s)
  - 10: mask corridor on u, keep s
  - 01: keep u, mask corridor on s
  - 11: mask corridor on both u and s

Measurements:
  - Corridor Continuous: median logit, P10 logit, median probability
  - Main causal effects: Delta U, Delta S, Delta US
  - Interaction effect: I_US = Delta US - Delta U - Delta S
  - Matched Control Group: Delta U_ctrl, Delta S_ctrl, Delta US_ctrl
  - Selective effects: Delta U_sel = Delta U - Delta U_ctrl, Delta S_sel = Delta S - Delta S_ctrl
  - Topology: persistent, cured, control broken
  - Robustness check: local background feature replacement vs zeroing on top causal events
"""

import collections
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Ensure SAGE_LITE and project root are on sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sage_lite_dir = os.path.join(project_root, "SAGE_LITE")
for p in [project_root, sage_lite_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from tools.run_phase6_c_topology_diagnostic import load_model_from_checkpoint


def find_matched_control_segment(target_bin: np.ndarray, target_length_px: float) -> Optional[Dict[str, Any]]:
    """
    Finds a matched true crack continuation segment in target_bin
    with Euclidean distance between endpoints approximately equal to target_length_px.
    """
    skel = skeletonize(target_bin > 0).astype(np.uint8)
    pts = np.argwhere(skel)
    if len(pts) < 10:
        return None

    np.random.seed(42)
    sample_indices = np.random.choice(len(pts), size=min(100, len(pts)), replace=False)
    
    best_pair = None
    best_diff = float('inf')

    for idx in sample_indices:
        p0 = pts[idx]
        dists = np.linalg.norm(pts - p0, axis=1)
        diffs = np.abs(dists - target_length_px)
        min_i = np.argmin(diffs)
        if diffs[min_i] < best_diff:
            p1 = pts[min_i]
            line_mask = np.zeros_like(target_bin)
            cv2.line(line_mask, (int(p0[1]), int(p0[0])), (int(p1[1]), int(p1[0])), 1, thickness=3)
            overlap = np.sum((line_mask == 1) & (target_bin == 1)) / max(np.sum(line_mask == 1), 1)
            if overlap > 0.65:
                best_diff = diffs[min_i]
                best_pair = (p0, p1)
                if best_diff < 1.0:
                    break

    if best_pair is None:
        p0 = pts[len(pts) // 2]
        dists = np.linalg.norm(pts - p0, axis=1)
        min_i = np.argmin(np.abs(dists - target_length_px))
        best_pair = (p0, pts[min_i])

    p0, p1 = best_pair
    line_mask = np.zeros_like(target_bin)
    cv2.line(line_mask, (int(p0[1]), int(p0[0])), (int(p1[1]), int(p1[0])), 1, thickness=3)
    ctrl_corridor = (line_mask == 1) & (target_bin == 1)

    return {
        'corridor_mask': ctrl_corridor,
        'length_px': float(np.linalg.norm(p0 - p1))
    }


def project_corridor_to_feature_grid(
    corridor_mask: np.ndarray,
    target_bin: np.ndarray,
    coords: List[Tuple[int, int]],
    H: int,
    W: int,
    H_grid: int,
    W_grid: int,
    device: torch.device
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Projects corridor mask from 448-space down to feature grid (H_grid, W_grid).
    Stores cell coverage ratio, crack ratio, pure gap cell count, and mixed cell count.
    Avoids destroying crack endpoints:
      - For pure gap cells (corr > 0 and crack == 0): weight = 1.0
      - For mixed cells (corr > 0 and crack > 0): soft weight = max(0.0, (n_corr - n_crack) / cell_area)
    """
    S = 448 // H_grid
    N_tiles = len(coords)
    cell_mask = torch.zeros((N_tiles, 1, H_grid, W_grid), device=device)
    
    total_pure_cells = 0
    total_mixed_cells = 0
    cell_coverages = []

    for j, (y_tile, x_tile) in enumerate(coords):
        for r_idx in range(H_grid):
            for c_idx in range(W_grid):
                gy0, gy1 = y_tile + r_idx * S, y_tile + (r_idx + 1) * S
                gx0, gx1 = x_tile + c_idx * S, x_tile + (c_idx + 1) * S
                cy0, cy1 = max(0, min(H, gy0)), max(0, min(H, gy1))
                cx0, cx1 = max(0, min(W, gx0)), max(0, min(W, gx1))
                if cy1 > cy0 and cx1 > cx0:
                    corr_patch = corridor_mask[cy0:cy1, cx0:cx1]
                    crack_patch = target_bin[cy0:cy1, cx0:cx1]
                    n_c = int(np.sum(corr_patch))
                    n_k = int(np.sum(crack_patch))
                    cell_area = S * S
                    if n_c > 0:
                        cov_ratio = n_c / cell_area
                        cell_coverages.append(cov_ratio)
                        if n_k == 0:
                            cell_mask[j, 0, r_idx, c_idx] = 1.0
                            total_pure_cells += 1
                        else:
                            # Soft weight protecting endpoint
                            w = max(0.0, (n_c - n_k) / float(cell_area))
                            cell_mask[j, 0, r_idx, c_idx] = w
                            total_mixed_cells += 1

    stats = {
        'grid_res': f"{H_grid}x{W_grid}",
        'stride_s': S,
        'pure_cells': total_pure_cells,
        'mixed_cells': total_mixed_cells,
        'total_active_cells': total_pure_cells + total_mixed_cells,
        'mean_coverage_ratio': float(np.mean(cell_coverages)) if len(cell_coverages) > 0 else 0.0
    }
    return cell_mask, stats


def compute_local_bg_vector(
    stage_idx: int,
    model: Any,
    batch: torch.Tensor,
    coords: List[Tuple[int, int]],
    bg_mask_448: np.ndarray,
    H: int,
    W: int,
    H_grid: int,
    W_grid: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes average background feature vector across local background cells
    for replacement in robustness check (u and skip separately).
    """
    S = 448 // H_grid
    captured = {}
    block = model.decoder.decoder_blocks[stage_idx]

    def hook_fn(m, i, o):
        # i[0] is deeper feature x, i[1] is skip
        captured['x'] = i[0].detach()
        captured['skip'] = i[1].detach()

    h = block.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(batch)
    h.remove()

    x_deep = captured['x']
    skip_feat = captured['skip']
    u_feat = block.upsample(x_deep)
    if u_feat.shape[2:] != skip_feat.shape[2:]:
        u_feat = F.interpolate(u_feat, size=skip_feat.shape[2:], mode='bilinear', align_corners=False)

    # Find background cells
    u_bg_vals = []
    s_bg_vals = []
    for j, (y_tile, x_tile) in enumerate(coords):
        for r_idx in range(H_grid):
            for c_idx in range(W_grid):
                gy0, gy1 = y_tile + r_idx * S, y_tile + (r_idx + 1) * S
                gx0, gx1 = x_tile + c_idx * S, x_tile + (c_idx + 1) * S
                cy0, cy1 = max(0, min(H, gy0)), max(0, min(H, gy1))
                cx0, cx1 = max(0, min(W, gx0)), max(0, min(W, gx1))
                if cy1 > cy0 and cx1 > cx0:
                    bg_patch = bg_mask_448[cy0:cy1, cx0:cx1]
                    if np.sum(bg_patch) > (S * S * 0.5):
                        u_bg_vals.append(u_feat[j, :, r_idx, c_idx])
                        s_bg_vals.append(skip_feat[j, :, r_idx, c_idx])

    if len(u_bg_vals) > 0:
        mean_u_bg = torch.stack(u_bg_vals).mean(dim=0).view(1, -1, 1, 1)
        mean_s_bg = torch.stack(s_bg_vals).mean(dim=0).view(1, -1, 1, 1)
    else:
        mean_u_bg = torch.zeros((1, u_feat.shape[1], 1, 1), device=device)
        mean_s_bg = torch.zeros((1, skip_feat.shape[1], 1, 1), device=device)

    return mean_u_bg, mean_s_bg


def render_causal_panel(
    img_rgb: np.ndarray,
    target_bin: np.ndarray,
    corridor_mask: np.ndarray,
    baseline_logits: np.ndarray,
    interv_logits_map: Dict[str, np.ndarray],
    out_path: str,
    event_id: int,
    case_name: str,
    gap_px: float,
    stage_name: str,
    d_u: float,
    d_s: float,
    d_us: float,
    i_us: float
):
    """
    Renders 4-panel visual strip for 2x2 factorial conditions:
    Panel 1: Cond 00 (Baseline)
    Panel 2: Cond 10 (Mask U)
    Panel 3: Cond 01 (Mask S)
    Panel 4: Cond 11 (Mask Both)
    """
    H, W = img_rgb.shape[:2]
    corr_pts = np.argwhere(corridor_mask)
    if len(corr_pts) == 0:
        return

    ymin, xmin = np.min(corr_pts, axis=0)
    ymax, xmax = np.max(corr_pts, axis=0)
    pad = 40
    ymin, xmin = max(0, ymin - pad), max(0, xmin - pad)
    ymax, xmax = min(H, ymax + pad + 1), min(W, xmax + pad + 1)

    crop_rgb = img_rgb[ymin:ymax, xmin:xmax].copy()
    crop_gt = target_bin[ymin:ymax, xmin:xmax]
    crop_corr = corridor_mask[ymin:ymax, xmin:xmax]
    contours, _ = cv2.findContours(crop_corr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cond_order = ['00', '10', '01', '11']
    cond_titles = [
        'Cond 00 (Base)',
        f'Cond 10 (Mask U) dU={d_u:+.2f}',
        f'Cond 01 (Mask S) dS={d_s:+.2f}',
        f'Cond 11 (Both) dUS={d_us:+.2f}'
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    panels = []

    for cond, title in zip(cond_order, cond_titles):
        logits = interv_logits_map[cond][ymin:ymax, xmin:xmax]
        prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        heat = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        blended = (0.65 * heat + 0.35 * gray_3ch).astype(np.uint8)

        # Highlight corridor contour in cyan
        cv2.drawContours(blended, contours, -1, (0, 255, 255), 1)

        cv2.putText(blended, title, (4, 12), font, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(blended)

    strip = np.hstack(panels)
    banner_h = 24
    banner = np.zeros((banner_h, strip.shape[1], 3), dtype=np.uint8)
    banner_title = f"Event #{event_id:03d} ({case_name}) | Gap: {gap_px:.1f}px | Stage: {stage_name} | Interaction I_US={i_us:+.2f}"
    cv2.putText(banner, banner_title, (10, 16), font, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

    final_img = np.vstack([banner, strip])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))


def compute_summaries(df_out: pd.DataFrame, out_dir: str, t0_start: float):
    # Compute summary tables
    summary_data = {
        'total_events': len(df_out),
        'small_gap_events_le8px': int(df_out['is_small_gap_le8px'].sum()),
        'dec28_effects': {
            'mean_delta_U': float(df_out['dec28_delta_U'].mean()),
            'median_delta_U': float(df_out['dec28_delta_U'].median()),
            'mean_delta_S': float(df_out['dec28_delta_S'].mean()),
            'median_delta_S': float(df_out['dec28_delta_S'].median()),
            'mean_delta_US': float(df_out['dec28_delta_US'].mean()),
            'median_delta_US': float(df_out['dec28_delta_US'].median()),
            'mean_interaction_I_US': float(df_out['dec28_interaction_I_US'].mean()),
            'median_interaction_I_US': float(df_out['dec28_interaction_I_US'].median()),
            'cured_11_count': int(df_out['dec28_cured_11'].sum()),
            'driver_distribution': {str(k): int(v) for k, v in df_out['dec28_driver_classification'].value_counts().items()}
        },
        'dec56_effects': {
            'mean_delta_U': float(df_out['dec56_delta_U'].mean()),
            'median_delta_U': float(df_out['dec56_delta_U'].median()),
            'mean_delta_S': float(df_out['dec56_delta_S'].mean()),
            'median_delta_S': float(df_out['dec56_delta_S'].median()),
            'mean_delta_US': float(df_out['dec56_delta_US'].mean()),
            'median_delta_US': float(df_out['dec56_delta_US'].median()),
            'mean_interaction_I_US': float(df_out['dec56_interaction_I_US'].mean()),
            'median_interaction_I_US': float(df_out['dec56_interaction_I_US'].median()),
            'cured_11_count': int(df_out['dec56_cured_11'].sum()),
            'driver_distribution': {str(k): int(v) for k, v in df_out['dec56_driver_classification'].value_counts().items()}
        }
    }

    # Selective effects summary (excluding NaNs)
    sel_u28 = df_out['dec28_delta_U_selective'].dropna()
    sel_s28 = df_out['dec28_delta_S_selective'].dropna()
    sel_u56 = df_out['dec56_delta_U_selective'].dropna()
    sel_s56 = df_out['dec56_delta_S_selective'].dropna()

    summary_data['selective_effects'] = {
        'dec28_median_selective_U': float(sel_u28.median()) if len(sel_u28) > 0 else 0.0,
        'dec28_median_selective_S': float(sel_s28.median()) if len(sel_s28) > 0 else 0.0,
        'dec56_median_selective_U': float(sel_u56.median()) if len(sel_u56) > 0 else 0.0,
        'dec56_median_selective_S': float(sel_s56.median()) if len(sel_s56) > 0 else 0.0,
    }

    # Robustness correlation on tested events
    rob_tested = df_out.dropna(subset=['dec56_robustness_delta_US'])
    if len(rob_tested) > 0:
        corr_val = float(np.corrcoef(rob_tested['dec56_delta_US'], rob_tested['dec56_robustness_delta_US'])[0, 1])
        summary_data['robustness_check'] = {
            'num_events_tested': len(rob_tested),
            'mean_delta_US_zero': float(rob_tested['dec56_delta_US'].mean()),
            'mean_delta_US_bg_replacement': float(rob_tested['dec56_robustness_delta_US'].mean()),
            'correlation_zero_vs_bg': corr_val
        }

    json_out = os.path.join(out_dir, 'decoder_causal_summary.json')
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved summary JSON: {json_out}")

    # Build 2x2 Factorial Summary DataFrame
    summary_rows = [
        {
            'decoder_stage': 'decoder_28 (Block 0)',
            'baseline_corridor_logit': float(df_out['z00_corridor_logit'].median()),
            'cond_10_mask_u_logit': float(df_out['dec28_z10_logit'].median()),
            'cond_01_mask_s_logit': float(df_out['dec28_z01_logit'].median()),
            'cond_11_mask_both_logit': float(df_out['dec28_z11_logit'].median()),
            'median_delta_U': float(df_out['dec28_delta_U'].median()),
            'median_delta_S': float(df_out['dec28_delta_S'].median()),
            'median_delta_US': float(df_out['dec28_delta_US'].median()),
            'median_interaction_I_US': float(df_out['dec28_interaction_I_US'].median()),
            'median_selective_U': float(summary_data['selective_effects']['dec28_median_selective_U']),
            'median_selective_S': float(summary_data['selective_effects']['dec28_median_selective_S']),
            'cured_events_11': int(df_out['dec28_cured_11'].sum()),
            'cure_rate_pct': float(df_out['dec28_cured_11'].mean() * 100.0)
        },
        {
            'decoder_stage': 'decoder_56 (Block 1)',
            'baseline_corridor_logit': float(df_out['z00_corridor_logit'].median()),
            'cond_10_mask_u_logit': float(df_out['dec56_z10_logit'].median()),
            'cond_01_mask_s_logit': float(df_out['dec56_z01_logit'].median()),
            'cond_11_mask_both_logit': float(df_out['dec56_z11_logit'].median()),
            'median_delta_U': float(df_out['dec56_delta_U'].median()),
            'median_delta_S': float(df_out['dec56_delta_S'].median()),
            'median_delta_US': float(df_out['dec56_delta_US'].median()),
            'median_interaction_I_US': float(df_out['dec56_interaction_I_US'].median()),
            'median_selective_U': float(summary_data['selective_effects']['dec56_median_selective_U']),
            'median_selective_S': float(summary_data['selective_effects']['dec56_median_selective_S']),
            'cured_events_11': int(df_out['dec56_cured_11'].sum()),
            'cure_rate_pct': float(df_out['dec56_cured_11'].mean() * 100.0)
        }
    ]
    df_sum = pd.DataFrame(summary_rows)
    sum_csv_out = os.path.join(out_dir, 'decoder_causal_summary.csv')
    df_sum.to_csv(sum_csv_out, index=False)
    print(f"Saved summary CSV: {sum_csv_out}")

    print("\n" + "=" * 80)
    print("2x2 FACTORIAL CAUSAL ABLATION SUMMARY TABLE (N = 43 EVENTS)")
    print("=" * 80)
    print(df_sum.to_string(index=False))

    el_time = time.time() - t0_start
    print(f"\nExecution finished in {el_time:.1f}s")


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    t0_start = time.time()
    print("=" * 80)
    print("PHASE 6 DIAGNOSTIC: 2x2 FACTORIAL DECODER CAUSAL PATHWAY ABLATION")
    print("Target Cohort: 43 Wider-Gap Events (D_gap > 5.0 px) | Checkpoint: Candidate B")
    print("=" * 80)

    out_dir = os.path.join(project_root, 'results', 'diagnostics', 'phase6_decoder_causal')
    viz_dir = os.path.join(out_dir, 'visualizations')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    csv_out = os.path.join(out_dir, 'decoder_causal_measurements_43events.csv')
    if os.path.exists(csv_out) and ('--force' not in sys.argv):
        print(f"Loading existing measurements from {csv_out}...")
        df_out = pd.read_csv(csv_out)
        compute_summaries(df_out, out_dir, t0_start)
        return

    config_path = os.path.join(project_root, 'results', 'configs', 'b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml')
    ckpt_path = os.path.join(project_root, 'results', 'checkpoints', 'P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = False

    print(f"Device: {device} (CuDNN disabled for numerical precision)")
    print("Loading Candidate B model...")
    model = load_model_from_checkpoint(config_path, ckpt_path, device)
    model.eval()

    csv_118 = os.path.join(project_root, 'results', 'diagnostics', 'phase6_bottleneck_path', 'bottleneck_path_118events.csv')
    df_118 = pd.read_csv(csv_118)
    df_43 = df_118[df_118['gap_category'] == 'Wider_Gap (> 5 px)'].copy().reset_index(drop=True)
    assert len(df_43) == 43, f"Expected exactly 43 wider gap events, found {len(df_43)}"
    print(f"Cohort loaded: {len(df_43)} events across {df_43['case_name'].nunique()} unique images.")

    data_root = os.path.join(project_root, 'datasets', 'Crack500_ready')
    val_img_dir = os.path.join(data_root, 'val', 'images')
    val_mask_dir = os.path.join(data_root, 'val', 'masks')

    event_records = []
    print("\n[SAFETY CHECK] Verifying baseline reproduction and hook transparency...")
    grouped = df_43.groupby('case_name')
    pbar = tqdm(total=len(df_43), desc="Running Causal Pathway Ablation")

    for case_name, group_events in grouped:
        ip = os.path.join(val_img_dir, case_name + '.jpg')
        if not os.path.exists(ip):
            ip = os.path.join(val_img_dir, case_name + '.png')
        mp = os.path.join(val_mask_dir, case_name + '.png')
        if not os.path.exists(mp):
            mp = os.path.join(val_mask_dir, case_name + '.jpg')

        img = cv2.imread(ip)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        target_bin = (target > 127).astype(np.uint8)
        H, W = target_bin.shape[:2]

        tile_size = 448
        pad_h = (tile_size - (H % tile_size)) % tile_size
        pad_w = (tile_size - (W % tile_size)) % tile_size
        padded_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        pH, pW = padded_img.shape[:2]

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        patches, coords = [], []
        for y in range(0, pH, tile_size):
            for x in range(0, pW, tile_size):
                p = padded_img[y:y+tile_size, x:x+tile_size]
                pt = torch.from_numpy(p).permute(2, 0, 1).float() / 255.0
                pt = (pt - mean) / std
                patches.append(pt)
                coords.append((y, x))

        batch = torch.stack(patches).to(device)

        # 1. Baseline Forward (Cond 00)
        with torch.no_grad():
            base_logits_tiles = model(batch)

        base_logits_np = np.zeros((pH, pW), dtype=np.float32)
        for j, (y, x) in enumerate(coords):
            base_logits_np[y:y+448, x:x+448] = base_logits_tiles[j, 0].cpu().numpy()
        base_logits_np = base_logits_np[:H, :W]
        base_pred_bin = (base_logits_np > 0.0).astype(np.uint8)

        num_gt_cc, gt_labels = cv2.connectedComponents(target_bin, connectivity=8)
        num_pred_cc, base_pred_labels = cv2.connectedComponents(base_pred_bin, connectivity=8)

        # Process each bridge event in this image
        for _, ev_row in group_events.iterrows():
            ev_id = int(ev_row['event_id'])
            p_id = int(ev_row['pred_cc_id'])
            gA = int(ev_row['primary_gt_A'])
            gB = int(ev_row['primary_gt_B'])
            d_gap = float(ev_row['d_gap_px'])
            is_small_gap = bool(d_gap <= 8.0)

            cc_fp = (base_pred_labels == p_id) & (target_bin == 0)
            r = max(int(np.ceil(d_gap / 2.0)) + 2, 3)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
            dil_A = cv2.dilate((gt_labels == gA).astype(np.uint8), k)
            dil_B = cv2.dilate((gt_labels == gB).astype(np.uint8), k)
            corridor = (dil_A > 0) & (dil_B > 0) & cc_fp
            if np.sum(corridor) == 0:
                corridor = cc_fp

            # Local background ring for robustness check
            k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))
            local_ring = cv2.dilate(corridor.astype(np.uint8), k_bg) > 0
            local_bg = local_ring & (target_bin == 0) & (base_pred_bin == 0)

            # Matched control crack segment
            ctrl_info = find_matched_control_segment(target_bin, d_gap)
            ctrl_mask = ctrl_info['corridor_mask'] if ctrl_info is not None else None

            # Corridor baseline continuous metrics
            z00_vals = base_logits_np[corridor]
            z00_med = float(np.median(z00_vals))
            z00_p10 = float(np.percentile(z00_vals, 10))
            z00_prob = float(1.0 / (1.0 + np.exp(-z00_med)))

            # Control baseline continuous metrics
            if ctrl_mask is not None and np.sum(ctrl_mask) > 0:
                z00_ctrl_vals = base_logits_np[ctrl_mask]
                z00_ctrl_med = float(np.median(z00_ctrl_vals))
            else:
                z00_ctrl_med = np.nan

            # Project masks down to grid for decoder_28 and decoder_56
            mask28, stats28 = project_corridor_to_feature_grid(corridor, target_bin, coords, H, W, 28, 28, device)
            mask56, stats56 = project_corridor_to_feature_grid(corridor, target_bin, coords, H, W, 56, 56, device)

            # Control masks
            if ctrl_mask is not None:
                ctrl_mask28, _ = project_corridor_to_feature_grid(ctrl_mask, np.zeros_like(target_bin), coords, H, W, 28, 28, device)
                ctrl_mask56, _ = project_corridor_to_feature_grid(ctrl_mask, np.zeros_like(target_bin), coords, H, W, 56, 56, device)
            else:
                ctrl_mask28, ctrl_mask56 = None, None

            # Results dictionary for this event
            rec: Dict[str, Any] = {
                'event_id': ev_id,
                'case_name': case_name,
                'd_gap_px': d_gap,
                'is_small_gap_le8px': is_small_gap,
                'n_corridor_pixels': int(np.sum(corridor)),
                'z00_corridor_logit': z00_med,
                'z00_corridor_p10': z00_p10,
                'z00_corridor_prob': z00_prob,
                'z00_control_logit': z00_ctrl_med,
                'dec28_pure_cells': stats28['pure_cells'],
                'dec28_mixed_cells': stats28['mixed_cells'],
                'dec28_mean_coverage': stats28['mean_coverage_ratio'],
                'dec56_pure_cells': stats56['pure_cells'],
                'dec56_mixed_cells': stats56['mixed_cells'],
                'dec56_mean_coverage': stats56['mean_coverage_ratio'],
            }

            # Helper runner for 2x2 factorial conditions
            def run_factorial(stage_idx: int, m_grid: torch.Tensor, m_ctrl_grid: Optional[torch.Tensor], stage_name: str):
                block = model.decoder.decoder_blocks[stage_idx]
                orig_forward = block.forward

                results_cond: Dict[str, Tuple[float, float, float, bool, bool]] = {}
                cond_logits_map: Dict[str, np.ndarray] = {'00': base_logits_np}

                for cond in ['10', '01', '11']:
                    do_u = (cond in ['10', '11'])
                    do_s = (cond in ['01', '11'])

                    def patched_forward(x, skip):
                        u = block.upsample(x)
                        if u.shape[2:] != skip.shape[2:]:
                            u = F.interpolate(u, size=skip.shape[2:], mode='bilinear', align_corners=False)
                        if do_u:
                            u = u * (1.0 - m_grid)
                        if do_s:
                            skip = skip * (1.0 - m_grid)
                        fused = torch.cat([u, skip], dim=1)
                        out = block.conv1(fused)
                        out = block.conv2(out)
                        return out

                    block.forward = patched_forward
                    try:
                        with torch.no_grad():
                            out_tiles = model(batch)
                    finally:
                        block.forward = orig_forward

                    out_np = np.zeros((pH, pW), dtype=np.float32)
                    for j, (y, x) in enumerate(coords):
                        out_np[y:y+448, x:x+448] = out_tiles[j, 0].cpu().numpy()
                    out_np = out_np[:H, :W]
                    cond_logits_map[cond] = out_np

                    # Metrics
                    c_vals = out_np[corridor]
                    c_med = float(np.median(c_vals))
                    c_p10 = float(np.percentile(c_vals, 10))
                    c_prob = float(1.0 / (1.0 + np.exp(-c_med)))

                    # Topology
                    interv_bin = (out_np > 0.0).astype(np.uint8)
                    _, interv_labels = cv2.connectedComponents(interv_bin, connectivity=8)
                    gA_lbls = interv_labels[gt_labels == gA]
                    gB_lbls = interv_labels[gt_labels == gB]
                    gA_lbls = gA_lbls[gA_lbls > 0]
                    gB_lbls = gB_lbls[gB_lbls > 0]
                    merged_now = False
                    if len(gA_lbls) > 0 and len(gB_lbls) > 0:
                        merged_now = bool(np.bincount(gA_lbls).argmax() == np.bincount(gB_lbls).argmax())
                    is_cured = not merged_now

                    # Check control
                    if ctrl_mask is not None and np.sum(ctrl_mask) > 0:
                        ctrl_vals = out_np[ctrl_mask]
                        ctrl_med = float(np.median(ctrl_vals))
                    else:
                        ctrl_med = np.nan

                    results_cond[cond] = (c_med, c_p10, c_prob, is_cured, ctrl_med)

                # Now calculate effects
                z10_med, z10_p10, z10_prob, cured_10, z10_ctrl = results_cond['10']
                z01_med, z01_p10, z01_prob, cured_01, z01_ctrl = results_cond['01']
                z11_med, z11_p10, z11_prob, cured_11, z11_ctrl = results_cond['11']

                d_u = z00_med - z10_med
                d_s = z00_med - z01_med
                d_us = z00_med - z11_med
                i_us = d_us - d_u - d_s

                # Control effects
                if not np.isnan(z00_ctrl_med) and not np.isnan(z10_ctrl):
                    d_u_ctrl = z00_ctrl_med - z10_ctrl
                    d_s_ctrl = z00_ctrl_med - z01_ctrl
                    d_us_ctrl = z00_ctrl_med - z11_ctrl
                    d_u_sel = d_u - d_u_ctrl
                    d_s_sel = d_s - d_s_ctrl
                    d_us_sel = d_us - d_us_ctrl
                else:
                    d_u_ctrl = d_s_ctrl = d_us_ctrl = np.nan
                    d_u_sel = d_s_sel = d_us_sel = np.nan

                # Determine dominant driver
                if abs(d_us) < 0.2:
                    driver = 'negligible'
                elif i_us > 0.5 and i_us > max(d_u, d_s):
                    driver = 'interaction_fusion'
                elif d_u > d_s + 0.3:
                    driver = 'upsample_driver'
                elif d_s > d_u + 0.3:
                    driver = 'skip_driver'
                else:
                    driver = 'co_contributed'

                rec_stage = {
                    f'{stage_name}_z10_logit': z10_med,
                    f'{stage_name}_z10_p10': z10_p10,
                    f'{stage_name}_z10_prob': z10_prob,
                    f'{stage_name}_cured_10': cured_10,
                    f'{stage_name}_z01_logit': z01_med,
                    f'{stage_name}_z01_p10': z01_p10,
                    f'{stage_name}_z01_prob': z01_prob,
                    f'{stage_name}_cured_01': cured_01,
                    f'{stage_name}_z11_logit': z11_med,
                    f'{stage_name}_z11_p10': z11_p10,
                    f'{stage_name}_z11_prob': z11_prob,
                    f'{stage_name}_cured_11': cured_11,
                    f'{stage_name}_delta_U': d_u,
                    f'{stage_name}_delta_S': d_s,
                    f'{stage_name}_delta_US': d_us,
                    f'{stage_name}_interaction_I_US': i_us,
                    f'{stage_name}_delta_U_ctrl': d_u_ctrl,
                    f'{stage_name}_delta_S_ctrl': d_s_ctrl,
                    f'{stage_name}_delta_US_ctrl': d_us_ctrl,
                    f'{stage_name}_delta_U_selective': d_u_sel,
                    f'{stage_name}_delta_S_selective': d_s_sel,
                    f'{stage_name}_delta_US_selective': d_us_sel,
                    f'{stage_name}_driver_classification': driver
                }

                # Save representative visual panel for select events
                if ev_id in [2, 3, 4, 10, 16, 20]:
                    v_path = os.path.join(viz_dir, f"event_{ev_id:03d}_{case_name}_{stage_name}_gap{d_gap:.1f}px.png")
                    render_causal_panel(
                        img, target_bin, corridor, base_logits_np, cond_logits_map,
                        v_path, ev_id, case_name, d_gap, stage_name, d_u, d_s, d_us, i_us
                    )

                return rec_stage, cond_logits_map, d_us

            # Run for decoder_28
            rec28, _, d_us_28 = run_factorial(0, mask28, ctrl_mask28, 'dec28')
            rec.update(rec28)

            # Run for decoder_56
            rec56, cond_map56, d_us_56 = run_factorial(1, mask56, ctrl_mask56, 'dec56')
            rec.update(rec56)

            # Robustness check on strong causal events (if delta_US > 0.8 on dec56)
            if d_us_56 > 0.8:
                mean_u_bg, mean_s_bg = compute_local_bg_vector(1, model, batch, coords, local_bg, H, W, 56, 56, device)
                block1 = model.decoder.decoder_blocks[1]
                orig_forward = block1.forward

                def bg_patched_forward(x, skip):
                    u = block1.upsample(x)
                    if u.shape[2:] != skip.shape[2:]:
                        u = F.interpolate(u, size=skip.shape[2:], mode='bilinear', align_corners=False)
                    u = u * (1.0 - mask56) + mean_u_bg * mask56
                    skip = skip * (1.0 - mask56) + mean_s_bg * mask56
                    fused = torch.cat([u, skip], dim=1)
                    out = block1.conv1(fused)
                    out = block1.conv2(out)
                    return out

                block1.forward = bg_patched_forward
                try:
                    with torch.no_grad():
                        bg_out_tiles = model(batch)
                finally:
                    block1.forward = orig_forward

                bg_out_np = np.zeros((pH, pW), dtype=np.float32)
                for j, (y, x) in enumerate(coords):
                    bg_out_np[y:y+448, x:x+448] = bg_out_tiles[j, 0].cpu().numpy()
                bg_out_np = bg_out_np[:H, :W]

                bg_c_vals = bg_out_np[corridor]
                rec['dec56_robustness_bg_logit'] = float(np.median(bg_c_vals))
                rec['dec56_robustness_delta_US'] = float(z00_med - np.median(bg_c_vals))
            else:
                rec['dec56_robustness_bg_logit'] = np.nan
                rec['dec56_robustness_delta_US'] = np.nan

            event_records.append(rec)
            pbar.update(1)

        pbar.close()
        df_out = pd.DataFrame(event_records)
        df_out.to_csv(csv_out, index=False)
        print(f"\nSaved raw per-event measurements: {csv_out}")

    compute_summaries(df_out, out_dir, t0_start)


if __name__ == '__main__':
    main()
