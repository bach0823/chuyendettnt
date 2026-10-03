#!/usr/bin/env python3
"""
scripts/diagnostics/run_skip_pathway_sensitivity_comparison.py

Controlled Skip-Pathway Sensitivity Comparison for All Direct Decoder Skips in Candidate B:
  - Stage-0 skip (S0): 112x112, C=48  -> Consumed by DecoderBlock 2
  - Stage-1 skip (S1): 56x56,   C=96  -> Consumed by DecoderBlock 1
  - Stage-2 skip (S2): 28x28,   C=192 -> Consumed by DecoderBlock 0

Note on Stage-3:
  In Candidate B, Stage-3 is the 14x14 feature map (C=384) feeding the ViT bottleneck.
  The ViT output is the input 'bottleneck' tensor to UNetDecoder.forward.
  Stage-3 is NOT a direct decoder skip connection; it is the bottleneck input.
  Therefore, direct decoder skips strictly comprise {S0, S1, S2}.

Methodology:
  Apples-to-apples generalization of the 2x2 causal pathway ablation:
  - Same checkpoint: Candidate B baseline
  - Same cohort: 43 wider-gap bridge events (D_gap > 5.0 px) and full 118 bridge events
  - Clean-control cohort: 43 clean validation images with length-matched crack segments
  - Same mask projection: project_corridor_to_feature_grid with endpoint protection
  - Same intervention mechanism: skip = skip * (1.0 - m_grid)
  - Same output logit readout: Delta = z_00(corridor) - z_interv(corridor)
  - Zero training, zero optimizer step, sealed test set.
"""

import argparse
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
import torch.nn as nn
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


# -----------------------------------------------------------------------------
# Matched Control Crack Segment Finding
# -----------------------------------------------------------------------------

def find_matched_control_segment(target_bin: np.ndarray, target_length_px: float, seed: int = 42) -> Optional[Dict[str, Any]]:
    """
    Finds a matched true crack continuation segment in target_bin
    with Euclidean distance between endpoints approximately equal to target_length_px.
    """
    skel = skeletonize(target_bin > 0).astype(np.uint8)
    pts = np.argwhere(skel)
    if len(pts) < 10:
        return None

    np.random.seed(seed)
    sample_indices = np.random.choice(len(pts), size=min(100, len(pts)), replace=False)

    best_pair = None
    best_diff = float("inf")

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
        "corridor_mask": ctrl_corridor,
        "length_px": float(np.linalg.norm(p0 - p1)),
    }


# -----------------------------------------------------------------------------
# Corridor Projection to Decoder Feature Grids
# -----------------------------------------------------------------------------

def project_corridor_to_feature_grid(
    corridor_mask: np.ndarray,
    target_bin: np.ndarray,
    coords: List[Tuple[int, int]],
    H: int,
    W: int,
    H_grid: int,
    W_grid: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Projects corridor mask from 448-space down to feature grid (H_grid, W_grid).
    Applies endpoint protection:
      - Pure gap cells (corr > 0 and crack == 0): weight = 1.0
      - Mixed cells (corr > 0 and crack > 0): soft weight = max(0.0, (n_corr - n_crack) / cell_area)
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
                            w = max(0.0, (n_c - n_k) / float(cell_area))
                            cell_mask[j, 0, r_idx, c_idx] = w
                            total_mixed_cells += 1

    stats = {
        "grid_res": f"{H_grid}x{W_grid}",
        "stride_s": S,
        "pure_cells": total_pure_cells,
        "mixed_cells": total_mixed_cells,
        "total_active_cells": total_pure_cells + total_mixed_cells,
        "mean_coverage_ratio": float(np.mean(cell_coverages)) if len(cell_coverages) > 0 else 0.0,
    }
    return cell_mask, stats


# -----------------------------------------------------------------------------
# Skip Intervention Execution
# -----------------------------------------------------------------------------

def evaluate_skip_intervention(
    model: Any,
    batch: torch.Tensor,
    coords: List[Tuple[int, int]],
    pH: int,
    pW: int,
    H: int,
    W: int,
    block_idx: int,
    mask_grid: torch.Tensor,
) -> np.ndarray:
    """
    Applies lateral skip masking on decoder_blocks[block_idx]:
      skip = skip * (1.0 - mask_grid)
    Runs forward pass through the entire model, stitches tiles, and returns (H, W) logits.
    """
    block = model.decoder.decoder_blocks[block_idx]
    orig_forward = block.forward

    def patched_forward(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        u = block.upsample(x)
        if u.shape[2:] != skip.shape[2:]:
            u = F.interpolate(u, size=skip.shape[2:], mode="bilinear", align_corners=False)
        skip = skip * (1.0 - mask_grid)
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
    return out_np[:H, :W]


# -----------------------------------------------------------------------------
# Main Evaluation Pipeline
# -----------------------------------------------------------------------------

def run_skip_comparison(
    config_path: str,
    ckpt_path: str,
    out_dir: str,
    max_events: Optional[int] = None,
    clean_sample_size: int = 43,
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False

    print("=" * 80)
    print("PHASE 6: CONTROLLED SKIP-PATHWAY SENSITIVITY COMPARISON (S0 vs S1 vs S2)")
    print(f"Device: {device} | Checkpoint: {os.path.basename(ckpt_path)}")
    print("Direct Decoder Skips:")
    print("  - S0: 112x112, C=48  -> DecoderBlock 2")
    print("  - S1: 56x56,   C=96  -> DecoderBlock 1")
    print("  - S2: 28x28,   C=192 -> DecoderBlock 0")
    print("=" * 80)

    model = load_model_from_checkpoint(config_path, ckpt_path, device, turn_off_asdw=False)
    model.eval()

    # Verify decoder structure
    assert len(model.decoder.decoder_blocks) == 3
    skip_configs = [
        {"name": "S0", "block_idx": 2, "res": 112, "channels": 48,  "stride": 4},
        {"name": "S1", "block_idx": 1, "res": 56,  "channels": 96,  "stride": 8},
        {"name": "S2", "block_idx": 0, "res": 28,  "channels": 192, "stride": 16},
    ]
    for sc in skip_configs:
        blk = model.decoder.decoder_blocks[sc["block_idx"]]
        assert blk.skip_channels == sc["channels"], f"Mismatch in {sc['name']} skip channels!"

    # Load 118 bridge events dataset
    csv_118 = os.path.join(project_root, "results", "diagnostics", "phase6_bottleneck_path", "bottleneck_path_118events.csv")
    df_118 = pd.read_csv(csv_118)
    if max_events is not None:
        df_118 = df_118.head(max_events)

    data_root = os.path.join(project_root, "datasets", "Crack500_ready")
    val_img_dir = os.path.join(data_root, "val", "images")
    val_mask_dir = os.path.join(data_root, "val", "masks")

    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tile_size = 448

    # -------------------------------------------------------------------------
    # PART 1: BRIDGE-EVENT COHORT EVALUATION
    # -------------------------------------------------------------------------
    print(f"\n[PART 1] Evaluating Bridge Cohort (N = {len(df_118)} events)...")
    event_records = []
    grouped = df_118.groupby("case_name")

    for case_name, group_events in tqdm(grouped, desc="Bridge Cohort Evaluation"):
        ip = os.path.join(val_img_dir, case_name + ".jpg")
        if not os.path.exists(ip):
            ip = os.path.join(val_img_dir, case_name + ".png")
        mp = os.path.join(val_mask_dir, case_name + ".png")
        if not os.path.exists(mp):
            mp = os.path.join(val_mask_dir, case_name + ".jpg")

        img = cv2.imread(ip)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        target_bin = (target > 127).astype(np.uint8)
        H, W = target_bin.shape[:2]

        pad_h = (tile_size - (H % tile_size)) % tile_size
        pad_w = (tile_size - (W % tile_size)) % tile_size
        padded_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        pH, pW = padded_img.shape[:2]

        patches, coords = [], []
        for y in range(0, pH, tile_size):
            for x in range(0, pW, tile_size):
                p = padded_img[y:y+tile_size, x:x+tile_size]
                pt = torch.from_numpy(p).permute(2, 0, 1).float() / 255.0
                pt = (pt - mean_t) / std_t
                patches.append(pt)
                coords.append((y, x))

        batch = torch.stack(patches).to(device)

        # Baseline prediction
        with torch.no_grad():
            base_tiles = model(batch)
        base_logits_np = np.zeros((pH, pW), dtype=np.float32)
        for j, (y, x) in enumerate(coords):
            base_logits_np[y:y+448, x:x+448] = base_tiles[j, 0].cpu().numpy()
        base_logits_np = base_logits_np[:H, :W]
        base_pred_bin = (base_logits_np > 0.0).astype(np.uint8)

        num_gt_cc, gt_labels = cv2.connectedComponents(target_bin, connectivity=8)
        num_pred_cc, base_pred_labels = cv2.connectedComponents(base_pred_bin, connectivity=8)

        # Count baseline bridges in this image
        base_bridge_count = 0
        for p_cc in range(1, num_pred_cc):
            gt_overlap = np.unique(gt_labels[base_pred_labels == p_cc])
            gt_overlap = gt_overlap[gt_overlap > 0]
            if len(gt_overlap) >= 2:
                base_bridge_count += 1

        for _, ev_row in group_events.iterrows():
            ev_id = int(ev_row["event_id"])
            p_id = int(ev_row["pred_cc_id"])
            gA = int(ev_row["primary_gt_A"])
            gB = int(ev_row["primary_gt_B"])
            d_gap = float(ev_row["d_gap_px"])
            gap_cat = str(ev_row["gap_category"])
            is_wider = bool(d_gap > 5.0)

            cc_fp = (base_pred_labels == p_id) & (target_bin == 0)
            r = max(int(np.ceil(d_gap / 2.0)) + 2, 3)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
            dil_A = cv2.dilate((gt_labels == gA).astype(np.uint8), k)
            dil_B = cv2.dilate((gt_labels == gB).astype(np.uint8), k)
            corridor = (dil_A > 0) & (dil_B > 0) & cc_fp
            if np.sum(corridor) == 0:
                corridor = cc_fp

            # Control crack segment
            ctrl_info = find_matched_control_segment(target_bin, d_gap)
            ctrl_mask = ctrl_info["corridor_mask"] if ctrl_info is not None else None

            z00_corridor_med = float(np.median(base_logits_np[corridor]))
            z00_control_med = float(np.median(base_logits_np[ctrl_mask])) if (ctrl_mask is not None and np.sum(ctrl_mask) > 0) else np.nan

            rec: Dict[str, Any] = {
                "event_id": ev_id,
                "case_name": case_name,
                "d_gap_px": d_gap,
                "gap_category": gap_cat,
                "is_wider_gap_gt5px": is_wider,
                "n_corridor_pixels": int(np.sum(corridor)),
                "z00_corridor_logit": z00_corridor_med,
                "z00_control_logit": z00_control_med,
                "base_image_bridges": base_bridge_count,
            }

            for sc in skip_configs:
                s_name = sc["name"]
                b_idx = sc["block_idx"]
                res = sc["res"]

                # Project masks
                m_grid, stats = project_corridor_to_feature_grid(corridor, target_bin, coords, H, W, res, res, device)
                if ctrl_mask is not None:
                    m_ctrl_grid, _ = project_corridor_to_feature_grid(ctrl_mask, np.zeros_like(target_bin), coords, H, W, res, res, device)
                else:
                    m_ctrl_grid = None

                # Forward with corridor skip masked
                out_np = evaluate_skip_intervention(model, batch, coords, pH, pW, H, W, b_idx, m_grid)

                # Logit measurements
                z_interv_med = float(np.median(out_np[corridor]))
                delta_s = z00_corridor_med - z_interv_med

                # Global MAD on whole image
                mad_full = float(np.mean(np.abs(base_logits_np - out_np)))

                # Topology cure check
                interv_bin = (out_np > 0.0).astype(np.uint8)
                num_interv_cc, interv_labels = cv2.connectedComponents(interv_bin, connectivity=8)
                gA_lbls = interv_labels[gt_labels == gA]
                gB_lbls = interv_labels[gt_labels == gB]
                gA_lbls = gA_lbls[gA_lbls > 0]
                gB_lbls = gB_lbls[gB_lbls > 0]
                merged_now = False
                if len(gA_lbls) > 0 and len(gB_lbls) > 0:
                    merged_now = bool(np.bincount(gA_lbls).argmax() == np.bincount(gB_lbls).argmax())
                is_cured = not merged_now

                # Count total bridges in intervened image
                interv_bridge_count = 0
                for p_cc in range(1, num_interv_cc):
                    gt_overlap = np.unique(gt_labels[interv_labels == p_cc])
                    gt_overlap = gt_overlap[gt_overlap > 0]
                    if len(gt_overlap) >= 2:
                        interv_bridge_count += 1
                new_bridges_created = max(0, interv_bridge_count - (base_bridge_count - (1 if is_cured else 0)))

                # Matched control crack segment
                if ctrl_mask is not None and m_ctrl_grid is not None:
                    out_ctrl_np = evaluate_skip_intervention(model, batch, coords, pH, pW, H, W, b_idx, m_ctrl_grid)
                    z_ctrl_interv = float(np.median(out_ctrl_np[ctrl_mask]))
                    delta_s_ctrl = z00_control_logit = z00_control_med - z_ctrl_interv
                else:
                    delta_s_ctrl = np.nan

                rec[f"{s_name}_pure_cells"] = stats["pure_cells"]
                rec[f"{s_name}_mixed_cells"] = stats["mixed_cells"]
                rec[f"{s_name}_z_logit"] = z_interv_med
                rec[f"{s_name}_delta"] = delta_s
                rec[f"{s_name}_cured"] = is_cured
                rec[f"{s_name}_mad_full"] = mad_full
                rec[f"{s_name}_interv_bridges"] = interv_bridge_count
                rec[f"{s_name}_new_bridges"] = new_bridges_created
                rec[f"{s_name}_delta_ctrl"] = delta_s_ctrl

            event_records.append(rec)

    df_events = pd.DataFrame(event_records)
    csv_events_out = os.path.join(out_dir, "skip_sensitivity_per_event_118all.csv")
    df_events.to_csv(csv_events_out, index=False)
    print(f"Saved full 118 events: {csv_events_out}")

    df_43_out = df_events[df_events["is_wider_gap_gt5px"]].copy().reset_index(drop=True)
    csv_43_out = os.path.join(out_dir, "skip_sensitivity_per_event_43wider.csv")
    df_43_out.to_csv(csv_43_out, index=False)
    print(f"Saved 43 wider gap events: {csv_43_out}")

    # -------------------------------------------------------------------------
    # PART 2: CLEAN CONTROL COHORT EVALUATION (Non-bridge images)
    # -------------------------------------------------------------------------
    print(f"\n[PART 2] Evaluating Clean Control Cohort (N = {clean_sample_size} clean images)...")
    bridge_cases_set = set(df_118["case_name"].unique())
    all_val_files = sorted([os.path.splitext(f)[0] for f in os.listdir(val_img_dir) if f.endswith((".jpg", ".png"))])
    clean_cases = [c for c in all_val_files if c not in bridge_cases_set]

    # Select deterministic subset of clean cases with true cracks
    np.random.seed(42)
    selected_clean_cases = []
    for c in clean_cases:
        mp = os.path.join(val_mask_dir, c + ".png")
        if not os.path.exists(mp):
            mp = os.path.join(val_mask_dir, c + ".jpg")
        tgt = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if tgt is not None and np.sum(tgt > 127) >= 100:
            selected_clean_cases.append(c)
        if len(selected_clean_cases) >= clean_sample_size:
            break

    # Sample matched gap lengths from bridge cohort
    sampled_lengths = df_118["d_gap_px"].sample(n=len(selected_clean_cases), random_state=42, replace=True).values

    clean_records = []
    for case_name, target_len in tqdm(zip(selected_clean_cases, sampled_lengths), total=len(selected_clean_cases), desc="Clean Control Evaluation"):
        ip = os.path.join(val_img_dir, case_name + ".jpg")
        if not os.path.exists(ip):
            ip = os.path.join(val_img_dir, case_name + ".png")
        mp = os.path.join(val_mask_dir, case_name + ".png")
        if not os.path.exists(mp):
            mp = os.path.join(val_mask_dir, case_name + ".jpg")

        img = cv2.imread(ip)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        target_bin = (target > 127).astype(np.uint8)
        H, W = target_bin.shape[:2]

        pad_h = (tile_size - (H % tile_size)) % tile_size
        pad_w = (tile_size - (W % tile_size)) % tile_size
        padded_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        pH, pW = padded_img.shape[:2]

        patches, coords = [], []
        for y in range(0, pH, tile_size):
            for x in range(0, pW, tile_size):
                p = padded_img[y:y+tile_size, x:x+tile_size]
                pt = torch.from_numpy(p).permute(2, 0, 1).float() / 255.0
                pt = (pt - mean_t) / std_t
                patches.append(pt)
                coords.append((y, x))

        batch = torch.stack(patches).to(device)

        with torch.no_grad():
            base_tiles = model(batch)
        base_logits_np = np.zeros((pH, pW), dtype=np.float32)
        for j, (y, x) in enumerate(coords):
            base_logits_np[y:y+448, x:x+448] = base_tiles[j, 0].cpu().numpy()
        base_logits_np = base_logits_np[:H, :W]
        base_pred_bin = (base_logits_np > 0.0).astype(np.uint8)

        num_gt_cc, gt_labels = cv2.connectedComponents(target_bin, connectivity=8)
        num_pred_cc, base_pred_labels = cv2.connectedComponents(base_pred_bin, connectivity=8)

        # Baseline fragmented count
        base_frag_count = 0
        for g_id in range(1, num_gt_cc):
            ov = np.unique(base_pred_labels[gt_labels == g_id])
            ov = ov[ov > 0]
            if len(ov) >= 2:
                base_frag_count += 1

        ctrl_info = find_matched_control_segment(target_bin, float(target_len))
        if ctrl_info is None or np.sum(ctrl_info["corridor_mask"]) == 0:
            continue
        crack_segment = ctrl_info["corridor_mask"]
        z00_crack_med = float(np.median(base_logits_np[crack_segment]))

        rec_c = {
            "case_name": case_name,
            "target_len_px": float(target_len),
            "gt_crack_pixels": int(np.sum(target_bin)),
            "segment_pixels": int(np.sum(crack_segment)),
            "z00_crack_logit": z00_crack_med,
            "base_frag_count": base_frag_count,
        }

        for sc in skip_configs:
            s_name = sc["name"]
            b_idx = sc["block_idx"]
            res = sc["res"]

            m_grid, _ = project_corridor_to_feature_grid(crack_segment, np.zeros_like(target_bin), coords, H, W, res, res, device)
            out_np = evaluate_skip_intervention(model, batch, coords, pH, pW, H, W, b_idx, m_grid)

            z_interv_med = float(np.median(out_np[crack_segment]))
            delta_crack = z00_crack_med - z_interv_med

            interv_bin = (out_np > 0.0).astype(np.uint8)
            num_interv_cc, interv_labels = cv2.connectedComponents(interv_bin, connectivity=8)

            # Check if crack broke
            interv_frag_count = 0
            for g_id in range(1, num_gt_cc):
                ov = np.unique(interv_labels[gt_labels == g_id])
                ov = ov[ov > 0]
                if len(ov) >= 2:
                    interv_frag_count += 1
            new_breakage = max(0, interv_frag_count - base_frag_count)

            # Check if any false bridge was created
            created_bridges = 0
            for p_cc in range(1, num_interv_cc):
                gt_ov = np.unique(gt_labels[interv_labels == p_cc])
                gt_ov = gt_ov[gt_ov > 0]
                if len(gt_ov) >= 2:
                    created_bridges += 1

            rec_c[f"{s_name}_delta_crack"] = delta_crack
            rec_c[f"{s_name}_new_breakage"] = new_breakage
            rec_c[f"{s_name}_created_bridges"] = created_bridges

        clean_records.append(rec_c)

    df_clean = pd.DataFrame(clean_records)
    csv_clean_out = os.path.join(out_dir, "skip_sensitivity_clean_control.csv")
    df_clean.to_csv(csv_clean_out, index=False)
    print(f"Saved clean control measurements: {csv_clean_out}")

    # -------------------------------------------------------------------------
    # PART 3: AGGREGATE SUMMARY & DELIVERABLES TABLE
    # -------------------------------------------------------------------------
    print("\n[PART 3] Computing Aggregate Metrics & Decision Statistics...")

    # Build summary rows matching user's exact specification:
    # | Skip | Resolution | Channels | median Δ | mean Δ | bridge Δ | bridge cured | bridge created | clean impact |
    summary_rows = []
    summary_dict: Dict[str, Any] = {}

    for sc in skip_configs:
        s_name = sc["name"]
        res_str = f"{sc['res']}x{sc['res']}"
        ch = sc["channels"]

        # 43 wider gap cohort
        delta_43 = df_43_out[f"{s_name}_delta"]
        med_delta_43 = float(delta_43.median())
        mean_delta_43 = float(delta_43.mean())
        p10_43 = float(delta_43.quantile(0.10))
        p25_43 = float(delta_43.quantile(0.25))
        p75_43 = float(delta_43.quantile(0.75))
        p90_43 = float(delta_43.quantile(0.90))

        cured_43 = int(df_43_out[f"{s_name}_cured"].sum())
        cured_pct_43 = float(df_43_out[f"{s_name}_cured"].mean() * 100.0)
        new_bridges_43 = int(df_43_out[f"{s_name}_new_bridges"].sum())

        # Full 118 cohort
        delta_118 = df_events[f"{s_name}_delta"]
        med_delta_118 = float(delta_118.median())
        mean_delta_118 = float(delta_118.mean())
        cured_118 = int(df_events[f"{s_name}_cured"].sum())
        cured_pct_118 = float(df_events[f"{s_name}_cured"].mean() * 100.0)
        new_bridges_118 = int(df_events[f"{s_name}_new_bridges"].sum())

        # Clean control impact
        delta_clean = df_clean[f"{s_name}_delta_crack"]
        med_clean_delta = float(delta_clean.median())
        mean_clean_delta = float(delta_clean.mean())
        clean_breakage_events = int(df_clean[f"{s_name}_new_breakage"].sum())
        clean_bridges_created = int(df_clean[f"{s_name}_created_bridges"].sum())

        summary_rows.append({
            "Skip": s_name,
            "Resolution": res_str,
            "Channels": ch,
            "Decoder_Block": f"Block {sc['block_idx']}",
            "median_Delta_43w": med_delta_43,
            "mean_Delta_43w": mean_delta_43,
            "P10_43w": p10_43,
            "P25_43w": p25_43,
            "P75_43w": p75_43,
            "P90_43w": p90_43,
            "bridge_Delta_43w": f"{med_delta_43:+.4f} (mean {mean_delta_43:+.4f})",
            "bridge_cured_43w": f"{cured_43}/43 ({cured_pct_43:.1f}%)",
            "bridge_created_43w": new_bridges_43,
            "median_Delta_118all": med_delta_118,
            "mean_Delta_118all": mean_delta_118,
            "bridge_cured_118all": f"{cured_118}/118 ({cured_pct_118:.1f}%)",
            "clean_crack_Delta": f"{med_clean_delta:+.4f} (mean {mean_clean_delta:+.4f})",
            "clean_breakage_events": clean_breakage_events,
            "clean_bridges_created": clean_bridges_created,
        })

        summary_dict[s_name] = {
            "resolution": res_str,
            "channels": ch,
            "decoder_block": f"DecoderBlock {sc['block_idx']}",
            "wider_gap_43": {
                "median_delta": med_delta_43,
                "mean_delta": mean_delta_43,
                "p10": p10_43,
                "p25": p25_43,
                "p75": p75_43,
                "p90": p90_43,
                "cured_count": cured_43,
                "cured_rate_pct": cured_pct_43,
                "new_bridges_created": new_bridges_43,
            },
            "full_118": {
                "median_delta": med_delta_118,
                "mean_delta": mean_delta_118,
                "cured_count": cured_118,
                "cured_rate_pct": cured_pct_118,
                "new_bridges_created": new_bridges_118,
            },
            "clean_control": {
                "median_crack_delta": med_clean_delta,
                "mean_crack_delta": mean_clean_delta,
                "breakage_events": clean_breakage_events,
                "bridges_created": clean_bridges_created,
            }
        }

    df_sum = pd.DataFrame(summary_rows)
    sum_csv_out = os.path.join(out_dir, "skip_sensitivity_summary.csv")
    df_sum.to_csv(sum_csv_out, index=False)

    sum_json_out = os.path.join(out_dir, "skip_sensitivity_summary.json")
    with open(sum_json_out, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    # Print the user-specified compact table
    print("\n" + "=" * 95)
    print("CONTROLLED SKIP-PATHWAY SENSITIVITY COMPARISON TABLE (CANDIDATE B)")
    print("=" * 95)
    compact_df = pd.DataFrame([
        {
            "Skip": r["Skip"],
            "Resolution": r["Resolution"],
            "Channels": r["Channels"],
            "median Delta (43w)": f"{r['median_Delta_43w']:+.4f}",
            "mean Delta (43w)": f"{r['mean_Delta_43w']:+.4f}",
            "bridge Delta (43w)": r["bridge_Delta_43w"],
            "bridge cured (43w)": r["bridge_cured_43w"],
            "bridge created": r["bridge_created_43w"],
            "clean impact": f"Delta_crack={r['clean_crack_Delta']}, break={r['clean_breakage_events']}",
        }
        for r in summary_rows
    ])
    print(compact_df.to_string(index=False))

    print("\n" + "=" * 95)
    print("FULL 118-EVENT COHORT RESULTS")
    print("=" * 95)
    compact_118 = pd.DataFrame([
        {
            "Skip": r["Skip"],
            "Resolution": r["Resolution"],
            "Channels": r["Channels"],
            "median Delta (118)": f"{r['median_Delta_118all']:+.4f}",
            "mean Delta (118)": f"{r['mean_Delta_118all']:+.4f}",
            "bridge cured (118)": r["bridge_cured_118all"],
        }
        for r in summary_rows
    ])
    print(compact_118.to_string(index=False))
    print("=" * 95)
    print(f"\nSaved all artifacts to: {out_dir}")


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description="Skip-Pathway Sensitivity Comparison")
    parser.add_argument("--config", type=str, default="results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml")
    parser.add_argument("--ckpt", type=str, default="results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth")
    parser.add_argument("--out-dir", type=str, default="results/diagnostics/skip_pathway_sensitivity")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--clean-samples", type=int, default=43)
    args = parser.parse_args()

    t0 = time.time()
    run_skip_comparison(
        config_path=os.path.join(project_root, args.config) if not os.path.isabs(args.config) else args.config,
        ckpt_path=os.path.join(project_root, args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt,
        out_dir=os.path.join(project_root, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir,
        max_events=args.max_events,
        clean_sample_size=args.clean_samples,
    )
    print(f"Total time elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
