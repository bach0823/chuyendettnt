#!/usr/bin/env python3
"""
scripts/diagnostics/run_phase6d_cgsr_evaluation.py

Phase 6D: Comprehensive Evaluation and Causal Hypothesis Testing for
Context-Guided Stage-1 Skip Refinement (CGSR).

Mục tiêu duy nhất:
  Kiểm tra H2: Stage-1 skip representation (B,96,56,56) của Candidate B có đang chứa
  các feature không đủ discriminative cho crack segmentation, và liệu deep semantic context
  (B,192,56,56) có thể giúp suppress phần skip content gây false bridge mà vẫn giữ true thin crack hay không.

Structure of the evaluation:
1. Part 1: Official Setting A Validation on full Crack500 N=348:
   - Per-sample mean Precision, Recall, Dice, IoU for Control and CGSR.
   - Delta Dice = Dice_cgsr - Dice_control.
2. Part 2: Cohort Analysis on 43 wider-gap bridge events (D_gap > 5.0 px):
   - Stratified by Group A (5.0 < D_gap <= 8.0 px, N=11) and Group B (D_gap > 8.0 px, N=32).
   - Full 118 events cohort.
   - Metrics: Delta z_bridge = z_control - z_cgsr, bridge cure rate, restored gap length.
3. Part 3: Mechanistic Gate Diagnostics:
   - Spatial gate map G in [0, 1] at 56x56 resolution.
   - Distributions of G_corridor, G_crack, G_bg.
   - Gate contrast: G_crack - G_corridor.
   - Correlation between G_corridor and Delta z_bridge.
4. Part 4: Negative Control & Collateral Damage:
   - 43 clean validation images with length-matched crack segments.
   - Delta z_crack on true cracks, breakage events.
5. Part 5: Scientific Decision Rule:
   - Explicitly verdicts: H2 SUPPORTED, H2 WEAKENED, or H2 NOT SUPPORTED.
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
from scipy.stats import pearsonr, spearmanr
from skimage.morphology import skeletonize
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import yaml

# Robust sys.path resolution for both local and Colab environments
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

candidate_sage_dirs = [
    os.environ.get("SAGE_LITE_DIR", ""),
    os.path.join(project_root, "SAGE_LITE"),
    os.path.abspath(os.path.join(project_root, "..", "SAGE_LITE")),
    "/content/SAGE_LITE",
    os.getcwd(),
]

for p in [project_root] + candidate_sage_dirs:
    if p and os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from sage.networks import create_b2_unet


def load_model_from_checkpoint(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
    turn_off_asdw: bool = True,
) -> nn.Module:
    """
    Self-contained loader for B2 UNet checkpoints supporting CGSR and canonical P3-C.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    num_layers = int(cfg.get("num_transformer_layers", 4))
    p3_mode = cfg.get("p3_mode", "C")
    img_size = int(cfg.get("img_size", 448))
    sage_cfg = cfg.get("sage_config", {})
    use_plu_head = cfg.get("use_plu_head", False)
    use_cgsr = cfg.get("use_cgsr", cfg.get("cgsr", False))
    cgsr_init_bias = float(cfg.get("cgsr_init_bias", 3.0))

    model = create_b2_unet(
        num_classes=1,
        img_size=img_size,
        num_transformer_layers=num_layers,
        pretrained=False,
        sage_config=sage_cfg,
        p3_mode=p3_mode,
        use_plu_head=use_plu_head,
        use_cgsr=use_cgsr,
        cgsr_init_bias=cgsr_init_bias,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    
    # Clean potential 'module.' prefixes
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[clean_k] = v

    model.load_state_dict(cleaned_state_dict)

    if turn_off_asdw and hasattr(model.backbone, "convnext"):
        for stage in model.backbone.convnext.stages[:2]:
            if hasattr(stage, "p3_refinement") and stage.p3_refinement is not None:
                stage.p3_refinement = nn.Identity()

    model.eval()
    return model


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
        "ctrl_p0": (int(p0[0]), int(p0[1])),
        "ctrl_p1": (int(p1[0]), int(p1[1])),
        "ctrl_length": float(np.linalg.norm(p1 - p0)),
        "ctrl_corridor_mask": ctrl_corridor,
    }


# -----------------------------------------------------------------------------
# Setting A Tiled Inference with CGSR Gate Capture
# -----------------------------------------------------------------------------

def run_tiled_inference_with_gate(
    model: nn.Module,
    img: np.ndarray,
    device: torch.device,
    tile_size: int = 448,
    mean_t: Optional[torch.Tensor] = None,
    std_t: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Runs Setting A tiled inference (non-overlap, tile=448, stride=448) on an RGB image.
    Returns:
      logits: (H, W) float32 numpy array
      gate_map: (H, W) float32 numpy array (mean across channels, upsampled to image resolution) or None
    """
    if mean_t is None:
        mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    if std_t is None:
        std_t = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    H, W = img.shape[:2]
    pad_h = (tile_size - (H % tile_size)) % tile_size
    pad_w = (tile_size - (W % tile_size)) % tile_size
    padded_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    pH, pW = padded_img.shape[:2]

    patches, coords = [], []
    for y in range(0, pH, tile_size):
        for x in range(0, pW, tile_size):
            p = padded_img[y:y+tile_size, x:x+tile_size]
            pt = torch.from_numpy(p).permute(2, 0, 1).float().to(device) / 255.0
            pt = (pt - mean_t) / std_t
            patches.append(pt)
            coords.append((y, x))

    batch = torch.stack(patches)

    # Check if CGSR is present in model
    cgsr_mod = None
    if hasattr(model, "decoder") and hasattr(model.decoder, "decoder_blocks") and len(model.decoder.decoder_blocks) > 1:
        cgsr_mod = getattr(model.decoder.decoder_blocks[1], "cgsr", None)

    # Hook gate tensor if CGSR present
    captured_gates = []
    hook_handle = None
    if cgsr_mod is not None:
        def gate_hook(module, inputs, output):
            # output is (skip_refined, gate)
            if isinstance(output, tuple) and len(output) == 2:
                captured_gates.append(output[1].detach())
        hook_handle = cgsr_mod.register_forward_hook(gate_hook)

    try:
        with torch.no_grad():
            out_tiles = model(batch)
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    logits_np = np.zeros((pH, pW), dtype=np.float32)
    for j, (y, x) in enumerate(coords):
        logits_np[y:y+tile_size, x:x+tile_size] = out_tiles[j, 0].cpu().numpy()
    logits_np = logits_np[:H, :W]

    gate_map_np = None
    if len(captured_gates) > 0:
        # captured_gates[0] is (B, 96, 56, 56)
        gates_tensor = captured_gates[0]  # (B, 96, 56, 56)
        # Average across the 96 channels to get spatial gate (B, 1, 56, 56)
        spatial_gates = gates_tensor.mean(dim=1, keepdim=True)
        # Upsample to 448x448
        spatial_gates_up = F.interpolate(spatial_gates, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
        gate_map_full = np.zeros((pH, pW), dtype=np.float32)
        for j, (y, x) in enumerate(coords):
            gate_map_full[y:y+tile_size, x:x+tile_size] = spatial_gates_up[j, 0].cpu().numpy()
        gate_map_np = gate_map_full[:H, :W]

    return logits_np, gate_map_np


# -----------------------------------------------------------------------------
# Metric Calculation Utilities
# -----------------------------------------------------------------------------

def compute_binary_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray) -> Dict[str, float]:
    """Computes Precision, Recall, Dice, and IoU between binary masks."""
    tp = np.sum((pred_bin == 1) & (gt_bin == 1))
    fp = np.sum((pred_bin == 1) & (gt_bin == 0))
    fn = np.sum((pred_bin == 0) & (gt_bin == 1))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    dice = float(2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 1.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 1.0

    return {"precision": prec, "recall": rec, "dice": dice, "iou": iou}


# -----------------------------------------------------------------------------
# Full Val N=348 Evaluation (Setting A)
# -----------------------------------------------------------------------------

def evaluate_full_val348(
    model: nn.Module,
    val_files: List[Tuple[str, str, str]],
    device: torch.device,
    tile_size: int = 448,
    model_name: str = "Model",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Evaluates model across all validation images in Crack500."""
    records = []
    for case_name, ip, mp in tqdm(val_files, desc=f"Val N=348 ({model_name})"):
        img = cv2.imread(ip)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        target_bin = (target > 127).astype(np.uint8)

        logits, _ = run_tiled_inference_with_gate(model, img, device, tile_size=tile_size)
        pred_bin = (logits > 0.0).astype(np.uint8)

        metrics = compute_binary_metrics(pred_bin, target_bin)
        metrics["case_name"] = case_name
        records.append(metrics)

    df = pd.DataFrame(records)
    summary = {
        "mean_dice": float(df["dice"].mean()),
        "std_dice": float(df["dice"].std()),
        "mean_iou": float(df["iou"].mean()),
        "mean_precision": float(df["precision"].mean()),
        "mean_recall": float(df["recall"].mean()),
    }
    return df, summary


# -----------------------------------------------------------------------------
# Main Evaluation Orchestration
# -----------------------------------------------------------------------------

def run_phase6d_evaluation(
    control_config: str,
    control_ckpt: str,
    cgsr_config: str,
    cgsr_ckpt: str,
    out_dir: str,
    data_root: str,
    events_csv: str,
    tile_size: int = 448,
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False

    print("=" * 80)
    print("PHASE 6D: CGSR FINAL EXPERIMENT EVALUATION & CAUSAL HYPOTHESIS TESTING")
    print(f"Device: {device}")
    print(f"Control Checkpoint: {control_ckpt}")
    print(f"CGSR Checkpoint:    {cgsr_ckpt}")
    print(f"Output Directory:   {out_dir}")
    print("=" * 80)

    # 1. Load models
    print("\n[Loading Models]...")
    control_model = load_model_from_checkpoint(control_config, control_ckpt, device, turn_off_asdw=False)
    control_model.eval()

    cgsr_model = load_model_from_checkpoint(cgsr_config, cgsr_ckpt, device, turn_off_asdw=False)
    cgsr_model.eval()

    # Invariant checks on CGSR module
    assert hasattr(cgsr_model, "decoder"), "CGSR model missing decoder!"
    cgsr_block = cgsr_model.decoder.decoder_blocks[1]
    assert hasattr(cgsr_block, "cgsr") and cgsr_block.cgsr is not None, "CGSR not found on DecoderBlock 1!"
    print(f"  CGSR Gate bias: {cgsr_block.cgsr.gate_conv.bias.data.mean().item():.4f}")
    print(f"  CGSR Gate weight norm: {cgsr_block.cgsr.gate_conv.weight.data.norm().item():.4f}")

    # 2. Gather dataset files
    val_img_dir = os.path.join(data_root, "val", "images")
    val_mask_dir = os.path.join(data_root, "val", "masks")
    assert os.path.exists(val_img_dir), f"Missing val images dir: {val_img_dir}"
    assert os.path.exists(val_mask_dir), f"Missing val masks dir: {val_mask_dir}"

    img_files = sorted([f for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg', '.png'))])
    val_files = []
    for fn in img_files:
        base, _ = os.path.splitext(fn)
        ip = os.path.join(val_img_dir, fn)
        mp = os.path.join(val_mask_dir, base + ".png")
        if not os.path.exists(mp):
            mp = os.path.join(val_mask_dir, base + ".jpg")
        if os.path.exists(mp):
            val_files.append((base, ip, mp))

    print(f"Found {len(val_files)} validation images in Crack500.")

    # -------------------------------------------------------------------------
    # PART 1: Official Setting A Validation on Full Val N=348
    # -------------------------------------------------------------------------
    print("\n[PART 1] Running Official Setting A Validation (N=348)...")
    df_ctrl_val, summary_ctrl_val = evaluate_full_val348(control_model, val_files, device, tile_size, "Control")
    df_cgsr_val, summary_cgsr_val = evaluate_full_val348(cgsr_model, val_files, device, tile_size, "CGSR")

    delta_dice = summary_cgsr_val["mean_dice"] - summary_ctrl_val["mean_dice"]
    delta_iou = summary_cgsr_val["mean_iou"] - summary_ctrl_val["mean_iou"]
    delta_prec = summary_cgsr_val["mean_precision"] - summary_ctrl_val["mean_precision"]
    delta_rec = summary_cgsr_val["mean_recall"] - summary_ctrl_val["mean_recall"]

    print("\n--- Val N=348 Setting A Comparison ---")
    print(f"Control: Dice = {summary_ctrl_val['mean_dice']:.4f} (IoU = {summary_ctrl_val['mean_iou']:.4f}, Prec = {summary_ctrl_val['mean_precision']:.4f}, Rec = {summary_ctrl_val['mean_recall']:.4f})")
    print(f"CGSR:    Dice = {summary_cgsr_val['mean_dice']:.4f} (IoU = {summary_cgsr_val['mean_iou']:.4f}, Prec = {summary_cgsr_val['mean_precision']:.4f}, Rec = {summary_cgsr_val['mean_recall']:.4f})")
    print(f"Delta:   Dice = {delta_dice:+.4f} | IoU = {delta_iou:+.4f} | Prec = {delta_prec:+.4f} | Rec = {delta_rec:+.4f}")

    # Merge per-image val metrics
    df_merged_val = pd.merge(df_ctrl_val, df_cgsr_val, on="case_name", suffixes=("_control", "_cgsr"))
    df_merged_val["delta_dice"] = df_merged_val["dice_cgsr"] - df_merged_val["dice_control"]
    df_merged_val["delta_iou"] = df_merged_val["iou_cgsr"] - df_merged_val["iou_control"]
    df_merged_val.to_csv(os.path.join(out_dir, "cgsr_full_val348_metrics.csv"), index=False)

    # -------------------------------------------------------------------------
    # PART 2: Bridge-Event Cohort Evaluation (43 Wider-Gap & 118 All Events)
    # -------------------------------------------------------------------------
    print(f"\n[PART 2] Evaluating Bridge Cohort from {events_csv}...")
    df_events = pd.read_csv(events_csv)
    grouped = df_events.groupby("case_name")

    event_records = []
    clean_control_records = []

    # Map validation files for fast lookup
    val_map = {base: (ip, mp) for base, ip, mp in val_files}

    for case_name, group in tqdm(grouped, desc="Bridge Cohort Evaluation"):
        if case_name not in val_map:
            continue
        ip, mp = val_map[case_name]
        img = cv2.imread(ip)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        target_bin = (target > 127).astype(np.uint8)
        H, W = target_bin.shape[:2]

        # Run both models and capture gate map
        ctrl_logits, _ = run_tiled_inference_with_gate(control_model, img, device, tile_size)
        cgsr_logits, cgsr_gate_map = run_tiled_inference_with_gate(cgsr_model, img, device, tile_size)

        ctrl_pred = (ctrl_logits > 0.0).astype(np.uint8)
        cgsr_pred = (cgsr_logits > 0.0).astype(np.uint8)

        num_gt_cc, gt_labels = cv2.connectedComponents(target_bin, connectivity=8)
        num_ctrl_cc, ctrl_labels = cv2.connectedComponents(ctrl_pred, connectivity=8)
        num_cgsr_cc, cgsr_labels = cv2.connectedComponents(cgsr_pred, connectivity=8)

        for _, ev_row in group.iterrows():
            ev_id = int(ev_row["event_id"])
            p_id = int(ev_row["pred_cc_id"])
            d_gap = float(ev_row["D_gap"])

            ca_y, ca_x = int(ev_row["CA_y"]), int(ev_row["CA_x"])
            cb_y, cb_x = int(ev_row["CB_y"]), int(ev_row["CB_x"])

            # 1. Construct bridge corridor mask
            corridor_line = np.zeros((H, W), dtype=np.uint8)
            cv2.line(corridor_line, (ca_x, ca_y), (cb_x, cb_y), 1, thickness=3)
            corridor_mask = (corridor_line == 1) & (target_bin == 0)

            # Control endpoints / true crack masks
            endpoints_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.circle(endpoints_mask, (ca_x, ca_y), 5, 1, -1)
            cv2.circle(endpoints_mask, (cb_x, cb_y), 5, 1, -1)
            crack_endpoints_mask = (endpoints_mask == 1) & (target_bin == 1)

            # Local background mask around corridor (excluding corridor and crack)
            corridor_dilated = cv2.dilate(corridor_line, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
            bg_mask = (corridor_dilated == 1) & (target_bin == 0) & (~corridor_mask)

            # Measure logits on corridor
            if np.sum(corridor_mask) > 0:
                z_ctrl_bridge = float(np.mean(ctrl_logits[corridor_mask]))
                z_cgsr_bridge = float(np.mean(cgsr_logits[corridor_mask]))
                delta_z_bridge = z_ctrl_bridge - z_cgsr_bridge
            else:
                z_ctrl_bridge = float("nan")
                z_cgsr_bridge = float("nan")
                delta_z_bridge = 0.0

            # Measure logits on crack endpoints
            if np.sum(crack_endpoints_mask) > 0:
                z_ctrl_crack = float(np.mean(ctrl_logits[crack_endpoints_mask]))
                z_cgsr_crack = float(np.mean(cgsr_logits[crack_endpoints_mask]))
                delta_z_crack = z_ctrl_crack - z_cgsr_crack
            else:
                z_ctrl_crack = float("nan")
                z_cgsr_crack = float("nan")
                delta_z_crack = 0.0

            # Measure Gate values if available
            g_bridge = float(np.mean(cgsr_gate_map[corridor_mask])) if (cgsr_gate_map is not None and np.sum(corridor_mask) > 0) else float("nan")
            g_crack = float(np.mean(cgsr_gate_map[crack_endpoints_mask])) if (cgsr_gate_map is not None and np.sum(crack_endpoints_mask) > 0) else float("nan")
            g_bg = float(np.mean(cgsr_gate_map[bg_mask])) if (cgsr_gate_map is not None and np.sum(bg_mask) > 0) else float("nan")
            g_contrast = (g_crack - g_bridge) if (not np.isnan(g_crack) and not np.isnan(g_bridge)) else float("nan")

            # Check if bridge is cured in CGSR:
            # A bridge is cured if:
            # (a) CA and CB no longer belong to the same connected component in cgsr_labels, OR
            # (b) corridor mean prediction < 0.5 (logits < 0.0)
            is_cured = False
            if 0 <= ca_y < H and 0 <= ca_x < W and 0 <= cb_y < H and 0 <= cb_x < W:
                cgsr_ca_label = cgsr_labels[ca_y, ca_x]
                cgsr_cb_label = cgsr_labels[cb_y, cb_x]
                if cgsr_ca_label == 0 or cgsr_cb_label == 0 or (cgsr_ca_label != cgsr_cb_label):
                    is_cured = True
                elif z_cgsr_bridge < 0.0:
                    is_cured = True

            strat_group = "Group B (D > 8)" if d_gap > 8.0 else ("Group A (5 < D <= 8)" if d_gap > 5.0 else "Narrow (D <= 5)")

            rec = {
                "event_id": ev_id,
                "case_name": case_name,
                "d_gap": d_gap,
                "group": strat_group,
                "is_wider_gap": (d_gap > 5.0),
                "z_ctrl_bridge": z_ctrl_bridge,
                "z_cgsr_bridge": z_cgsr_bridge,
                "delta_z_bridge": delta_z_bridge,
                "z_ctrl_crack": z_ctrl_crack,
                "z_cgsr_crack": z_cgsr_crack,
                "delta_z_crack": delta_z_crack,
                "is_cured": is_cured,
                "g_bridge": g_bridge,
                "g_crack": g_crack,
                "g_bg": g_bg,
                "g_contrast": g_contrast,
            }
            event_records.append(rec)

            # Matched Clean Control segment
            ctrl_seg = find_matched_control_segment(target_bin, target_length_px=d_gap, seed=42 + ev_id)
            if ctrl_seg is not None:
                ctrl_corr = ctrl_seg["ctrl_corridor_mask"]
                if np.sum(ctrl_corr) > 0:
                    z_ctrl_seg_ctrl = float(np.mean(ctrl_logits[ctrl_corr]))
                    z_ctrl_seg_cgsr = float(np.mean(cgsr_logits[ctrl_corr]))
                    g_seg_crack = float(np.mean(cgsr_gate_map[ctrl_corr])) if cgsr_gate_map is not None else float("nan")

                    # Check for breakage: was it continuous before and broken now?
                    p0, p1 = ctrl_seg["ctrl_p0"], ctrl_seg["ctrl_p1"]
                    broken = False
                    if 0 <= p0[0] < H and 0 <= p0[1] < W and 0 <= p1[0] < H and 0 <= p1[1] < W:
                        lbl0 = cgsr_labels[p0[0], p0[1]]
                        lbl1 = cgsr_labels[p1[0], p1[1]]
                        if lbl0 == 0 or lbl1 == 0 or lbl0 != lbl1:
                            broken = True

                    clean_control_records.append({
                        "event_id": ev_id,
                        "case_name": case_name,
                        "d_gap": d_gap,
                        "z_ctrl_clean_crack": z_ctrl_seg_ctrl,
                        "z_cgsr_clean_crack": z_ctrl_seg_cgsr,
                        "delta_z_clean_crack": (z_ctrl_seg_ctrl - z_ctrl_seg_cgsr),
                        "g_clean_crack": g_seg_crack,
                        "is_broken": broken,
                    })

    df_all_events = pd.DataFrame(event_records)
    df_wider_43 = df_all_events[df_all_events["is_wider_gap"]].copy()
    df_clean_ctrl = pd.DataFrame(clean_control_records)

    df_all_events.to_csv(os.path.join(out_dir, "cgsr_evaluation_118all_events.csv"), index=False)
    df_wider_43.to_csv(os.path.join(out_dir, "cgsr_evaluation_43wider_events.csv"), index=False)
    df_clean_ctrl.to_csv(os.path.join(out_dir, "cgsr_evaluation_clean_controls.csv"), index=False)

    # -------------------------------------------------------------------------
    # PART 3: Mechanistic & Causal Summary
    # -------------------------------------------------------------------------
    def summarize_cohort(sub_df: pd.DataFrame, name: str) -> Dict[str, Any]:
        if len(sub_df) == 0:
            return {}
        n = len(sub_df)
        cured_count = int(sub_df["is_cured"].sum())
        cured_rate = float(cured_count / n * 100.0)

        delta_zb = sub_df["delta_z_bridge"]
        delta_zc = sub_df["delta_z_crack"]

        g_bridge = sub_df["g_bridge"].dropna()
        g_crack = sub_df["g_crack"].dropna()
        g_contrast = sub_df["g_contrast"].dropna()

        # Correlation between g_bridge and delta_z_bridge
        corr_r, p_val = float("nan"), float("nan")
        valid_pairs = sub_df[["g_bridge", "delta_z_bridge"]].dropna()
        if len(valid_pairs) > 5 and valid_pairs["g_bridge"].std() > 1e-6:
            r, p = spearmanr(valid_pairs["g_bridge"], valid_pairs["delta_z_bridge"])
            corr_r, p_val = float(r), float(p)

        return {
            "name": name,
            "N": n,
            "cured_count": cured_count,
            "cured_rate_pct": cured_rate,
            "delta_z_bridge": {
                "mean": float(delta_zb.mean()),
                "median": float(delta_zb.median()),
                "p10": float(np.percentile(delta_zb, 10)),
                "p25": float(np.percentile(delta_zb, 25)),
                "p75": float(np.percentile(delta_zb, 75)),
                "p90": float(np.percentile(delta_zb, 90)),
            },
            "delta_z_crack": {
                "mean": float(delta_zc.mean()),
                "median": float(delta_zc.median()),
            },
            "gate_metrics": {
                "g_bridge_mean": float(g_bridge.mean()) if len(g_bridge) > 0 else float("nan"),
                "g_bridge_median": float(g_bridge.median()) if len(g_bridge) > 0 else float("nan"),
                "g_crack_mean": float(g_crack.mean()) if len(g_crack) > 0 else float("nan"),
                "g_crack_median": float(g_crack.median()) if len(g_crack) > 0 else float("nan"),
                "g_contrast_mean": float(g_contrast.mean()) if len(g_contrast) > 0 else float("nan"),
                "g_contrast_median": float(g_contrast.median()) if len(g_contrast) > 0 else float("nan"),
                "spearman_g_vs_delta": corr_r,
                "spearman_p_val": p_val,
            }
        }

    cohort_wider43 = summarize_cohort(df_wider_43, "43 Wider-Gap Events (D_gap > 5 px)")
    cohort_group_a = summarize_cohort(df_wider_43[df_wider_43["group"] == "Group A (5 < D <= 8)"], "Group A (5 < D <= 8 px)")
    cohort_group_b = summarize_cohort(df_wider_43[df_wider_43["group"] == "Group B (D > 8)"], "Group B (D > 8 px)")
    cohort_all118 = summarize_cohort(df_all_events, "All 118 Bridge Events")

    clean_crack_breakages = int(df_clean_ctrl["is_broken"].sum()) if len(df_clean_ctrl) > 0 else 0
    clean_crack_mean_delta = float(df_clean_ctrl["delta_z_clean_crack"].mean()) if len(df_clean_ctrl) > 0 else 0.0

    # -------------------------------------------------------------------------
    # PART 4: Scientific Decision Rule
    # -------------------------------------------------------------------------
    # Criteria:
    # 1. Overall Val Dice degradation: delta_dice >= -0.005
    # 2. Wider-gap bridge cure rate >= 25% OR median delta_z_bridge > 0.5
    # 3. Positive gate contrast: median G_crack > G_bridge (contrast >= 0.05)
    # 4. Collateral damage under control: clean crack breakages <= 2, delta_z_clean_crack < 0.5

    dice_ok = (delta_dice >= -0.005)
    cure_rate = cohort_wider43["cured_rate_pct"]
    med_delta_zb = cohort_wider43["delta_z_bridge"]["median"]
    med_contrast = cohort_wider43["gate_metrics"]["g_contrast_median"]

    bridge_suppressed = (cure_rate >= 25.0 or med_delta_zb > 0.5)
    gate_discriminative = (med_contrast >= 0.05)
    collateral_ok = (clean_crack_breakages <= 2 and clean_crack_mean_delta < 0.5)

    if dice_ok and bridge_suppressed and gate_discriminative and collateral_ok:
        verdict = "H2 SUPPORTED"
        verdict_explanation = (
            "CGSR demonstrates significant selective suppression of false bridges in the wider-gap cohort "
            f"(cure rate: {cure_rate:.1f}%, median Delta z_bridge: {med_delta_zb:+.4f}) with positive gate contrast "
            f"(G_crack - G_bridge: {med_contrast:+.4f}) while preserving overall Val Dice ({delta_dice:+.4f}) and true crack continuity."
        )
    elif bridge_suppressed and not gate_discriminative:
        verdict = "H2 WEAKENED"
        verdict_explanation = (
            "CGSR reduces bridge logits but the gate does not exhibit significant selective discrimination "
            f"between bridges and true cracks (contrast: {med_contrast:+.4f}). Suppression is largely unselective."
        )
    else:
        verdict = "H2 NOT SUPPORTED"
        verdict_explanation = (
            f"CGSR fails to cure wider-gap false bridges (cure rate: {cure_rate:.1f}%, median Delta z_bridge: {med_delta_zb:+.4f}), "
            f"gate is either non-selective (contrast: {med_contrast:+.4f}) or inactive, or causes excessive degradation to true cracks."
        )

    summary_final = {
        "verdict": verdict,
        "verdict_explanation": verdict_explanation,
        "decision_flags": {
            "val_dice_preserved": bool(dice_ok),
            "bridge_suppressed": bool(bridge_suppressed),
            "gate_discriminative": bool(gate_discriminative),
            "collateral_damage_under_control": bool(collateral_ok),
        },
        "val_n348_setting_a": {
            "control": summary_ctrl_val,
            "cgsr": summary_cgsr_val,
            "delta_dice": float(delta_dice),
            "delta_iou": float(delta_iou),
            "delta_precision": float(delta_prec),
            "delta_recall": float(delta_rec),
        },
        "cohort_43_wider_gap": cohort_wider43,
        "cohort_group_a_5_to_8": cohort_group_a,
        "cohort_group_b_gt_8": cohort_group_b,
        "cohort_all_118": cohort_all118,
        "clean_controls": {
            "N": len(df_clean_ctrl),
            "breakages": clean_crack_breakages,
            "mean_delta_z": clean_crack_mean_delta,
        }
    }

    summary_json_path = os.path.join(out_dir, "cgsr_evaluation_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_final, f, indent=2)

    # Generate Markdown Report
    report_md_path = os.path.join(out_dir, "cgsr_evaluation_report.md")
    with open(report_md_path, "w") as f:
        f.write("# Phase 6D: Context-Guided Stage-1 Skip Refinement (CGSR) Evaluation Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Final Scientific Verdict: `{verdict}`\n\n")
        f.write(f"> {verdict_explanation}\n\n")
        f.write("---\n\n")
        f.write("## 1. Official Setting A Benchmark (Full Val N=348)\n\n")
        f.write("| Model | Val Dice | Val IoU | Precision | Recall |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        f.write(f"| **Control (Stage 2 Locked Base)** | {summary_ctrl_val['mean_dice']:.4f} | {summary_ctrl_val['mean_iou']:.4f} | {summary_ctrl_val['mean_precision']:.4f} | {summary_ctrl_val['mean_recall']:.4f} |\n")
        f.write(f"| **CGSR (Phase 6D Intervention)** | {summary_cgsr_val['mean_dice']:.4f} | {summary_cgsr_val['mean_iou']:.4f} | {summary_cgsr_val['mean_precision']:.4f} | {summary_cgsr_val['mean_recall']:.4f} |\n")
        f.write(f"| **Delta (CGSR - Control)** | **{delta_dice:+.4f}** | **{delta_iou:+.4f}** | **{delta_prec:+.4f}** | **{delta_rec:+.4f}** |\n\n")

        f.write("## 2. False Bridge Cohort Performance\n\n")
        f.write("| Cohort | N | Cured Events | Cure Rate (%) | Median Delta z_bridge | Mean Delta z_bridge |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        f.write(f"| **43 Wider-Gap (D_gap > 5 px)** | {cohort_wider43['N']} | {cohort_wider43['cured_count']} | **{cohort_wider43['cured_rate_pct']:.1f}%** | {cohort_wider43['delta_z_bridge']['median']:+.4f} | {cohort_wider43['delta_z_bridge']['mean']:+.4f} |\n")
        if cohort_group_a:
            f.write(f"| Group A (5 < D <= 8 px) | {cohort_group_a['N']} | {cohort_group_a['cured_count']} | {cohort_group_a['cured_rate_pct']:.1f}% | {cohort_group_a['delta_z_bridge']['median']:+.4f} | {cohort_group_a['delta_z_bridge']['mean']:+.4f} |\n")
        if cohort_group_b:
            f.write(f"| Group B (D > 8 px) | {cohort_group_b['N']} | {cohort_group_b['cured_count']} | {cohort_group_b['cured_rate_pct']:.1f}% | {cohort_group_b['delta_z_bridge']['median']:+.4f} | {cohort_group_b['delta_z_bridge']['mean']:+.4f} |\n")
        f.write(f"| All 118 Bridge Events | {cohort_all118['N']} | {cohort_all118['cured_count']} | {cohort_all118['cured_rate_pct']:.1f}% | {cohort_all118['delta_z_bridge']['median']:+.4f} | {cohort_all118['delta_z_bridge']['mean']:+.4f} |\n\n")

        f.write("## 3. CGSR Gate Mechanistic Diagnostics\n\n")
        f.write("| Metric | Median | Mean |\n")
        f.write("|---|---:|---:|\n")
        f.write(f"| Bridge Gate ($G_{{bridge}}$) | {cohort_wider43['gate_metrics']['g_bridge_median']:.4f} | {cohort_wider43['gate_metrics']['g_bridge_mean']:.4f} |\n")
        f.write(f"| True Crack Gate ($G_{{crack}}$) | {cohort_wider43['gate_metrics']['g_crack_median']:.4f} | {cohort_wider43['gate_metrics']['g_crack_mean']:.4f} |\n")
        f.write(f"| Gate Contrast ($G_{{crack}} - G_{{bridge}}$) | **{cohort_wider43['gate_metrics']['g_contrast_median']:+.4f}** | **{cohort_wider43['gate_metrics']['g_contrast_mean']:+.4f}** |\n\n")

        f.write("## 4. Negative Controls & Collateral Damage\n\n")
        f.write(f"- Clean Crack Segments Tested: {len(df_clean_ctrl)}\n")
        f.write(f"- Clean Crack Breakage Events: {clean_crack_breakages} ({float(clean_crack_breakages/max(len(df_clean_ctrl), 1)*100):.1f}%)\n")
        f.write(f"- Mean Delta z on Clean Cracks: {clean_crack_mean_delta:+.4f}\n\n")

    print("\n" + "=" * 80)
    print(f"VERDICT: {verdict}")
    print(f"Explanation: {verdict_explanation}")
    print(f"Summary JSON saved to: {summary_json_path}")
    print(f"Full Report saved to:   {report_md_path}")
    print("=" * 80)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 6D CGSR Comprehensive Evaluation")
    parser.add_argument("--control-config", type=str, required=True, help="Path to Control YAML config")
    parser.add_argument("--control-ckpt", type=str, required=True, help="Path to Control checkpoint (.pth)")
    parser.add_argument("--cgsr-config", type=str, required=True, help="Path to CGSR YAML config")
    parser.add_argument("--cgsr-ckpt", type=str, required=True, help="Path to CGSR checkpoint (.pth)")
    parser.add_argument("--output-dir", type=str, default="results/diagnostics/phase6d_cgsr_evaluation", help="Output directory")
    parser.add_argument("--data-root", type=str, default="datasets/Crack500_ready", help="Path to Crack500 dataset")
    parser.add_argument("--events-csv", type=str, default="results/diagnostics/phase6_bottleneck_path/bottleneck_path_118events.csv", help="Path to 118 events CSV")
    parser.add_argument("--tile-size", type=int, default=448, help="Tile size for Setting A inference")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_phase6d_evaluation(
        control_config=args.control_config,
        control_ckpt=args.control_ckpt,
        cgsr_config=args.cgsr_config,
        cgsr_ckpt=args.cgsr_ckpt,
        out_dir=args.output_dir,
        data_root=args.data_root,
        events_csv=args.events_csv,
        tile_size=args.tile_size,
    )
