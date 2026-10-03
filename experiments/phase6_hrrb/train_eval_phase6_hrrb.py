#!/usr/bin/env python3
"""
experiments/phase6_hrrb/train_eval_phase6_hrrb.py

Phase 6: High-Resolution Residual Bypass (HRRB) Training & Evaluation Suite
============================================================================

Scientific Objective:
    Test whether Candidate B's persistent topology failure (false bridges) stems from
    the lack of an independent high-resolution feature pathway bypassing the aggressive
    stride-4 stem downsampling.

Intervention Architecture:
    Main Model: Candidate B 100% strictly FROZEN.
    Bypass:
        RGB [B, 3, 448, 448]
            ↓
        Conv 3x3 s1 p1 (3 -> 8)
            ↓
        GELU
            ↓
        Conv 3x3 s2 p1 (8 -> 16)
            ↓ [B, 16, 224, 224]
        Conv 1x1 s1 p0 (16 -> 1)
            ↓ [B, 1, 224, 224]
        Bilinear x2 (align_corners=False)
            ↓
        detail_residual [B, 1, 448, 448]

    final_logits = main_logits_448 + detail_residual

Identity Guarantee:
    Conv 1x1 zero-initialized (weights=0, bias=0).
    At t=0: detail_residual == 0, final_logits == main_logits bit-exactly.
    Trainable parameters: EXACTLY 1,409.

Protocol:
    Physical batch = 14, seed = 42, FP32 ONLY, no GA, 8 epochs.
    AdamW(lr=1e-4, wd=1e-2), CosineAnnealingLR.
    Loss = BCE + Dice (no topology/clDice loss).
    Validation: N=348 Setting A (tau=0.5).
    Test: Strictly SEALED.
"""

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('SAGE_LITE'))

from sage_lite.tools.run_phase6_c_topology_diagnostic import (
    load_model_from_checkpoint,
    compute_topology_metrics,
)
from scripts.diagnostics.phase6_stem_factorization_provenance import (
    isolate_bridged_pairs_and_rois,
    isolate_clean_pairs_and_rois,
)
from scripts.diagnostics.train_eval_phase6_stem_genesis import (
    Crack500TrainDataset,
    compute_seg_loss,
)
from experiments.phase6_hrrb.hrrb_module import (
    HighResolutionResidualBypass,
    CandidateBWithHRRB,
)
from experiments.phase6_hrrb.test_hrrb_invariants import (
    CANDIDATE_B_CHECKPOINT_SHA256,
    DEFAULT_CONFIG,
    DEFAULT_CHECKPOINT,
    compute_file_sha256,
)


# ---------------------------------------------------------------------------
# Setting A Inference with Full Component Logging
# ---------------------------------------------------------------------------

def predict_setting_a_components(
    model: CandidateBWithHRRB,
    img_bgr: np.ndarray,
    device: torch.device,
    tile_size: int = 448,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs canonical Setting A inference, returning full-resolution maps:
        prob_final: (H, W) in [0, 1]
        logits_main: (H, W)
        logits_res: (H, W)
        logits_final: (H, W)
    """
    H, W, _ = img_bgr.shape
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    pad_h = (tile_size - (H % tile_size)) % tile_size
    pad_w = (tile_size - (W % tile_size)) % tile_size

    img_padded = cv2.copyMakeBorder(img_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    H_pad, W_pad, _ = img_padded.shape

    img_norm = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_norm = (img_norm - mean) / std

    map_main = np.zeros((H_pad, W_pad), dtype=np.float32)
    map_res  = np.zeros((H_pad, W_pad), dtype=np.float32)
    map_final= np.zeros((H_pad, W_pad), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for y in range(0, H_pad, tile_size):
            for x in range(0, W_pad, tile_size):
                tile = img_norm[y:y+tile_size, x:x+tile_size]
                tile_t = torch.from_numpy(tile.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

                final_log, main_log, detail_res = model(tile_t, return_components=True)

                map_main[y:y+tile_size, x:x+tile_size]  = main_log.squeeze().cpu().numpy()
                map_res[y:y+tile_size, x:x+tile_size]   = detail_res.squeeze().cpu().numpy()
                map_final[y:y+tile_size, x:x+tile_size] = final_log.squeeze().cpu().numpy()

    # Crop to original image dimensions
    crop_main = map_main[:H, :W]
    crop_res  = map_res[:H, :W]
    crop_final= map_final[:H, :W]

    prob_final = 1.0 / (1.0 + np.exp(-crop_final))
    return prob_final, crop_main, crop_res, crop_final


# ---------------------------------------------------------------------------
# Evaluation Routine
# ---------------------------------------------------------------------------

def evaluate_hrrb_full(
    model: CandidateBWithHRRB,
    val_img_paths: List[str],
    val_mask_dir: str,
    device: torch.device,
    desc: str = "Evaluating HRRB",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluates model across all validation images.
    Returns:
        df_topo: per-sample topology metrics
        df_residual: per-sample residual magnitude and ratio audit
        df_standalone: per-sample standalone residual diagnostics (AUC, pseudo-Dice)
    """
    topo_rows = []
    residual_rows = []
    standalone_rows = []

    for img_p in tqdm(val_img_paths, desc=desc, ncols=80):
        stem = os.path.splitext(os.path.basename(img_p))[0]
        mask_p = os.path.join(val_mask_dir, f"{stem}.png")

        img = cv2.imread(img_p)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        gt_binary = (mask > 127).astype(np.uint8)

        prob_final, logits_main, logits_res, logits_final = predict_setting_a_components(
            model, img, device
        )
        pred_binary = (prob_final >= 0.5).astype(np.uint8)

        # 1. Topology metrics
        topo = compute_topology_metrics(pred_binary, gt_binary)
        topo["image_id"] = stem
        topo_rows.append(topo)

        # 2. Residual magnitude audit across regions
        abs_res = np.abs(logits_res)
        abs_main = np.abs(logits_main)
        ratio_map = abs_res / (abs_main + 1e-4)

        crack_mask = (gt_binary == 1)
        bg_mask = (gt_binary == 0)
        boundary_mask = (abs_main < 0.5)  # Near decision boundary

        res_row = {
            "image_id": stem,
            "res_mean_global": float(np.mean(abs_res)),
            "res_max_global": float(np.max(abs_res)),
            "res_mean_crack": float(np.mean(abs_res[crack_mask])) if np.any(crack_mask) else 0.0,
            "res_mean_bg": float(np.mean(abs_res[bg_mask])) if np.any(bg_mask) else 0.0,
            "res_mean_boundary": float(np.mean(abs_res[boundary_mask])) if np.any(boundary_mask) else 0.0,
            "main_mean_global": float(np.mean(abs_main)),
            "main_mean_crack": float(np.mean(abs_main[crack_mask])) if np.any(crack_mask) else 0.0,
            "main_mean_bg": float(np.mean(abs_main[bg_mask])) if np.any(bg_mask) else 0.0,
            "ratio_mean_global": float(np.mean(ratio_map)),
            "ratio_mean_crack": float(np.mean(ratio_map[crack_mask])) if np.any(crack_mask) else 0.0,
            "ratio_mean_bg": float(np.mean(ratio_map[bg_mask])) if np.any(bg_mask) else 0.0,
            "ratio_mean_boundary": float(np.mean(ratio_map[boundary_mask])) if np.any(boundary_mask) else 0.0,
        }
        residual_rows.append(res_row)

        # 3. Standalone residual diagnostics
        res_binary = (logits_res > 0).astype(np.uint8)
        tp_res = int(np.sum((res_binary == 1) & (gt_binary == 1)))
        fp_res = int(np.sum((res_binary == 1) & (gt_binary == 0)))
        fn_res = int(np.sum((res_binary == 0) & (gt_binary == 1)))
        dice_standalone = float(2.0 * tp_res / (2.0 * tp_res + fp_res + fn_res + 1e-8))

        # Standalone AUC
        try:
            # Flatten arrays for AUC
            auc_val = float(roc_auc_score(gt_binary.flatten(), logits_res.flatten()))
        except Exception:
            auc_val = float("nan")

        std_row = {
            "image_id": stem,
            "residual_auc": auc_val,
            "residual_standalone_dice": dice_standalone,
            "res_positive_area": int(np.sum(res_binary)),
            "gt_area": int(np.sum(gt_binary)),
        }
        standalone_rows.append(std_row)

    df_topo = pd.DataFrame(topo_rows)
    df_residual = pd.DataFrame(residual_rows)
    df_standalone = pd.DataFrame(standalone_rows)
    return df_topo, df_residual, df_standalone


# ---------------------------------------------------------------------------
# Subgroup Aggregator
# ---------------------------------------------------------------------------

def compute_subgroup_summary(
    df_hrrb: pd.DataFrame,
    df_c0: pd.DataFrame,
    cohort_dict: Dict[str, set],
) -> pd.DataFrame:
    rows = []
    for cohort_name, stems in cohort_dict.items():
        if cohort_name == "Full_Validation_348":
            sub_hrrb = df_hrrb
            sub_c0 = df_c0
        else:
            sub_hrrb = df_hrrb[df_hrrb["image_id"].isin(stems)]
            sub_c0 = df_c0[df_c0["image_id"].isin(stems)]

        n_samples = len(sub_hrrb)
        if n_samples == 0:
            continue

        c0_b_img = int((sub_c0["bridge_events"] > 0).sum())
        c0_b_evt = int(sub_c0["bridge_events"].sum())
        c0_brk   = int(sub_c0["fragmented_gt_components"].sum())
        c0_dice  = float(sub_c0["dice"].mean())
        c0_cldice= float(sub_c0["cldice"].mean())
        c0_recall= float(sub_c0["recall"].mean())
        c0_prec  = float(sub_c0["precision"].mean())

        hrrb_b_img = int((sub_hrrb["bridge_events"] > 0).sum())
        hrrb_b_evt = int(sub_hrrb["bridge_events"].sum())
        hrrb_brk   = int(sub_hrrb["fragmented_gt_components"].sum())
        hrrb_dice  = float(sub_hrrb["dice"].mean())
        hrrb_cldice= float(sub_hrrb["cldice"].mean())
        hrrb_recall= float(sub_hrrb["recall"].mean())
        hrrb_prec  = float(sub_hrrb["precision"].mean())

        # Cured & Created images
        m_c0 = sub_c0.set_index("image_id")["bridge_events"] > 0
        m_hrrb = sub_hrrb.set_index("image_id")["bridge_events"] > 0
        cured = int((m_c0 & ~m_hrrb).sum())
        created = int((~m_c0 & m_hrrb).sum())

        row = {
            "cohort": cohort_name,
            "N": n_samples,
            "C0_Bridge_Events": c0_b_evt,
            "C0_Bridge_Images": c0_b_img,
            "C0_Break_Events": c0_brk,
            "C0_Dice": c0_dice,
            "C0_clDice": c0_cldice,
            "C0_Recall": c0_recall,
            "C0_Precision": c0_prec,
            "HRRB_Bridge_Events": hrrb_b_evt,
            "HRRB_Bridge_Images": hrrb_b_img,
            "HRRB_Break_Events": hrrb_brk,
            "HRRB_Dice": hrrb_dice,
            "HRRB_clDice": hrrb_cldice,
            "HRRB_Recall": hrrb_recall,
            "HRRB_Precision": hrrb_prec,
            "Delta_Bridge_Events": hrrb_b_evt - c0_b_evt,
            "Delta_Bridge_Images": hrrb_b_img - c0_b_img,
            "Delta_Break_Events": hrrb_brk - c0_brk,
            "Delta_Dice": hrrb_dice - c0_dice,
            "Delta_clDice": hrrb_cldice - c0_cldice,
            "Delta_Recall": hrrb_recall - c0_recall,
            "Delta_Precision": hrrb_prec - c0_prec,
            "Cured_Images": cured,
            "Created_Images": created,
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main Training & Evaluation Loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 6: HRRB Training & Evaluation")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=str, default="datasets/Crack500_ready")
    parser.add_argument("--out-dir", type=str, default="results/diagnostics/phase6_hrrb")
    parser.add_argument("--mode", type=str, choices=["all", "train", "eval"], default="all")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=14)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 80)
    print("PHASE 6: HIGH-RESOLUTION RESIDUAL BYPASS (HRRB)")
    print(f"Mode: {args.mode}, Epochs: {args.epochs}, Batch: {args.batch_size}, Device: {device}")
    print("=" * 80)

    # 1. Preflight Invariants Check
    assert os.path.exists(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"
    actual_hash = compute_file_sha256(args.checkpoint)
    print(f"Candidate B Checkpoint SHA256: {actual_hash}")
    assert actual_hash == CANDIDATE_B_CHECKPOINT_SHA256, "VIOLATION: Candidate B Checkpoint hash mismatch!"

    # 2. Instantiate Model
    print("Loading Candidate B base model...")
    candidate_b = load_model_from_checkpoint(args.config, args.checkpoint, device=device)
    model = CandidateBWithHRRB(candidate_b).to(device)

    # Invariant: 1,409 trainable parameters
    trainable_numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_numel == 1409, f"Expected 1,409 trainable parameters, got {trainable_numel}"
    print(f">> Preflight Invariant PASS: Exactly {trainable_numel} parameters trainable.")

    # Invariant: t=0 bitwise identity
    dummy_x = torch.randn(2, 3, 448, 448, device=device)
    with torch.no_grad():
        c0_out = candidate_b(dummy_x)
        hrrb_out, _, detail_res0 = model(dummy_x, return_components=True)
        assert torch.max(torch.abs(detail_res0)).item() == 0.0, "detail_residual at t=0 must be 0!"
        diff0 = torch.max(torch.abs(hrrb_out - c0_out)).item()
        assert diff0 == 0.0, f"Identity failure: diff={diff0}"
    print(">> Preflight Invariant PASS: t=0 Bitwise Identity with Candidate B verified.")

    # 3. Datasets Setup
    train_img_dir = os.path.join(args.data_root, "train", "images")
    train_mask_dir = os.path.join(args.data_root, "train", "masks")
    val_img_dir   = os.path.join(args.data_root, "val", "images")
    val_mask_dir  = os.path.join(args.data_root, "val", "masks")

    val_img_paths = sorted(glob.glob(os.path.join(val_img_dir, "*.jpg")))
    assert len(val_img_paths) == 348, f"Expected 348 validation images, got {len(val_img_paths)}"

    weights_path = os.path.join(args.out_dir, "phase6_hrrb_weights.pth")
    training_log_path = os.path.join(args.out_dir, "hrrb_training_log.csv")

    # -----------------------------------------------------------------------
    # Step 1: Training HRRB
    # -----------------------------------------------------------------------
    if args.mode in ["all", "train"]:
        if os.path.exists(weights_path):
            print(f"\nFound existing HRRB weights at {weights_path}, loading...")
            model.hrrb.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            print("\n" + "=" * 80)
            print("TRAINING HRRB: High-Resolution Residual Bypass (3->8 @448 -> 8->16 @224 -> 16->1 @224 -> bil.x2)")
            print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, Precision: FP32, LR: {args.lr}, WD: {args.wd}")
            print("=" * 80)

            train_dataset = Crack500TrainDataset(train_img_dir, train_mask_dir, img_size=448, seed=42)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True if device.type == "cuda" else False,
            )

            optimizer = torch.optim.AdamW(model.hrrb.parameters(), lr=args.lr, weight_decay=args.wd)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

            log_rows = []
            for epoch in range(1, args.epochs + 1):
                model.train()
                total_loss_accum = 0.0
                bce_accum = 0.0
                dice_accum = 0.0
                n_batches = 0

                pbar = tqdm(train_loader, desc=f"HRRB Ep {epoch}/{args.epochs}", ncols=90)
                for batch in pbar:
                    images = batch["image"].to(device)
                    masks  = batch["mask"].to(device)

                    optimizer.zero_grad()
                    final_logits, _, detail_res = model(images, return_components=True)
                    loss, metrics = compute_seg_loss(final_logits, masks)

                    loss.backward()
                    optimizer.step()

                    total_loss_accum += metrics["loss_total"]
                    bce_accum += metrics["loss_bce"]
                    dice_accum += metrics["loss_dice"]
                    n_batches += 1

                    pbar.set_postfix({
                        "tot": f"{metrics['loss_total']:.3f}",
                        "bce": f"{metrics['loss_bce']:.3f}",
                        "dice": f"{metrics['loss_dice']:.3f}",
                        "res_max": f"{torch.max(torch.abs(detail_res)).item():.3f}",
                    })

                scheduler.step()

                # Audit weight norms
                with torch.no_grad():
                    c1_w_norm = float(torch.norm(model.hrrb.conv1.weight).item())
                    c2_w_norm = float(torch.norm(model.hrrb.conv2.weight).item())
                    proj_w_norm = float(torch.norm(model.hrrb.proj.weight).item())
                    proj_b_norm = float(torch.norm(model.hrrb.proj.bias).item())

                log_entry = {
                    "epoch": epoch,
                    "loss_total": total_loss_accum / n_batches,
                    "loss_bce": bce_accum / n_batches,
                    "loss_dice": dice_accum / n_batches,
                    "conv1_weight_norm": c1_w_norm,
                    "conv2_weight_norm": c2_w_norm,
                    "proj_weight_norm": proj_w_norm,
                    "proj_bias_norm": proj_b_norm,
                    "lr": scheduler.get_last_lr()[0],
                }
                log_rows.append(log_entry)
                print(f"  Epoch {epoch:02d} Summary: Loss={log_entry['loss_total']:.4f} | proj_w_norm={proj_w_norm:.4f}, proj_b_norm={proj_b_norm:.4f}")

            # Save weights & log
            torch.save(model.hrrb.state_dict(), weights_path)
            pd.DataFrame(log_rows).to_csv(training_log_path, index=False)
            print(f">> Saved HRRB weights to {weights_path}")
            print(f">> Saved training log to {training_log_path}")

    # -----------------------------------------------------------------------
    # Step 2: Evaluation on Validation Cohorts
    # -----------------------------------------------------------------------
    if args.mode in ["all", "eval"]:
        print("\n" + "=" * 80)
        print("EVALUATION: HRRB on Validation Set (N=348, Setting A, tau=0.5)")
        print("=" * 80)

        # Load weights if not already in model
        if os.path.exists(weights_path):
            model.hrrb.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()

        # Run evaluation
        df_topo, df_residual, df_standalone = evaluate_hrrb_full(
            model, val_img_paths, val_mask_dir, device, desc="Evaluating HRRB"
        )

        val_topo_path = os.path.join(args.out_dir, "validation_hrrb_per_sample.csv")
        res_audit_path = os.path.join(args.out_dir, "residual_magnitude_audit.csv")
        standalone_path = os.path.join(args.out_dir, "standalone_residual_diagnostics.csv")

        df_topo.to_csv(val_topo_path, index=False)
        df_residual.to_csv(res_audit_path, index=False)
        df_standalone.to_csv(standalone_path, index=False)
        print(f">> Saved per-sample topology to {val_topo_path}")
        print(f">> Saved residual magnitude audit to {res_audit_path}")
        print(f">> Saved standalone residual diagnostics to {standalone_path}")

        # Subgroup Cohorts
        master_path = "results/diagnostics/phase6_d2_topology/topology_7models_master_paired.csv"
        df_master = pd.read_csv(master_path)
        models_list = ["Base", "A1", "A2", "B1", "C1", "D1", "D2"]
        bcols = [f"{m}_bridge_events" for m in models_list]
        c7_stems = set(df_master[(df_master[bcols] > 0).all(axis=1)]["case_name"].tolist())
        clean_stems = set(df_master[(df_master[bcols] == 0).all(axis=1) & (df_master["gt_cc"] >= 2)]["case_name"].tolist())

        ac_atten_csv = "results/diagnostics/phase6_stem_ac_attenuation/ac_attenuation_per_sample_alpha.csv"
        df_prev = pd.read_csv(ac_atten_csv)
        c7a0 = df_prev[(df_prev["cohort"] == "Consensus_7of7") & (df_prev["alpha"] == 0.0)]
        resistant_stems = set(c7a0[c7a0["has_bridge"] == True]["image_id"].tolist())
        sensitive_stems = set(c7a0[c7a0["has_bridge"] == False]["image_id"].tolist())

        cohort_dict = {
            "Full_Validation_348": set(df_topo["image_id"].tolist()),
            "Consensus_7of7_101": c7_stems,
            "Consensus_Resistant_82": resistant_stems,
            "Consensus_Sensitive_19": sensitive_stems,
            "Clean_Control_56": clean_stems,
        }

        # Load C0 baseline for direct comparison
        c0_csv = "results/diagnostics/phase6_stem_genesis/validation_c0_per_sample.csv"
        assert os.path.exists(c0_csv), f"Candidate B evaluation missing at {c0_csv}"
        df_c0 = pd.read_csv(c0_csv)

        df_subgroup = compute_subgroup_summary(df_topo, df_c0, cohort_dict)
        subgroup_path = os.path.join(args.out_dir, "validation_subgroup_comparison_hrrb_vs_c0.csv")
        df_subgroup.to_csv(subgroup_path, index=False)
        print(f"\n>> Subgroup Comparison Summary saved to {subgroup_path}:")
        print(df_subgroup[["cohort", "N", "C0_Bridge_Events", "HRRB_Bridge_Events", "Delta_Bridge_Events", "C0_Break_Events", "HRRB_Break_Events", "Delta_Break_Events", "C0_Dice", "HRRB_Dice"]].to_string())

        # Residual magnitude summary
        print("\n" + "=" * 80)
        print("RESIDUAL MAGNITUDE AUDIT SUMMARY")
        print("=" * 80)
        print(f"Global Mean |detail_residual|:  {df_residual['res_mean_global'].mean():.4f}")
        print(f"Global Max |detail_residual|:   {df_residual['res_max_global'].mean():.4f}")
        print(f"Crack Mean |detail_residual|:   {df_residual['res_mean_crack'].mean():.4f}")
        print(f"Background Mean |detail_residual|: {df_residual['res_mean_bg'].mean():.4f}")
        print(f"Boundary Mean |detail_residual|:   {df_residual['res_mean_boundary'].mean():.4f}")
        print(f"Ratio (|res| / (|main| + 1e-4)) Crack:    {df_residual['ratio_mean_crack'].mean():.4f}")
        print(f"Ratio (|res| / (|main| + 1e-4)) BG:       {df_residual['ratio_mean_bg'].mean():.4f}")
        print(f"Ratio (|res| / (|main| + 1e-4)) Boundary: {df_residual['ratio_mean_boundary'].mean():.4f}")

        # Standalone residual diagnostics
        print("\n" + "=" * 80)
        print("STANDALONE RESIDUAL DIAGNOSTICS")
        print("=" * 80)
        valid_aucs = df_standalone["residual_auc"].dropna()
        print(f"Mean Standalone AUC:  {valid_aucs.mean():.4f}")
        print(f"Mean Standalone Dice: {df_standalone['residual_standalone_dice'].mean():.4f}")


if __name__ == "__main__":
    main()
