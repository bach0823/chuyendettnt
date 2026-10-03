#!/usr/bin/env python3
"""
scripts/diagnostics/run_candidate_b_routing_ablation.py

Controlled Routing Ablation Study for Candidate B Baseline (Top-k=2, Router Hidden Dim=64):
  Evaluates whether adaptive routing provides downstream segmentation utility,
  thin crack preservation, or bridge suppression compared to:
  1. Adaptive Baseline (Official Candidate B, learned dynamic router)
  2. Static Top-2 (Training usage accumulation)
  3. Static Top-2 (Validation empirical preference)
  4. Random Top-2 (Uniform random sampling across multiple seeds)
  5. SAGE-Off / Zero-Residual (Bypassing expert path entirely: residual_scale = 0.0)

Evaluates:
  - Full Crack500 validation set N=348 in Setting A (non-overlapping 448x448 tiling)
  - Paired statistical metrics: paired t-test, Wilcoxon signed-rank, 95% CI, win/loss rate
  - Thinness quartiles (Q1 to Q4)
  - False bridge cohort impact:
      * 43 wider-gap bridge events (D_gap > 5.0 px)
      * 118 total bridge events
      * Delta z_bridge and cure rates across routing regimes.
"""

import argparse
import collections
import datetime
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy import stats
from skimage.morphology import skeletonize
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import yaml

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sage_lite_dir = os.path.join(project_root, "SAGE_LITE")
for p in [project_root, sage_lite_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sage.components.router import SageRouter
from sage.networks import create_b2_unet
from scripts.evaluate_crack_official import predict_full_image_tiling_setting_a

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("candidate_b_routing_ablation")


# ==============================================================================
# 1. Modulated Logits Computation & Intervention Context
# ==============================================================================

def compute_modulated_logits(router: SageRouter, input_tensor: torch.Tensor) -> torch.Tensor:
    """Computes modulated logits for a SageRouter instance in eval mode."""
    with torch.no_grad():
        agg = router._aggregate_features_with_adaptation(input_tensor)
        g_s = torch.sigmoid(router.shared_expert_gate(agg))
        query = router.query_projection(agg)
        base_logits = torch.matmul(query, router.expert_keys.T) / router.temperature

        if router.logit_modulation:
            eps = 1e-5
            orig_dtype = base_logits.dtype
            g_s_clamped = torch.clamp(g_s.float(), min=eps, max=1.0 - eps)
            log_g_s = torch.log(g_s_clamped)
            log_one_minus_g_s = torch.log(1.0 - g_s_clamped)
            mask_f32 = router.shared_mask.float()
            modulated = (
                base_logits.float() +
                mask_f32 * log_g_s +
                (1.0 - mask_f32) * log_one_minus_g_s
            ).to(orig_dtype)
        else:
            modulated = base_logits
        return modulated


class RoutingInterventionContext:
    """Context manager for non-invasive routing intervention on SageRouter modules."""
    def __init__(
        self,
        router_modules: Dict[str, SageRouter],
        mode: str = "adaptive",
        static_policy: Optional[Dict[str, List[int]]] = None,
        random_seed: int = 42,
    ):
        self.router_modules = router_modules
        self.mode = mode.lower()
        self.static_policy = static_policy or {}
        self.random_seed = random_seed
        self.generator = torch.Generator(device="cpu").manual_seed(random_seed)
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, alias: str, router: SageRouter):
        def hook(module: SageRouter, inputs: Tuple[torch.Tensor, ...], output: Tuple[Any, ...]):
            if self.mode == "adaptive":
                return output

            input_tensor = inputs[0]
            B = input_tensor.shape[0]
            device = input_tensor.device
            M = module.expert_pool_size
            K = module.top_k

            mod_logits = compute_modulated_logits(module, input_tensor)

            if self.mode == "static":
                fixed_k = self.static_policy.get(alias)
                if not fixed_k or len(fixed_k) != K:
                    raise ValueError(f"Static policy missing or invalid for router '{alias}': {fixed_k}")
                sel_indices = torch.tensor(fixed_k, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
            elif self.mode == "random":
                batch_perms = []
                for _ in range(B):
                    perm = torch.randperm(M, generator=self.generator)[:K]
                    batch_perms.append(perm)
                sel_indices = torch.stack(batch_perms, dim=0).to(device)
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

            sel_logits = mod_logits.gather(dim=-1, index=sel_indices)
            if module.gating_type == "softmax":
                gating_weights = F.softmax(sel_logits, dim=-1)
            else:
                gating_weights = torch.sigmoid(sel_logits)

            orig_indices, orig_weights, orig_info = output
            new_info = dict(orig_info)
            new_info["intervention_mode"] = self.mode
            return sel_indices, gating_weights, new_info

        return hook

    def __enter__(self):
        self.hook_handles = []
        for alias, router in self.router_modules.items():
            h = router.register_forward_hook(self._make_hook(alias, router))
            self.hook_handles.append(h)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()


class SageOffContext:
    """Context manager to completely zero out the SAGE expert residual paths."""
    def __init__(self, model: nn.Module):
        self.model = model
        self.orig_scales: Dict[str, float] = {}

    def __enter__(self):
        self.orig_scales = {}
        for name, mod in self.model.named_modules():
            if hasattr(mod, "residual_scale"):
                self.orig_scales[name] = mod.residual_scale
                mod.residual_scale = 0.0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, mod in self.model.named_modules():
            if name in self.orig_scales:
                mod.residual_scale = self.orig_scales[name]


# ==============================================================================
# 2. Helpers: Metrics, Router Discovery & Policy Extraction
# ==============================================================================

def compute_thinness_score(target_mask: np.ndarray) -> float:
    gt_area = int(np.sum(target_mask > 0))
    if gt_area == 0:
        return 0.0
    contours, _ = cv2.findContours(target_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, closed=True) for c in contours)
    return float(perimeter / (2.0 * gt_area))


def compute_sample_metrics(pred_bin: np.ndarray, target_bin: np.ndarray) -> Dict[str, float]:
    tp = np.sum((pred_bin == 1) & (target_bin == 1))
    fp = np.sum((pred_bin == 1) & (target_bin == 0))
    fn = np.sum((pred_bin == 0) & (target_bin == 1))
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    dice = float(2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 1.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 1.0
    return {"dice": dice, "iou": iou, "precision": prec, "recall": rec}


def discover_routers(model: nn.Module) -> Dict[str, Tuple[str, SageRouter]]:
    routers = {}
    if hasattr(model, "backbone"):
        if hasattr(model.backbone, "convnext") and hasattr(model.backbone.convnext, "stages"):
            for idx, stage in enumerate(model.backbone.convnext.stages):
                if hasattr(stage, "router") and isinstance(stage.router, SageRouter):
                    routers[f"S{idx}"] = (f"backbone.convnext.stages.{idx}.router", stage.router)
        if hasattr(model.backbone, "transformer_blocks"):
            for idx, blk in enumerate(model.backbone.transformer_blocks):
                if hasattr(blk, "router") and isinstance(blk.router, SageRouter):
                    routers[f"B{idx}"] = (f"backbone.transformer_blocks.{idx}.router", blk.router)
    return routers


def compute_paired_stats(a_scores: np.ndarray, b_scores: np.ndarray, label_a: str, label_b: str) -> Dict[str, Any]:
    diffs = a_scores - b_scores
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    sem = std_diff / np.sqrt(n) if n > 0 else 0.0
    t_crit = stats.t.ppf(0.975, df=n - 1) if n > 1 else 1.96
    ci_lower = float(mean_diff - t_crit * sem)
    ci_upper = float(mean_diff + t_crit * sem)
    ttest_res = stats.ttest_rel(a_scores, b_scores)
    try:
        w_res = stats.wilcoxon(a_scores, b_scores, alternative="two-sided")
        w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
    except Exception:
        w_stat, w_p = float("nan"), float("nan")

    wins = int(np.sum(diffs > 1e-6))
    losses = int(np.sum(diffs < -1e-6))
    ties = n - wins - losses
    return {
        "comparison": f"{label_a} vs {label_b}",
        "mean_diff": round(mean_diff, 6),
        "median_diff": round(median_diff, 6),
        "std_diff": round(std_diff, 6),
        "ci_95_lower": round(ci_lower, 6),
        "ci_95_upper": round(ci_upper, 6),
        "paired_t_stat": round(float(ttest_res.statistic), 4),
        "paired_t_pvalue": float(ttest_res.pvalue),
        "wilcoxon_stat": round(w_stat, 2),
        "wilcoxon_pvalue": float(w_p),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": round(float(wins / n), 4),
    }


# ==============================================================================
# 3. Main Evaluation Orchestrator
# ==============================================================================

def run_routing_ablation(
    config_path: str,
    ckpt_path: str,
    data_root: str,
    events_csv: str,
    out_dir: str,
    random_seeds: List[int],
    batch_size: int = 8,
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False

    logger.info("=" * 80)
    logger.info("CANDIDATE B ROUTING ABLATION: STATIC vs RANDOM vs ADAPTIVE vs SAGE-OFF")
    logger.info(f"Device: {device} | Checkpoint: {os.path.basename(ckpt_path)}")
    logger.info("=" * 80)

    # 1. Load config & model
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    model = create_b2_unet(
        num_classes=1,
        img_size=int(cfg.get("img_size", 448)),
        num_transformer_layers=int(cfg.get("num_transformer_layers", 4)),
        pretrained=False,
        sage_config=cfg.get("sage_config", {}),
        p3_mode="C",
    ).to(device)

    raw_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_sd = raw_ckpt.get("model_state_dict", raw_ckpt)
    model.load_state_dict(ckpt_sd, strict=True)

    # Turn off ASDW (Candidate B official baseline)
    for stage in model.backbone.convnext.stages[:2]:
        if hasattr(stage, "p3_refinement") and stage.p3_refinement is not None:
            stage.p3_refinement = nn.Identity()
    model.eval()

    # Discover routers
    router_map = discover_routers(model)
    assert len(router_map) == 8, f"Expected 8 routers, found {len(router_map)}!"
    router_modules = {alias: router for alias, (_, router) in router_map.items()}

    # 2. Derive Static Policies
    # Policy A: Training Usage Top-2
    static_policy_train = {}
    for alias, (full_name, router) in router_map.items():
        buf_name = f"{full_name}.expert_usage_count"
        counts = ckpt_sd[buf_name].cpu().numpy()
        top_k_indices = counts.argsort()[::-1][:router.top_k].tolist()
        static_policy_train[alias] = top_k_indices

    # Policy B: Validation Empirical Top-2 (from full_val per_router_usage.csv)
    static_policy_val = {
        "S0": [4, 6],
        "S1": [1, 6],
        "S2": [4, 6],
        "S3": [5, 4],
        "B0": [7, 1],
        "B1": [7, 4],
        "B2": [1, 2],
        "B3": [5, 7],
    }

    logger.info("Static Policy (Training Usage Top-2):")
    for a, p in static_policy_train.items():
        logger.info(f"  {a}: {p}")
    logger.info("Static Policy (Val Empirical Top-2):")
    for a, p in static_policy_val.items():
        logger.info(f"  {a}: {p}")

    # 3. Load Validation Dataset into memory for speed
    val_img_dir = os.path.join(data_root, "val", "images")
    val_mask_dir = os.path.join(data_root, "val", "masks")
    img_files = sorted([f for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg', '.png'))])

    logger.info(f"Preloading {len(img_files)} validation images into RAM...")
    val_cache = []
    thinness_scores = []
    for fn in tqdm(img_files, desc="Preloading"):
        stem, _ = os.path.splitext(fn)
        ip = os.path.join(val_img_dir, fn)
        mp = os.path.join(val_mask_dir, stem + ".png")
        if not os.path.exists(mp):
            mp = os.path.join(val_mask_dir, stem + ".jpg")
        img = cv2.imread(ip)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        target_bin = (target > 127).astype(np.uint8)
        val_cache.append((stem, img_rgb, target_bin))
        thinness_scores.append(compute_thinness_score(target_bin))

    # Quartiles
    q25 = float(np.percentile(thinness_scores, 25))
    q50 = float(np.percentile(thinness_scores, 50))
    q75 = float(np.percentile(thinness_scores, 75))
    quartiles = []
    for s in thinness_scores:
        if s <= q25:
            quartiles.append("Q1")
        elif s <= q50:
            quartiles.append("Q2")
        elif s <= q75:
            quartiles.append("Q3")
        else:
            quartiles.append("Q4")

    # Load 118 Bridge Events
    df_events = pd.read_csv(events_csv)
    wider_events = df_events[df_events["d_gap_px"] > 5.0].copy()
    logger.info(f"Loaded {len(df_events)} bridge events ({len(wider_events)} wider-gap D > 5 px).")

    # 4. Evaluation Function for any mode
    def run_inference_pass(mode_name: str, context_obj) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        records = []
        logits_cache = {}
        with context_obj:
            for idx, (stem, img_rgb, target_bin) in enumerate(tqdm(val_cache, desc=f"Eval [{mode_name}]")):
                logits = predict_full_image_tiling_setting_a(
                    model, img_rgb, device, tile_size=448, batch_size=batch_size
                )
                pred_bin = (logits > 0.0).astype(np.uint8)
                H, W = target_bin.shape[:2]
                pred_bin = pred_bin[:H, :W]
                logits = logits[:H, :W]

                m = compute_sample_metrics(pred_bin, target_bin)
                m["case_name"] = stem
                m["thinness"] = thinness_scores[idx]
                m["quartile"] = quartiles[idx]
                records.append(m)
                logits_cache[stem] = logits

        return pd.DataFrame(records), logits_cache

    # 5. Run Modes
    results_dfs = {}
    logits_dicts = {}

    # Mode 1: Adaptive Baseline
    logger.info("\n[1/5] Running Adaptive Baseline...")
    df_adap, logits_adap = run_inference_pass(
        "Adaptive", RoutingInterventionContext(router_modules, mode="adaptive")
    )
    results_dfs["Adaptive"] = df_adap
    logits_dicts["Adaptive"] = logits_adap

    # Mode 2: Static (Val Preference)
    logger.info("\n[2/5] Running Static (Val Empirical Top-2)...")
    df_stat_val, logits_stat_val = run_inference_pass(
        "Static_Val", RoutingInterventionContext(router_modules, mode="static", static_policy=static_policy_val)
    )
    results_dfs["Static_Val"] = df_stat_val
    logits_dicts["Static_Val"] = logits_stat_val

    # Mode 3: Static (Train Usage)
    logger.info("\n[3/5] Running Static (Train Usage Top-2)...")
    df_stat_train, logits_stat_train = run_inference_pass(
        "Static_Train", RoutingInterventionContext(router_modules, mode="static", static_policy=static_policy_train)
    )
    results_dfs["Static_Train"] = df_stat_train
    logits_dicts["Static_Train"] = logits_stat_train

    # Mode 4: Random Top-2 (across seeds)
    rand_dfs = []
    rand_logits_list = []
    for s_idx, seed in enumerate(random_seeds):
        logger.info(f"\n[4/5] Running Random Top-2 (Seed {seed} [{s_idx+1}/{len(random_seeds)}])...")
        df_r, log_r = run_inference_pass(
            f"Random_s{seed}", RoutingInterventionContext(router_modules, mode="random", random_seed=seed)
        )
        rand_dfs.append(df_r)
        rand_logits_list.append(log_r)

    # Average random metrics per sample across seeds
    df_rand_avg = df_adap.copy()
    df_rand_avg["dice"] = np.mean([df["dice"].values for df in rand_dfs], axis=0)
    df_rand_avg["iou"] = np.mean([df["iou"].values for df in rand_dfs], axis=0)
    df_rand_avg["precision"] = np.mean([df["precision"].values for df in rand_dfs], axis=0)
    df_rand_avg["recall"] = np.mean([df["recall"].values for df in rand_dfs], axis=0)
    results_dfs["Random_Avg"] = df_rand_avg

    # Mode 5: SAGE-Off (Zero residual)
    logger.info("\n[5/5] Running SAGE-Off (Pure Backbone, residual_scale=0.0)...")
    df_sage_off, logits_sage_off = run_inference_pass(
        "SAGE_Off", SageOffContext(model)
    )
    results_dfs["SAGE_Off"] = df_sage_off
    logits_dicts["SAGE_Off"] = logits_sage_off

    # 6. Overall Metrics Table
    summary_rows = []
    for mode_name, df_m in results_dfs.items():
        summary_rows.append({
            "mode": mode_name,
            "mean_dice": float(df_m["dice"].mean()),
            "std_dice": float(df_m["dice"].std()),
            "median_dice": float(df_m["dice"].median()),
            "mean_iou": float(df_m["iou"].mean()),
            "mean_precision": float(df_m["precision"].mean()),
            "mean_recall": float(df_m["recall"].mean()),
            "delta_dice_vs_adaptive": float(df_m["dice"].mean() - df_adap["dice"].mean()),
            "delta_iou_vs_adaptive": float(df_m["iou"].mean() - df_adap["iou"].mean()),
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(out_dir, "overall_metrics.csv"), index=False)

    # 7. Paired Comparisons against Adaptive Baseline
    paired_rows = []
    for mode_name in ["Static_Val", "Static_Train", "Random_Avg", "SAGE_Off"]:
        stats_res = compute_paired_stats(
            results_dfs["Adaptive"]["dice"].values,
            results_dfs[mode_name]["dice"].values,
            "Adaptive",
            mode_name,
        )
        paired_rows.append(stats_res)
    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv(os.path.join(out_dir, "paired_comparisons.csv"), index=False)

    # 8. Thinness Quartile Breakdown
    quartile_rows = []
    for q_label in ["Q1", "Q2", "Q3", "Q4"]:
        row = {"quartile": q_label}
        for mode_name, df_m in results_dfs.items():
            sub = df_m[df_m["quartile"] == q_label]
            row[f"{mode_name}_dice"] = float(sub["dice"].mean())
        quartile_rows.append(row)
    df_quartiles = pd.DataFrame(quartile_rows)
    df_quartiles.to_csv(os.path.join(out_dir, "quartile_metrics.csv"), index=False)

    # 9. False Bridge Cohort Impact
    logger.info("\nEvaluating False Bridge Cohort Impact across regimes...")
    val_map = {stem: target for stem, _, target in val_cache}
    bridge_modes = {
        "Adaptive": logits_adap,
        "Static_Val": logits_stat_val,
        "Static_Train": logits_stat_train,
        "Random_Seed42": rand_logits_list[0],
        "SAGE_Off": logits_sage_off,
    }

    bridge_results = []
    for _, ev_row in df_events.iterrows():
        ev_id = int(ev_row["event_id"])
        case_name = str(ev_row["case_name"])
        p_id = int(ev_row["pred_cc_id"])
        gA = int(ev_row["primary_gt_A"])
        gB = int(ev_row["primary_gt_B"])
        d_gap = float(ev_row["d_gap_px"])
        is_wider = bool(d_gap > 5.0)

        if case_name not in val_map:
            continue
        target_bin = val_map[case_name]
        H, W = target_bin.shape[:2]

        num_gt_cc, gt_labels = cv2.connectedComponents(target_bin, connectivity=8)

        # Baseline corridor mask
        base_logits = logits_adap[case_name]
        base_pred = (base_logits > 0.0).astype(np.uint8)
        _, base_labels = cv2.connectedComponents(base_pred, connectivity=8)

        cc_fp = (base_labels == p_id) & (target_bin == 0)
        r = max(int(np.ceil(d_gap / 2.0)) + 2, 3)
        k_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
        dil_A = cv2.dilate((gt_labels == gA).astype(np.uint8), k_elem)
        dil_B = cv2.dilate((gt_labels == gB).astype(np.uint8), k_elem)
        corridor_mask = (dil_A > 0) & (dil_B > 0) & cc_fp
        if np.sum(corridor_mask) == 0:
            corridor_mask = cc_fp

        ev_rec = {
            "event_id": ev_id,
            "case_name": case_name,
            "d_gap_px": d_gap,
            "is_wider_gap": is_wider,
        }

        for m_name, log_dict in bridge_modes.items():
            mode_logits = log_dict[case_name]
            mode_pred = (mode_logits > 0.0).astype(np.uint8)
            num_m_cc, m_labels = cv2.connectedComponents(mode_pred, connectivity=8)

            z_corr = float(np.mean(mode_logits[corridor_mask])) if np.sum(corridor_mask) > 0 else float("nan")

            # Check if merged
            merged = False
            for p_m in range(1, num_m_cc):
                gt_over = np.unique(gt_labels[m_labels == p_m])
                if gA in gt_over and gB in gt_over:
                    merged = True
                    break

            ev_rec[f"z_{m_name}"] = z_corr
            ev_rec[f"merged_{m_name}"] = merged
            ev_rec[f"cured_{m_name}"] = (not merged) or (z_corr < 0.0)

        bridge_results.append(ev_rec)

    df_bridge_ev = pd.DataFrame(bridge_results)
    df_bridge_ev.to_csv(os.path.join(out_dir, "bridge_cohort_per_event.csv"), index=False)

    # Summarize Bridge Cohort
    df_wider = df_bridge_ev[df_bridge_ev["is_wider_gap"]].copy()
    bridge_summary_rows = []
    for m_name in bridge_modes.keys():
        wider_cured = int(df_wider[f"cured_{m_name}"].sum())
        all_cured = int(df_bridge_ev[f"cured_{m_name}"].sum())
        wider_cure_rate = float(wider_cured / len(df_wider) * 100.0)
        all_cure_rate = float(all_cured / len(df_bridge_ev) * 100.0)
        mean_z_wider = float(df_wider[f"z_{m_name}"].mean())
        mean_z_all = float(df_bridge_ev[f"z_{m_name}"].mean())
        bridge_summary_rows.append({
            "mode": m_name,
            "wider_43_cured": wider_cured,
            "wider_43_cure_rate_pct": wider_cure_rate,
            "wider_43_mean_z_corridor": mean_z_wider,
            "all_118_cured": all_cured,
            "all_118_cure_rate_pct": all_cure_rate,
            "all_118_mean_z_corridor": mean_z_all,
        })
    df_bridge_summary = pd.DataFrame(bridge_summary_rows)
    df_bridge_summary.to_csv(os.path.join(out_dir, "bridge_cohort_summary.csv"), index=False)

    # 10. Save Detailed JSON & Markdown Report
    summary_final = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": os.path.abspath(ckpt_path),
        "overall_metrics": summary_rows,
        "paired_comparisons": paired_rows,
        "thinness_quartiles": quartile_rows,
        "bridge_cohort_summary": bridge_summary_rows,
    }
    with open(os.path.join(out_dir, "candidate_b_routing_ablation_summary.json"), "w") as f:
        json.dump(summary_final, f, indent=2)

    # Generate Markdown Report
    report_path = os.path.join(out_dir, "candidate_b_routing_ablation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Candidate B Routing Ablation Study: Static vs Random vs Adaptive vs SAGE-Off\n\n")
        f.write(f"**Date:** {summary_final['date']}\n\n")
        f.write(f"**Model Checkpoint:** `{os.path.basename(ckpt_path)}`\n\n")
        f.write(f"**Dataset:** Crack500 Val ($N=348$), Setting A (tile=448, stride=448, threshold=0.5)\n\n")
        f.write("---\n\n")
        f.write("## 1. Overall Setting A Validation Performance\n\n")
        f.write("| Mode | Val Dice | Val IoU | Precision | Recall | Delta Dice vs Adaptive | Delta IoU vs Adaptive |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in summary_rows:
            f.write(f"| **{r['mode']}** | {r['mean_dice']:.4f} ± {r['std_dice']:.4f} | {r['mean_iou']:.4f} | {r['mean_precision']:.4f} | {r['mean_recall']:.4f} | {r['delta_dice_vs_adaptive']:+.5f} | {r['delta_iou_vs_adaptive']:+.5f} |\n")
        f.write("\n")

        f.write("## 2. Paired Statistical Tests against Adaptive Baseline ($N=348$)\n\n")
        f.write("| Comparison | Mean Delta | Median Delta | 95% CI | Paired t-stat (p-val) | Wilcoxon W (p-val) | Win Rate (Win/Tie/Loss) |\n")
        f.write("|---|---:|---:|:---:|:---:|:---:|:---:|\n")
        for r in paired_rows:
            f.write(f"| **{r['comparison']}** | {r['mean_diff']:+.5f} | {r['median_diff']:+.5f} | [{r['ci_95_lower']:+.5f}, {r['ci_95_upper']:+.5f}] | t={r['paired_t_stat']:+.2f} (p={r['paired_t_pvalue']:.2e}) | W={r['wilcoxon_stat']} (p={r['wilcoxon_pvalue']:.2e}) | **{r['win_rate']*100:.1f}%** ({r['wins']}/{r['ties']}/{r['losses']}) |\n")
        f.write("\n")

        f.write("## 3. Thinness Quartile Breakdown (Q1: Thicked -> Q4: Thinnest Cracks)\n\n")
        f.write("| Thinness Quartile | Adaptive Dice | Static (Val) Dice | Static (Train) Dice | Random Dice | SAGE-Off Dice |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in quartile_rows:
            f.write(f"| **{r['quartile']}** | {r['Adaptive_dice']:.4f} | {r['Static_Val_dice']:.4f} | {r['Static_Train_dice']:.4f} | {r['Random_Avg_dice']:.4f} | {r['SAGE_Off_dice']:.4f} |\n")
        f.write("\n")

        f.write("## 4. False Bridge Cohort Impact\n\n")
        f.write("| Routing Mode | 43 Wider-Gap Cured (%) | 43 Wider Mean Corridor Logit | 118 All Cured (%) | 118 All Mean Corridor Logit |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for r in bridge_summary_rows:
            f.write(f"| **{r['mode']}** | {r['wider_43_cured']}/43 (**{r['wider_43_cure_rate_pct']:.1f}%**) | {r['wider_43_mean_z_corridor']:+.4f} | {r['all_118_cured']}/118 (**{r['all_118_cure_rate_pct']:.1f}%**) | {r['all_118_mean_z_corridor']:+.4f} |\n")
        f.write("\n")

    logger.info("=" * 80)
    logger.info(f"CANDIDATE B ROUTING ABLATION COMPLETED SUCCESSFULLY!")
    logger.info(f"Report saved to: {report_path}")
    logger.info("=" * 80)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Candidate B Routing Ablation Study")
    parser.add_argument("--config", type=str, default="SAGE_LITE/configs/p3_ablation/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth")
    parser.add_argument("--data-root", type=str, default="datasets/Crack500_ready")
    parser.add_argument("--events-csv", type=str, default="results/diagnostics/phase6_bottleneck_path/bottleneck_path_118events.csv")
    parser.add_argument("--output-dir", type=str, default="results/diagnostics/candidate_b_routing_ablation")
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_routing_ablation(
        config_path=args.config,
        ckpt_path=args.checkpoint,
        data_root=args.data_root,
        events_csv=args.events_csv,
        out_dir=args.output_dir,
        random_seeds=args.random_seeds,
        batch_size=args.batch_size,
    )
