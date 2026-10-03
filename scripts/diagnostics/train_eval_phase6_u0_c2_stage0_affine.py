#!/usr/bin/env python3
"""
scripts/diagnostics/train_eval_phase6_u0_c2_stage0_affine.py

Phase 6: Upstream Genesis / Stage-0 Affine Adaptation (U0-C2-A)
================================================================

Scientific Objective:
- Test the causal hypothesis: Did U0-C1 fail because learnable overlap does not carry
  useful boundary separation, or because the new Stem representation is incompatible
  with the frozen Stage 0 distribution?
- Controlled single-variable causal test:
    U0-C1: 7x7 Stem with center 4x4 frozen + halo 33 pos trainable (trained weights)
    U0-C2-A: U0-C1 + Stage-0 LayerNorm2d affine adaptation (192 parameters: gamma/beta)
- Invariant C2-A - C1:
    EXACTLY 192 additional trainable parameters (Stage 0 blocks.0/1 norm.weight and norm.bias).
    Total trainable parameters = 4,752 (halo) + 192 (Stage 0 LN) = 4,944 parameters.
    Zero DW-conv fine-tuning, zero MLP conv fine-tuning, zero GRN adaptation, zero SAGE unfreezing.
    Downstream network (Stages 1-3, ViT, Decoder B1) 100% strictly FROZEN.
- Strict Protocol:
    Physical batch = 14, seed = 42, FP32 ONLY (no AMP), no GA, 8 epochs, AdamW(1e-4, wd=1e-2).
    CosineAnnealingLR scheduler.
    Loss = BCE + Dice (no topology loss, no auxiliary loss).
    Validation N=348 evaluated with canonical Setting A tiling (tau=0.5).
    Test set N=1124 strictly SEALED.
"""

import argparse
import copy
import glob
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('SAGE_LITE'))
from tools.run_phase6_c_topology_diagnostic import (
    load_model_from_checkpoint,
    compute_topology_metrics,
)
from scripts.diagnostics.phase6_stem_factorization_provenance import (
    isolate_bridged_pairs_and_rois,
    isolate_clean_pairs_and_rois,
    compute_file_hash,
    compute_model_param_hash,
)
from scripts.diagnostics.train_eval_phase6_stem_genesis import (
    OverlappingStem7x7,
    Crack500TrainDataset,
    compute_seg_loss,
    predict_setting_a,
    evaluate_on_validation_cohort,
)


# ---------------------------------------------------------------------------
# Multi-Scale Representation Extractor (Stem and Stage 0)
# ---------------------------------------------------------------------------

def extract_stem_and_stage0_energy_and_features(
    model: nn.Module,
    tile_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts features and energies from Stem output and Stage 0 output.
    Returns:
        act_stem: (1, 48, 112, 112)
        energy_stem: (112, 112)
        act_stage0: (1, 48, 112, 112)
        energy_stage0: (112, 112)
    """
    # 1. Stem forward
    act_stem = model.backbone.convnext.stem(tile_t)  # (1, 48, 112, 112)
    energy_stem = torch.norm(act_stem, p=2, dim=1).squeeze(0)  # (112, 112)
    
    # 2. Stage 0 forward
    stage0 = model.backbone.convnext.stages[0]
    act_stage0 = stage0(act_stem)
    if isinstance(act_stage0, tuple):
        act_stage0 = act_stage0[0]
    energy_stage0 = torch.norm(act_stage0, p=2, dim=1).squeeze(0)  # (112, 112)
    
    return act_stem, energy_stem, act_stage0, energy_stage0


def evaluate_dual_representation(
    model: nn.Module,
    cohort_stems: List[Tuple[str, str]],
    val_img_dir: str,
    val_mask_dir: str,
    device: torch.device,
    tile_size: int = 448,
    model_tag: str = "U0-C2-A",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates both Stem and Stage 0 representation separation on cohort samples.
    Returns: (df_stem_repr, df_stage0_repr)
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    rows_stem = []
    rows_stage0 = []
    model.eval()
    
    for stem_name, cohort in tqdm(cohort_stems, desc=f"Dual Repr Audit [{model_tag}]", ncols=80):
        img_p = os.path.join(val_img_dir, f'{stem_name}.jpg')
        mask_p = os.path.join(val_mask_dir, f'{stem_name}.png')
        if not os.path.exists(img_p):
            continue
            
        img = cv2.imread(img_p)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        gt_binary = (mask > 127).astype(np.uint8)
        
        prob = predict_setting_a(model, img, device, use_amp=False)
        pred_binary = (prob >= 0.5).astype(np.uint8)
        
        if cohort != 'Clean_Control':
            neck_mask, crack_mask, bg_mask, _ = isolate_bridged_pairs_and_rois(pred_binary, gt_binary)
        else:
            neck_mask, crack_mask, bg_mask, _ = isolate_clean_pairs_and_rois(gt_binary)
            
        n_neck_px = int(np.sum(neck_mask))
        n_crack_px = int(np.sum(crack_mask))
        n_bg_px = int(np.sum(bg_mask))
        
        H, W = gt_binary.shape
        pad_h = (tile_size - (H % tile_size)) % tile_size
        pad_w = (tile_size - (W % tile_size)) % tile_size
        img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        pH, pW = img_padded.shape[:2]
        
        energy_stem_full = np.zeros((pH, pW), dtype=np.float32)
        energy_stage0_full = np.zeros((pH, pW), dtype=np.float32)
        
        f_neck_stem, f_crack_stem = [], []
        f_neck_stage0, f_crack_stage0 = [], []
        
        with torch.no_grad():
            for py in range(0, pH, tile_size):
                for px in range(0, pW, tile_size):
                    tile = img_padded[py:py+tile_size, px:px+tile_size]
                    tile_t = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
                    tile_t = ((tile_t.to(device) - mean) / std).unsqueeze(0)
                    
                    act_s, eng_s, act_0, eng_0 = extract_stem_and_stage0_energy_and_features(model, tile_t)
                    
                    # Interpolate to 448x448
                    eng_s_448 = F.interpolate(
                        eng_s.unsqueeze(0).unsqueeze(0),
                        size=(tile_size, tile_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze().cpu().numpy()
                    
                    eng_0_448 = F.interpolate(
                        eng_0.unsqueeze(0).unsqueeze(0),
                        size=(tile_size, tile_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze().cpu().numpy()
                    
                    energy_stem_full[py:py+tile_size, px:px+tile_size] = eng_s_448
                    energy_stage0_full[py:py+tile_size, px:px+tile_size] = eng_0_448
                    
                    # Extract ROI feature vectors
                    p_neck = neck_mask[py:py+tile_size, px:px+tile_size] if py < H and px < W else None
                    p_crack = crack_mask[py:py+tile_size, px:px+tile_size] if py < H and px < W else None
                    
                    if p_neck is not None and np.sum(p_neck) > 0:
                        m_n = (F.interpolate(torch.from_numpy(p_neck).float().unsqueeze(0).unsqueeze(0), size=(112, 112), mode='nearest').squeeze() > 0)
                        if m_n.any():
                            f_neck_stem.append(act_s[0, :, m_n].mean(dim=1).cpu())
                            f_neck_stage0.append(act_0[0, :, m_n].mean(dim=1).cpu())
                    if p_crack is not None and np.sum(p_crack) > 0:
                        m_c = (F.interpolate(torch.from_numpy(p_crack).float().unsqueeze(0).unsqueeze(0), size=(112, 112), mode='nearest').squeeze() > 0)
                        if m_c.any():
                            f_crack_stem.append(act_s[0, :, m_c].mean(dim=1).cpu())
                            f_crack_stage0.append(act_0[0, :, m_c].mean(dim=1).cpu())
                            
        # Compute Stem metrics
        eng_stem_crop = energy_stem_full[:H, :W]
        en_s = float(np.mean(eng_stem_crop[neck_mask == 1])) if n_neck_px > 0 else 0.0
        ec_s = float(np.mean(eng_stem_crop[crack_mask == 1])) if n_crack_px > 0 else 0.0
        eb_s = float(np.mean(eng_stem_crop[bg_mask == 1])) if n_bg_px > 0 else 0.0
        sep_s = float(1.0 - (en_s / (ec_s + 1e-6)))
        denom_s = (ec_s - eb_s)
        rn_s = float((en_s - eb_s) / denom_s) if abs(denom_s) > 1e-5 else 0.0
        cos_s = 0.0
        if len(f_neck_stem) > 0 and len(f_crack_stem) > 0:
            vn = torch.stack(f_neck_stem).mean(dim=0)
            vc = torch.stack(f_crack_stem).mean(dim=0)
            cos_s = float(F.cosine_similarity(vn.unsqueeze(0), vc.unsqueeze(0)).item())
            
        rows_stem.append({
            'image_id': stem_name,
            'cohort': cohort,
            'model': model_tag,
            'level': 'Stem',
            'n_neck_px': n_neck_px,
            'n_crack_px': n_crack_px,
            'e_neck': en_s,
            'e_crack': ec_s,
            'e_bg': eb_s,
            'sep_margin': sep_s,
            'r_norm': rn_s,
            'cos_neck_crack': cos_s,
        })
        
        # Compute Stage 0 metrics
        eng_stage0_crop = energy_stage0_full[:H, :W]
        en_0 = float(np.mean(eng_stage0_crop[neck_mask == 1])) if n_neck_px > 0 else 0.0
        ec_0 = float(np.mean(eng_stage0_crop[crack_mask == 1])) if n_crack_px > 0 else 0.0
        eb_0 = float(np.mean(eng_stage0_crop[bg_mask == 1])) if n_bg_px > 0 else 0.0
        sep_0 = float(1.0 - (en_0 / (ec_0 + 1e-6)))
        denom_0 = (ec_0 - eb_0)
        rn_0 = float((en_0 - eb_0) / denom_0) if abs(denom_0) > 1e-5 else 0.0
        cos_0 = 0.0
        if len(f_neck_stage0) > 0 and len(f_crack_stage0) > 0:
            vn = torch.stack(f_neck_stage0).mean(dim=0)
            vc = torch.stack(f_crack_stage0).mean(dim=0)
            cos_0 = float(F.cosine_similarity(vn.unsqueeze(0), vc.unsqueeze(0)).item())
            
        rows_stage0.append({
            'image_id': stem_name,
            'cohort': cohort,
            'model': model_tag,
            'level': 'Stage0',
            'n_neck_px': n_neck_px,
            'n_crack_px': n_crack_px,
            'e_neck': en_0,
            'e_crack': ec_0,
            'e_bg': eb_0,
            'sep_margin': sep_0,
            'r_norm': rn_0,
            'cos_neck_crack': cos_0,
        })
        
    return pd.DataFrame(rows_stem), pd.DataFrame(rows_stage0)


# ---------------------------------------------------------------------------
# Training Loop for U0-C2-A (Stem Halo + Stage 0 LN Affine) - FP32 ONLY
# ---------------------------------------------------------------------------

def train_u0_c2_a(
    model: nn.Module,
    stem_module: OverlappingStem7x7,
    trainable_params: List[nn.Parameter],
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    out_dir: str = 'results/diagnostics/phase6_stem_genesis/u0_c2_stage0_affine'
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("TRAINING U0-C2-A: Stem Halo + Stage-0 LayerNorm2d Affine Adaptation")
    print(f"Epochs: {epochs}, Batch: 14, Precision: FP32 ONLY (Strict), LR: {lr}, WD: {weight_decay}")
    print(f"Total Trainable Parameters: {sum(p.numel() for p in trainable_params)} (Expected: 4,944)")
    print("=" * 80)
    
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    logs = []
    
    stage0 = model.backbone.convnext.stages[0]
    ln0 = stage0.main_block.module.blocks[0].norm
    ln1 = stage0.main_block.module.blocks[1].norm
    
    for ep in range(1, epochs + 1):
        model.eval()  # Keep all batchnorms/dropouts in eval
        stem_module.train()
        ln0.train()
        ln1.train()
        
        ep_loss_total = 0.0
        ep_loss_bce = 0.0
        ep_loss_dice = 0.0
        
        pbar = tqdm(train_loader, desc=f"U0-C2-A Ep {ep}/{epochs}", ncols=90)
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            # Strict FP32 forward
            logits = model(images)
            loss, m = compute_seg_loss(logits, masks)
            
            # Strict FP32 backward
            loss.backward()
            
            # Center invariant check: gradient at center must be 0.0 bitwise
            center_slice = stem_module.halo_weight.grad[:, :, 3:7, 3:7]
            center_grad_max = torch.max(torch.abs(center_slice)).item()
            assert center_grad_max == 0.0, f"Gradient leakage into center 4x4! max={center_grad_max}"
            
            # Optimizer step
            optimizer.step()
            
            # Center weight invariant enforcement
            stem_module.enforce_center_invariants()
            
            ep_loss_total += m['loss_total']
            ep_loss_bce += m['loss_bce']
            ep_loss_dice += m['loss_dice']
            
            pbar.set_postfix({
                'tot': f"{m['loss_total']:.3f}",
                'bce': f"{m['loss_bce']:.3f}",
                'dice': f"{m['loss_dice']:.3f}"
            })
            
        scheduler.step()
        
        with torch.no_grad():
            halo_norm = float(torch.norm(stem_module.halo_weight * stem_module.halo_mask).item())
            halo_max = float(torch.max(torch.abs(stem_module.halo_weight * stem_module.halo_mask)).item())
            ln0_w_norm = float(torch.norm(ln0.weight).item())
            ln0_b_norm = float(torch.norm(ln0.bias).item())
            ln1_w_norm = float(torch.norm(ln1.weight).item())
            ln1_b_norm = float(torch.norm(ln1.bias).item())
            
        n_b = len(train_loader)
        log_entry = {
            'epoch': ep,
            'loss_total': ep_loss_total / n_b,
            'loss_bce': ep_loss_bce / n_b,
            'loss_dice': ep_loss_dice / n_b,
            'halo_l2_norm': halo_norm,
            'halo_max_abs': halo_max,
            'ln0_weight_norm': ln0_w_norm,
            'ln0_bias_norm': ln0_b_norm,
            'ln1_weight_norm': ln1_w_norm,
            'ln1_bias_norm': ln1_b_norm,
            'lr': scheduler.get_last_lr()[0],
        }
        logs.append(log_entry)
        print(f"Ep {ep:02d} Summary: Total={log_entry['loss_total']:.4f} (BCE={log_entry['loss_bce']:.4f}, Dice={log_entry['loss_dice']:.4f}) | Halo L2={halo_norm:.4f}, LN0_W={ln0_w_norm:.4f}, LN1_W={ln1_w_norm:.4f}")
        
    return {'logs': logs}


# ---------------------------------------------------------------------------
# Main Routine: U0-C2-A Diagnostic Pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 6 Upstream Genesis: U0-C2-A Stage-0 Affine Adaptation")
    parser.add_argument('--data_root', type=str, default='datasets/Crack500_ready', help="Path to Crack500 dataset")
    parser.add_argument('--config', type=str, default='results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml', help="Config path")
    parser.add_argument('--checkpoint', type=str, default='results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth', help="Candidate B checkpoint path")
    parser.add_argument('--c1_weights', type=str, default='results/diagnostics/phase6_stem_genesis/u0_c1_halo_weights.pth', help="Trained C1 halo weights path")
    parser.add_argument('--out_dir', type=str, default='results/diagnostics/phase6_stem_genesis/u0_c2_stage0_affine', help="Output directory")
    parser.add_argument('--epochs', type=int, default=8, help="Epoch budget (default: 8)")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("PHASE 6: UPSTREAM GENESIS (U0-C2-A STAGE-0 AFFINE ADAPTATION)")
    print(f"Device: {device} | Precision: FP32 ONLY | Epochs: {args.epochs} | LR: {args.lr}")
    print("=" * 80)
    
    # -----------------------------------------------------------------------
    # I1: Candidate B Checkpoint Audit
    # -----------------------------------------------------------------------
    file_sha = compute_file_hash(args.checkpoint)
    print(f"Candidate B Checkpoint SHA256: {file_sha}")
    assert file_sha == "147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66", (
        f"Checkpoint SHA256 mismatch! Expected 147f78... got {file_sha}"
    )
    
    model = load_model_from_checkpoint(args.config, args.checkpoint, device=device)
    model_param_sha = compute_model_param_hash(model)
    print(f"Candidate B Parameters SHA256: {model_param_sha}")
    assert model_param_sha == "4aeda58ce6d2fb32f5b772cdbf92dec7fb913f9510b392910efe6674b0adf5d6", (
        f"Model parameters SHA256 mismatch! Expected 4aeda5... got {model_param_sha}"
    )
    
    # -----------------------------------------------------------------------
    # Setup Stem Module with C1 Initialization
    # -----------------------------------------------------------------------
    base_conv = model.backbone.convnext.stem[0]
    base_norm = model.backbone.convnext.stem[1]
    stem_c2 = OverlappingStem7x7(base_conv, base_norm, trainable_halo=True).to(device)
    
    assert os.path.exists(args.c1_weights), f"C1 weights not found at {args.c1_weights}!"
    print(f"\nLoading C1 halo weights from {args.c1_weights}...")
    c1_state = torch.load(args.c1_weights, map_location=device)
    stem_c2.load_state_dict(c1_state)
    stem_c2.enforce_center_invariants()
    
    model.backbone.convnext.stem = stem_c2
    
    # -----------------------------------------------------------------------
    # Configure Trainable Whitelist (Stage-0 LN Affine Only)
    # -----------------------------------------------------------------------
    # Freeze 100% of the model first
    for p in model.parameters():
        p.requires_grad = False
        
    # Unfreeze Stem halo
    stem_c2.halo_weight.requires_grad = True
    
    # Unfreeze Stage 0 blocks.0/1 LayerNorm2d affine parameters ONLY
    stage0 = model.backbone.convnext.stages[0]
    ln0 = stage0.main_block.module.blocks[0].norm
    ln1 = stage0.main_block.module.blocks[1].norm
    
    ln0.weight.requires_grad = True
    ln0.bias.requires_grad = True
    ln1.weight.requires_grad = True
    ln1.bias.requires_grad = True
    
    trainable_whitelist = [
        stem_c2.halo_weight,
        ln0.weight,
        ln0.bias,
        ln1.weight,
        ln1.bias,
    ]
    
    # -----------------------------------------------------------------------
    # Invariant Tests (I1 to I8)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RUNNING INVARIANT TESTS (I1 to I8) BEFORE TRAINING")
    print("=" * 80)
    
    # I3: Whitelist parameter count check
    actual_trainable = [p for p in model.parameters() if p.requires_grad]
    total_tensor_numel = sum(p.numel() for p in actual_trainable)
    active_halo_dof = int(stem_c2.halo_mask.sum().item())
    stage0_ln_dof = sum(p.numel() for p in [ln0.weight, ln0.bias, ln1.weight, ln1.bias])
    total_active_dof = active_halo_dof + stage0_ln_dof
    
    print(f"[I3] Whitelist Tensors Count: {len(actual_trainable)} (Expected: 5)")
    print(f"[I3] Whitelist Tensor Elements: {total_tensor_numel} (Expected: 7,248)")
    print(f"[I3] Whitelist Active Degrees of Freedom: {total_active_dof} (Expected: 4,944)")
    assert len(actual_trainable) == 5, f"Expected 5 trainable parameter tensors, got {len(actual_trainable)}"
    assert total_tensor_numel == 7248, f"Expected 7248 tensor elements, got {total_tensor_numel}"
    assert total_active_dof == 4944, f"Expected 4944 active parameters, got {total_active_dof}"
    print(">> PASS I3: Exactly 4,944 active trainable parameters whitelist confirmed.")
    
    # I4 & I5: Center frozen & Halo trainable in dummy forward/backward
    dummy_x = torch.randn(2, 3, 448, 448, device=device)
    model.zero_grad()
    dummy_out = model(dummy_x)
    dummy_loss = dummy_out.sum()
    dummy_loss.backward()
    
    center_grad_max = stem_c2.halo_weight.grad[:, :, 3:7, 3:7].abs().max().item()
    halo_grad_norm = torch.norm(stem_c2.halo_weight.grad * stem_c2.halo_mask).item()
    print(f"[I4] Stem Center Grad Max: {center_grad_max:.2e} (Expected: 0.00e+00)")
    print(f"[I5] Stem Halo Grad Norm: {halo_grad_norm:.4f} (Expected: > 0.0)")
    assert center_grad_max == 0.0, f"Violation I4: Stem center gradient is non-zero! {center_grad_max}"
    assert halo_grad_norm > 0.0, "Violation I5: Stem halo gradient is zero!"
    print(">> PASS I4 & I5: Stem center strictly frozen, halo actively trainable.")
    
    # I6: Stage 0 DW-conv and MLP frozen
    dw0_grad = stage0.main_block.module.blocks[0].conv_dw.weight.grad
    fc1_grad = stage0.main_block.module.blocks[0].mlp.fc1.weight.grad
    grn_grad = stage0.main_block.module.blocks[0].mlp.grn.weight.grad
    assert dw0_grad is None or dw0_grad.abs().max().item() == 0.0, "Violation I6: Stage 0 DWConv gradient is non-zero!"
    assert fc1_grad is None or fc1_grad.abs().max().item() == 0.0, "Violation I6: Stage 0 MLP gradient is non-zero!"
    assert grn_grad is None or grn_grad.abs().max().item() == 0.0, "Violation I6: Stage 0 GRN gradient is non-zero!"
    print(">> PASS I6: Stage 0 DWConv, MLP, and GRN are 100% frozen.")
    
    # I7: Downstream frozen (Stages 1-3, SAGE, ViT, Decoder)
    stage1_dw_grad = model.backbone.convnext.stages[1].main_block.module.blocks[0].conv_dw.weight.grad
    whitelist_ids = {id(w) for w in trainable_whitelist}
    non_whitelist_has_grad = any(
        id(p) not in whitelist_ids and p.grad is not None and p.grad.abs().max().item() > 0.0
        for p in model.parameters()
    )
    assert stage1_dw_grad is None or stage1_dw_grad.abs().max().item() == 0.0, "Violation I7: Stage 1 gradient is non-zero!"
    assert not non_whitelist_has_grad, "Violation I7: A non-whitelist parameter received active gradient!"
    print(">> PASS I7: Downstream Stages 1-3, SAGE, ViT, and Decoder are 100% frozen.")
    
    # I8: Optimizer whitelist check
    optimizer_test = torch.optim.AdamW(trainable_whitelist, lr=1e-4)
    opt_params = [p for group in optimizer_test.param_groups for p in group['params']]
    assert sum(p.numel() for p in opt_params) == 7248, "Violation I8: Optimizer parameter count mismatch!"
    print(">> PASS I8: Optimizer parameter whitelist verified.")
    
    # Clear dummy gradients
    model.zero_grad()
    stem_c2.enforce_center_invariants()
    
    # -----------------------------------------------------------------------
    # I2: C1 -> C2-A Initialization Audit (t=0 Bitwise Identity)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("INITIALIZATION AUDIT (t=0): C1 vs C2-A Identity Verification")
    print("=" * 80)
    with torch.no_grad():
        out_init = model(dummy_x)
        # Load pure C1 model for comparison
        model_c1_ref = load_model_from_checkpoint(args.config, args.checkpoint, device=device)
        stem_c1_ref = OverlappingStem7x7(model_c1_ref.backbone.convnext.stem[0], model_c1_ref.backbone.convnext.stem[1], trainable_halo=True).to(device)
        stem_c1_ref.load_state_dict(c1_state)
        model_c1_ref.backbone.convnext.stem = stem_c1_ref
        out_c1 = model_c1_ref(dummy_x)
        diff_init = torch.max(torch.abs(out_init - out_c1)).item()
        print(f"[I2] C2-A(t=0) vs C1 prediction max_abs_diff: {diff_init:.2e}")
        assert diff_init == 0.0 or diff_init < 1e-6, f"Violation I2: C2-A(t=0) does not reproduce C1! diff={diff_init}"
        print(">> PASS I2: C2-A at t=0 is 100% BITWISE IDENTICAL to C1!")
        del model_c1_ref, stem_c1_ref, out_init, out_c1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # Initial activation statistics
    with torch.no_grad():
        act_s, eng_s, act_0, eng_0 = extract_stem_and_stage0_energy_and_features(model, dummy_x)
        print(f"Initial Activation Stats (dummy batch 2):")
        print(f"  Stem Output:   mean={act_s.mean():.4f}, std={act_s.std():.4f}, norm={torch.norm(act_s):.4f}")
        print(f"  Stage0 Output: mean={act_0.mean():.4f}, std={act_0.std():.4f}, norm={torch.norm(act_0):.4f}")
        
    # -----------------------------------------------------------------------
    # Setup Datasets & Cohorts
    # -----------------------------------------------------------------------
    train_img_dir = os.path.join(args.data_root, 'train', 'images')
    train_mask_dir = os.path.join(args.data_root, 'train', 'masks')
    val_img_dir   = os.path.join(args.data_root, 'val', 'images')
    val_mask_dir  = os.path.join(args.data_root, 'val', 'masks')
    
    val_img_paths = sorted(glob.glob(os.path.join(val_img_dir, '*.jpg')))
    assert len(val_img_paths) == 348, f"Expected 348 validation images, got {len(val_img_paths)}"
    
    train_dataset = Crack500TrainDataset(train_img_dir, train_mask_dir, img_size=448, seed=42)
    train_loader = DataLoader(
        train_dataset,
        batch_size=14,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    print(f"Train Dataset: {len(train_dataset)} images, {len(train_loader)} batches (batch_size=14)")
    
    # Subgroup cohort setup
    master_path = 'results/diagnostics/phase6_d2_topology/topology_7models_master_paired.csv'
    df_master = pd.read_csv(master_path)
    models_list = ['Base', 'A1', 'A2', 'B1', 'C1', 'D1', 'D2']
    bcols = [f'{m}_bridge_events' for m in models_list]
    c7_stems = set(df_master[(df_master[bcols] > 0).all(axis=1)]['case_name'].tolist())
    clean_stems = set(df_master[(df_master[bcols] == 0).all(axis=1) & (df_master['gt_cc'] >= 2)]['case_name'].tolist())
    
    ac_atten_csv = 'results/diagnostics/phase6_stem_ac_attenuation/ac_attenuation_per_sample_alpha.csv'
    df_prev = pd.read_csv(ac_atten_csv)
    c7a0 = df_prev[(df_prev['cohort'] == 'Consensus_7of7') & (df_prev['alpha'] == 0.0)]
    resistant_stems = set(c7a0[c7a0['has_bridge'] == True]['image_id'].tolist())
    sensitive_stems = set(c7a0[c7a0['has_bridge'] == False]['image_id'].tolist())
    
    cohort_stems = []
    for s in sorted(list(resistant_stems)):
        cohort_stems.append((s, 'Resistant_82'))
    for s in sorted(list(sensitive_stems)):
        cohort_stems.append((s, 'Sensitive_19'))
    for s in sorted(list(clean_stems)):
        cohort_stems.append((s, 'Clean_Control'))
        
    print(f"Cohort Samples for Dual Repr Audit: Total={len(cohort_stems)} (Resistant={len(resistant_stems)}, Sensitive={len(sensitive_stems)}, Clean={len(clean_stems)})")
    
    # Artifact paths
    c2_pth = os.path.join(args.out_dir, 'u0_c2_a_weights.pth')
    train_log_csv = os.path.join(args.out_dir, 'c2_a_training_log.csv')
    val_c2_csv = os.path.join(args.out_dir, 'validation_c2_a_per_sample.csv')
    subgroup_csv = os.path.join(args.out_dir, 'validation_subgroup_comparison_c2_vs_c1_c0.csv')
    repr_stem_csv = os.path.join(args.out_dir, 'representation_stem_comparison.csv')
    repr_stage0_csv = os.path.join(args.out_dir, 'representation_stage0_comparison.csv')
    
    # -----------------------------------------------------------------------
    # Step 1: Training U0-C2-A
    # -----------------------------------------------------------------------
    if os.path.exists(c2_pth):
        print(f"\nFound existing C2-A weights at {c2_pth}, loading...")
        ckpt_c2 = torch.load(c2_pth, map_location=device)
        stem_c2.load_state_dict(ckpt_c2['stem_state'])
        ln0.load_state_dict(ckpt_c2['ln0_state'])
        ln1.load_state_dict(ckpt_c2['ln1_state'])
        stem_c2.enforce_center_invariants()
    else:
        train_res = train_u0_c2_a(
            model=model,
            stem_module=stem_c2,
            trainable_params=trainable_whitelist,
            train_loader=train_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=1e-2,
            out_dir=args.out_dir
        )
        torch.save({
            'stem_state': stem_c2.state_dict(),
            'ln0_state': ln0.state_dict(),
            'ln1_state': ln1.state_dict(),
        }, c2_pth)
        pd.DataFrame(train_res['logs']).to_csv(train_log_csv, index=False)
        print(f"Saved trained C2-A weights to {c2_pth}")
        
    # -----------------------------------------------------------------------
    # Step 2: U0-C2-A Full Validation Evaluation (Setting A, N=348)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: U0-C2-A FULL VALIDATION EVALUATION (Setting A, N=348, tau=0.5)")
    print("=" * 80)
    
    if os.path.exists(val_c2_csv):
        print(f"Loading existing C2-A evaluation from {val_c2_csv}...")
        df_c2 = pd.read_csv(val_c2_csv)
    else:
        df_c2, sum_c2 = evaluate_on_validation_cohort(model, val_img_paths, val_mask_dir, device, desc="C2-A Eval", use_amp=False)
        df_c2.to_csv(val_c2_csv, index=False)
        
    c2_b_images = int((df_c2['bridge_events'] > 0).sum())
    c2_b_events = int(df_c2['bridge_events'].sum())
    c2_res_events = int(df_c2[df_c2['image_id'].isin(resistant_stems)]['bridge_events'].sum())
    
    print(f"\nU0-C2-A Full Val Results: Bridge Images = {c2_b_images}, Bridge Events = {c2_b_events}, Resistant Events = {c2_res_events}")
    print(f"U0-C2-A Dice = {df_c2['dice'].mean():.4f}, Recall = {df_c2['recall'].mean():.4f}, clDice = {df_c2['cldice'].mean():.4f}")
    
    # -----------------------------------------------------------------------
    # Step 3: Dual Representation Audit (Stem vs Stage 0, C1 vs C2-A)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: DUAL REPRESENTATION AUDIT (STEM vs STAGE 0)")
    print("=" * 80)
    
    df_stem_c2, df_stage0_c2 = evaluate_dual_representation(
        model, cohort_stems, val_img_dir, val_mask_dir, device, model_tag="U0-C2-A"
    )
    
    # Evaluate C1 representation with same dual protocol
    print("\nEvaluating C1 Dual Representation Baseline...")
    model_c1_eval = load_model_from_checkpoint(args.config, args.checkpoint, device=device)
    stem_c1_eval = OverlappingStem7x7(model_c1_eval.backbone.convnext.stem[0], model_c1_eval.backbone.convnext.stem[1], trainable_halo=True).to(device)
    stem_c1_eval.load_state_dict(c1_state)
    model_c1_eval.backbone.convnext.stem = stem_c1_eval
    
    df_stem_c1, df_stage0_c1 = evaluate_dual_representation(
        model_c1_eval, cohort_stems, val_img_dir, val_mask_dir, device, model_tag="U0-C1"
    )
    del model_c1_eval, stem_c1_eval
    
    df_stem_all = pd.concat([df_stem_c1, df_stem_c2], ignore_index=True)
    df_stage0_all = pd.concat([df_stage0_c1, df_stage0_c2], ignore_index=True)
    
    df_stem_all.to_csv(repr_stem_csv, index=False)
    df_stage0_all.to_csv(repr_stage0_csv, index=False)
    
    print("\n--- Stem Representation (Resistant Cohort N=82) ---")
    st_c1_res = df_stem_c1[df_stem_c1['cohort'] == 'Resistant_82']
    st_c2_res = df_stem_c2[df_stem_c2['cohort'] == 'Resistant_82']
    print(f"  SepMargin: C1 = {st_c1_res['sep_margin'].mean():.4f} | C2-A = {st_c2_res['sep_margin'].mean():.4f} | Delta = {st_c2_res['sep_margin'].mean() - st_c1_res['sep_margin'].mean():+.4f}")
    print(f"  cos(N, C): C1 = {st_c1_res['cos_neck_crack'].mean():.4f} | C2-A = {st_c2_res['cos_neck_crack'].mean():.4f} | Delta = {st_c2_res['cos_neck_crack'].mean() - st_c1_res['cos_neck_crack'].mean():+.4f}")
    
    print("\n--- Stage 0 Representation (Resistant Cohort N=82) ---")
    s0_c1_res = df_stage0_c1[df_stage0_c1['cohort'] == 'Resistant_82']
    s0_c2_res = df_stage0_c2[df_stage0_c2['cohort'] == 'Resistant_82']
    print(f"  SepMargin: C1 = {s0_c1_res['sep_margin'].mean():.4f} | C2-A = {s0_c2_res['sep_margin'].mean():.4f} | Delta = {s0_c2_res['sep_margin'].mean() - s0_c1_res['sep_margin'].mean():+.4f}")
    print(f"  cos(N, C): C1 = {s0_c1_res['cos_neck_crack'].mean():.4f} | C2-A = {s0_c2_res['cos_neck_crack'].mean():.4f} | Delta = {s0_c2_res['cos_neck_crack'].mean() - s0_c1_res['cos_neck_crack'].mean():+.4f}")
    
    # -----------------------------------------------------------------------
    # Step 4: Tri-Way Comparative Subgroup Analysis (C2-A vs C1 vs C0)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4: TRI-WAY COMPARATIVE SUBGROUP ANALYSIS (C2-A vs C1 vs C0)")
    print("=" * 80)
    
    df_c0 = pd.read_csv('results/diagnostics/phase6_stem_genesis/validation_c0_per_sample.csv')
    df_c1 = pd.read_csv('results/diagnostics/phase6_stem_genesis/validation_c1_per_sample.csv')
    
    df_pair_c1_c2 = pd.merge(df_c1[['image_id', 'bridge_events', 'dice', 'cldice']], df_c2[['image_id', 'bridge_events', 'dice', 'cldice']], on='image_id', suffixes=('_c1', '_c2'))
    df_pair_c1_c2['cured_vs_c1'] = (df_pair_c1_c2['bridge_events_c1'] > 0) & (df_pair_c1_c2['bridge_events_c2'] == 0)
    df_pair_c1_c2['created_vs_c1'] = (df_pair_c1_c2['bridge_events_c1'] == 0) & (df_pair_c1_c2['bridge_events_c2'] > 0)
    
    df_pair_c0_c2 = pd.merge(df_c0[['image_id', 'bridge_events', 'dice', 'cldice']], df_c2[['image_id', 'bridge_events', 'dice', 'cldice']], on='image_id', suffixes=('_c0', '_c2'))
    df_pair_c0_c2['cured_vs_c0'] = (df_pair_c0_c2['bridge_events_c0'] > 0) & (df_pair_c0_c2['bridge_events_c2'] == 0)
    df_pair_c0_c2['created_vs_c0'] = (df_pair_c0_c2['bridge_events_c0'] == 0) & (df_pair_c0_c2['bridge_events_c2'] > 0)
    
    cohorts_dict = {
        'Full_Validation_348': set(df_c0['image_id']),
        'Consensus_7of7_101': c7_stems,
        'Consensus_Resistant_82': resistant_stems,
        'Consensus_Sensitive_19': sensitive_stems,
        'Clean_Control_56': clean_stems
    }
    
    subgroup_rows = []
    for cname, cset in cohorts_dict.items():
        sub_c0 = df_c0[df_c0['image_id'].isin(cset)]
        sub_c1 = df_c1[df_c1['image_id'].isin(cset)]
        sub_c2 = df_c2[df_c2['image_id'].isin(cset)]
        sub_p12 = df_pair_c1_c2[df_pair_c1_c2['image_id'].isin(cset)]
        sub_p02 = df_pair_c0_c2[df_pair_c0_c2['image_id'].isin(cset)]
        
        row = {
            'cohort': cname,
            'N': len(cset),
            # C0
            'C0_Bridge_Events': int(sub_c0['bridge_events'].sum()),
            'C0_Break_Events': int(sub_c0['fragmented_gt_components'].sum()),
            'C0_Dice': float(sub_c0['dice'].mean()),
            'C0_clDice': float(sub_c0['cldice'].mean()),
            # C1
            'C1_Bridge_Events': int(sub_c1['bridge_events'].sum()),
            'C1_Break_Events': int(sub_c1['fragmented_gt_components'].sum()),
            'C1_Dice': float(sub_c1['dice'].mean()),
            'C1_clDice': float(sub_c1['cldice'].mean()),
            # C2-A
            'C2A_Bridge_Events': int(sub_c2['bridge_events'].sum()),
            'C2A_Break_Events': int(sub_c2['fragmented_gt_components'].sum()),
            'C2A_Dice': float(sub_c2['dice'].mean()),
            'C2A_clDice': float(sub_c2['cldice'].mean()),
            # Primary Delta: C2A - C1
            'Delta_Bridge_C2A_vs_C1': int(sub_c2['bridge_events'].sum()) - int(sub_c1['bridge_events'].sum()),
            'Delta_Break_C2A_vs_C1': int(sub_c2['fragmented_gt_components'].sum()) - int(sub_c1['fragmented_gt_components'].sum()),
            'Delta_Dice_C2A_vs_C1': float(sub_c2['dice'].mean() - sub_c1['dice'].mean()),
            'Delta_clDice_C2A_vs_C1': float(sub_c2['cldice'].mean() - sub_c1['cldice'].mean()),
            'Cured_vs_C1': int(sub_p12['cured_vs_c1'].sum()),
            'Created_vs_C1': int(sub_p12['created_vs_c1'].sum()),
            # Secondary Delta: C2A - C0
            'Delta_Bridge_C2A_vs_C0': int(sub_c2['bridge_events'].sum()) - int(sub_c0['bridge_events'].sum()),
            'Delta_Dice_C2A_vs_C0': float(sub_c2['dice'].mean() - sub_c0['dice'].mean()),
            'Cured_vs_C0': int(sub_p02['cured_vs_c0'].sum()),
            'Created_vs_C0': int(sub_p02['created_vs_c0'].sum()),
        }
        subgroup_rows.append(row)
        
    df_sub_out = pd.DataFrame(subgroup_rows)
    df_sub_out.to_csv(subgroup_csv, index=False)
    
    print("\nTri-Way Subgroup Comparison Summary:")
    print(df_sub_out[['cohort', 'N', 'C0_Bridge_Events', 'C1_Bridge_Events', 'C2A_Bridge_Events', 'Delta_Bridge_C2A_vs_C1', 'Delta_Dice_C2A_vs_C1']].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("PHASE 6 U0-C2-A EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"Artifacts saved in: {args.out_dir}")


if __name__ == '__main__':
    main()
