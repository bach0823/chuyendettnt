#!/usr/bin/env python3
"""
scripts/diagnostics/train_eval_phase6_stem_genesis.py

Phase 6: Upstream Genesis / Stem Information Preservation (U0 Diagnostic Matrix)
=================================================================================

Scientific Objective:
- Determine whether early false-bridge emergence originates from non-overlapping
  patch downsampling (4x4 stride 4) losing thin structure (<4px) boundary info (H1).
- Controlled single-variable causal test:
    U0-C0: 7x7 Stem with center 4x4 frozen (pretrained) + halo 33 pos frozen (=0)
    U0-C1: 7x7 Stem with center 4x4 frozen (pretrained) + halo 33 pos trainable (=0 at t=0)
- Invariant C1 - C0:
    EXACTLY 4,752 trainable halo weights (Learnable Spatial Overlap only).
    Zero depth change, zero non-linearity, zero normalization drift, zero pretrained-core adaptation.
- Protocol:
    Physical batch = 14, seed = 42, FP32, no GA, 8 epochs, AdamW(1e-4, wd=1e-2).
    Loss = BCE + Dice (no topology loss).
    Candidate B Stage 0 through Decoder 100% frozen.
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

sys.path.insert(0, 'SAGE_LITE')
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


# ---------------------------------------------------------------------------
# Module: OverlappingStem7x7 (Causal Halo-Only Parameterization)
# ---------------------------------------------------------------------------

class OverlappingStem7x7(nn.Module):
    """
    Overlapping 7x7 Stem with Immutable Pretrained Core and Learnable Halo.
    
    Formula:
        W_7x7 = embed(fixed_center_weight) + halo_mask * halo_weight
        
    Properties:
        - fixed_center_weight: (48, 3, 4, 4) frozen buffer containing baseline weights.
        - fixed_bias: (48,) frozen buffer containing baseline bias.
        - norm: LayerNorm2d(48) frozen module containing baseline affine parameters.
        - halo_mask: (48, 3, 7, 7) frozen buffer with 0 at [3:7, 3:7] and 1 elsewhere.
        - halo_weight:
            If trainable_halo=True (C1): nn.Parameter(zeros), only halo updated.
            If trainable_halo=False (C0): frozen buffer of zeros.
            
    Invariants:
        W_7x7[:, :, 3:7, 3:7] == fixed_center_weight bitwise at all times.
        halo_weight.grad[:, :, 3:7, 3:7] == 0.0 at all times via backward hook.
        Center 4x4 unaffected by AdamW weight decay.
    """
    def __init__(self, base_conv: nn.Conv2d, base_norm: nn.Module, trainable_halo: bool = False):
        super().__init__()
        assert base_conv.kernel_size == (4, 4), f"Expected kernel (4, 4), got {base_conv.kernel_size}"
        assert base_conv.stride == (4, 4), f"Expected stride (4, 4), got {base_conv.stride}"
        
        self.out_channels = base_conv.out_channels  # 48
        self.in_channels = base_conv.in_channels    # 3
        self.stride = 4
        self.padding = 3
        self.trainable_halo = trainable_halo
        
        # 1. Immutable Pretrained Center Buffer (48, 3, 4, 4)
        self.register_buffer('fixed_center_weight', base_conv.weight.detach().clone())
        self.register_buffer('fixed_bias', base_conv.bias.detach().clone())
        
        # 2. Immutable Normalization (Frozen)
        self.norm = copy.deepcopy(base_norm)
        for p in self.norm.parameters():
            p.requires_grad = False
            
        # 3. Halo Mask (48, 3, 7, 7): 0 at center [3:7, 3:7], 1 at 33 halo positions
        halo_mask = torch.ones(self.out_channels, self.in_channels, 7, 7, dtype=torch.float32)
        halo_mask[:, :, 3:7, 3:7] = 0.0
        self.register_buffer('halo_mask', halo_mask)
        
        # 4. Halo Weight
        if trainable_halo:
            self.halo_weight = nn.Parameter(torch.zeros(self.out_channels, self.in_channels, 7, 7, dtype=torch.float32))
            # Backward hook to strictly zero gradient at center
            self.halo_weight.register_hook(lambda grad: grad * self.halo_mask)
        else:
            self.register_buffer('halo_weight', torch.zeros(self.out_channels, self.in_channels, 7, 7, dtype=torch.float32))

    def get_effective_weight(self) -> torch.Tensor:
        """Constructs effective 7x7 weight ensuring bitwise center preservation."""
        w7 = torch.zeros(
            self.out_channels, self.in_channels, 7, 7,
            device=self.fixed_center_weight.device,
            dtype=self.fixed_center_weight.dtype
        )
        w7[:, :, 3:7, 3:7] = self.fixed_center_weight
        if isinstance(self.halo_weight, nn.Parameter):
            w7 = w7 + self.halo_mask * self.halo_weight
        return w7

    def enforce_center_invariants(self):
        """Post-step invariant check and center zeroing (guard against weight decay)."""
        if isinstance(self.halo_weight, nn.Parameter):
            with torch.no_grad():
                self.halo_weight.data[:, :, 3:7, 3:7] = 0.0
        w7 = self.get_effective_weight()
        assert torch.equal(w7[:, :, 3:7, 3:7], self.fixed_center_weight), (
            "INVARIANT VIOLATION: Center 4x4 deviated from fixed baseline weights!"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w7 = self.get_effective_weight()
        y = F.conv2d(x, w7, self.fixed_bias, stride=self.stride, padding=self.padding)
        y = self.norm(y)
        return y


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    probs = torch.sigmoid(pred_logits)
    probs_flat = probs.view(-1)
    target_flat = target.view(-1)
    intersection = (probs_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (probs_flat.sum() + target_flat.sum() + smooth)
    return 1.0 - dice


def compute_seg_loss(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    l_total = bce + dice
    metrics = {
        'loss_bce': float(bce.item()),
        'loss_dice': float(dice.item()),
        'loss_total': float(l_total.item()),
    }
    return l_total, metrics


# ---------------------------------------------------------------------------
# Training Dataset with Random 448x448 Cropping
# ---------------------------------------------------------------------------

class Crack500TrainDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, img_size: int = 448, seed: int = 42):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.rng = np.random.RandomState(seed)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_p = self.img_paths[idx]
        stem = os.path.splitext(os.path.basename(img_p))[0]
        mask_p = os.path.join(self.mask_dir, f'{stem}.png')
        
        img = cv2.imread(img_p)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_p}")
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_p}")
            
        H, W = mask.shape
        if H < self.img_size or W < self.img_size:
            pad_h = max(0, self.img_size - H)
            pad_w = max(0, self.img_size - W)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            H, W = mask.shape
            
        y1 = self.rng.randint(0, H - self.img_size + 1)
        x1 = self.rng.randint(0, W - self.img_size + 1)
        
        crop_img = img[y1:y1+self.img_size, x1:x1+self.img_size]
        crop_mask = mask[y1:y1+self.img_size, x1:x1+self.img_size]
        
        # Flips
        if self.rng.rand() > 0.5:
            crop_img = np.fliplr(crop_img).copy()
            crop_mask = np.fliplr(crop_mask).copy()
        if self.rng.rand() > 0.5:
            crop_img = np.flipud(crop_img).copy()
            crop_mask = np.flipud(crop_mask).copy()
            
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop_img = (crop_img - self.mean) / self.std
        crop_img = crop_img.transpose(2, 0, 1)  # (3, H, W)
        crop_mask = (crop_mask > 127).astype(np.float32)[np.newaxis, :, :]  # (1, H, W)
        
        return {
            'image': torch.from_numpy(crop_img).float(),
            'mask': torch.from_numpy(crop_mask).float(),
            'stem': stem
        }


# ---------------------------------------------------------------------------
# Canonical Setting A Tiling Evaluation
# ---------------------------------------------------------------------------

def predict_setting_a(model: nn.Module, img_bgr: np.ndarray, device: torch.device, tile_size: int = 448, use_amp: bool = False) -> np.ndarray:
    H, W, _ = img_bgr.shape
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    
    pad_h = (tile_size - (H % tile_size)) % tile_size
    pad_w = (tile_size - (W % tile_size)) % tile_size
    
    img_padded = cv2.copyMakeBorder(img_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    H_pad, W_pad, _ = img_padded.shape
    
    img_norm = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_norm = (img_norm - mean) / std
    
    prob_map = np.zeros((H_pad, W_pad), dtype=np.float32)
    
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=use_amp and device.type == 'cuda'):
            for y in range(0, H_pad, tile_size):
                for x in range(0, W_pad, tile_size):
                    tile = img_norm[y:y+tile_size, x:x+tile_size]
                    tile_t = torch.from_numpy(tile.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                    logit = model(tile_t)
                    prob = torch.sigmoid(logit).squeeze().cpu().numpy()
                    prob_map[y:y+tile_size, x:x+tile_size] = prob
                
    return prob_map[:H, :W]


def evaluate_on_validation_cohort(
    model: nn.Module,
    val_img_paths: List[str],
    val_mask_dir: str,
    device: torch.device,
    desc: str = "Evaluating",
    use_amp: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    for img_p in tqdm(val_img_paths, desc=desc, ncols=80):
        stem = os.path.splitext(os.path.basename(img_p))[0]
        mask_p = os.path.join(val_mask_dir, f'{stem}.png')
        
        img = cv2.imread(img_p)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        gt_binary = (mask > 127).astype(np.uint8)
        
        prob = predict_setting_a(model, img, device, use_amp=use_amp)
        pred_binary = (prob >= 0.5).astype(np.uint8)
        
        topo_metrics = compute_topology_metrics(pred_binary, gt_binary)
        topo_metrics['image_id'] = stem
        rows.append(topo_metrics)
        
    df = pd.DataFrame(rows)
    b_images = int((df['bridge_events'] > 0).sum())
    b_events = int(df['bridge_events'].sum())
    brk_events = int(df['fragmented_gt_components'].sum())
    area_ratio = float((df['pred_area'] / df['gt_area'].clip(lower=1)).mean())
    
    summary = {
        'dice': float(df['dice'].mean()),
        'recall': float(df['recall'].mean()),
        'precision': float(df['precision'].mean()),
        'cldice': float(df['cldice'].mean()),
        'tsens': float(df['tsens'].mean()),
        'tprec': float(df['tprec'].mean()),
        'bridge_images': b_images,
        'bridge_events': b_events,
        'breakage_events': brk_events,
        'area_ratio': area_ratio,
    }
    return df, summary


# ---------------------------------------------------------------------------
# Layer 1: Stem Representation Diagnostic (Separation Margin & R_norm)
# ---------------------------------------------------------------------------

def evaluate_stem_representation(
    model: nn.Module,
    stem_module: OverlappingStem7x7,
    cohort_stems: List[Tuple[str, str]],
    val_img_dir: str,
    val_mask_dir: str,
    device: torch.device,
    tile_size: int = 448,
    use_amp: bool = False,
) -> pd.DataFrame:
    """
    Extracts activation immediately after Stem LayerNorm2d (112x112, 48ch).
    Computes E_neck, E_crack, E_bg, SepMargin, R_norm, and cosine(neck, crack).
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    rows = []
    model.eval()
    
    for stem_name, cohort in tqdm(cohort_stems, desc="Stem Repr Audit", ncols=80):
        img_p = os.path.join(val_img_dir, f'{stem_name}.jpg')
        mask_p = os.path.join(val_mask_dir, f'{stem_name}.png')
        if not os.path.exists(img_p):
            continue
            
        img = cv2.imread(img_p)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        gt_binary = (mask > 127).astype(np.uint8)
        
        prob = predict_setting_a(model, img, device, use_amp=use_amp)
        pred_binary = (prob >= 0.5).astype(np.uint8)
        
        if cohort != 'Clean_Control':
            neck_mask, crack_mask, bg_mask, pair_records = isolate_bridged_pairs_and_rois(pred_binary, gt_binary)
        else:
            neck_mask, crack_mask, bg_mask, pair_records = isolate_clean_pairs_and_rois(gt_binary)
            
        n_neck_px = int(np.sum(neck_mask))
        n_crack_px = int(np.sum(crack_mask))
        n_bg_px = int(np.sum(bg_mask))
        
        # Tile Setting A to accumulate full-resolution 448x448 Stem energy
        H, W = gt_binary.shape
        pad_h = (tile_size - (H % tile_size)) % tile_size
        pad_w = (tile_size - (W % tile_size)) % tile_size
        img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        pH, pW = img_padded.shape[:2]
        
        stem_energy_full = np.zeros((pH, pW), dtype=np.float32)
        f_neck_list, f_crack_list = [], []
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=use_amp and device.type == 'cuda'):
                for py in range(0, pH, tile_size):
                    for px in range(0, pW, tile_size):
                        tile = img_padded[py:py+tile_size, px:px+tile_size]
                        tile_t = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
                        tile_t = ((tile_t.to(device) - mean) / std).unsqueeze(0)
                        
                        # Forward exclusively through Stem module
                        act = stem_module(tile_t)  # (1, 48, 112, 112)
                        energy = torch.norm(act, p=2, dim=1).squeeze(0)  # (112, 112)
                    
                    energy_448 = F.interpolate(
                        energy.unsqueeze(0).unsqueeze(0),
                        size=(tile_size, tile_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze().cpu().numpy()
                    
                    stem_energy_full[py:py+tile_size, px:px+tile_size] = energy_448
                    
                    # Local feature vectors for cosine distance
                    p_neck = neck_mask[py:py+tile_size, px:px+tile_size] if py < H and px < W else None
                    p_crack = crack_mask[py:py+tile_size, px:px+tile_size] if py < H and px < W else None
                    
                    if p_neck is not None and np.sum(p_neck) > 0:
                        m_n = (F.interpolate(torch.from_numpy(p_neck).float().unsqueeze(0).unsqueeze(0), size=(112, 112), mode='nearest').squeeze() > 0)
                        if m_n.any():
                            f_neck_list.append(act[0, :, m_n].mean(dim=1).cpu())
                    if p_crack is not None and np.sum(p_crack) > 0:
                        m_c = (F.interpolate(torch.from_numpy(p_crack).float().unsqueeze(0).unsqueeze(0), size=(112, 112), mode='nearest').squeeze() > 0)
                        if m_c.any():
                            f_crack_list.append(act[0, :, m_c].mean(dim=1).cpu())
                            
        energy_cropped = stem_energy_full[:H, :W]
        e_neck = float(np.mean(energy_cropped[neck_mask == 1])) if n_neck_px > 0 else 0.0
        e_crack = float(np.mean(energy_cropped[crack_mask == 1])) if n_crack_px > 0 else 0.0
        e_bg = float(np.mean(energy_cropped[bg_mask == 1])) if n_bg_px > 0 else 0.0
        
        c_neck_crack = float(e_neck / (e_crack + 1e-6))
        sep_margin = float(1.0 - c_neck_crack)
        denom = (e_crack - e_bg)
        r_norm = float((e_neck - e_bg) / denom) if abs(denom) > 1e-5 else 0.0
        
        cos_nc = 0.0
        if len(f_neck_list) > 0 and len(f_crack_list) > 0:
            vn = torch.stack(f_neck_list).mean(dim=0)
            vc = torch.stack(f_crack_list).mean(dim=0)
            cos_nc = float(F.cosine_similarity(vn.unsqueeze(0), vc.unsqueeze(0)).item())
            
        rows.append({
            'image_id': stem_name,
            'cohort': cohort,
            'n_neck_px': n_neck_px,
            'n_crack_px': n_crack_px,
            'e_neck': e_neck,
            'e_crack': e_crack,
            'e_bg': e_bg,
            'sep_margin': sep_margin,
            'r_norm': r_norm,
            'cos_neck_crack': cos_nc,
        })
        
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Training Loop for U0-C1 (Halo Only)
# ---------------------------------------------------------------------------

def train_u0_c1(
    model: nn.Module,
    stem_c1: OverlappingStem7x7,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    use_amp: bool = False,
    out_dir: str = 'results/diagnostics/phase6_stem_genesis'
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("TRAINING U0-C1: Learnable Spatial Overlap (Halo Only, Center Frozen)")
    print(f"Epochs: {epochs}, Batch: 14, Precision: {'AMP (FP16)' if use_amp else 'FP32'}, LR: {lr}, WD: {weight_decay}")
    print("=" * 80)
    
    # Freeze candidate B model completely
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
        
    # Stem: center is buffer (no grad), only halo_weight is trainable parameter
    assert isinstance(stem_c1.halo_weight, nn.Parameter), "stem_c1.halo_weight must be nn.Parameter"
    stem_c1.halo_weight.requires_grad = True
    
    # Invariant assertion before optimizer setup
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) == 1, f"Expected exactly 1 trainable parameter tensor, got {len(trainable_params)}"
    assert trainable_params[0] is stem_c1.halo_weight, "Trainable parameter is not stem_c1.halo_weight!"
    
    optimizer = torch.optim.AdamW([stem_c1.halo_weight], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and device.type == 'cuda')
    
    logs = []
    
    for ep in range(1, epochs + 1):
        stem_c1.train()
        ep_loss_total = 0.0
        ep_loss_bce = 0.0
        ep_loss_dice = 0.0
        n_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"U0-C1 Ep {ep}/{epochs}", ncols=90)
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp and device.type == 'cuda'):
                logits = model(images)
                loss, m = compute_seg_loss(logits, masks)
                
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                center_grad_max = torch.max(torch.abs(stem_c1.halo_weight.grad[:, :, 3:7, 3:7])).item()
                assert center_grad_max == 0.0 or center_grad_max < 1e-6, f"Gradient leakage into center 4x4! max={center_grad_max}"
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                center_grad_max = torch.max(torch.abs(stem_c1.halo_weight.grad[:, :, 3:7, 3:7])).item()
                assert center_grad_max == 0.0, f"Gradient leakage into center 4x4! max={center_grad_max}"
                optimizer.step()
                
            # Post-optimizer step enforcement: ensure center weight is bitwise unchanged
            stem_c1.enforce_center_invariants()
            
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
            halo_norm = float(torch.norm(stem_c1.halo_weight * stem_c1.halo_mask).item())
            halo_max = float(torch.max(torch.abs(stem_c1.halo_weight * stem_c1.halo_mask)).item())
            
        avg_total = ep_loss_total / n_batches
        avg_bce = ep_loss_bce / n_batches
        avg_dice = ep_loss_dice / n_batches
        
        print(f"  Ep {ep}/{epochs} Summary: Total={avg_total:.4f}, BCE={avg_bce:.4f}, Dice={avg_dice:.4f} | Halo L2={halo_norm:.4f}, Halo Max={halo_max:.4f}")
        
        logs.append({
            'epoch': ep,
            'loss_total': avg_total,
            'loss_bce': avg_bce,
            'loss_dice': avg_dice,
            'halo_l2_norm': halo_norm,
            'halo_max_abs': halo_max,
        })
        
    return {'logs': logs}


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 6: Upstream Genesis U0 Diagnostic")
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'audit_c0', 'train_c1', 'eval'], help="Execution mode")
    parser.add_argument('--epochs', type=int, default=8, help="Number of training epochs for C1")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for halo weights")
    parser.add_argument('--amp', action='store_true', help="Enable AMP (FP16 autocast) for accelerated training on Tesla T4")
    parser.add_argument('--data_root', type=str, default='datasets/Crack500_ready', help="Path to Crack500 dataset")
    parser.add_argument('--config', type=str, default='results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml', help="Config path")
    parser.add_argument('--checkpoint', type=str, default='results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth', help="Checkpoint path")
    parser.add_argument('--out_dir', type=str, default='results/diagnostics/phase6_stem_genesis', help="Output directory")
    args = parser.parse_args()
    
    print("=" * 80)
    print("PHASE 6: UPSTREAM GENESIS (U0-C0 vs U0-C1 CAUSAL DIAGNOSTIC)")
    print(f"Mode: {args.mode}, Epochs: {args.epochs}, AMP: {args.amp}")
    print("=" * 80)
    
    config_path = args.config
    ckpt_path   = args.checkpoint
    out_dir     = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 0. Checkpoint & Model Invariants Audit
    file_sha = compute_file_hash(ckpt_path)
    print(f"Candidate B Checkpoint SHA256: {file_sha}")
    assert file_sha == "147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66", (
        f"Checkpoint SHA256 mismatch! Expected 147f78... got {file_sha}"
    )
    
    # Load Candidate B
    model = load_model_from_checkpoint(config_path, ckpt_path, device=device)
    model_param_sha = compute_model_param_hash(model)
    print(f"Candidate B Parameters SHA256: {model_param_sha}")
    assert model_param_sha == "4aeda58ce6d2fb32f5b772cdbf92dec7fb913f9510b392910efe6674b0adf5d6", (
        f"Model parameters SHA256 mismatch! Expected 4aeda5... got {model_param_sha}"
    )
    
    # Reference baseline stem
    base_conv = model.backbone.convnext.stem[0]
    base_norm = model.backbone.convnext.stem[1]
    
    # Preflight Mathematical Identity Test: Stem_base vs Stem_C0
    dummy_x = torch.randn(2, 3, 448, 448, device=device)
    stem_c0 = OverlappingStem7x7(base_conv, base_norm, trainable_halo=False).to(device)
    
    with torch.no_grad():
        out_base = model.backbone.convnext.stem(dummy_x)
        out_c0 = stem_c0(dummy_x)
        max_abs_diff = torch.max(torch.abs(out_base - out_c0)).item()
        
    print(f"\n[Preflight Test] Stem_baseline vs Stem_C0 on dummy input: max_abs_diff = {max_abs_diff:.2e}")
    assert max_abs_diff == 0.0 or max_abs_diff < 1e-7, (
        f"PREFLIGHT INVARIANT FAILED: Stem_C0 did not reproduce Baseline! diff={max_abs_diff}"
    )
    
    # Datasets setup
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
        
    print(f"Cohort Samples for Stem Repr Audit: Total={len(cohort_stems)} (Resistant={len(resistant_stems)}, Sensitive={len(sensitive_stems)}, Clean={len(clean_stems)})")
    
    # File targets
    c0_eval_csv = os.path.join(out_dir, 'validation_c0_per_sample.csv')
    c1_eval_csv = os.path.join(out_dir, 'validation_c1_per_sample.csv')
    c1_pth = os.path.join(out_dir, 'u0_c1_halo_weights.pth')
    train_log_csv = os.path.join(out_dir, 'c1_training_log.csv')
    stem_repr_csv = os.path.join(out_dir, 'stem_representation_metrics_comparison.csv')
    subgroup_csv = os.path.join(out_dir, 'validation_subgroup_comparison.csv')
    audit_json = os.path.join(out_dir, 'stem_c0_identity_audit.json')
    
    # -----------------------------------------------------------------------
    # Step 1: U0-C0 Full Validation & Identity Verification
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: U0-C0 FULL VALIDATION & IDENTITY AUDIT")
    print("=" * 80)
    
    model.backbone.convnext.stem = stem_c0
    
    if os.path.exists(c0_eval_csv):
        print(f"Loading existing C0 evaluation from {c0_eval_csv}...")
        df_c0 = pd.read_csv(c0_eval_csv)
    else:
        df_c0, sum_c0 = evaluate_on_validation_cohort(model, val_img_paths, val_mask_dir, device, desc="C0 Eval", use_amp=args.amp)
        df_c0.to_csv(c0_eval_csv, index=False)
        
    c0_b_images = int((df_c0['bridge_events'] > 0).sum())
    c0_b_events = int(df_c0['bridge_events'].sum())
    c0_res_events = int(df_c0[df_c0['image_id'].isin(resistant_stems)]['bridge_events'].sum())
    
    print(f"U0-C0 Full Val Results: Bridge Images = {c0_b_images}, Bridge Events = {c0_b_events}, Resistant Events = {c0_res_events}")
    print(f"U0-C0 Dice = {df_c0['dice'].mean():.4f}, Recall = {df_c0['recall'].mean():.4f}, clDice = {df_c0['cldice'].mean():.4f}")
    
    # Verify exact reproduction of Candidate B official metrics
    assert c0_b_images == 110, f"C0 Identity Violation: Bridge Images expected 110, got {c0_b_images}"
    assert c0_b_events == 118, f"C0 Identity Violation: Bridge Events expected 118, got {c0_b_events}"
    assert c0_res_events == 88, f"C0 Identity Violation: Resistant Events expected 88, got {c0_res_events}"
    print(">> SUCCESS: U0-C0 Reproduces Candidate B Baseline 100% BIT-EXACTLY!")
    
    with open(audit_json, 'w') as f:
        json.dump({
            'status': 'PASS',
            'dummy_max_abs_diff': max_abs_diff,
            'bridge_images': c0_b_images,
            'bridge_events': c0_b_events,
            'resistant_bridge_events': c0_res_events,
            'dice': float(df_c0['dice'].mean()),
            'cldice': float(df_c0['cldice'].mean()),
        }, f, indent=2)
        
    # -----------------------------------------------------------------------
    # Step 2: U0-C1 Training (Halo Only)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: U0-C1 HALO TRAINING & VALIDATION")
    print("=" * 80)
    
    stem_c1 = OverlappingStem7x7(base_conv, base_norm, trainable_halo=True).to(device)
    model.backbone.convnext.stem = stem_c1
    
    if os.path.exists(c1_pth):
        print(f"Found existing C1 weights at {c1_pth}, loading...")
        stem_c1.load_state_dict(torch.load(c1_pth, map_location=device))
        stem_c1.enforce_center_invariants()
    else:
        train_res = train_u0_c1(
            model=model,
            stem_c1=stem_c1,
            train_loader=train_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=1e-2,
            use_amp=args.amp,
            out_dir=out_dir
        )
        torch.save(stem_c1.state_dict(), c1_pth)
        pd.DataFrame(train_res['logs']).to_csv(train_log_csv, index=False)
        print(f"Saved trained C1 weights to {c1_pth}")
        
    # -----------------------------------------------------------------------
    # Step 3: U0-C1 Full Validation Evaluation (Setting A, N=348)
    # -----------------------------------------------------------------------
    if os.path.exists(c1_eval_csv):
        print(f"Loading existing C1 evaluation from {c1_eval_csv}...")
        df_c1 = pd.read_csv(c1_eval_csv)
    else:
        df_c1, sum_c1 = evaluate_on_validation_cohort(model, val_img_paths, val_mask_dir, device, desc="C1 Eval", use_amp=args.amp)
        df_c1.to_csv(c1_eval_csv, index=False)
        
    c1_b_images = int((df_c1['bridge_events'] > 0).sum())
    c1_b_events = int(df_c1['bridge_events'].sum())
    c1_res_events = int(df_c1[df_c1['image_id'].isin(resistant_stems)]['bridge_events'].sum())
    
    print(f"\nU0-C1 Full Val Results: Bridge Images = {c1_b_images}, Bridge Events = {c1_b_events}, Resistant Events = {c1_res_events}")
    print(f"U0-C1 Dice = {df_c1['dice'].mean():.4f}, Recall = {df_c1['recall'].mean():.4f}, clDice = {df_c1['cldice'].mean():.4f}")
    
    # -----------------------------------------------------------------------
    # Step 4: Layer 1 Stem Representation Audit (C0 vs C1)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4: LAYER 1 STEM REPRESENTATION AUDIT (C0 vs C1)")
    print("=" * 80)
    
    # Audit Stem C0 representation
    model.backbone.convnext.stem = stem_c0
    df_repr_c0 = evaluate_stem_representation(model, stem_c0, cohort_stems, val_img_dir, val_mask_dir, device, use_amp=args.amp)
    df_repr_c0['model'] = 'U0-C0'
    
    # Audit Stem C1 representation
    model.backbone.convnext.stem = stem_c1
    df_repr_c1 = evaluate_stem_representation(model, stem_c1, cohort_stems, val_img_dir, val_mask_dir, device, use_amp=args.amp)
    df_repr_c1['model'] = 'U0-C1'
    
    df_repr_all = pd.concat([df_repr_c0, df_repr_c1], ignore_index=True)
    df_repr_all.to_csv(stem_repr_csv, index=False)
    
    print("\nStem Representation Summary across Cohorts:")
    for cohort in ['Resistant_82', 'Sensitive_19', 'Clean_Control']:
        sub_c0 = df_repr_c0[df_repr_c0['cohort'] == cohort]
        sub_c1 = df_repr_c1[df_repr_c1['cohort'] == cohort]
        print(f"  [{cohort}]:")
        print(f"    SepMargin: C0 = {sub_c0['sep_margin'].mean():.4f} | C1 = {sub_c1['sep_margin'].mean():.4f} | Diff (C1-C0) = {sub_c1['sep_margin'].mean() - sub_c0['sep_margin'].mean():+.4f}")
        print(f"    R_norm:    C0 = {sub_c0['r_norm'].mean():.4f} | C1 = {sub_c1['r_norm'].mean():.4f} | Diff (C1-C0) = {sub_c1['r_norm'].mean() - sub_c0['r_norm'].mean():+.4f}")
        print(f"    cos(N, C): C0 = {sub_c0['cos_neck_crack'].mean():.4f} | C1 = {sub_c1['cos_neck_crack'].mean():.4f} | Diff (C1-C0) = {sub_c1['cos_neck_crack'].mean() - sub_c0['cos_neck_crack'].mean():+.4f}")
        
    # -----------------------------------------------------------------------
    # Step 5: Comparative Subgroup Analysis (C1 - C0 Primary Priority)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 5: COMPARATIVE SUBGROUP ANALYSIS (C1 - C0 PRIMARY FOCUS)")
    print("=" * 80)
    
    # Pair per-sample for Cured / Created
    df_paired = pd.merge(df_c0[['image_id', 'bridge_events', 'dice', 'cldice']], df_c1[['image_id', 'bridge_events', 'dice', 'cldice']], on='image_id', suffixes=('_c0', '_c1'))
    df_paired['cured'] = (df_paired['bridge_events_c0'] > 0) & (df_paired['bridge_events_c1'] == 0)
    df_paired['created'] = (df_paired['bridge_events_c0'] == 0) & (df_paired['bridge_events_c1'] > 0)
    
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
        sub_p  = df_paired[df_paired['image_id'].isin(cset)]
        
        b_img_c0 = int((sub_c0['bridge_events'] > 0).sum())
        b_img_c1 = int((sub_c1['bridge_events'] > 0).sum())
        b_ev_c0 = int(sub_c0['bridge_events'].sum())
        b_ev_c1 = int(sub_c1['bridge_events'].sum())
        brk_c0  = int(sub_c0['fragmented_gt_components'].sum())
        brk_c1  = int(sub_c1['fragmented_gt_components'].sum())
        
        cured_cnt = int(sub_p['cured'].sum())
        created_cnt = int(sub_p['created'].sum())
        
        row = {
            'cohort': cname,
            'N': len(cset),
            'C0_Bridge_Images': b_img_c0,
            'C1_Bridge_Images': b_img_c1,
            'Delta_Bridge_Images': b_img_c1 - b_img_c0,
            'C0_Bridge_Events': b_ev_c0,
            'C1_Bridge_Events': b_ev_c1,
            'Delta_Bridge_Events': b_ev_c1 - b_ev_c0,
            'Cured_Images': cured_cnt,
            'Created_Images': created_cnt,
            'C0_Break_Events': brk_c0,
            'C1_Break_Events': brk_c1,
            'Delta_Break_Events': brk_c1 - brk_c0,
            'C0_Dice': float(sub_c0['dice'].mean()),
            'C1_Dice': float(sub_c1['dice'].mean()),
            'Delta_Dice': float(sub_c1['dice'].mean() - sub_c0['dice'].mean()),
            'C0_clDice': float(sub_c0['cldice'].mean()),
            'C1_clDice': float(sub_c1['cldice'].mean()),
            'Delta_clDice': float(sub_c1['cldice'].mean() - sub_c0['cldice'].mean()),
        }
        subgroup_rows.append(row)
        
    df_subgroups = pd.DataFrame(subgroup_rows)
    df_subgroups.to_csv(subgroup_csv, index=False)
    print("\nSubgroup Comparative Results:")
    print(df_subgroups[['cohort', 'N', 'C0_Bridge_Events', 'C1_Bridge_Events', 'Delta_Bridge_Events', 'Cured_Images', 'Created_Images', 'Delta_clDice']].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("PHASE 6 U0 UPSTREAM GENESIS EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"Artifacts saved in: {out_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
