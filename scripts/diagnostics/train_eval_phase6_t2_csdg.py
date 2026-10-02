#!/usr/bin/env python3
"""
scripts/diagnostics/train_eval_phase6_t2_csdg.py

Phase 6: Contextual Spatial Dual-Stream Gate (T2-CSDG, K=3) Training & Evaluation
==================================================================================

Scientific Objective:
- Distinguish whether pointwise gating (T2-SDSG) failed due to lack of 2D spatial
  context (H2) or because T2 representations are already irreparably corrupted upstream (H1).
- Controlled single-variable change:
    SDSG (1x1 pointwise gate) -> CSDG (3x3 depthwise spatial context gate)
- Loss strictly pinned to Arm B: L = L_seg + 0.1 * L_aux (no loss change, no lambda sweep).
- Physical batch size = 14 (no gradient accumulation), seed = 42, 8 epochs, AdamW(1e-4).
- 100% Candidate B parameters frozen (in eval mode). Exact identity initialization.
- Full Validation N=348 evaluated with canonical Setting A tiling.
- Explicit gate spatial distribution audit on: true crack vs bridge corridor vs background.
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
from scripts.diagnostics.phase6_stem_factorization_provenance import isolate_bridged_pairs_and_rois


# ---------------------------------------------------------------------------
# Module: T2-CSDG (K=3)
# ---------------------------------------------------------------------------

class T2CSDG(nn.Module):
    """
    Contextual Spatial Dual-Stream Gate (T2-CSDG) with Depthwise Separable Spatial Context
    
    Architecture:
        Input: T2 = [B (192ch); S (96ch)] -> (B, 288, H, W)
        1. Channel Reduction: 1x1 Conv(288 -> 64) + BN + ReLU
        2. Spatial Context: DWConv KxK(64 -> 64, groups=64, padding=K//2) + BN + ReLU
        3. Gate Projection: 1x1 Conv(64 -> 2)
        4. Gate Scaling: g = 2.0 * Sigmoid(a) -> range (0, 2)
        Output: T2' = [g_B * B; g_S * S]
        
    Identity Initialization:
        W_gate = 0, b_gate = 0 -> a = 0 -> g = 2.0 * 0.5 = 1.0 (EXACT IDENTITY).
    """
    def __init__(self, in_channels: int = 192, skip_channels: int = 96, hidden_dim: int = 64, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        total_channels = in_channels + skip_channels
        padding = kernel_size // 2
        
        # 1. 1x1 Channel reduction
        self.reduce_conv = nn.Conv2d(total_channels, hidden_dim, kernel_size=1, bias=False)
        self.reduce_bn = nn.BatchNorm2d(hidden_dim)
        self.reduce_relu = nn.ReLU(inplace=True)
        
        # 2. KxK Depthwise spatial context
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, groups=hidden_dim, bias=False)
        self.dw_bn = nn.BatchNorm2d(hidden_dim)
        self.dw_relu = nn.ReLU(inplace=True)
        
        # 3. 1x1 Gate output (2 channels: g_B, g_S)
        self.gate_conv = nn.Conv2d(hidden_dim, 2, kernel_size=1, bias=True)
        
        # Exact Identity Initialization
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.zeros_(self.gate_conv.bias)
        
        self.last_gate = None

    def forward(self, b_stream: torch.Tensor, s_stream: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t2 = torch.cat([b_stream, s_stream], dim=1)
        h = self.reduce_relu(self.reduce_bn(self.reduce_conv(t2)))
        h_spatial = self.dw_relu(self.dw_bn(self.dw_conv(h)))
        a = self.gate_conv(h_spatial)
        g = 2.0 * torch.sigmoid(a)
        self.last_gate = g
        
        g_b = g[:, 0:1, :, :]
        g_s = g[:, 1:2, :, :]
        
        b_prime = g_b * b_stream
        s_prime = g_s * s_stream
        
        return torch.cat([b_prime, s_prime], dim=1), g


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


def compute_arm_b_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gates: torch.Tensor,
    neck_masks_56: Optional[torch.Tensor],
    lambda_aux: float = 0.1,
    lambda_id: float = 0.05,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    l_seg = bce + dice
    
    crack_56 = (F.max_pool2d(targets, 8, 8) > 0.5).float()
    
    # 1. True crack region: target gate = 1.0
    crack_count = crack_56.sum()
    if crack_count > 0:
        l_crack = ((gates - 1.0) ** 2 * crack_56).sum() / (crack_count * 2.0 + 1e-6)
    else:
        l_crack = torch.tensor(0.0, device=gates.device)
        
    # 2. False bridge neck corridor: target gate = 0.0
    if neck_masks_56 is not None and neck_masks_56.sum() > 0:
        neck_count = neck_masks_56.sum()
        l_neck = (gates ** 2 * neck_masks_56).sum() / (neck_count * 2.0 + 1e-6)
    else:
        l_neck = torch.tensor(0.0, device=gates.device)
        
    # 3. Mild minimal-deviation / identity penalty: anchor non-neck regions to 1.0
    l_id = ((gates - 1.0) ** 2).mean()
    
    l_aux = l_crack + l_neck + lambda_id * l_id
    l_total = l_seg + lambda_aux * l_aux
    
    metrics = {
        'loss_bce': bce.item(),
        'loss_dice': dice.item(),
        'loss_seg': l_seg.item(),
        'loss_crack': l_crack.item() if isinstance(l_crack, torch.Tensor) else l_crack,
        'loss_neck': l_neck.item() if isinstance(l_neck, torch.Tensor) else l_neck,
        'loss_id': l_id.item(),
        'loss_aux': l_aux.item(),
        'loss_total': l_total.item(),
    }
    return l_total, metrics


# ---------------------------------------------------------------------------
# Dataset with Random 448x448 Cropping & Aligned Neck Masks
# ---------------------------------------------------------------------------

class Crack500TrainDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        neck_coords_path: Optional[str] = None,
        img_size: int = 448,
        seed: int = 42,
    ):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.rng = np.random.RandomState(seed)
        
        self.neck_coords = {}
        if neck_coords_path and os.path.exists(neck_coords_path):
            self.neck_coords = torch.load(neck_coords_path, map_location='cpu')
            print(f"Loaded {len(self.neck_coords)} images with neck coordinates for training.")

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
        neck_mask = np.zeros((H, W), dtype=np.uint8)
        if stem in self.neck_coords:
            ys = self.neck_coords[stem]['ys']
            xs = self.neck_coords[stem]['xs']
            neck_mask[ys, xs] = 1
            
        if H < self.img_size or W < self.img_size:
            pad_h = max(0, self.img_size - H)
            pad_w = max(0, self.img_size - W)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            neck_mask = cv2.copyMakeBorder(neck_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            H, W = mask.shape
            
        max_y = H - self.img_size
        max_x = W - self.img_size
        y0 = self.rng.randint(0, max_y + 1) if max_y > 0 else 0
        x0 = self.rng.randint(0, max_x + 1) if max_x > 0 else 0
        
        crop_img = img[y0:y0+self.img_size, x0:x0+self.img_size]
        crop_mask = mask[y0:y0+self.img_size, x0:x0+self.img_size]
        crop_neck = neck_mask[y0:y0+self.img_size, x0:x0+self.img_size]
        
        if self.rng.rand() > 0.5:
            crop_img = np.fliplr(crop_img).copy()
            crop_mask = np.fliplr(crop_mask).copy()
            crop_neck = np.fliplr(crop_neck).copy()
        if self.rng.rand() > 0.5:
            crop_img = np.flipud(crop_img).copy()
            crop_mask = np.flipud(crop_mask).copy()
            crop_neck = np.flipud(crop_neck).copy()
            
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop_img = (crop_img - self.mean) / self.std
        crop_img = crop_img.transpose(2, 0, 1) # (3, H, W)
        
        crop_mask = (crop_mask > 127).astype(np.float32)[np.newaxis, :, :] # (1, H, W)
        
        # Downsample neck mask to 56x56 for T2 feature grid
        crop_neck_56 = cv2.resize(crop_neck, (56, 56), interpolation=cv2.INTER_NEAREST)
        crop_neck_56 = crop_neck_56.astype(np.float32)[np.newaxis, :, :] # (1, 56, 56)
        
        return {
            'image': torch.from_numpy(crop_img).float(),
            'mask': torch.from_numpy(crop_mask).float(),
            'neck_56': torch.from_numpy(crop_neck_56).float(),
            'stem': stem
        }


# ---------------------------------------------------------------------------
# Canonical Setting A Tiling Evaluation
# ---------------------------------------------------------------------------

def predict_setting_a(model: nn.Module, img_bgr: np.ndarray, device: torch.device, tile_size: int = 448) -> np.ndarray:
    H, W, _ = img_bgr.shape
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    
    pad_h = (tile_size - (H % tile_size)) % tile_size
    pad_w = (tile_size - (W % tile_size)) % tile_size
    
    img_padded = cv2.copyMakeBorder(img_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    H_pad, W_pad, _ = img_padded.shape
    
    img_norm = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_norm = (img_norm - mean) / std
    
    prob_map = np.zeros((H_pad, W_pad), dtype=np.float32)
    
    model.eval()
    with torch.no_grad():
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
    desc: str = "Evaluating"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    
    for img_p in tqdm(val_img_paths, desc=desc, ncols=80):
        stem = os.path.splitext(os.path.basename(img_p))[0]
        mask_p = os.path.join(val_mask_dir, f'{stem}.png')
        
        img = cv2.imread(img_p)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        gt_binary = (mask > 127).astype(np.uint8)
        
        prob = predict_setting_a(model, img, device)
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
# Training Loop
# ---------------------------------------------------------------------------

def train_csdg_k3(
    model: nn.Module,
    csdg: T2CSDG,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 8,
    lr: float = 1e-4,
    lambda_aux: float = 0.1,
    out_dir: str = 'results/diagnostics/phase6_t2_csdg',
) -> Dict[str, Any]:
    print(f"\nTraining T2-CSDG (K=3, lambda_aux={lambda_aux}, epochs={epochs}, batch=14)...")
    
    # Freeze candidate B model completely
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
        
    # Optimizer strictly on CSDG parameters
    optimizer = torch.optim.AdamW(csdg.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Setup hooked forward on Decoder Block 1
    dec_b1 = model.decoder.decoder_blocks[1]
    
    def hooked_forward(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = dec_b1.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        t2_prime, g = csdg(x, skip)
        x = dec_b1.conv1(t2_prime)
        x = dec_b1.conv2(x)
        return x
        
    dec_b1.forward = hooked_forward
    
    logs = []
    
    for ep in range(1, epochs + 1):
        csdg.train()
        ep_loss_total = 0.0
        ep_loss_seg = 0.0
        ep_loss_aux = 0.0
        n_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"CSDG-K3 Ep {ep}/{epochs}", ncols=90)
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            necks_56 = batch['neck_56'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(images)
            gates = csdg.last_gate
            
            loss, m = compute_arm_b_loss(
                logits=logits,
                targets=masks,
                gates=gates,
                neck_masks_56=necks_56,
                lambda_aux=lambda_aux,
            )
            
            loss.backward()
            optimizer.step()
            
            ep_loss_total += m['loss_total']
            ep_loss_seg += m['loss_seg']
            ep_loss_aux += m['loss_aux']
            
            pbar.set_postfix({
                'tot': f"{m['loss_total']:.3f}",
                'seg': f"{m['loss_seg']:.3f}",
                'aux': f"{m['loss_aux']:.3f}",
            })
            
        scheduler.step()
        
        # Epoch level gate modulation statistics
        with torch.no_grad():
            last_g = csdg.last_gate.detach().cpu()
            g_b_mean = float(last_g[:, 0].mean())
            g_s_mean = float(last_g[:, 1].mean())
            g_min = float(last_g.min())
            g_max = float(last_g.max())
            
        avg_total = ep_loss_total / n_batches
        avg_seg = ep_loss_seg / n_batches
        avg_aux = ep_loss_aux / n_batches
        
        print(f"  Ep {ep}/{epochs} Summary: Total={avg_total:.4f}, Seg={avg_seg:.4f}, Aux={avg_aux:.4f} | g_B={g_b_mean:.3f}, g_S={g_s_mean:.3f}, range=[{g_min:.3f}, {g_max:.3f}]")
        
        logs.append({
            'model': 'CSDG_K3',
            'epoch': ep,
            'loss_total': avg_total,
            'loss_seg': avg_seg,
            'loss_aux': avg_aux,
            'g_b_mean': g_b_mean,
            'g_s_mean': g_s_mean,
            'g_min': g_min,
            'g_max': g_max,
        })
        
    final_path = os.path.join(out_dir, 'csdg_k3_final.pth')
    torch.save(csdg.state_dict(), final_path)
    print(f"Saved final CSDG-K3 weights to {final_path}")
    
    return {'logs': logs, 'csdg_state': csdg.state_dict()}


# ---------------------------------------------------------------------------
# Spatial Gate Distribution Audit (Crack vs Bridge Corridor vs Background)
# ---------------------------------------------------------------------------

def audit_gate_spatial_distribution(
    model: nn.Module,
    csdg: T2CSDG,
    val_img_paths: List[str],
    val_mask_dir: str,
    device: torch.device,
    stems_to_audit: set[str],
    out_dir: str,
) -> pd.DataFrame:
    print("\nAuditing spatial gate distribution across Crack vs Neck vs Background...")
    
    dec_b1 = model.decoder.decoder_blocks[1]
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    
    audit_rows = []
    
    for img_p in val_img_paths:
        stem = os.path.splitext(os.path.basename(img_p))[0]
        if stem not in stems_to_audit:
            continue
            
        mask_p = os.path.join(val_mask_dir, f'{stem}.png')
        img = cv2.imread(img_p)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        H, W = mask.shape
        gt_binary = (mask > 127).astype(np.uint8)
        
        # Run tile-based or direct 448 center crop for feature gate extraction
        # Since T2 is 56x56, let's take a 448x448 crop centered on the first bridge if possible
        # Or standard canonical first tile [0:448, 0:448]
        crop_img = img[:448, :448]
        crop_gt = gt_binary[:448, :448]
        if crop_img.shape[0] < 448 or crop_img.shape[1] < 448:
            crop_img = cv2.copyMakeBorder(crop_img, 0, 448 - crop_img.shape[0], 0, 448 - crop_img.shape[1], cv2.BORDER_REFLECT)
            crop_gt = cv2.copyMakeBorder(crop_gt, 0, 448 - crop_gt.shape[0], 0, 448 - crop_gt.shape[1], cv2.BORDER_REFLECT)
            
        norm_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm_img = (norm_img - mean) / std
        input_t = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        
        model.eval()
        with torch.no_grad():
            logits = model(input_t)
            g = csdg.last_gate.squeeze(0).cpu().numpy() # (2, 56, 56)
            g_b = g[0]
            g_s = g[1]
            pred_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            
        # Ground truth downsampled to 56x56
        gt_56 = cv2.resize(crop_gt, (56, 56), interpolation=cv2.INTER_NEAREST)
        crack_mask = (gt_56 > 0)
        bg_mask = (gt_56 == 0)
        
        # Check false-bridge neck corridor if available
        pred_binary = (pred_prob >= 0.5).astype(np.uint8)
        pred_56 = cv2.resize(pred_binary, (56, 56), interpolation=cv2.INTER_NEAREST)
        
        # False bridge corridor region: pred=1 & gt=0
        fp_mask = (pred_56 > 0) & bg_mask
        
        row = {
            'image_id': stem,
            'g_b_crack_mean': float(g_b[crack_mask].mean()) if crack_mask.any() else np.nan,
            'g_b_crack_median': float(np.median(g_b[crack_mask])) if crack_mask.any() else np.nan,
            'g_b_corridor_mean': float(g_b[fp_mask].mean()) if fp_mask.any() else np.nan,
            'g_b_corridor_median': float(np.median(g_b[fp_mask])) if fp_mask.any() else np.nan,
            'g_b_bg_mean': float(g_b[bg_mask].mean()) if bg_mask.any() else np.nan,
            
            'g_s_crack_mean': float(g_s[crack_mask].mean()) if crack_mask.any() else np.nan,
            'g_s_crack_median': float(np.median(g_s[crack_mask])) if crack_mask.any() else np.nan,
            'g_s_corridor_mean': float(g_s[fp_mask].mean()) if fp_mask.any() else np.nan,
            'g_s_corridor_median': float(np.median(g_s[fp_mask])) if fp_mask.any() else np.nan,
            'g_s_bg_mean': float(g_s[bg_mask].mean()) if bg_mask.any() else np.nan,
        }
        audit_rows.append(row)
        
    df_audit = pd.DataFrame(audit_rows)
    audit_csv = os.path.join(out_dir, 'gate_spatial_distribution_audit.csv')
    df_audit.to_csv(audit_csv, index=False)
    print(f"Saved spatial gate audit to {audit_csv}")
    
    print("\n--- Gate Spatial Distribution Summary (Across Audited Bridge Cohort) ---")
    print(f"Crack Region:     g_B mean={df_audit['g_b_crack_mean'].mean():.4f}, g_S mean={df_audit['g_s_crack_mean'].mean():.4f}")
    print(f"Bridge Corridor:  g_B mean={df_audit['g_b_corridor_mean'].mean():.4f}, g_S mean={df_audit['g_s_corridor_mean'].mean():.4f}")
    print(f"Background:       g_B mean={df_audit['g_b_bg_mean'].mean():.4f}, g_S mean={df_audit['g_s_bg_mean'].mean():.4f}")
    print(f"Selectivity Ratio (Crack / Corridor): g_B={df_audit['g_b_crack_mean'].mean() / df_audit['g_b_corridor_mean'].mean():.4f}, g_S={df_audit['g_s_crack_mean'].mean() / df_audit['g_s_corridor_mean'].mean():.4f}")
    print("------------------------------------------------------------------------\n")
    
    return df_audit


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Phase 6: T2-CSDG (K=3) Intervention Experiment & Spatial Evaluation")
    print("=" * 80)
    
    config_path = 'results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml'
    ckpt_path   = 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'
    train_neck_coords = 'results/diagnostics/phase6_t2_sdsg/train_neck_coords.pt'
    out_dir = 'results/diagnostics/phase6_t2_csdg'
    os.makedirs(out_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Dataset & Dataloader setup (batch 14, seed 42)
    train_img_dir  = 'datasets/Crack500_ready/train/images'
    train_mask_dir = 'datasets/Crack500_ready/train/masks'
    val_img_dir    = 'datasets/Crack500_ready/val/images'
    val_mask_dir   = 'datasets/Crack500_ready/val/masks'
    
    val_img_paths = sorted(glob.glob(os.path.join(val_img_dir, '*.jpg')))
    assert len(val_img_paths) == 348, f"Expected 348 validation images, got {len(val_img_paths)}"
    
    train_dataset = Crack500TrainDataset(
        img_dir=train_img_dir,
        mask_dir=train_mask_dir,
        neck_coords_path=train_neck_coords,
        img_size=448,
        seed=42,
    )
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
    models = ['Base', 'A1', 'A2', 'B1', 'C1', 'D1', 'D2']
    bcols = [f'{m}_bridge_events' for m in models]
    c7_stems = set(df_master[(df_master[bcols] > 0).all(axis=1)]['case_name'].tolist())
    clean_stems = set(df_master[(df_master[bcols] == 0).all(axis=1) & (df_master['gt_cc'] >= 2)]['case_name'].tolist())
    
    ac_atten_csv = 'results/diagnostics/phase6_stem_ac_attenuation/ac_attenuation_per_sample_alpha.csv'
    df_prev = pd.read_csv(ac_atten_csv)
    c7a0 = df_prev[(df_prev['cohort'] == 'Consensus_7of7') & (df_prev['alpha'] == 0.0)]
    resistant_stems = set(c7a0[c7a0['has_bridge'] == True]['image_id'].tolist())
    sensitive_stems = set(c7a0[c7a0['has_bridge'] == False]['image_id'].tolist())
    
    # 2. Setup CSDG (K=3) & Model
    model = load_model_from_checkpoint(config_path, ckpt_path, device=device)
    csdg_k3 = T2CSDG(in_channels=192, skip_channels=96, hidden_dim=64, kernel_size=3).to(device)
    final_pth = os.path.join(out_dir, 'csdg_k3_final.pth')
    eval_csv = os.path.join(out_dir, 'validation_csdg_k3_per_sample.csv')
    
    if os.path.exists(final_pth):
        print(f"Found existing CSDG-K3 weights at {final_pth}, loading...")
        csdg_k3.load_state_dict(torch.load(final_pth, map_location=device))
    else:
        train_res = train_csdg_k3(
            model=model,
            csdg=csdg_k3,
            train_loader=train_loader,
            device=device,
            epochs=8,
            lr=1e-4,
            lambda_aux=0.1,
            out_dir=out_dir,
        )
        # Save training logs
        df_logs = pd.DataFrame(train_res['logs'])
        df_logs.to_csv(os.path.join(out_dir, 'csdg_k3_training_log.csv'), index=False)
    
    # Hook forward
    dec_b1 = model.decoder.decoder_blocks[1]
    def hooked_forward(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = dec_b1.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        t2_prime, g = csdg_k3(x, skip)
        x = dec_b1.conv1(t2_prime)
        x = dec_b1.conv2(x)
        return x
    dec_b1.forward = hooked_forward
    
    # 3. Full Validation Evaluation (Setting A, N=348)
    if os.path.exists(eval_csv):
        print(f"Found existing validation evaluation at {eval_csv}, loading...")
        df_eval = pd.read_csv(eval_csv)
    else:
        print("\n" + "=" * 80)
        print("Evaluating T2-CSDG (K=3) on Full Validation N=348 (Setting A)...")
        print("=" * 80)
        df_eval, sum_eval = evaluate_on_validation_cohort(
            model, val_img_paths, val_mask_dir, device, desc="CSDG-K3 Eval"
        )
        df_eval.to_csv(eval_csv, index=False)
    
    # 4. Spatial Gate Distribution Audit
    df_audit = audit_gate_spatial_distribution(
        model=model,
        csdg=csdg_k3,
        val_img_paths=val_img_paths,
        val_mask_dir=val_mask_dir,
        device=device,
        stems_to_audit=c7_stems,
        out_dir=out_dir,
    )
    
    # 5. Comparative Subgroup Analysis (Baseline Candidate B vs Arm B SDSG vs CSDG K=3)
    df_base = pd.read_csv('results/diagnostics/phase6_t2_sdsg/validation_baseline_per_sample.csv')
    df_sdsg_b = pd.read_csv('results/diagnostics/phase6_t2_sdsg/validation_arm_B_per_sample.csv')
    
    cohorts = [
        ('Full_Validation_348', set(df_base['image_id'])),
        ('Consensus_7of7_101', c7_stems),
        ('Consensus_Resistant_82', resistant_stems),
        ('Consensus_Sensitive_19', sensitive_stems),
        ('Clean_Control_56', clean_stems)
    ]
    
    rows_comp = []
    for model_name, df_m in [
        ('Baseline_Candidate_B', df_base),
        ('Arm_B_SDSG_1x1', df_sdsg_b),
        ('T2_CSDG_K3', df_eval),
    ]:
        for cohort_name, stems in cohorts:
            sub_m = df_m[df_m['image_id'].isin(stems)]
            sub_b = df_base[df_base['image_id'].isin(stems)]
            
            b_ev = int(sub_m['bridge_events'].sum())
            base_b_ev = int(sub_b['bridge_events'].sum())
            b_img = int((sub_m['bridge_events'] > 0).sum())
            base_b_img = int((sub_b['bridge_events'] > 0).sum())
            
            merged = sub_b[['image_id', 'bridge_events']].merge(sub_m[['image_id', 'bridge_events']], on='image_id', suffixes=('_base', '_eval'))
            cured_img = int(((merged['bridge_events_base'] > 0) & (merged['bridge_events_eval'] == 0)).sum())
            created_img = int(((merged['bridge_events_base'] == 0) & (merged['bridge_events_eval'] > 0)).sum())
            persist_img = int(((merged['bridge_events_base'] > 0) & (merged['bridge_events_eval'] > 0)).sum())
            
            brk_ev = int(sub_m['fragmented_gt_components'].sum())
            base_brk_ev = int(sub_b['fragmented_gt_components'].sum())
            
            area_ratio = float((sub_m['pred_area'] / sub_m['gt_area'].clip(lower=1)).mean())
            
            rows_comp.append({
                'model': model_name,
                'cohort': cohort_name,
                'n_samples': len(stems),
                'dice': float(sub_m['dice'].mean()),
                'recall': float(sub_m['recall'].mean()),
                'precision': float(sub_m['precision'].mean()),
                'cldice': float(sub_m['cldice'].mean()),
                'tsens': float(sub_m['tsens'].mean()),
                'bridge_images': b_img,
                'bridge_images_diff': b_img - base_b_img,
                'bridge_events': b_ev,
                'bridge_events_diff': b_ev - base_b_ev,
                'cured_images': cured_img,
                'created_images': created_img,
                'persistent_images': persist_img,
                'breakage_events': brk_ev,
                'breakage_events_diff': brk_ev - base_brk_ev,
                'area_ratio': area_ratio,
            })
            
    df_comparison = pd.DataFrame(rows_comp)
    comp_path = os.path.join(out_dir, 'validation_subgroup_comparison.csv')
    df_comparison.to_csv(comp_path, index=False)
    print(f"Saved comparison to {comp_path}")
    
    # Print formatted table
    print("\n" + "=" * 108)
    print(f"{'Model':<22} {'Cohort':<22} | {'Dice':>6} {'Recall':>6} {'clDice':>6} | {'BrImgs':>6} {'Diff':>5} {'BrEvts':>6} {'Diff':>5} | {'Break':>5} {'Diff':>5} | {'AreaR':>6}")
    print("-" * 108)
    for _, r in df_comparison.iterrows():
        print(f"{r['model']:<22} {r['cohort']:<22} | {r['dice']:>6.4f} {r['recall']:>6.4f} {r['cldice']:>6.4f} | {r['bridge_images']:>6} {r['bridge_images_diff']:>+5} {r['bridge_events']:>6} {r['bridge_events_diff']:>+5} | {r['breakage_events']:>5} {r['breakage_events_diff']:>+5} | {r['area_ratio']:>6.4f}")
    print("=" * 108)
    
    print("\nPhase 6 T2-CSDG (K=3) Pipeline Completed Successfully!")


if __name__ == '__main__':
    main()
