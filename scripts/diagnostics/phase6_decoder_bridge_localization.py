#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_decoder_bridge_localization.py

Task C: Decoder-Stage Bridge Localization Diagnostic
Answers:
- Where in the network hierarchy is the false bridge committed?
- Does the bridge signal originate in:
    1. S3 / Bottleneck (14x14) - deep ViT/CNN representation
    2. S2 Decoder Block 0 (28x28) - 1st upsampling stage
    3. S1 Decoder Block 1 (56x56) - 2nd upsampling stage
    4. S0 Decoder Block 2 (112x112) - shallowest decoder stage
    5. Pre-logit Head (112x112) - 1x1 conv before interpolation
    6. Final Logits (448x448) - 4x bilinear upsample
- Is the bridge connector already positive at the 112x112 pre-upsample stage,
  or is it created by the 4x bilinear interpolation?

Sample sets (N=25 representative deterministic cases):
- 10 Consensus 7/7 cases (seed=42)
- 5 Base->D2 cured cases
- 5 Base-clean->D2 created cases
- 5 Low-consensus cases (from 1/7, 2/7)
"""

import gc
import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, 'SAGE_LITE')
from tools.run_phase6_c_topology_diagnostic import load_model_from_checkpoint

def isolate_bridge_connector_neck(pred_bin: np.ndarray, target_bin: np.ndarray):
    """
    Isolates connector neck mask and crack mask.
    """
    num_gt_cc, gt_labels = cv2.connectedComponents(target_bin.astype(np.uint8), connectivity=8)
    num_pred_cc, pred_labels = cv2.connectedComponents(pred_bin.astype(np.uint8), connectivity=8)
    pred_cc = int(max(num_pred_cc - 1, 0))

    bridge_fp_mask = np.zeros_like(target_bin, dtype=np.uint8)
    connector_neck_mask = np.zeros_like(target_bin, dtype=np.uint8)

    for p_id in range(1, pred_cc + 1):
        p_mask = (pred_labels == p_id)
        overlapping_gt = np.unique(gt_labels[p_mask])
        overlapping_gt = overlapping_gt[overlapping_gt > 0]

        if len(overlapping_gt) < 2:
            continue

        cc_fp = p_mask & (target_bin == 0)
        bridge_fp_mask = bridge_fp_mask | cc_fp.astype(np.uint8)

        gt_list = list(overlapping_gt)
        for i_idx in range(len(gt_list)):
            gi = gt_list[i_idx]
            mask_i = (gt_labels == gi)
            dt_i = distance_transform_edt(1 - mask_i)

            for j_idx in range(i_idx + 1, len(gt_list)):
                gj = gt_list[j_idx]
                mask_j = (gt_labels == gj)
                gap_dist = float(np.min(dt_i[mask_j]))
                radius = max(int(np.ceil(gap_dist / 2.0)) + 2, 3)

                ksize = 2 * radius + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

                dil_i = cv2.dilate(mask_i.astype(np.uint8), kernel)
                dil_j = cv2.dilate(mask_j.astype(np.uint8), kernel)

                corridor = (dil_i > 0) & (dil_j > 0)
                neck = cc_fp & corridor
                connector_neck_mask = connector_neck_mask | neck.astype(np.uint8)

    if np.sum(bridge_fp_mask) > 0 and np.sum(connector_neck_mask) == 0:
        connector_neck_mask = bridge_fp_mask.copy()

    return bridge_fp_mask, connector_neck_mask

def extract_multi_scale_decoder_features(model, image, device, tile_size=448):
    """
    Runs Setting A tiling while hooking intermediate decoder feature maps.
    Reconstructs full-image feature activations at:
    - S3: Bottleneck (scale 1/32)
    - S2: DecoderBlock 0 (scale 1/16)
    - S1: DecoderBlock 1 (scale 1/8)
    - S0: DecoderBlock 2 (scale 1/4)
    - Head112: Pre-upsample logits (scale 1/4)
    - Final448: Final logits (scale 1)
    """
    H, W = image.shape[:2]
    pad_h = (tile_size - (H % tile_size)) % tile_size
    pad_w = (tile_size - (W % tile_size)) % tile_size
    padded_img = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    pH, pW = padded_img.shape[:2]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    patches = []
    coords = []
    for y in range(0, pH, tile_size):
        for x in range(0, pW, tile_size):
            patch = padded_img[y:y+tile_size, x:x+tile_size]
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            patch_tensor = (patch_tensor - mean) / std
            patches.append(patch_tensor)
            coords.append((y, x))

    # Reconstructed feature grids (mean absolute activation across channels)
    act_s3 = np.zeros((pH // 32, pW // 32), dtype=np.float32)
    act_s2 = np.zeros((pH // 16, pW // 16), dtype=np.float32)
    act_s1 = np.zeros((pH // 8, pW // 8), dtype=np.float32)
    act_s0 = np.zeros((pH // 4, pW // 4), dtype=np.float32)
    logits_head112 = np.zeros((pH // 4, pW // 4), dtype=np.float32)
    logits_final448 = np.zeros((pH, pW), dtype=np.float32)

    # Hooks
    cache = {}

    def hook_bottleneck(module, inp, out):
        # inp[0] is bottleneck feature map (B, 384, 14, 14)
        cache['s3'] = inp[0].detach()

    def hook_block0(module, inp, out):
        cache['s2'] = out.detach()  # (B, 192, 28, 28)

    def hook_block1(module, inp, out):
        cache['s1'] = out.detach()  # (B, 96, 56, 56)

    def hook_block2(module, inp, out):
        cache['s0'] = out.detach()  # (B, 48, 112, 112)

    def hook_head(module, inp, out):
        cache['head112'] = out.detach()  # (B, 1, 112, 112)

    h_s3 = model.decoder.decoder_blocks[0].register_forward_hook(hook_bottleneck)
    h_s2 = model.decoder.decoder_blocks[0].register_forward_hook(hook_block0)
    h_s1 = model.decoder.decoder_blocks[1].register_forward_hook(hook_block1)
    h_s0 = model.decoder.decoder_blocks[2].register_forward_hook(hook_block2)
    h_hd = model.decoder.segmentation_head.register_forward_hook(hook_head)

    batch_size = 4
    try:
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch = torch.stack(patches[i:i+batch_size]).to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    out_logits = model(batch)

                b_final = out_logits.squeeze(1).float().cpu().numpy()
                b_s3 = cache['s3'].abs().mean(dim=1).float().cpu().numpy()  # (B, 14, 14)
                b_s2 = cache['s2'].abs().mean(dim=1).float().cpu().numpy()  # (B, 28, 28)
                b_s1 = cache['s1'].abs().mean(dim=1).float().cpu().numpy()  # (B, 56, 56)
                b_s0 = cache['s0'].abs().mean(dim=1).float().cpu().numpy()  # (B, 112, 112)
                b_hd = cache['head112'].squeeze(1).float().cpu().numpy()     # (B, 112, 112)

                for j in range(len(b_final)):
                    y, x = coords[i + j]
                    # S3
                    act_s3[y//32 : y//32 + 14, x//32 : x//32 + 14] = b_s3[j]
                    # S2
                    act_s2[y//16 : y//16 + 28, x//16 : x//16 + 28] = b_s2[j]
                    # S1
                    act_s1[y//8 : y//8 + 56, x//8 : x//8 + 56] = b_s1[j]
                    # S0
                    act_s0[y//4 : y//4 + 112, x//4 : x//4 + 112] = b_s0[j]
                    # Head112
                    logits_head112[y//4 : y//4 + 112, x//4 : x//4 + 112] = b_hd[j]
                    # Final448
                    logits_final448[y : y + 448, x : x + 448] = b_final[j]
    finally:
        h_s3.remove()
        h_s2.remove()
        h_s1.remove()
        h_s0.remove()
        h_hd.remove()

    # Crop back to original dimensions
    h_s3_dim, w_s3_dim = H // 32, W // 32
    h_s2_dim, w_s2_dim = H // 16, W // 16
    h_s1_dim, w_s1_dim = H // 8, W // 8
    h_s0_dim, w_s0_dim = H // 4, W // 4

    return {
        's3': act_s3[:h_s3_dim, :w_s3_dim],
        's2': act_s2[:h_s2_dim, :w_s2_dim],
        's1': act_s1[:h_s1_dim, :w_s1_dim],
        's0': act_s0[:h_s0_dim, :w_s0_dim],
        'head112': logits_head112[:h_s0_dim, :w_s0_dim],
        'final448': logits_final448[:H, :W],
    }

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    out_dir = 'results/diagnostics/phase6_bridge_localization_v2'
    os.makedirs(out_dir, exist_ok=True)

    config_path = 'results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml'
    ckpt_path = 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'
    consensus_csv = 'results/diagnostics/phase6_bridge_localization/bridge_consensus_per_image.csv'
    df_meta = pd.read_csv(consensus_csv)

    # 1. Deterministic Selection of N=25 representative samples
    rng = np.random.RandomState(42)

    # 10 Consensus 7/7
    c7_all = sorted(df_meta[df_meta['n_models_with_bridge'] == 7]['image_id'].tolist())
    c7_sel = sorted(list(rng.choice(c7_all, size=10, replace=False)))

    # 5 Cured by D2
    cured_all = sorted(df_meta[df_meta['d2_transition_type'] == 'Cured_by_D2']['image_id'].tolist())
    cured_sel = cured_all[:5]

    # 5 Created by D2
    created_all = sorted(df_meta[df_meta['d2_transition_type'] == 'Created_by_D2']['image_id'].tolist())
    created_sel = created_all[:5]

    # 5 Low consensus (1/7 or 2/7)
    low_all = sorted(df_meta[df_meta['n_models_with_bridge'].isin([1, 2])]['image_id'].tolist())
    # Exclude any already in cured_sel
    low_avail = [c for c in low_all if c not in cured_sel]
    if len(low_avail) < 5:
        low_3 = sorted(df_meta[df_meta['n_models_with_bridge'] == 3]['image_id'].tolist())
        low_avail.extend(low_3)
    low_sel = low_avail[:5]

    selected_samples = []
    for c in c7_sel:
        selected_samples.append((c, 'Consensus_7of7'))
    for c in cured_sel:
        selected_samples.append((c, 'Cured_by_D2'))
    for c in created_sel:
        selected_samples.append((c, 'Created_by_D2'))
    for c in low_sel:
        selected_samples.append((c, 'Low_Consensus'))

    assert len(selected_samples) == 25, f"Expected 25 samples, got {len(selected_samples)}"
    print(f"Selected 25 representative evaluation samples:")
    for c, g in selected_samples:
        print(f"  [{g:16s}] {c}")

    data_root = 'datasets/Crack500_ready'
    val_img_dir = os.path.join(data_root, 'val', 'images')
    val_mask_dir = os.path.join(data_root, 'val', 'masks')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = False

    print("\nLoading Candidate B model for Decoder Feature Map Extraction...")
    model = load_model_from_checkpoint(config_path, ckpt_path, device)
    model.eval()

    per_sample_rows = []

    for case_name, grp in tqdm(selected_samples, desc="Decoder Feature Extraction"):
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
        H, W = target_bin.shape

        feats = extract_multi_scale_decoder_features(model, img, device, tile_size=448)
        final_logits = feats['final448']
        pred_bin = (final_logits > 0.0).astype(np.uint8)

        # Connector neck mask on full resolution
        whole_bridge_fp, neck_mask = isolate_bridge_connector_neck(pred_bin, target_bin)
        crack_tp_mask = (pred_bin == 1) & (target_bin == 1)
        bg_mask = (pred_bin == 0) & (target_bin == 0)

        has_neck = (np.sum(neck_mask) > 0)
        # If created_by_D2, Base doesn't have bridge, so fallback neck is GT gap corridor
        if not has_neck and grp == 'Created_by_D2':
            # Use whole target background near GT
            num_gt, gt_lbls = cv2.connectedComponents(target_bin, connectivity=8)
            if num_gt > 2:
                # Find corridor between closest pair of GT CCs
                m1 = (gt_lbls == 1)
                m2 = (gt_lbls == 2)
                dt1 = distance_transform_edt(1 - m1)
                gap = float(np.min(dt1[m2]))
                rad = max(int(np.ceil(gap / 2.0)) + 2, 3)
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*rad+1, 2*rad+1))
                d1 = cv2.dilate(m1.astype(np.uint8), k)
                d2 = cv2.dilate(m2.astype(np.uint8), k)
                neck_mask = (d1 > 0) & (d2 > 0) & (target_bin == 0)

        # Evaluate at each stage
        stages = [
            ('S3_Bottleneck_14x14', feats['s3'], 32, False),
            ('S2_DecoderBlock0_28x28', feats['s2'], 16, False),
            ('S1_DecoderBlock1_56x56', feats['s1'], 8, False),
            ('S0_DecoderBlock2_112x112', feats['s0'], 4, False),
            ('Head_PreUpsample_112x112', feats['head112'], 4, True),
            ('Final_Logits_448x448', feats['final448'], 1, True),
        ]

        row = {
            'image_id': case_name,
            'stratum': grp,
            'has_base_bridge': int(np.sum(whole_bridge_fp) > 0),
            'neck_pixels_448': int(np.sum(neck_mask)),
        }

        for s_name, f_map, scale, is_logit in stages:
            h_s, w_s = f_map.shape
            neck_s = (cv2.resize(neck_mask.astype(np.float32), (w_s, h_s), interpolation=cv2.INTER_AREA) > 0.05)
            crack_s = (cv2.resize(crack_tp_mask.astype(np.float32), (w_s, h_s), interpolation=cv2.INTER_AREA) > 0.05)
            bg_s = (cv2.resize(bg_mask.astype(np.float32), (w_s, h_s), interpolation=cv2.INTER_AREA) > 0.5)

            if not is_logit:
                # Activation magnitudes
                neck_val = float(np.mean(f_map[neck_s])) if np.sum(neck_s) > 0 else np.nan
                crack_val = float(np.mean(f_map[crack_s])) if np.sum(crack_s) > 0 else np.nan
                bg_val = float(np.mean(f_map[bg_s])) if np.sum(bg_s) > 0 else np.nan
                contrast = float(neck_val / (bg_val + 1e-6)) if not np.isnan(neck_val) and not np.isnan(bg_val) else np.nan
                crack_ratio = float(neck_val / (crack_val + 1e-6)) if not np.isnan(neck_val) and not np.isnan(crack_val) else np.nan

                row[f'{s_name}_neck_act'] = round(neck_val, 4)
                row[f'{s_name}_crack_act'] = round(crack_val, 4)
                row[f'{s_name}_bg_act'] = round(bg_val, 4)
                row[f'{s_name}_contrast_neck_over_bg'] = round(contrast, 4)
                row[f'{s_name}_neck_over_crack_ratio'] = round(crack_ratio, 4)
            else:
                # Logits and Probabilities
                prob_map = 1.0 / (1.0 + np.exp(-f_map.astype(np.float64)))
                neck_p = prob_map[neck_s]
                crack_p = prob_map[crack_s]
                bg_p = prob_map[bg_s]

                p_mean = float(np.mean(neck_p)) if len(neck_p) > 0 else np.nan
                p_med = float(np.median(neck_p)) if len(neck_p) > 0 else np.nan
                pct_ge50 = float(np.sum(neck_p >= 0.50) / len(neck_p) * 100.0) if len(neck_p) > 0 else np.nan
                pct_ge75 = float(np.sum(neck_p >= 0.75) / len(neck_p) * 100.0) if len(neck_p) > 0 else np.nan

                row[f'{s_name}_neck_prob_mean'] = round(p_mean, 4)
                row[f'{s_name}_neck_prob_median'] = round(p_med, 4)
                row[f'{s_name}_neck_pct_ge50'] = round(pct_ge50, 2)
                row[f'{s_name}_neck_pct_ge75'] = round(pct_ge75, 2)
                row[f'{s_name}_crack_prob_median'] = round(float(np.median(crack_p)), 4) if len(crack_p) > 0 else np.nan
                row[f'{s_name}_bg_prob_median'] = round(float(np.median(bg_p)), 4) if len(bg_p) > 0 else np.nan

        per_sample_rows.append(row)

    df_per_sample = pd.DataFrame(per_sample_rows)
    out_per_sample_path = os.path.join(out_dir, 'decoder_bridge_localization_per_stage.csv')
    df_per_sample.to_csv(out_per_sample_path, index=False)
    print(f"Saved: {out_per_sample_path}")

    # Summary table across strata
    summary_rows = []
    strata = [
        ('Consensus_7of7_10cases', df_per_sample[df_per_sample['stratum'] == 'Consensus_7of7']),
        ('Cured_by_D2_5cases', df_per_sample[df_per_sample['stratum'] == 'Cured_by_D2']),
        ('Created_by_D2_5cases', df_per_sample[df_per_sample['stratum'] == 'Created_by_D2']),
        ('Low_Consensus_5cases', df_per_sample[df_per_sample['stratum'] == 'Low_Consensus']),
        ('All_25_Representative_Cases', df_per_sample),
    ]

    for s_name, sub in strata:
        summary_rows.append({
            'stratum': s_name,
            'N_samples': len(sub),
            'S3_14x14_contrast_neck_over_bg': round(sub['S3_Bottleneck_14x14_contrast_neck_over_bg'].mean(), 4),
            'S2_28x28_contrast_neck_over_bg': round(sub['S2_DecoderBlock0_28x28_contrast_neck_over_bg'].mean(), 4),
            'S1_56x56_contrast_neck_over_bg': round(sub['S1_DecoderBlock1_56x56_contrast_neck_over_bg'].mean(), 4),
            'S0_112x112_contrast_neck_over_bg': round(sub['S0_DecoderBlock2_112x112_contrast_neck_over_bg'].mean(), 4),
            'Head_112x112_neck_prob_median': round(sub['Head_PreUpsample_112x112_neck_prob_median'].mean(), 4),
            'Head_112x112_neck_pct_ge50': round(sub['Head_PreUpsample_112x112_neck_pct_ge50'].mean(), 2),
            'Head_112x112_neck_pct_ge75': round(sub['Head_PreUpsample_112x112_neck_pct_ge75'].mean(), 2),
            'Final_448x448_neck_prob_median': round(sub['Final_Logits_448x448_neck_prob_median'].mean(), 4),
            'Final_448x448_neck_pct_ge50': round(sub['Final_Logits_448x448_neck_pct_ge50'].mean(), 2),
            'Final_448x448_neck_pct_ge75': round(sub['Final_Logits_448x448_neck_pct_ge75'].mean(), 2),
        })

    df_summary = pd.DataFrame(summary_rows)
    out_summary_path = os.path.join(out_dir, 'decoder_bridge_localization_summary.csv')
    df_summary.to_csv(out_summary_path, index=False)
    print(f"Saved: {out_summary_path}")

    print("\n" + "=" * 80)
    print("TASK C DECODER STAGE LOCALIZATION SUMMARY:")
    print("=" * 80)
    for r in summary_rows:
        print(f"Stratum: {r['stratum']:30s} (N={r['N_samples']})")
        print(f"  Contrast (Neck / BG):")
        print(f"    - S3 (14x14):    {r['S3_14x14_contrast_neck_over_bg']:.4f}")
        print(f"    - S2 (28x28):    {r['S2_28x28_contrast_neck_over_bg']:.4f}")
        print(f"    - S1 (56x56):    {r['S1_56x56_contrast_neck_over_bg']:.4f}")
        print(f"    - S0 (112x112):  {r['S0_112x112_contrast_neck_over_bg']:.4f}")
        print(f"  Pre-Upsample (112x112 Grid) Probability:")
        print(f"    - Median Prob:   {r['Head_112x112_neck_prob_median']:.4f}")
        print(f"    - Neck >= 0.50:  {r['Head_112x112_neck_pct_ge50']:.1f}%")
        print(f"    - Neck >= 0.75:  {r['Head_112x112_neck_pct_ge75']:.1f}%")
        print(f"  Final Output (448x448 Grid) Probability:")
        print(f"    - Median Prob:   {r['Final_448x448_neck_prob_median']:.4f}")
        print(f"    - Neck >= 0.50:  {r['Final_448x448_neck_pct_ge50']:.1f}%")
        print(f"    - Neck >= 0.75:  {r['Final_448x448_neck_pct_ge75']:.1f}%")
        print("-" * 50)

if __name__ == '__main__':
    main()
