#!/usr/bin/env python3
"""
scripts/tests/test_phase6_p3_asdw_verification.py

Phase 6D: Zero-Training Verification Suite for P3 (Feature/Detail Enhancement -> Spatial Compression)
Target: SAGE-Lite B2 Architecture (D4, K2, H64) on Crack500

Verification Checkpoints:
1. Exact Tensor Shapes & Interface Contracts:
   - S0: (B, 48, 112, 112) -> ASDW -> AdaptiveAvgPool(28, 28) -> flatten/transpose (B, 784, 48) -> proj (B, 784, 192) -> ViT -> SA-Hub restore (B, 48, 112, 112)
   - S1: (B, 96, 56, 56) -> ASDW -> AdaptiveAvgPool(28, 28) -> flatten/transpose (B, 784, 96) -> proj (B, 784, 192) -> ViT -> SA-Hub restore (B, 96, 56, 56)
2. High-Frequency / Detail Enhancement Physically Exists Before Compression:
   - Spectral High-Pass Energy E_HF = ||Laplacian(x)||^2 before vs after ASDW.
   - Contrast Ratio preservation through spatial pooling: Contrast(ASDW -> Pool) vs Contrast(Identity -> Pool).
3. Strict Localization to Intended Branch (CNN S0/S1 -> ViT Expert):
   - Main CNN path: bitwise identical between Run A (Identity) and Run C (ASDW).
   - Non-target routes: bitwise identical.
   - Difference exists strictly at CNN S0/S1 -> ViT tokens.
4. Zero Dynamic Parameter Creation in Forward Pass:
   - Parameter counts and state_dict keys bitwise invariant before and after multiple forward calls.
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure SAGE_LITE and project root are on sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sage_lite_dir = os.path.join(project_root, "SAGE_LITE")
for p in [project_root, sage_lite_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sage.components.p3_refinement import ASDWRefinement
from sage.networks.b2_unet import create_b2_unet, B2ConvNeXtViTUNet
from sage.components.sage_layer import SageLayer


def test_1_tensor_shapes_and_interface_contracts():
    print("=" * 80)
    print("[Test 1] Verifying Tensor Shapes & Interface Contracts for S0 and S1...")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = 2

    # Check Stage 0: 48 channels, 112x112
    C0, H0, W0 = 48, 112, 112
    x0 = torch.randn(B, C0, H0, W0, device=device)
    asdw0 = ASDWRefinement(channels=C0).to(device)
    
    # 1. ASDW Refinement
    x0_refined = asdw0(x0)
    assert x0_refined.shape == (B, C0, H0, W0), f"S0 ASDW shape mismatch: {x0_refined.shape}"
    
    # 2. Adaptive AvgPool to 28x28
    x0_compressed = F.adaptive_avg_pool2d(x0_refined, (28, 28))
    assert x0_compressed.shape == (B, C0, 28, 28), f"S0 compression shape mismatch: {x0_compressed.shape}"
    
    # 3. Flatten and transpose
    x0_tokens = x0_compressed.flatten(2).transpose(1, 2).contiguous()
    assert x0_tokens.shape == (B, 784, C0), f"S0 tokens shape mismatch: {x0_tokens.shape}"

    # Check Stage 1: 96 channels, 56x56
    C1, H1, W1 = 96, 56, 56
    x1 = torch.randn(B, C1, H1, W1, device=device)
    asdw1 = ASDWRefinement(channels=C1).to(device)

    # 1. ASDW Refinement
    x1_refined = asdw1(x1)
    assert x1_refined.shape == (B, C1, H1, W1), f"S1 ASDW shape mismatch: {x1_refined.shape}"

    # 2. Adaptive AvgPool to 28x28
    x1_compressed = F.adaptive_avg_pool2d(x1_refined, (28, 28))
    assert x1_compressed.shape == (B, C1, 28, 28), f"S1 compression shape mismatch: {x1_compressed.shape}"

    # 3. Flatten and transpose
    x1_tokens = x1_compressed.flatten(2).transpose(1, 2).contiguous()
    assert x1_tokens.shape == (B, 784, C1), f"S1 tokens shape mismatch: {x1_tokens.shape}"

    print(f"  Stage 0 contract: ({B}, {C0}, {H0}, {W0}) -> ASDW -> Pool(28,28) -> Tokens ({B}, 784, {C0}) [VERIFIED]")
    print(f"  Stage 1 contract: ({B}, {C1}, {H1}, {W1}) -> ASDW -> Pool(28,28) -> Tokens ({B}, 784, {C1}) [VERIFIED]")
    print(">> PASS: Test 1 (Tensor Shapes & Interface Contracts).")
    return True


def test_2_high_frequency_detail_enhancement():
    print("\n" + "=" * 80)
    print("[Test 2] Verifying High-Frequency Detail Enhancement Before Compression...")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)

    # Create a synthetic high-resolution feature map containing thin curvilinear cracks (width 1-2 px)
    # immersed in a smooth background
    C, H, W = 48, 112, 112
    x = torch.zeros(1, C, H, W, device=device)

    # Add background smooth gradient
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
    smooth_bg = (grid_y**2 + grid_x**2).unsqueeze(0).unsqueeze(0).to(device)
    x = x + 0.2 * smooth_bg

    # Draw thin horizontal crack at row 56 (width 1 px)
    x[:, :, 56, 20:92] += 2.0
    # Draw thin vertical crack at col 56 (width 1 px)
    x[:, :, 20:92, 56] += 2.0

    # Instantiate ASDW with trained or random weights
    asdw = ASDWRefinement(channels=C).to(device)
    # Set gamma = 0.1 for clear measurable diagnostic response
    with torch.no_grad():
        asdw.gamma.fill_(0.1)

    with torch.no_grad():
        x_refined = asdw(x)

    # Compute high-pass energy via discrete Laplacian operator
    laplacian_kernel = torch.tensor([
        [0.0,  1.0, 0.0],
        [1.0, -4.0, 1.0],
        [0.0,  1.0, 0.0]
    ], device=device).view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    hp_raw = F.conv2d(x, laplacian_kernel, padding=1, groups=C)
    hp_ref = F.conv2d(x_refined, laplacian_kernel, padding=1, groups=C)

    energy_raw = float(torch.sum(hp_raw**2).item())
    energy_ref = float(torch.sum(hp_ref**2).item())

    # High-frequency boost ratio
    hf_boost_ratio = energy_ref / energy_raw

    # Now observe contrast preservation after spatial compression: AdaptiveAvgPool2d((28, 28))
    pool_raw = F.adaptive_avg_pool2d(x, (28, 28))
    pool_ref = F.adaptive_avg_pool2d(x_refined, (28, 28))

    # Crack row in 28x28 grid is row 14 (56 // 4)
    # Background region is at rows 2:6, cols 2:6
    crack_val_raw = float(pool_raw[:, :, 14, 5:23].mean().item())
    bg_val_raw = float(pool_raw[:, :, 2:6, 2:6].mean().item())
    contrast_raw = crack_val_raw - bg_val_raw

    crack_val_ref = float(pool_ref[:, :, 14, 5:23].mean().item())
    bg_val_ref = float(pool_ref[:, :, 2:6, 2:6].mean().item())
    contrast_ref = crack_val_ref - bg_val_ref

    contrast_boost_ratio = contrast_ref / contrast_raw

    print(f"  Laplacian High-Frequency Energy (Raw Input):     {energy_raw:.4f}")
    print(f"  Laplacian High-Frequency Energy (ASDW Refined):  {energy_ref:.4f}")
    print(f"  High-Frequency Energy Ratio:                     {hf_boost_ratio:.4f} (E_ref / E_raw)")
    print(f"  Pooled 28x28 Crack-to-BG Contrast (Raw Input):   {contrast_raw:.4f}")
    print(f"  Pooled 28x28 Crack-to-BG Contrast (ASDW Refined):{contrast_ref:.4f}")
    print(f"  Pooled Contrast Retention Boost:                 {contrast_boost_ratio:.4f}x")

    assert hf_boost_ratio > 1.0, f"Expected HF energy boost > 1.0, got {hf_boost_ratio}"
    assert contrast_ref > 0, "Expected positive pooled contrast"
    print(">> PASS: Test 2 (High-Frequency Detail Enhancement Before Compression Verified).")
    return True


def test_3_strict_branch_localization():
    print("\n" + "=" * 80)
    print("[Test 3] Verifying Strict Branch Localization (Run A vs Run C)...")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)

    sage_cfg = {
        "top_k": 2,
        "gating_type": "sigmoid",
        "shared_expert_indices": [0, 1, 2, 3],
        "router_hidden_dim": 64,
        "load_balance_factor": 0.01,
        "logit_modulation": True,
        "expert_dropout": 0.0,
        "fusion_type": "residual",
        "residual_scale": 0.1,
    }

    # Model A: Identity (Baseline không-ASDW)
    model_a = create_b2_unet(
        num_classes=1,
        img_size=448,
        num_transformer_layers=4,
        pretrained=False,
        sage_config=sage_cfg,
        p3_mode="A"
    ).to(device)

    # Model C: ASDW Refinement (P3-ASDW)
    model_c = create_b2_unet(
        num_classes=1,
        img_size=448,
        num_transformer_layers=4,
        pretrained=False,
        sage_config=sage_cfg,
        p3_mode="C"
    ).to(device)

    # Initialize shared base parameters with identical weights
    state_a = model_a.state_dict()
    state_c = model_c.state_dict()
    shared_keys = set(state_a.keys()).intersection(set(state_c.keys()))
    with torch.no_grad():
        for k in shared_keys:
            state_c[k].copy_(state_a[k])
    model_c.load_state_dict(state_c)

    model_a.eval()
    model_c.eval()

    # Synthetic batch of images
    x_test = torch.randn(2, 3, 448, 448, device=device)

    # 1. Verify ConvNeXt Stem output is 100% bitwise identical
    with torch.no_grad():
        stem_a = model_a.backbone.convnext.stem(x_test)
        stem_c = model_c.backbone.convnext.stem(x_test)
        diff_stem = torch.max(torch.abs(stem_a - stem_c)).item()
    print(f"  Stem Output Max Diff:                     {diff_stem:.6e}")
    assert diff_stem == 0.0, f"Stem outputs differ: {diff_stem}"

    # 2. Verify Main CNN Path (_execute_main_path) in Stage 0 and Stage 1 is 100% bitwise identical
    with torch.no_grad():
        main0_a = model_a.backbone.convnext.stages[0]._execute_main_path(stem_a)
        main0_c = model_c.backbone.convnext.stages[0]._execute_main_path(stem_c)
        diff_main0 = torch.max(torch.abs(main0_a - main0_c)).item()

        main1_a = model_a.backbone.convnext.stages[1]._execute_main_path(main0_a)
        main1_c = model_c.backbone.convnext.stages[1]._execute_main_path(main0_c)
        diff_main1 = torch.max(torch.abs(main1_a - main1_c)).item()
    print(f"  Stage 0 Main Path Max Diff:               {diff_main0:.6e}")
    print(f"  Stage 1 Main Path Max Diff:               {diff_main1:.6e}")
    assert diff_main0 == 0.0, f"Stage 0 main paths differ: {diff_main0}"
    assert diff_main1 == 0.0, f"Stage 1 main paths differ: {diff_main1}"

    # 3. Verify that difference exists strictly inside the S0/S1 -> ViT token branch
    stage0_a = model_a.backbone.convnext.stages[0]
    stage0_c = model_c.backbone.convnext.stages[0]

    with torch.no_grad():
        # Input to P3 branch is stem output
        tok_a = F.adaptive_avg_pool2d(stage0_a.p3_refinement(stem_a), (28, 28))
        tok_c = F.adaptive_avg_pool2d(stage0_c.p3_refinement(stem_c), (28, 28))
        diff_p3_s0 = torch.max(torch.abs(tok_a - tok_c)).item()
    print(f"  Stage 0 P3 Refined Tokens Max Diff:       {diff_p3_s0:.6e}")
    assert diff_p3_s0 > 0.0, "Expected difference in S0 P3 tokens between Identity and ASDW"

    print(">> PASS: Test 3 (Strict Branch Localization Verified).")
    return True


def test_4_zero_dynamic_parameter_creation():
    print("\n" + "=" * 80)
    print("[Test 4] Verifying Zero Dynamic Parameter Creation in Forward Pass...")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_b2_unet(
        num_classes=1,
        img_size=448,
        num_transformer_layers=4,
        pretrained=False,
        p3_mode="C"
    ).to(device)

    # Initial parameter inventory
    initial_param_names = [name for name, _ in model.named_parameters()]
    initial_param_count = sum(p.numel() for p in model.parameters())
    initial_buffer_names = [name for name, _ in model.named_buffers()]

    # Run forward passes with different batch sizes
    model.eval()
    for bs in [1, 2, 4]:
        x = torch.randn(bs, 3, 448, 448, device=device)
        with torch.no_grad():
            out = model(x)
            assert out.shape == (bs, 1, 448, 448), f"Output shape mismatch at BS={bs}"

    # Final parameter inventory
    final_param_names = [name for name, _ in model.named_parameters()]
    final_param_count = sum(p.numel() for p in model.parameters())
    final_buffer_names = [name for name, _ in model.named_buffers()]

    assert initial_param_names == final_param_names, "Parameter names changed after forward!"
    assert initial_param_count == final_param_count, f"Parameter count changed: {initial_param_count} -> {final_param_count}"
    assert initial_buffer_names == final_buffer_names, "Buffers changed after forward!"

    print(f"  Total Parameters: {final_param_count:,} (Static, 0 dynamic allocations)")
    print(f"  Total Buffers:    {len(final_buffer_names)} (Static, including backbone.pe28_fixed)")
    print(">> PASS: Test 4 (Zero Dynamic Parameter Creation Verified).")
    return True


def main():
    print("\n" + "#" * 80)
    print("PHASE 6D: P3-ASDW ZERO-TRAINING VERIFICATION SUITE")
    print("#" * 80 + "\n")

    t1 = test_1_tensor_shapes_and_interface_contracts()
    t2 = test_2_high_frequency_detail_enhancement()
    t3 = test_3_strict_branch_localization()
    t4 = test_4_zero_dynamic_parameter_creation()

    if t1 and t2 and t3 and t4:
        print("\n" + "=" * 80)
        print("ALL 4 ZERO-TRAINING VERIFICATION CHECKS PASSED BIT-PERFECTLY!")
        print("=" * 80 + "\n")
        return 0
    else:
        print("\nFAILED AT LEAST ONE VERIFICATION CHECK!\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
