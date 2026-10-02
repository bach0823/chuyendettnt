#!/usr/bin/env python3
"""
scripts/diagnostics/test_t2_csdg_invariants.py

Pre-training Invariant & Identity Smoke Test for T2-CSDG (K=3)
=============================================================
Protocol:
- Verify Candidate B checkpoint SHA256 bitwise invariant.
- Verify Candidate B model parameter SHA256 bitwise invariant.
- Verify T2-CSDG (K=3) architecture:
    T2 [288, 56, 56]
      -> Conv1x1(288 -> 64) -> BN -> ReLU
      -> DWConv3x3(64 -> 64, groups=64, pad=1) -> BN -> ReLU
      -> Conv1x1(64 -> 2)
      -> g = 2.0 * sigmoid(a)
      -> T2' = [g_B * B; g_S * S]
- Verify Exact Identity Initialization:
    W_gate = 0, b_gate = 0 -> a = 0 -> g_B = g_S = 1.0
    Bitwise forward diff on dummy features: max_abs_diff == 0.0
- Verify Full Model Forward Identity:
    Model output with hooked CSDG at epoch 0 must be BITWISE IDENTICAL
    to unmodified Candidate B model output (diff == 0.0).
- Verify 100% parameter freezing on Candidate B (requires_grad = False).
- Verify only CSDG parameters exist in optimizer.
"""

import hashlib
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, 'SAGE_LITE')
from tools.run_phase6_c_topology_diagnostic import load_model_from_checkpoint


def compute_model_param_hash(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode('utf-8'))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


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
        The final gate_conv (1x1 Conv(64 -> 2)) has weight=0, bias=0.
        At initialization:
            a = 0 everywhere
            sigmoid(0) = 0.5
            g = 2.0 * 0.5 = 1.0 EXACTLY.
            T2' = [1.0 * B; 1.0 * S] = T2 EXACTLY.
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
        
        # Cache for diagnostic inspection
        self.last_gate = None

    def forward(self, b_stream: torch.Tensor, s_stream: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def run_checks():
    print("=" * 80)
    print("T2-CSDG (K=3) Pre-training Smoke Test & System Invariant Verification")
    print("=" * 80)
    
    config_path = 'results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml'
    ckpt_path   = 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Checkpoint integrity check
    print("\n[Check 1/10] Verifying Candidate B checkpoint SHA256...")
    with open(ckpt_path, 'rb') as f:
        file_sha = hashlib.sha256(f.read()).hexdigest()
    EXPECTED_SHA = '147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66'
    assert file_sha == EXPECTED_SHA, f"Checkpoint SHA mismatch: {file_sha}"
    print(f"  PASS: Checkpoint SHA256 verified ({file_sha[:16]}...)")
    
    # 2. Load model & verify parameter hash
    print("\n[Check 2/10] Loading Candidate B and computing initial param hash...")
    model = load_model_from_checkpoint(config_path, ckpt_path, device=device)
    init_param_hash = compute_model_param_hash(model)
    EXPECTED_PARAM_SHA = '4aeda58ce6d2fb32f5b772cdbf92dec7fb913f9510b392910efe6674b0adf5d6'
    assert init_param_hash == EXPECTED_PARAM_SHA, f"Param SHA mismatch: {init_param_hash}"
    print(f"  PASS: Initial parameter hash verified ({init_param_hash[:16]}...)")
    
    # 3. Freeze all model parameters
    print("\n[Check 3/10] Freezing all model parameters...")
    for p in model.parameters():
        p.requires_grad = False
    assert sum(p.requires_grad for p in model.parameters()) == 0, "Model params must not require grad"
    print("  PASS: All 100% of Candidate B parameters frozen (requires_grad=False)")
    
    # 4. Instantiate T2-CSDG & verify identity init
    print("\n[Check 4/10] Verifying T2-CSDG (K=3) exact identity initialization...")
    csdg = T2CSDG(in_channels=192, skip_channels=96, hidden_dim=64, kernel_size=3).to(device)
    dummy_b = torch.randn(2, 192, 56, 56, device=device)
    dummy_s = torch.randn(2, 96, 56, 56, device=device)
    with torch.no_grad():
        dummy_t2_prime, dummy_g = csdg(dummy_b, dummy_s)
    
    max_gate_diff = (dummy_g - 1.0).abs().max().item()
    print(f"  Max gate difference from 1.0: {max_gate_diff:.10f}")
    assert max_gate_diff == 0.0, f"Identity init must give exact 1.0, got diff {max_gate_diff}"
    
    # Verify exact input-output identity on dummy features
    expected_t2 = torch.cat([dummy_b, dummy_s], dim=1)
    feat_diff = (dummy_t2_prime - expected_t2).abs().max().item()
    print(f"  Max feature difference (T2' vs baseline T2): {feat_diff:.10f}")
    assert feat_diff == 0.0, f"T2' must be exactly identical to T2 at epoch 0, got diff {feat_diff}"
    print("  PASS: Exact Identity Initialization verified (g_B=g_S=1.0, T2'=T2, max diff = 0.0)")
    
    # 5. Integrate CSDG into Decoder Block 1
    print("\n[Check 5/10] Integrating CSDG into Decoder Block 1 forward path...")
    dec_b1 = model.decoder.decoder_blocks[1]
    
    def hooked_forward(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = dec_b1.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        t2_gated, _ = csdg(x, skip)
        x = dec_b1.conv1(t2_gated)
        x = dec_b1.conv2(x)
        return x
        
    original_forward = dec_b1.forward
    dec_b1.forward = hooked_forward
    print("  PASS: Decoder Block 1 forward hooked successfully")
    
    # 6. Full model end-to-end forward bitwise identity test
    print("\n[Check 6/10] Verifying full model end-to-end output identity against clean baseline...")
    clean_model = load_model_from_checkpoint(config_path, ckpt_path, device=device)
    clean_model.eval()
    model.eval()
    
    dummy_input = torch.randn(2, 3, 448, 448, device=device)
    with torch.no_grad():
        clean_out = clean_model(dummy_input)
        hooked_out = model(dummy_input)
        
    e2e_diff = (clean_out - hooked_out).abs().max().item()
    print(f"  Max End-to-End Logit Difference at Epoch 0: {e2e_diff:.10f}")
    assert e2e_diff == 0.0, f"End-to-end forward must be bitwise identical at epoch 0, got diff {e2e_diff}"
    print("  PASS: End-to-End Bitwise Identity Verified (max diff = 0.0)")
    
    # 7. Optimizer parameter containment check
    print("\n[Check 7/10] Verifying optimizer parameters...")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) == 0, "No model parameters must be trainable"
    
    csdg_trainable = [p for p in csdg.parameters() if p.requires_grad]
    # csdg has: reduce_conv.weight (1), reduce_bn (weight, bias: 2), dw_conv.weight (1), dw_bn (weight, bias: 2), gate_conv (weight, bias: 2) -> 8 parameter tensors
    assert len(csdg_trainable) == 8, f"CSDG must have exactly 8 parameter tensors, got {len(csdg_trainable)}"
    print(f"  PASS: Optimizer contains exactly {len(csdg_trainable)} CSDG parameter tensors (0 model params)")
    
    # 8. Gradient leakage check during backward pass
    print("\n[Check 8/10] Verifying backward pass & gradient isolation...")
    optimizer = torch.optim.AdamW(csdg.parameters(), lr=1e-4)
    model.eval() # Keep frozen modules in eval mode
    csdg.train() # Only CSDG in train mode
    
    out = model(dummy_input)
    dummy_loss = out.mean()
    dummy_loss.backward()
    
    for name, p in model.named_parameters():
        assert p.grad is None, f"Gradient leaked into frozen model parameter: {name}"
    print("  PASS: Zero gradient leak into frozen model parameters")
    
    for name, p in csdg.named_parameters():
        assert p.grad is not None, f"Missing gradient on CSDG parameter: {name}"
    print("  PASS: All CSDG parameters received valid gradients")
    
    # 9. Parameter invariance post optimizer step
    print("\n[Check 9/10] Verifying model parameter invariance after optimizer.step()...")
    optimizer.step()
    post_step_param_hash = compute_model_param_hash(model)
    assert post_step_param_hash == EXPECTED_PARAM_SHA, "Model params must remain bitwise identical after optimizer step"
    print("  PASS: Model parameter hash remains 100% BITWISE INVARIANT post-step")
    
    # 10. FP32 Numerical Stability Check
    print("\n[Check 10/10] Verifying FP32 stability under physical batch 14...")
    batch14_input = torch.randn(14, 3, 448, 448, device=device)
    with torch.no_grad():
        out14 = model(batch14_input)
    assert not torch.isnan(out14).any(), "NaN detected in batch 14 output"
    assert not torch.isinf(out14).any(), "Inf detected in batch 14 output"
    vram_used = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    print(f"  PASS: Batch 14 forward numerically stable, VRAM allocated: {vram_used:.1f} MB")
    
    print("\n" + "=" * 80)
    print("ALL 10/10 PRE-TRAINING INVARIANT CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    run_checks()
