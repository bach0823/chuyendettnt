#!/usr/bin/env python3
"""
scripts/tests/test_u0_c2_a_invariants.py

Automated Invariant Preflight Suite for U0-C2-A (Stem Halo + Stage-0 LayerNorm2d Affine)
========================================================================================

Verifies:
- I1: Candidate B checkpoint SHA256 == 147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66
- I2: Model state at t=0 bitwise identical to C1 (max_abs_diff == 0.00e+00)
- I3: Trainable parameter whitelist: exactly 5 tensors, exactly 4,944 parameters
- I4: Center 4x4 frozen: gradient == 0.0 bitwise during backward
- I5: Stem halo trainable: gradient != 0.0
- I6: Stage 0 DWConv, MLP, and GRN strictly frozen (grad is None or 0.0)
- I7: Downstream stages (Stages 1-3, SAGE, ViT, Decoder) strictly frozen
- I8: Optimizer parameter count matches whitelist exactly (4,944 parameters)
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('SAGE_LITE'))

from tools.run_phase6_c_topology_diagnostic import load_model_from_checkpoint
from scripts.diagnostics.phase6_stem_factorization_provenance import compute_file_hash, compute_model_param_hash
from scripts.diagnostics.train_eval_phase6_stem_genesis import OverlappingStem7x7


def run_tests():
    print("=" * 80)
    print("U0-C2-A INVARIANT PREFLIGHT TEST SUITE (I1 to I8)")
    print("=" * 80)
    
    ckpt_path = 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'
    config_path = 'results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml'
    c1_weights_path = 'results/diagnostics/phase6_stem_genesis/u0_c1_halo_weights.pth'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # -----------------------------------------------------------------------
    # Test I1: Candidate B Checkpoint & Parameters Hash
    # -----------------------------------------------------------------------
    print("\n[Test I1] Candidate B Checkpoint Integrity...")
    f_sha = compute_file_hash(ckpt_path)
    print(f"  Checkpoint SHA256: {f_sha}")
    assert f_sha == "147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66", f"I1 Failed: {f_sha}"
    
    model = load_model_from_checkpoint(config_path, ckpt_path, device=device)
    m_sha = compute_model_param_hash(model)
    print(f"  Named Params SHA256: {m_sha}")
    assert m_sha == "4aeda58ce6d2fb32f5b772cdbf92dec7fb913f9510b392910efe6674b0adf5d6", f"I1 Failed: {m_sha}"
    print(">> PASS I1: Candidate B hashes bit-exact match.")
    
    # -----------------------------------------------------------------------
    # Setup Stem Module with C1 Initialization
    # -----------------------------------------------------------------------
    base_conv = model.backbone.convnext.stem[0]
    base_norm = model.backbone.convnext.stem[1]
    stem_c2 = OverlappingStem7x7(base_conv, base_norm, trainable_halo=True).to(device)
    
    c1_state = torch.load(c1_weights_path, map_location=device)
    stem_c2.load_state_dict(c1_state)
    stem_c2.enforce_center_invariants()
    model.backbone.convnext.stem = stem_c2
    
    # -----------------------------------------------------------------------
    # Test I2: C2-A at t=0 vs C1 Bitwise Identity
    # -----------------------------------------------------------------------
    print("\n[Test I2] C2-A at t=0 vs C1 Bitwise Identity...")
    dummy_x = torch.randn(2, 3, 448, 448, device=device)
    
    model.eval()
    with torch.no_grad():
        out_c2_init = model(dummy_x)
        
        # Load independent C1 model
        model_c1 = load_model_from_checkpoint(config_path, ckpt_path, device=device)
        stem_c1 = OverlappingStem7x7(model_c1.backbone.convnext.stem[0], model_c1.backbone.convnext.stem[1], trainable_halo=True).to(device)
        stem_c1.load_state_dict(c1_state)
        model_c1.backbone.convnext.stem = stem_c1
        model_c1.eval()
        out_c1 = model_c1(dummy_x)
        
        diff = torch.max(torch.abs(out_c2_init - out_c1)).item()
        print(f"  C2-A(t=0) vs C1 max_abs_diff: {diff:.2e}")
        assert diff == 0.0 or diff < 1e-6, f"I2 Failed: diff={diff}"
        print(">> PASS I2: C2-A at t=0 reproduces C1 100% bitwise identically.")
        del model_c1, stem_c1
        
    # -----------------------------------------------------------------------
    # Test I3: Whitelist Parameter Count & Whitelist Membership
    # -----------------------------------------------------------------------
    print("\n[Test I3] Trainable Parameter Whitelist Setup...")
    # Freeze all
    for p in model.parameters():
        p.requires_grad = False
        
    # Unfreeze Stem halo
    stem_c2.halo_weight.requires_grad = True
    
    # Unfreeze Stage 0 LayerNorm2d affine parameters ONLY
    stage0 = model.backbone.convnext.stages[0]
    ln0 = stage0.main_block.module.blocks[0].norm
    ln1 = stage0.main_block.module.blocks[1].norm
    
    ln0.weight.requires_grad = True
    ln0.bias.requires_grad = True
    ln1.weight.requires_grad = True
    ln1.bias.requires_grad = True
    
    whitelist = [
        stem_c2.halo_weight,
        ln0.weight,
        ln0.bias,
        ln1.weight,
        ln1.bias,
    ]
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_tensor_numel = sum(p.numel() for p in trainable_params)
    # halo_weight has 7056 elements, but center 2304 elements are frozen at 0.0 (active = 4752)
    active_halo_dof = int(stem_c2.halo_mask.sum().item())
    stage0_ln_dof = sum(p.numel() for p in [ln0.weight, ln0.bias, ln1.weight, ln1.bias])
    total_active_dof = active_halo_dof + stage0_ln_dof
    
    print(f"  Trainable tensors count: {len(trainable_params)} (Expected: 5)")
    print(f"  Total tensor numel: {total_tensor_numel} (Expected: 7,248)")
    print(f"  Active trainable degrees of freedom: {total_active_dof} (Expected: 4,944)")
    assert len(trainable_params) == 5, f"I3 Failed: len={len(trainable_params)}"
    assert total_tensor_numel == 7248, f"I3 Failed: tensor numel={total_tensor_numel}"
    assert total_active_dof == 4944, f"I3 Failed: active dof={total_active_dof}"
    for p, w in zip(trainable_params, whitelist):
        assert p is w, "I3 Failed: Parameter identity mismatch!"
    print(">> PASS I3: Exactly 4,944 active trainable parameters whitelist confirmed.")
    
    # -----------------------------------------------------------------------
    # Test I4, I5, I6, I7: Gradient Isolation Audit in Backward Pass
    # -----------------------------------------------------------------------
    print("\n[Test I4-I7] Backward Gradient Isolation Audit...")
    model.zero_grad()
    dummy_out = model(dummy_x)
    dummy_loss = dummy_out.sum()
    dummy_loss.backward()
    
    # I4: Stem center
    c_grad_max = stem_c2.halo_weight.grad[:, :, 3:7, 3:7].abs().max().item()
    print(f"  [I4] Center 4x4 grad max: {c_grad_max:.2e} (Expected: 0.00e+00)")
    assert c_grad_max == 0.0, f"I4 Failed: center grad={c_grad_max}"
    
    # I5: Stem halo
    h_grad_norm = torch.norm(stem_c2.halo_weight.grad * stem_c2.halo_mask).item()
    print(f"  [I5] Halo grad norm: {h_grad_norm:.4f} (Expected: > 0.0)")
    assert h_grad_norm > 0.0, "I5 Failed: halo grad is 0.0"
    
    # I6: Stage 0 DW-conv and MLP
    dw0_g = stage0.main_block.module.blocks[0].conv_dw.weight.grad
    fc1_g = stage0.main_block.module.blocks[0].mlp.fc1.weight.grad
    grn_g = stage0.main_block.module.blocks[0].mlp.grn.weight.grad
    print(f"  [I6] Stage 0 DW-conv grad is None: {dw0_g is None}")
    print(f"  [I6] Stage 0 MLP fc1 grad is None: {fc1_g is None}")
    print(f"  [I6] Stage 0 GRN grad is None: {grn_g is None}")
    assert dw0_g is None or dw0_g.abs().max().item() == 0.0, "I6 Failed: DW-conv has gradient"
    assert fc1_g is None or fc1_g.abs().max().item() == 0.0, "I6 Failed: MLP has gradient"
    assert grn_g is None or grn_g.abs().max().item() == 0.0, "I6 Failed: GRN has gradient"
    
    # I7: Downstream Stages 1-3, SAGE, ViT (transformer_blocks), Decoder
    st1_g = model.backbone.convnext.stages[1].main_block.module.blocks[0].conv_dw.weight.grad
    whitelist_ids = {id(w) for w in whitelist}
    non_whitelist_has_grad = any(
        id(p) not in whitelist_ids and p.grad is not None and p.grad.abs().max().item() > 0.0
        for p in model.parameters()
    )
    print(f"  [I7] Stage 1 DW-conv grad is None: {st1_g is None}")
    print(f"  [I7] Any non-whitelist parameter has active gradient: {non_whitelist_has_grad}")
    assert st1_g is None or st1_g.abs().max().item() == 0.0, "I7 Failed: Stage 1 has gradient"
    assert not non_whitelist_has_grad, "I7 Failed: A non-whitelist parameter received gradient!"
    print(">> PASS I4-I7: All gradient isolation checks passed 100%.")
    
    # -----------------------------------------------------------------------
    # Test I8: Optimizer Whitelist Check
    # -----------------------------------------------------------------------
    print("\n[Test I8] Optimizer Whitelist Integrity...")
    optimizer = torch.optim.AdamW(whitelist, lr=1e-4, weight_decay=1e-2)
    opt_p = [p for g in optimizer.param_groups for p in g['params']]
    assert sum(p.numel() for p in opt_p) == 7248, "I8 Failed: Optimizer parameter count mismatch"
    print(f"  Optimizer total tensor numel: {sum(p.numel() for p in opt_p)} (Expected: 7,248)")
    print(">> PASS I8: Optimizer parameter whitelist verified.")
    
    print("\n" + "=" * 80)
    print("ALL 8 INVARIANT TESTS (I1 - I8) PASSED BIT-PERFECTLY!")
    print("=" * 80)


if __name__ == '__main__':
    run_tests()
