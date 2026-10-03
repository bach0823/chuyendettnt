#!/usr/bin/env python3
"""
experiments/phase6_hrrb/test_hrrb_invariants.py

Preflight Invariant Test Suite for Phase 6: High-Resolution Residual Bypass (HRRB)
==================================================================================

Enforces all 7 mandatory scientific invariants:
    Test 1: Trainable Whitelist (Only HRRB parameters trainable, exactly 1,409 params).
    Test 2: Frozen Main Network (Zero gradient propagation to main model).
    Test 3: Zero Residual at Initialization (detail_residual == 0.0 bit-exactly).
    Test 4: End-to-End Bitwise Identity (final_logits == candidate_b_logits at t=0).
    Test 5: Strict Tensor Shape Contract ([B, 1, 448, 448]).
    Test 6: Optimizer Isolation (Optimizer parameter groups contain ONLY HRRB params).
    Test 7: Candidate B Checkpoint Integrity (SHA256 bit-exact invariant).
"""

import hashlib
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure SAGE_LITE and project root are accessible
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('SAGE_LITE'))

from sage_lite.tools.run_phase6_c_topology_diagnostic import load_model_from_checkpoint
from experiments.phase6_hrrb.hrrb_module import HighResolutionResidualBypass, CandidateBWithHRRB


CANDIDATE_B_CHECKPOINT_SHA256 = "147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66"
DEFAULT_CONFIG = "results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml"
DEFAULT_CHECKPOINT = "results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth"


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run_preflight_invariants():
    print("=" * 80)
    print("PHASE 6: HIGH-RESOLUTION RESIDUAL BYPASS (HRRB) PREFLIGHT INVARIANT AUDIT")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # TEST 7: Main Checkpoint Integrity
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Candidate B Checkpoint Integrity...")
    assert os.path.exists(DEFAULT_CHECKPOINT), f"Checkpoint not found at {DEFAULT_CHECKPOINT}"
    actual_hash = compute_file_sha256(DEFAULT_CHECKPOINT)
    print(f"  Checkpoint SHA256: {actual_hash}")
    assert actual_hash == CANDIDATE_B_CHECKPOINT_SHA256, (
        f"VIOLATION TEST 7: Checkpoint SHA256 mismatch!\nExpected: {CANDIDATE_B_CHECKPOINT_SHA256}\nGot: {actual_hash}"
    )
    print(">> PASS TEST 7: Candidate B Checkpoint SHA256 invariant verified.")

    # Load Candidate B
    print("\nLoading Candidate B baseline model...")
    candidate_b = load_model_from_checkpoint(DEFAULT_CONFIG, DEFAULT_CHECKPOINT, device=device)
    model = CandidateBWithHRRB(candidate_b).to(device)

    # -------------------------------------------------------------------------
    # TEST 1: Trainable Whitelist
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Trainable Whitelist Audit...")
    trainable_named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    trainable_names = [name for name, p in trainable_named]
    trainable_numel = sum(p.numel() for _, p in trainable_named)

    expected_whitelist = [
        "hrrb.conv1.weight",  # 8 * 3 * 3 * 3 = 216
        "hrrb.conv1.bias",    # 8
        "hrrb.conv2.weight",  # 16 * 8 * 3 * 3 = 1152
        "hrrb.conv2.bias",    # 16
        "hrrb.proj.weight",   # 1 * 16 * 1 * 1 = 16
        "hrrb.proj.bias",     # 1
    ]
    print(f"  Trainable parameters count: {trainable_numel}")
    for name, p in trainable_named:
        print(f"    - {name}: shape={list(p.shape)}, numel={p.numel()}")

    assert trainable_names == expected_whitelist, (
        f"VIOLATION TEST 1: Trainable names mismatch!\nExpected: {expected_whitelist}\nGot: {trainable_names}"
    )
    assert trainable_numel == 1409, f"VIOLATION TEST 1: Expected 1,409 trainable parameters, got {trainable_numel}"
    print(">> PASS TEST 1: Whitelist verified (exactly 1,409 trainable parameters).")

    # -------------------------------------------------------------------------
    # TEST 3: Zero Residual at Initialization
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Zero Residual at Initialization Audit...")
    dummy_x = torch.randn(2, 3, 448, 448, device=device)
    with torch.no_grad():
        detail_res_0 = model.hrrb(dummy_x)
        max_res_abs = torch.max(torch.abs(detail_res_0)).item()
        print(f"  Max absolute detail_residual at t=0: {max_res_abs:.2e}")
        assert max_res_abs == 0.0, f"VIOLATION TEST 3: detail_residual at t=0 is non-zero ({max_res_abs})"
    print(">> PASS TEST 3: detail_residual at t=0 is 0.0 bit-exactly.")

    # -------------------------------------------------------------------------
    # TEST 4: End-to-End Bitwise Identity with Candidate B
    # -------------------------------------------------------------------------
    print("\n[TEST 4] End-to-End Identity with Candidate B Audit...")
    with torch.no_grad():
        candidate_b_logits = candidate_b(dummy_x)
        final_logits, main_logits, detail_res = model(dummy_x, return_components=True)
        diff_main = torch.max(torch.abs(main_logits - candidate_b_logits)).item()
        diff_final = torch.max(torch.abs(final_logits - candidate_b_logits)).item()
        print(f"  Max abs diff (main_logits vs candidate_b_logits): {diff_main:.2e}")
        print(f"  Max abs diff (final_logits vs candidate_b_logits): {diff_final:.2e}")
        assert diff_main == 0.0, f"VIOLATION TEST 4: main_logits does not match Candidate B! diff={diff_main}"
        assert diff_final == 0.0, f"VIOLATION TEST 4: final_logits does not match Candidate B at t=0! diff={diff_final}"
    print(">> PASS TEST 4: Composite model at t=0 reproduces Candidate B bit-exactly.")

    # -------------------------------------------------------------------------
    # TEST 5: Strict Tensor Shape Contract
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Strict Tensor Shape Contract...")
    assert main_logits.shape == (2, 1, 448, 448), f"VIOLATION TEST 5: main_logits shape {main_logits.shape}"
    assert detail_res.shape == (2, 1, 448, 448), f"VIOLATION TEST 5: detail_residual shape {detail_res.shape}"
    assert final_logits.shape == (2, 1, 448, 448), f"VIOLATION TEST 5: final_logits shape {final_logits.shape}"
    print(f"  Shapes verified: main={main_logits.shape}, res={detail_res.shape}, final={final_logits.shape}")
    print(">> PASS TEST 5: Strict tensor shape contract verified.")

    # -------------------------------------------------------------------------
    # TEST 6: Optimizer Isolation
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Optimizer Isolation...")
    optimizer = torch.optim.AdamW(model.hrrb.parameters(), lr=1e-4, weight_decay=1e-2)
    opt_params = [p for grp in optimizer.param_groups for p in grp["params"]]
    opt_numel = sum(p.numel() for p in opt_params)
    print(f"  Optimizer parameter count: {opt_numel}")
    assert opt_numel == 1409, f"VIOLATION TEST 6: Expected 1,409 optimizer parameters, got {opt_numel}"
    print(">> PASS TEST 6: Optimizer isolation verified.")

    # -------------------------------------------------------------------------
    # TEST 2: Frozen Main Network (Zero Gradient Leakage)
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Frozen Main Network Gradient Audit...")
    model.train()
    optimizer.zero_grad()
    dummy_out = model(dummy_x)
    dummy_loss = dummy_out.sum()
    dummy_loss.backward()

    # Step 1: proj receives non-zero gradient, main receives none
    hrrb_grads_step1 = [p.grad for p in model.hrrb.parameters()]
    assert all(g is not None for g in hrrb_grads_step1), "VIOLATION TEST 2: HRRB parameters missing gradients!"
    assert hrrb_grads_step1[4].abs().sum().item() > 0, "VIOLATION TEST 2: proj.weight missing gradient!"
    assert hrrb_grads_step1[5].abs().sum().item() > 0, "VIOLATION TEST 2: proj.bias missing gradient!"

    # Step optimizer so proj.weight becomes non-zero, then check step 2
    optimizer.step()
    optimizer.zero_grad()
    dummy_out2 = model(dummy_x)
    dummy_loss2 = dummy_out2.sum()
    dummy_loss2.backward()

    hrrb_grads_step2 = [p.grad for p in model.hrrb.parameters()]
    assert all(g is not None and g.abs().sum().item() > 0 for g in hrrb_grads_step2), (
        "VIOLATION TEST 2: Not all HRRB parameters receive non-zero gradients on active step!"
    )
    print("  HRRB all 6 parameter gradients non-zero on active step: True")

    # Verify Candidate B parameters got ZERO gradients (grad is None or zero)
    for name, p in model.main_model.named_parameters():
        assert p.grad is None or torch.all(p.grad == 0), (
            f"VIOLATION TEST 2: Gradient leaked to frozen main parameter {name}!"
        )
    print(">> PASS TEST 2: Main network is 100% strictly frozen (zero gradient leakage).")
    # Reset model to zero-init for subsequent tests
    model.hrrb.reset_parameters()
    optimizer.zero_grad()

    # -------------------------------------------------------------------------
    # Compute & Memory Benchmark
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPUTE & MEMORY BENCHMARK (Batch 14, FP32, 448x448)")
    print("=" * 80)
    # Theoretical FLOPs:
    # Conv1 (3->8, 3x3, 448x448): 8 * (3*3*3) * 448*448 = 43,352,064 MACs
    # Conv2 (8->16, 3x3, s2, 224x224): 16 * (8*3*3) * 224*224 = 57,802,752 MACs
    # Proj (16->1, 1x1, 224x224): 1 * (16*1*1) * 224*224 = 802,816 MACs
    # Total MACs = 101,957,632 MACs (~0.102 GMACs)
    # Total FLOPs = 2 * MACs = 203,915,264 FLOPs (~0.204 GFLOPs)
    macs_per_image = 101957632
    flops_per_image = 2 * macs_per_image
    print(f"HRRB Trainable Parameters: {trainable_numel}")
    print(f"HRRB Theoretical MACs/image:  {macs_per_image:,} ({macs_per_image / 1e9:.4f} GMACs)")
    print(f"HRRB Theoretical FLOPs/image: {flops_per_image:,} ({flops_per_image / 1e9:.4f} GFLOPs)")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        batch_14_x = torch.randn(14, 3, 448, 448, device=device)
        mem_init = torch.cuda.memory_allocated() / (1024 ** 2)

        # Forward
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        out_14 = model(batch_14_x)
        t1.record()
        torch.cuda.synchronize()
        mem_fwd_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        fwd_time_ms = t0.elapsed_time(t1)

        # Backward
        loss_14 = out_14.sum()
        t2 = torch.cuda.Event(enable_timing=True)
        t3 = torch.cuda.Event(enable_timing=True)
        t2.record()
        loss_14.backward()
        t3.record()
        torch.cuda.synchronize()
        mem_fwd_bwd_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        bwd_time_ms = t2.elapsed_time(t3)

        print(f"Peak VRAM Forward:          {mem_fwd_peak:.2f} MB")
        print(f"Peak VRAM Forward+Backward: {mem_fwd_bwd_peak:.2f} MB")
        print(f"Measured Forward Time (B=14):  {fwd_time_ms:.2f} ms ({fwd_time_ms/14:.2f} ms/img)")
        print(f"Measured Backward Time (B=14): {bwd_time_ms:.2f} ms ({bwd_time_ms/14:.2f} ms/img)")
        del batch_14_x, out_14, loss_14
        torch.cuda.empty_cache()
    else:
        print("CUDA not available; memory benchmark skipped on CPU.")

    print("\n" + "=" * 80)
    print(">> ALL 7 PREFLIGHT INVARIANTS PASSED 100% BIT-EXACTLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_preflight_invariants()
