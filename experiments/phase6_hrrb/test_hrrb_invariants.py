#!/usr/bin/env python3
"""
experiments/phase6_hrrb/test_hrrb_invariants.py

Preflight Invariant Test Suite for Phase 6: High-Resolution Residual Bypass (HRRB)
==================================================================================

Enforces all mandatory scientific invariants:
    Test 1: Trainable Whitelist (Only HRRB parameters trainable, exactly 1,409 params).
    Test 2: Frozen Main Network (Zero gradient propagation to main model).
    Test 3: Zero Residual at Initialization (detail_residual == 0.0 bit-exactly).
    Test 4: End-to-End Bitwise Identity (final_logits == candidate_b_logits at t=0).
    Test 5: Strict Tensor Shape Contract ([B, 1, 448, 448]).
    Test 6: Optimizer Isolation (Optimizer parameter groups contain ONLY HRRB params).
    Test 7: 3-Tiered Candidate B Integrity:
            - Tier 1: Checkpoint file SHA256 (147f7840...)
            - Tier 2: Loaded named-parameters SHA256 (4aeda58c...)
            - Tier 3: Deterministic probe-input output SHA256 (1d35f0d6...)
    
Multi-Iteration Differential Benchmark:
    Warmup: 10 iterations
    Measurement: 30 iterations (Median, Mean, P90, Std)
    Three Conditions:
        1. Candidate B Baseline (forward, torch.no_grad)
        2. HRRB Standalone (forward + backward)
        3. Composite Candidate B + HRRB (forward + HRRB backward)
    Differential Overhead:
        Delta_VRAM = Peak(Composite) - Peak(Candidate B)
        Delta_t_forward = Median(Composite Forward) - Median(Candidate B Forward)
        t_backward = Median(Composite Backward)
"""

import hashlib
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure SAGE_LITE and project root are accessible
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sage_lite_dir = os.path.join(project_root, "SAGE_LITE")

for p in [project_root, sage_lite_dir, os.path.abspath('.'), os.path.abspath('SAGE_LITE')]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from tools.run_phase6_c_topology_diagnostic import load_model_from_checkpoint
except ModuleNotFoundError:
    from SAGE_LITE.tools.run_phase6_c_topology_diagnostic import load_model_from_checkpoint
from experiments.phase6_hrrb.hrrb_module import HighResolutionResidualBypass, CandidateBWithHRRB


CANDIDATE_B_CHECKPOINT_SHA256 = "147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66"
CANDIDATE_B_PARAM_SHA256      = "4aeda58ce6d2fb32f5b772cdbf92dec7fb913f9510b392910efe6674b0adf5d6"
# Cross-platform deterministic probe output hashes (accounting for GPU PRNG / cuDNN architecture across T4 vs RTX)
CANDIDATE_B_PROBE_SHA256_WHITELIST = {
    "1d35f0d6f53681097d8f6cfc870c9fd0e1697f6ad9d4f2d3ccdb4b4f27a6d646",  # Ada / RTX (Compute Capability 8.9)
    "2beabebe5f8808bbb3cbe39d10ecdf167b710e4ee89bfae8b62724e1cc4af3f0",  # Turing / Tesla T4 (Compute Capability 7.5, Colab)
}

DEFAULT_CONFIG = "results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml"
DEFAULT_CHECKPOINT = "results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth"


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_model_param_hash(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(p.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def run_preflight_invariants():
    print("=" * 80)
    print("PHASE 6: HIGH-RESOLUTION RESIDUAL BYPASS (HRRB) PREFLIGHT INVARIANT AUDIT")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # TEST 7: 3-Tiered Candidate B Integrity Audit
    # -------------------------------------------------------------------------
    print("\n[TEST 7] 3-Tiered Candidate B Integrity Audit...")
    assert os.path.exists(DEFAULT_CHECKPOINT), f"Checkpoint not found at {DEFAULT_CHECKPOINT}"

    # Tier 1: Checkpoint file SHA256
    actual_file_hash = compute_file_sha256(DEFAULT_CHECKPOINT)
    print(f"  Tier 1 - Checkpoint File SHA256:     {actual_file_hash}")
    assert actual_file_hash == CANDIDATE_B_CHECKPOINT_SHA256, (
        f"VIOLATION TEST 7 Tier 1: File hash mismatch!\nExpected: {CANDIDATE_B_CHECKPOINT_SHA256}\nGot: {actual_file_hash}"
    )

    # Load Candidate B
    candidate_b = load_model_from_checkpoint(DEFAULT_CONFIG, DEFAULT_CHECKPOINT, device=device)

    # Tier 2: Loaded named-parameters SHA256
    actual_param_hash = compute_model_param_hash(candidate_b)
    print(f"  Tier 2 - Named Parameters SHA256:    {actual_param_hash}")
    assert actual_param_hash == CANDIDATE_B_PARAM_SHA256, (
        f"VIOLATION TEST 7 Tier 2: Named param hash mismatch!\nExpected: {CANDIDATE_B_PARAM_SHA256}\nGot: {actual_param_hash}"
    )

    # Tier 3: Deterministic probe-input output SHA256
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    probe_x = torch.randn(1, 3, 448, 448, device=device)
    with torch.no_grad():
        probe_out = candidate_b(probe_x)
    actual_probe_hash = hashlib.sha256(probe_out.detach().cpu().numpy().tobytes()).hexdigest()
    assert actual_probe_hash in CANDIDATE_B_PROBE_SHA256_WHITELIST, (
        f"VIOLATION TEST 7 Tier 3: Probe output hash mismatch!\nExpected one of: {CANDIDATE_B_PROBE_SHA256_WHITELIST}\nGot: {actual_probe_hash}"
    )
    print(">> PASS TEST 7: Candidate B 3-Tiered Integrity verified 100% bit-exactly.")

    # Instantiate Composite Model
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
    # Multi-Iteration Differential Compute & Memory Benchmark
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MULTI-ITERATION DIFFERENTIAL BENCHMARK (Batch 14, FP32, 448x448)")
    print("Protocol: 10 warmup iterations + 30 measured iterations with CUDA sync")
    print("=" * 80)

    # Theoretical Complexity
    # Conv1 (3->8, 3x3, 448x448): 8 * (3*3*3) * 448*448 = 43,352,064 MACs
    # Conv2 (8->16, 3x3, s2, 224x224): 16 * (8*3*3) * 224*224 = 57,802,752 MACs
    # Proj (16->1, 1x1, 224x224): 1 * (16*1*1) * 224*224 = 802,816 MACs
    # Total MACs = 101,957,632 MACs (~0.102 GMACs)
    # Total FLOPs = 2 * MACs = 203,915,264 FLOPs (~0.204 GFLOPs)
    macs_per_image = 101957632
    flops_per_image = 2 * macs_per_image
    print(f"HRRB Trainable Parameters:    {trainable_numel}")
    print(f"HRRB Theoretical MACs/image:  {macs_per_image:,} ({macs_per_image / 1e9:.4f} GMACs)")
    print(f"HRRB Theoretical FLOPs/image: {flops_per_image:,} ({flops_per_image / 1e9:.4f} GFLOPs)")

    if torch.cuda.is_available():
        warmup_iters = 10
        measure_iters = 30
        batch_14_x = torch.randn(14, 3, 448, 448, device=device)

        def measure_condition(fwd_fn, bwd_fn=None):
            # Warmup
            for _ in range(warmup_iters):
                out = fwd_fn(batch_14_x)
                if bwd_fn is not None:
                    bwd_fn(out)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            fwd_times = []
            bwd_times = []
            for _ in range(measure_iters):
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                out = fwd_fn(batch_14_x)
                t1.record()
                torch.cuda.synchronize()
                fwd_times.append(t0.elapsed_time(t1))

                if bwd_fn is not None:
                    t2 = torch.cuda.Event(enable_timing=True)
                    t3 = torch.cuda.Event(enable_timing=True)
                    t2.record()
                    bwd_fn(out)
                    t3.record()
                    torch.cuda.synchronize()
                    bwd_times.append(t2.elapsed_time(t3))

            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
            fwd_arr = np.array(fwd_times)
            bwd_arr = np.array(bwd_times) if bwd_times else None

            stats = {
                "peak_vram_mb": peak_vram,
                "fwd_median_ms": float(np.median(fwd_arr)),
                "fwd_mean_ms": float(np.mean(fwd_arr)),
                "fwd_p90_ms": float(np.percentile(fwd_arr, 90)),
                "fwd_std_ms": float(np.std(fwd_arr)),
            }
            if bwd_arr is not None:
                stats.update({
                    "bwd_median_ms": float(np.median(bwd_arr)),
                    "bwd_mean_ms": float(np.mean(bwd_arr)),
                    "bwd_p90_ms": float(np.percentile(bwd_arr, 90)),
                    "bwd_std_ms": float(np.std(bwd_arr)),
                })
            return stats

        # 1. Condition A: Candidate B Standalone (forward only under no_grad)
        candidate_b.eval()
        with torch.no_grad():
            stats_b = measure_condition(lambda x: candidate_b(x))

        # 2. Condition B: HRRB Standalone
        hrrb_standalone = HighResolutionResidualBypass().to(device)
        hrrb_opt = torch.optim.AdamW(hrrb_standalone.parameters(), lr=1e-4)
        def hrrb_fwd(x):
            return hrrb_standalone(x)
        def hrrb_bwd(out):
            hrrb_opt.zero_grad()
            l = out.sum()
            l.backward()
        stats_hrrb = measure_condition(hrrb_fwd, hrrb_bwd)

        # 3. Condition C: Composite Model (Candidate B in no_grad forward + HRRB forward + backward)
        model.train()
        comp_opt = torch.optim.AdamW(model.hrrb.parameters(), lr=1e-4)
        def comp_fwd(x):
            return model(x)
        def comp_bwd(out):
            comp_opt.zero_grad()
            l = out.sum()
            l.backward()
        stats_comp = measure_condition(comp_fwd, comp_bwd)

        # Compute Differentials
        delta_vram = stats_comp["peak_vram_mb"] - stats_b["peak_vram_mb"]
        delta_fwd_median = stats_comp["fwd_median_ms"] - stats_b["fwd_median_ms"]
        delta_fwd_mean = stats_comp["fwd_mean_ms"] - stats_b["fwd_mean_ms"]

        print("\n--- MEASURED BENCHMARK SUMMARY (N=30 iterations after 10 warmup) ---")
        print("1. Candidate B Baseline (eval, torch.no_grad):")
        print(f"   Peak VRAM:        {stats_b['peak_vram_mb']:.2f} MB")
        print(f"   Forward Median:   {stats_b['fwd_median_ms']:.2f} ms ({stats_b['fwd_median_ms']/14:.2f} ms/img)")
        print(f"   Forward Mean:     {stats_b['fwd_mean_ms']:.2f} +/- {stats_b['fwd_std_ms']:.2f} ms")
        print(f"   Forward P90:      {stats_b['fwd_p90_ms']:.2f} ms")

        print("\n2. HRRB Standalone (train, forward + backward):")
        print(f"   Peak VRAM:        {stats_hrrb['peak_vram_mb']:.2f} MB")
        print(f"   Forward Median:   {stats_hrrb['fwd_median_ms']:.2f} ms ({stats_hrrb['fwd_median_ms']/14:.2f} ms/img)")
        print(f"   Backward Median:  {stats_hrrb['bwd_median_ms']:.2f} ms ({stats_hrrb['bwd_median_ms']/14:.2f} ms/img)")

        print("\n3. Composite Candidate B + HRRB (Candidate B no_grad, HRRB trainable):")
        print(f"   Peak VRAM:        {stats_comp['peak_vram_mb']:.2f} MB")
        print(f"   Forward Median:   {stats_comp['fwd_median_ms']:.2f} ms ({stats_comp['fwd_median_ms']/14:.2f} ms/img)")
        print(f"   Forward Mean:     {stats_comp['fwd_mean_ms']:.2f} +/- {stats_comp['fwd_std_ms']:.2f} ms")
        print(f"   Forward P90:      {stats_comp['fwd_p90_ms']:.2f} ms")
        print(f"   Backward Median:  {stats_comp['bwd_median_ms']:.2f} ms ({stats_comp['bwd_median_ms']/14:.2f} ms/img)")
        print(f"   Backward Mean:    {stats_comp['bwd_mean_ms']:.2f} +/- {stats_comp['bwd_std_ms']:.2f} ms")

        print("\n4. Differential Overhead of HRRB:")
        print(f"   Delta VRAM Peak:        {delta_vram:+.2f} MB (Peak Composite - Peak Candidate B)")
        print(f"   Delta Forward (Median): {delta_fwd_median:+.2f} ms ({delta_fwd_median/14:+.2f} ms/img)")
        print(f"   Delta Forward (Mean):   {delta_fwd_mean:+.2f} ms ({delta_fwd_mean/14:+.2f} ms/img)")
        print(f"   HRRB Backward (Median): {stats_comp['bwd_median_ms']:.2f} ms ({stats_comp['bwd_median_ms']/14:.2f} ms/img)")

        del batch_14_x
        torch.cuda.empty_cache()
    else:
        print("CUDA not available; memory & timing benchmark skipped on CPU.")

    print("\n" + "=" * 80)
    print(">> ALL PREFLIGHT INVARIANTS & 3-TIER INTEGRITY PASSED 100% BIT-EXACTLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_preflight_invariants()
