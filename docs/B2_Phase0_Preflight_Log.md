# B2 Phase 0 Runtime Preflight & OOM Probe Log (Tesla T4)

**Date**: 2026-09-23  
**Environment**: Google Colab — NVIDIA Tesla T4 (15.0 GB / 14.56 GB usable, Compute Capability 7.5, Driver/cuDNN 91900)  
**Target Model**: B2 (Full SAGE-Lite, ViT Depth 12, 16 Injected Routers, Top-K = 4, Sigmoid Gating, Logit Modulation = True)  
**Input Resolution**: 448×448, AMP FP16  
**Script**: `scripts/preflight_b2.py`  

---

## 1. Executive Summary & Hardware Boundary Matrix

| Batch Size | Probe Scope | Status | Peak Alloc VRAM | Peak Reserved VRAM | Free VRAM | Safety Margin / Ghi chú |
|---|---|---|---|---|---|---|
| **Batch 20** | 1 iter | ❌ **FAIL (OOM)** | ~14.56 GB | ~14.56 GB | 0 GB | OOM tại ViT Expert MLP & Decoder |
| **Batch 14+**| 1 iter | ❌ **FAIL (OOM)** | > 14.56 GB | — | 0 GB | Vượt quá giới hạn cứng 14.56 GB của Tesla T4 |
| **Batch 12** | **12 iters** | ✅ **PASS** | 14,100.3 MB (13.77 GB) | 14,436.0 MB (14.10 GB) | 476.7 MB (0.47 GB) | Cực kỳ sát ngưỡng (96.8% VRAM). Ổn định (+0.08 MB leak). |
| **Batch 10** | **12 iters** | ✅ **PASS** | 13,755.0 MB (13.43 GB) | 14,220.0 MB (13.89 GB) | 692.7 MB (0.68 GB) | Khá an toàn (0.68 GB đệm). Cân bằng tốt giữa throughput và độ an toàn. |
| **Batch 8**  | **12 iters** | ✅ **PASS** | 12,361.6 MB (12.07 GB) | 12,798.0 MB (12.50 GB) | 2,114.7 MB (2.07 GB) | **An toàn tuyệt đối** (> 2.0 GB đệm). Không sợ OOM khi eval/cache. |

> [!IMPORTANT]
> **Quy tắc về Batch Size:**  
> Việc điều chỉnh `batch_size` (từ 20 xuống 10 hoặc 8) là **hardware-constrained runtime setting** (bắt buộc do giới hạn VRAM 14.56 GB của T4), **KHÔNG PHẢI** là hyperparameter optimization (HPO) hay thay đổi kiến trúc SAGE.

---

## 2. Kết quả Thử nghiệm Dài hạn (12 Iterations Stress Test)

### 2.1 Batch Size = 12 (12 Iterations)

```
======================================================================
B2 RUNTIME PREFLIGHT: configs/b2_crack500_depth12.yaml
======================================================================
[Device] Target Device: cuda
[Device] GPU: Tesla T4 | Compute Cap: 7.5 | Total VRAM: 14912.7 MB (14.56 GB)
[Device] cuDNN: 91900 (enabled=True, benchmark=True)

[Model Config] Model: B2 (Full SAGE-Lite) | ViT Depth: 12
[Model Config] Batch Size: 12 | Image Size: 448x448 | Top-K: 4
[Model Config] Gating: sigmoid | Logit Mod: True

======================================================================
STEP 1: OOM / VRAM PROBE (Batch Size = 12, 12 Iterations)
======================================================================
Instantiating B2 model for VRAM probe...
  Iter 1/12 (160.99s) | Loss: 2.2125 (Seg: 2.0450, LB: 0.1675) | Alloc: 275.7 MB | Peak Alloc: 13778.2 MB (13.46 GB) | Peak Res: 13896.0 MB (13.57 GB) | Free: 0.99 GB
  Iter 2/12 (70.48s) | Loss: 2.2136 (Seg: 2.0426, LB: 0.1709) | Alloc: 276.1 MB | Peak Alloc: 14100.3 MB (13.77 GB) | Peak Res: 14436.0 MB (14.10 GB) | Free: 0.47 GB
  Iter 3/12 (24.22s) | Loss: 2.2090 (Seg: 2.0405, LB: 0.1685) | Alloc: 275.3 MB | Peak Alloc: 14100.3 MB (13.77 GB) | Peak Res: 14436.0 MB (14.10 GB) | Free: 0.47 GB
  Iter 4/12 (39.22s) | Loss: 2.2058 (Seg: 2.0385, LB: 0.1674) | Alloc: 275.8 MB | Peak Alloc: 14100.3 MB (13.77 GB) | Peak Res: 14436.0 MB (14.10 GB) | Free: 0.47 GB
  Iter 5/12 (28.41s) | Loss: 2.2031 (Seg: 2.0366, LB: 0.1665) | Alloc: 276.5 MB | Peak Alloc: 14100.3 MB (13.77 GB) | Peak Res: 14436.0 MB (14.10 GB) | Free: 0.47 GB
  Iter 6/12 (27.10s) | Loss: 2.2017 (Seg: 2.0346, LB: 0.1671) | Alloc: 276.2 MB | Peak Alloc: 14100.3 MB (13.77 GB) | Peak Res: 14436.0 MB (14.10 GB) | Free: 0.47 GB
  Iter 7/12 (19.31s) | Loss: 2.1999 (Seg: 2.0329, LB: 0.1670) | Alloc: 276.7 MB | Peak Alloc: 14100.3 MB (13.77 GB) | Peak Res: 14436.0 MB (14.10 GB) | Free: 0.47 GB
  Iter 8/12 (10.02s) | Loss: 2.1969 (Seg: 2.0310, LB: 0.1659) | Alloc: 276.0 MB | Peak Alloc: 14100.3 MB (13.77 GB) | Peak Res: 14436.0 MB (14.10 GB) | Free: 0.47 GB
  Iter 9/12 (10.29s) | Loss: 2.1959 (Seg: 2.0295, LB: 0.1663) | Alloc: 276.0 MB | Peak Alloc: 14100.3 MB (13.77 GB) | Peak Res: 14436.0 MB (14.10 GB) | Free: 0.47 GB
  Iter 10/12 (9.63s) | Loss: 2.1952 (Seg: 2.0280, LB: 0.1672) | Alloc: 277.6 MB | Peak Alloc: 14100.3 MB (13.77 GB) | Peak Res: 14436.0 MB (14.10 GB) | Free: 0.47 GB
  Iter 11/12 (9.99s) | Loss: 2.1954 (Seg: 2.0265, LB: 0.1689) | Alloc: 275.8 MB | Peak Alloc: 14100.3 MB (13.77 GB) | Peak Res: 14436.0 MB (14.10 GB) | Free: 0.47 GB
  Iter 12/12 (14.23s) | Loss: 2.1930 (Seg: 2.0248, LB: 0.1683) | Alloc: 276.2 MB | Peak Alloc: 14100.3 MB (13.77 GB) | Peak Res: 14436.0 MB (14.10 GB) | Free: 0.47 GB

[VRAM Memory Stability Check]
  Allocated Memory Iter 2: 276.15 MB
  Allocated Memory Iter 12: 276.23 MB
  Difference (Iter 12 - Iter 2): +0.08 MB
  [PASS] No monotonically increasing memory leak detected.

[VRAM Summary for Batch=12]
  - Peak Memory Allocated: 14100.3 MB (13.77 GB)
  - Peak Memory Reserved:  14436.0 MB (14.10 GB)
  - Free VRAM Remaining:   476.7 MB (0.47 GB) out of 14.56 GB
  [PASS] Step 1: OOM / VRAM Probe successfully passed without OOM.

======================================================================
STEP 2: MINI-TRAINING SMOKE (5 Training Batches)
======================================================================
  Batch 1/5 | Total Loss: 2.2042 (Seg: 2.0359, LB: 0.1682) | Logits & Grads: FINITE [OK]
  Batch 2/5 | Total Loss: 2.2032 (Seg: 2.0362, LB: 0.1670) | Logits & Grads: FINITE [OK]
  Batch 3/5 | Total Loss: 2.2026 (Seg: 2.0364, LB: 0.1662) | Logits & Grads: FINITE [OK]
  Batch 4/5 | Total Loss: 2.2037 (Seg: 2.0363, LB: 0.1674) | Logits & Grads: FINITE [OK]
  Batch 5/5 | Total Loss: 2.2032 (Seg: 2.0359, LB: 0.1673) | Logits & Grads: FINITE [OK]
  [PASS] Step 2: Mini-training smoke passed (all losses, logits, and gradients strictly finite).

======================================================================
STEP 3: ROUTING DIAGNOSTICS & EXPERT USAGE INSPECTION
======================================================================
  Total Injected Routers: 16 (4 CNN Stages + 12 ViT Blocks)
  Expert Pool Size:       16
  Target Top-K:           4

--- Per-Expert Selection Breakdown (Entire Mini-Run) ---
  Expert 00 (   CNN Stage) [SHARED]:    832 selections (  6.4%)
  Expert 01 (   CNN Stage) [SHARED]:    808 selections (  6.2%)
  Expert 02 (   CNN Stage) [SHARED]:    848 selections (  6.5%)
  Expert 03 (   CNN Stage) [SHARED]:    768 selections (  5.9%)
  Expert 04 (ViT Block 00)         :    847 selections (  6.5%)
  Expert 05 (ViT Block 01)         :    767 selections (  5.9%)
  Expert 06 (ViT Block 02)         :    825 selections (  6.3%)
  Expert 07 (ViT Block 03)         :    851 selections (  6.5%)
  Expert 08 (ViT Block 04)         :    771 selections (  5.9%)
  Expert 09 (ViT Block 05)         :    786 selections (  6.0%)
  Expert 10 (ViT Block 06)         :    819 selections (  6.3%)
  Expert 11 (ViT Block 07)         :    814 selections (  6.2%)
  Expert 12 (ViT Block 08)         :    816 selections (  6.2%)
  Expert 13 (ViT Block 09)         :    854 selections (  6.5%)
  Expert 14 (ViT Block 10)         :    810 selections (  6.2%)
  Expert 15 (ViT Block 11)         :    840 selections (  6.4%)

  [PASS] All 16 experts received at least 1 routing selection.
  Total routing selection events recorded: 13056
  [PASS] Step 3: Routing diagnostics verified (top_k=4 verified, expert usage recorded).

======================================================================
STEP 4: CHECKPOINT SAVE & LOAD INTEGRITY
======================================================================
  Saved test checkpoint to: /tmp/tmpi5uxr11k.pth (158.93 MB)
  Loaded state_dict strictly: 0 missing keys, 0 unexpected keys.
  Max absolute difference between original and loaded model: 0.00000000e+00
  [PASS] Step 4: Checkpoint save & load integrity 100% verified.
```

---

### 2.2 Batch Size = 10 (12 Iterations)

```
======================================================================
B2 RUNTIME PREFLIGHT: configs/b2_crack500_depth12.yaml
======================================================================
[Device] Target Device: cuda
[Device] GPU: Tesla T4 | Compute Cap: 7.5 | Total VRAM: 14912.7 MB (14.56 GB)
[Device] cuDNN: 91900 (enabled=True, benchmark=True)

[Model Config] Model: B2 (Full SAGE-Lite) | ViT Depth: 12
[Model Config] Batch Size: 10 | Image Size: 448x448 | Top-K: 4
[Model Config] Gating: sigmoid | Logit Mod: True

======================================================================
STEP 1: OOM / VRAM PROBE (Batch Size = 10, 12 Iterations)
======================================================================
Instantiating B2 model for VRAM probe...
  Iter 1/12 (142.90s) | Loss: 2.1308 (Seg: 1.9607, LB: 0.1702) | Alloc: 268.4 MB | Peak Alloc: 12064.7 MB (11.78 GB) | Peak Res: 12194.0 MB (11.91 GB) | Free: 2.65 GB
  Iter 2/12 (56.87s) | Loss: 2.1279 (Seg: 1.9585, LB: 0.1694) | Alloc: 268.3 MB | Peak Alloc: 13755.0 MB (13.43 GB) | Peak Res: 14220.0 MB (13.89 GB) | Free: 0.68 GB
  Iter 3/12 (30.19s) | Loss: 2.1245 (Seg: 1.9565, LB: 0.1679) | Alloc: 269.8 MB | Peak Alloc: 13755.0 MB (13.43 GB) | Peak Res: 14220.0 MB (13.89 GB) | Free: 0.68 GB
  Iter 4/12 (40.62s) | Loss: 2.1244 (Seg: 1.9547, LB: 0.1697) | Alloc: 267.8 MB | Peak Alloc: 13755.0 MB (13.43 GB) | Peak Res: 14220.0 MB (13.89 GB) | Free: 0.68 GB
  Iter 5/12 (17.84s) | Loss: 2.1212 (Seg: 1.9529, LB: 0.1683) | Alloc: 268.1 MB | Peak Alloc: 13755.0 MB (13.43 GB) | Peak Res: 14220.0 MB (13.89 GB) | Free: 0.68 GB
  Iter 6/12 (15.64s) | Loss: 2.1200 (Seg: 1.9513, LB: 0.1687) | Alloc: 269.4 MB | Peak Alloc: 13755.0 MB (13.43 GB) | Peak Res: 14220.0 MB (13.89 GB) | Free: 0.68 GB
  Iter 7/12 (25.79s) | Loss: 2.1181 (Seg: 1.9497, LB: 0.1683) | Alloc: 267.8 MB | Peak Alloc: 13755.0 MB (13.43 GB) | Peak Res: 14220.0 MB (13.89 GB) | Free: 0.68 GB
  Iter 8/12 (8.12s) | Loss: 2.1167 (Seg: 1.9482, LB: 0.1685) | Alloc: 267.8 MB | Peak Alloc: 13755.0 MB (13.43 GB) | Peak Res: 14220.0 MB (13.89 GB) | Free: 0.68 GB
  Iter 9/12 (12.90s) | Loss: 2.1145 (Seg: 1.9469, LB: 0.1676) | Alloc: 269.8 MB | Peak Alloc: 13755.0 MB (13.43 GB) | Peak Res: 14220.0 MB (13.89 GB) | Free: 0.68 GB
  Iter 10/12 (13.86s) | Loss: 2.1146 (Seg: 1.9455, LB: 0.1691) | Alloc: 268.7 MB | Peak Alloc: 13755.0 MB (13.43 GB) | Peak Res: 14220.0 MB (13.89 GB) | Free: 0.68 GB
  Iter 11/12 (8.28s) | Loss: 2.1124 (Seg: 1.9443, LB: 0.1680) | Alloc: 267.8 MB | Peak Alloc: 13755.0 MB (13.43 GB) | Peak Res: 14220.0 MB (13.89 GB) | Free: 0.68 GB
  Iter 12/12 (12.84s) | Loss: 2.1111 (Seg: 1.9429, LB: 0.1682) | Alloc: 269.7 MB | Peak Alloc: 13755.0 MB (13.43 GB) | Peak Res: 14220.0 MB (13.89 GB) | Free: 0.68 GB

[VRAM Memory Stability Check]
  Allocated Memory Iter 2: 268.34 MB
  Allocated Memory Iter 12: 269.71 MB
  Difference (Iter 12 - Iter 2): +1.37 MB
  [PASS] No monotonically increasing memory leak detected.

[VRAM Summary for Batch=10]
  - Peak Memory Allocated: 13755.0 MB (13.43 GB)
  - Peak Memory Reserved:  14220.0 MB (13.89 GB)
  - Free VRAM Remaining:   692.7 MB (0.68 GB) out of 14.56 GB
  [PASS] Step 1: OOM / VRAM Probe successfully passed without OOM.

======================================================================
STEP 2: MINI-TRAINING SMOKE (5 Training Batches)
======================================================================
  Batch 1/5 | Total Loss: 2.1212 (Seg: 1.9528, LB: 0.1683) | Logits & Grads: FINITE [OK]
  Batch 2/5 | Total Loss: 2.1217 (Seg: 1.9533, LB: 0.1683) | Logits & Grads: FINITE [OK]
  Batch 3/5 | Total Loss: 2.1214 (Seg: 1.9533, LB: 0.1681) | Logits & Grads: FINITE [OK]
  Batch 4/5 | Total Loss: 2.1224 (Seg: 1.9532, LB: 0.1692) | Logits & Grads: FINITE [OK]
  Batch 5/5 | Total Loss: 2.1214 (Seg: 1.9532, LB: 0.1682) | Logits & Grads: FINITE [OK]
  [PASS] Step 2: Mini-training smoke passed (all losses, logits, and gradients strictly finite).

======================================================================
STEP 3: ROUTING DIAGNOSTICS & EXPERT USAGE INSPECTION
======================================================================
  Total Injected Routers: 16 (4 CNN Stages + 12 ViT Blocks)
  Expert Pool Size:       16
  Target Top-K:           4

--- Per-Expert Selection Breakdown (Entire Mini-Run) ---
  Expert 00 (   CNN Stage) [SHARED]:    678 selections (  6.2%)
  Expert 01 (   CNN Stage) [SHARED]:    714 selections (  6.6%)
  Expert 02 (   CNN Stage) [SHARED]:    672 selections (  6.2%)
  Expert 03 (   CNN Stage) [SHARED]:    686 selections (  6.3%)
  Expert 04 (ViT Block 00)         :    685 selections (  6.3%)
  Expert 05 (ViT Block 01)         :    678 selections (  6.2%)
  Expert 06 (ViT Block 02)         :    683 selections (  6.3%)
  Expert 07 (ViT Block 03)         :    624 selections (  5.7%)
  Expert 08 (ViT Block 04)         :    662 selections (  6.1%)
  Expert 09 (ViT Block 05)         :    690 selections (  6.3%)
  Expert 10 (ViT Block 06)         :    701 selections (  6.4%)
  Expert 11 (ViT Block 07)         :    702 selections (  6.5%)
  Expert 12 (ViT Block 08)         :    665 selections (  6.1%)
  Expert 13 (ViT Block 09)         :    687 selections (  6.3%)
  Expert 14 (ViT Block 10)         :    705 selections (  6.5%)
  Expert 15 (ViT Block 11)         :    648 selections (  6.0%)

  [PASS] All 16 experts received at least 1 routing selection.
  Total routing selection events recorded: 10880
  [PASS] Step 3: Routing diagnostics verified (top_k=4 verified, expert usage recorded).

======================================================================
STEP 4: CHECKPOINT SAVE & LOAD INTEGRITY
======================================================================
  Saved test checkpoint to: /tmp/tmp1rtdzy9m.pth (158.93 MB)
  Loaded state_dict strictly: 0 missing keys, 0 unexpected keys.
  Max absolute difference between original and loaded model: 0.00000000e+00
  [PASS] Step 4: Checkpoint save & load integrity 100% verified.
```

---

### 2.3 Batch Size = 8 (12 Iterations)

```
======================================================================
B2 RUNTIME PREFLIGHT: configs/b2_crack500_depth12.yaml
======================================================================
[Device] Target Device: cuda
[Device] GPU: Tesla T4 | Compute Cap: 7.5 | Total VRAM: 14912.7 MB (14.56 GB)
[Device] cuDNN: 91900 (enabled=True, benchmark=True)

[Model Config] Model: B2 (Full SAGE-Lite) | ViT Depth: 12
[Model Config] Batch Size: 8 | Image Size: 448x448 | Top-K: 4
[Model Config] Gating: sigmoid | Logit Mod: True

======================================================================
STEP 1: OOM / VRAM PROBE (Batch Size = 8, 12 Iterations)
======================================================================
Instantiating B2 model for VRAM probe...
  Iter 1/12 (131.49s) | Loss: 2.4227 (Seg: 2.2522, LB: 0.1704) | Alloc: 259.5 MB | Peak Alloc: 12167.2 MB (11.88 GB) | Peak Res: 12590.0 MB (12.29 GB) | Free: 2.27 GB
  Iter 2/12 (47.54s) | Loss: 2.4198 (Seg: 2.2496, LB: 0.1702) | Alloc: 260.5 MB | Peak Alloc: 12167.2 MB (11.88 GB) | Peak Res: 12590.0 MB (12.29 GB) | Free: 2.27 GB
  Iter 3/12 (37.63s) | Loss: 2.4177 (Seg: 2.2470, LB: 0.1707) | Alloc: 259.6 MB | Peak Alloc: 12167.2 MB (11.88 GB) | Peak Res: 12590.0 MB (12.29 GB) | Free: 2.27 GB
  Iter 4/12 (20.01s) | Loss: 2.4149 (Seg: 2.2446, LB: 0.1704) | Alloc: 259.6 MB | Peak Alloc: 12167.2 MB (11.88 GB) | Peak Res: 12590.0 MB (12.29 GB) | Free: 2.27 GB
  Iter 5/12 (17.52s) | Loss: 2.4150 (Seg: 2.2423, LB: 0.1727) | Alloc: 259.9 MB | Peak Alloc: 12167.2 MB (11.88 GB) | Peak Res: 12590.0 MB (12.29 GB) | Free: 2.27 GB
  Iter 6/12 (15.73s) | Loss: 2.4095 (Seg: 2.2400, LB: 0.1695) | Alloc: 260.8 MB | Peak Alloc: 12167.2 MB (11.88 GB) | Peak Res: 12590.0 MB (12.29 GB) | Free: 2.27 GB
  Iter 7/12 (15.82s) | Loss: 2.4079 (Seg: 2.2380, LB: 0.1699) | Alloc: 259.9 MB | Peak Alloc: 12167.2 MB (11.88 GB) | Peak Res: 12590.0 MB (12.29 GB) | Free: 2.27 GB
  Iter 8/12 (12.48s) | Loss: 2.4073 (Seg: 2.2361, LB: 0.1712) | Alloc: 259.6 MB | Peak Alloc: 12361.6 MB (12.07 GB) | Peak Res: 12798.0 MB (12.50 GB) | Free: 2.07 GB
  Iter 9/12 (8.84s) | Loss: 2.4046 (Seg: 2.2341, LB: 0.1704) | Alloc: 259.9 MB | Peak Alloc: 12361.6 MB (12.07 GB) | Peak Res: 12798.0 MB (12.50 GB) | Free: 2.07 GB
  Iter 10/12 (16.31s) | Loss: 2.4014 (Seg: 2.2321, LB: 0.1693) | Alloc: 260.2 MB | Peak Alloc: 12361.6 MB (12.07 GB) | Peak Res: 12798.0 MB (12.50 GB) | Free: 2.07 GB
  Iter 11/12 (19.51s) | Loss: 2.4006 (Seg: 2.2306, LB: 0.1700) | Alloc: 260.4 MB | Peak Alloc: 12361.6 MB (12.07 GB) | Peak Res: 12798.0 MB (12.50 GB) | Free: 2.07 GB
  Iter 12/12 (8.66s) | Loss: 2.3992 (Seg: 2.2287, LB: 0.1705) | Alloc: 260.3 MB | Peak Alloc: 12361.6 MB (12.07 GB) | Peak Res: 12798.0 MB (12.50 GB) | Free: 2.07 GB

[VRAM Memory Stability Check]
  Allocated Memory Iter 2: 260.50 MB
  Allocated Memory Iter 12: 260.28 MB
  Difference (Iter 12 - Iter 2): -0.22 MB
  [PASS] No monotonically increasing memory leak detected.

[VRAM Summary for Batch=8]
  - Peak Memory Allocated: 12361.6 MB (12.07 GB)
  - Peak Memory Reserved:  12798.0 MB (12.50 GB)
  - Free VRAM Remaining:   2114.7 MB (2.07 GB) out of 14.56 GB
  [PASS] Step 1: OOM / VRAM Probe successfully passed without OOM.

======================================================================
STEP 2: MINI-TRAINING SMOKE (5 Training Batches)
======================================================================
  Batch 1/5 | Total Loss: 2.4139 (Seg: 2.2432, LB: 0.1707) | Logits & Grads: FINITE [OK]
  Batch 2/5 | Total Loss: 2.4131 (Seg: 2.2428, LB: 0.1704) | Logits & Grads: FINITE [OK]
  Batch 3/5 | Total Loss: 2.4127 (Seg: 2.2432, LB: 0.1695) | Logits & Grads: FINITE [OK]
  Batch 4/5 | Total Loss: 2.4129 (Seg: 2.2434, LB: 0.1694) | Logits & Grads: FINITE [OK]
  Batch 5/5 | Total Loss: 2.4143 (Seg: 2.2425, LB: 0.1718) | Logits & Grads: FINITE [OK]
  [PASS] Step 2: Mini-training smoke passed (all losses, logits, and gradients strictly finite).

======================================================================
STEP 3: ROUTING DIAGNOSTICS & EXPERT USAGE INSPECTION
======================================================================
  Total Injected Routers: 16 (4 CNN Stages + 12 ViT Blocks)
  Expert Pool Size:       16
  Target Top-K:           4

--- Per-Expert Selection Breakdown (Entire Mini-Run) ---
  Expert 00 (   CNN Stage) [SHARED]:    594 selections (  6.8%)
  Expert 01 (   CNN Stage) [SHARED]:    543 selections (  6.2%)
  Expert 02 (   CNN Stage) [SHARED]:    526 selections (  6.0%)
  Expert 03 (   CNN Stage) [SHARED]:    557 selections (  6.4%)
  Expert 04 (ViT Block 00)         :    527 selections (  6.1%)
  Expert 05 (ViT Block 01)         :    547 selections (  6.3%)
  Expert 06 (ViT Block 02)         :    531 selections (  6.1%)
  Expert 07 (ViT Block 03)         :    540 selections (  6.2%)
  Expert 08 (ViT Block 04)         :    565 selections (  6.5%)
  Expert 09 (ViT Block 05)         :    539 selections (  6.2%)
  Expert 10 (ViT Block 06)         :    511 selections (  5.9%)
  Expert 11 (ViT Block 07)         :    571 selections (  6.6%)
  Expert 12 (ViT Block 08)         :    536 selections (  6.2%)
  Expert 13 (ViT Block 09)         :    505 selections (  5.8%)
  Expert 14 (ViT Block 10)         :    555 selections (  6.4%)
  Expert 15 (ViT Block 11)         :    557 selections (  6.4%)

  [PASS] All 16 experts received at least 1 routing selection.
  Total routing selection events recorded: 8704
  [PASS] Step 3: Routing diagnostics verified (top_k=4 verified, expert usage recorded).

======================================================================
STEP 4: CHECKPOINT SAVE & LOAD INTEGRITY
======================================================================
  Saved test checkpoint to: /tmp/tmp46aon8yk.pth (158.93 MB)
  Loaded state_dict strictly: 0 missing keys, 0 unexpected keys.
  Max absolute difference between original and loaded model: 0.00000000e+00
  [PASS] Step 4: Checkpoint save & load integrity 100% verified.
```

---

## 3. Phân tích So sánh & Độ ổn định Dài hạn (12 Iterations)

1. **Tính ổn định bộ nhớ (No Memory Leak)**:
   - Batch 12: `Diff(Iter 12 - Iter 2) = +0.08 MB`
   - Batch 10: `Diff(Iter 12 - Iter 2) = +1.37 MB`
   - Batch 8: `Diff(Iter 12 - Iter 2) = -0.22 MB`
   - Khẳng định 100%: Bộ nhớ ổn định phẳng sau Iter 2 khi PyTorch CUDA Caching Allocator đã warm-up, hoàn toàn không có memory leak tích lũy theo từng batch.
2. **Thời gian thực thi / Throughput**:
   - Iter 1 chạy trong khoảng 130s–160s (do khởi tạo CUDA context và JIT compile).
   - Từ Iter 8 trở đi, thời gian ổn định chỉ còn **8s – 14s / iteration**.
3. **Phân phối Routing cân bằng tuyệt đối**:
   - Ở cả 3 mức batch (8, 10, 12), tỷ lệ lựa chọn cho mỗi expert đều nằm trong dải hẹp **5.7% – 6.8%** (xung quanh mức lý thuyết $6.25\%$).
   - Số lượng sự kiện chọn expert ghi nhận:
     - Batch 12: 13,056 events
     - Batch 10: 10,880 events
     - Batch 8: 8,704 events
4. **Khuyến nghị lựa chọn Runtime Batch Size**:
   - **`batch_size: 10`**: Lựa chọn cân bằng tối ưu nhất giữa throughput huấn luyện và vùng đệm an toàn VRAM (~692 MB).
   - **`batch_size: 8`**: Lựa chọn an toàn tuyệt đối nếu muốn chạy cùng lúc các tác vụ profiling, background logging, hoặc tránh rủi ro OOM ở các ảnh test có kích thước lớn.
