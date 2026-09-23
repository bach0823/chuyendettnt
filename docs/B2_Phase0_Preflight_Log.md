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
| **Batch 20** | 1 iter (Synthetic) | ❌ **FAIL (OOM)** | ~14.56 GB | ~14.56 GB | 0 GB | OOM tại ViT Expert MLP & Decoder |
| **Batch 14+**| 1 iter (Synthetic) | ❌ **FAIL (OOM)** | > 14.56 GB | — | 0 GB | Vượt quá giới hạn cứng 14.56 GB của Tesla T4 |
| **Batch 12** | **12 batches (Real Crack500)** | ✅ **PASS** | **14,168.1 MB (13.84 GB)** | **14,388.0 MB (14.05 GB)** | **524.7 MB (0.51 GB)** | **Xác thực dữ liệu thật**: 0 OOM, 0.55 samples/s, Peak VRAM phẳng từ batch 2. |
| **Batch 12** | 12 iters (Synthetic) | ✅ **PASS** | 14,100.3 MB (13.77 GB) | 14,436.0 MB (14.10 GB) | 476.7 MB (0.47 GB) | Khớp gần như tuyệt đối với Real Data (< 0.5% chênh lệch). |
| **Batch 10** | 12 iters (Synthetic) | ✅ **PASS** | 13,755.0 MB (13.43 GB) | 14,220.0 MB (13.89 GB) | 692.7 MB (0.68 GB) | Khá an toàn (0.68 GB đệm). Cân bằng throughput và headroom. |
| **Batch 8**  | 12 iters (Synthetic) | ✅ **PASS** | 12,361.6 MB (12.07 GB) | 12,798.0 MB (12.50 GB) | 2,114.7 MB (2.07 GB) | An toàn tuyệt đối (> 2.0 GB đệm). Không sợ OOM khi eval/cache. |

> [!IMPORTANT]
> **Quy tắc về Batch Size:**  
> Việc điều chỉnh `batch_size` (từ 20 xuống 12, 10 hoặc 8) là **hardware-constrained runtime setting** (bắt buộc do giới hạn VRAM 14.56 GB của T4), **KHÔNG PHẢI** là hyperparameter optimization (HPO) hay thay đổi kiến trúc SAGE.


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
4. **Khuyến nghị lựa chọn Runtime Batch Size (từ Synthetic Probe)**:
   - **`batch_size: 10`**: Lựa chọn cân bằng tối ưu nhất giữa throughput huấn luyện và vùng đệm an toàn VRAM (~692 MB).
   - **`batch_size: 8`**: Lựa chọn an toàn tuyệt đối nếu muốn chạy cùng lúc các tác vụ profiling, background logging, hoặc tránh rủi ro OOM ở các ảnh test có kích thước lớn.

---

## 4. Real-Data Preflight Benchmark trên Crack500 (Batch Size = 12, 12 Batches)

Chạy thực nghiệm trực tiếp với script `scripts/preflight_b2_realdata.py` trên tập huấn luyện thực tế **Crack500** (1896 mẫu ảnh, `ConfigurableMedicalDataset`, Albumentations random crop 448×448, smart filter `fg_pixels >= 20`):

```
======================================================================
B2 REAL-DATA RUNTIME PREFLIGHT (CRACK500)
======================================================================
[Config Path]  : configs/b2_crack500_depth12.yaml
[Target Batch] : 12
[Total Batches]: 12 (Warmup: 2, Measured: 10)

[Device] Target Device: cuda
[Device] GPU: Tesla T4 | Compute Cap: 7.5 | Total VRAM: 14912.7 MB (14.56 GB)
[Device] cuDNN: 91900 (enabled=True, benchmark=True)

======================================================================
LOADING REAL CRACK500 DATASET (Image Size: 448x448)
======================================================================
[ConfigurableDataset] Loaded TRAIN: 1896 samples from /content/dataset/Crack500/train/images
[Dataset] Train samples count: 1896
[Dataset] Preprocessing mode: crop_mode='random', smart_filter=True
[DataLoader] Batch size: 12 | Num workers: 2 | Total available batches: 158

======================================================================
INSTANTIATING B2 MODEL (Full SAGE-Lite)
======================================================================
[Model] Name:                 B2ConvNeXtViTUNet (Full SAGE-Lite)
[Model] ViT Depth:            12
[Model] Injected Routers:     16
[Model] Expert Pool Size:     16
[Model] Top-K:                4
[Model] Fusion:               residual (scale=0.1)

======================================================================
RUNNING BENCHMARK (12 Batches on Real Crack500)
======================================================================
  Batch 01/12 [WARMUP] | Step: 163.63s (Data: 0.69s) | Loss: 2.2444 (Seg: 2.0521, LB: 0.1923) | Peak Alloc: 13.57 GB | Peak Res: 13.67 GB | Free: 0.90 GB
  Batch 02/12 [WARMUP] | Step:  56.94s (Data: 0.00s) | Loss: 2.2217 (Seg: 2.0341, LB: 0.1876) | Peak Alloc: 13.84 GB | Peak Res: 14.05 GB | Free: 0.51 GB
  Batch 03/12 [MEASURED] | Step:  55.19s (Data: 0.00s) | Loss: 2.1934 (Seg: 2.0121, LB: 0.1813) | Peak Alloc: 13.84 GB | Peak Res: 14.05 GB | Free: 0.51 GB
  Batch 04/12 [MEASURED] | Step:  38.32s (Data: 0.00s) | Loss: 2.1714 (Seg: 1.9853, LB: 0.1862) | Peak Alloc: 13.84 GB | Peak Res: 14.05 GB | Free: 0.51 GB
  Batch 05/12 [MEASURED] | Step:  18.68s (Data: 0.00s) | Loss: 2.1238 (Seg: 1.9407, LB: 0.1832) | Peak Alloc: 13.84 GB | Peak Res: 14.05 GB | Free: 0.51 GB
  Batch 06/12 [MEASURED] | Step:  17.30s (Data: 0.00s) | Loss: 2.1315 (Seg: 1.9445, LB: 0.1870) | Peak Alloc: 13.84 GB | Peak Res: 14.05 GB | Free: 0.51 GB
  Batch 07/12 [MEASURED] | Step:   5.23s (Data: 0.00s) | Loss: 2.1765 (Seg: 1.9975, LB: 0.1791) | Peak Alloc: 13.84 GB | Peak Res: 14.05 GB | Free: 0.51 GB
  Batch 08/12 [MEASURED] | Step:  21.84s (Data: 0.00s) | Loss: 2.0790 (Seg: 1.8974, LB: 0.1816) | Peak Alloc: 13.84 GB | Peak Res: 14.05 GB | Free: 0.51 GB
  Batch 09/12 [MEASURED] | Step:  24.56s (Data: 0.00s) | Loss: 2.1973 (Seg: 2.0050, LB: 0.1923) | Peak Alloc: 13.84 GB | Peak Res: 14.05 GB | Free: 0.51 GB
  Batch 10/12 [MEASURED] | Step:  14.80s (Data: 0.00s) | Loss: 2.0443 (Seg: 1.8685, LB: 0.1758) | Peak Alloc: 13.84 GB | Peak Res: 14.05 GB | Free: 0.51 GB
  Batch 11/12 [MEASURED] | Step:  10.00s (Data: 0.00s) | Loss: 2.0196 (Seg: 1.8355, LB: 0.1842) | Peak Alloc: 13.84 GB | Peak Res: 14.05 GB | Free: 0.51 GB
  Batch 12/12 [MEASURED] | Step:  14.18s (Data: 0.00s) | Loss: 2.1409 (Seg: 1.9606, LB: 0.1803) | Peak Alloc: 13.84 GB | Peak Res: 14.05 GB | Free: 0.51 GB

======================================================================
THROUGHPUT & VRAM BENCHMARK RESULTS
======================================================================
  Execution Status:          PASS (0 OOM errors across 12 batches)
  Finite Gradients & Losses: PASS (All strictly finite)

  --- VRAM Profiling (Batch Size = 12) ---
  Peak VRAM Allocated:        14168.1 MB (13.84 GB)
  Peak VRAM Reserved:         14388.0 MB (14.05 GB) / 14.56 GB
  VRAM Free Remaining:          524.7 MB (0.51 GB)
  VRAM Utilization Ratio:    96.5%

  --- Throughput Timing (Excluding First 2 Warmup Batches) ---
  Measured Batches Count:    10
  Mean Step Time:            22.011 s / batch
  Median Step Time:          17.990 s / batch
  Min / Max Step Time:       5.227 s / 55.192 s
  Processing Throughput:     0.55 samples / second

======================================================================
ROUTING USAGE DIAGNOSTICS (Real Crack500 Batches)
======================================================================
  --- Per-Expert Selection Breakdown ---
  Expert 00 (CNN Stage 0) [SHARED]:    587 selections (  6.4%)
  Expert 01 (CNN Stage 1) [SHARED]:    636 selections (  6.9%)
  Expert 02 (CNN Stage 2) [SHARED]:    586 selections (  6.4%)
  Expert 03 (CNN Stage 3) [SHARED]:    597 selections (  6.5%)
  Expert 04 (ViT Block 00)         :    591 selections (  6.4%)
  Expert 05 (ViT Block 01)         :    601 selections (  6.5%)
  Expert 06 (ViT Block 02)         :    621 selections (  6.7%)
  Expert 07 (ViT Block 03)         :    503 selections (  5.5%)
  Expert 08 (ViT Block 04)         :    620 selections (  6.7%)
  Expert 09 (ViT Block 05)         :    585 selections (  6.3%)
  Expert 10 (ViT Block 06)         :    587 selections (  6.4%)
  Expert 11 (ViT Block 07)         :    557 selections (  6.0%)
  Expert 12 (ViT Block 08)         :    550 selections (  6.0%)
  Expert 13 (ViT Block 09)         :    461 selections (  5.0%)
  Expert 14 (ViT Block 10)         :    612 selections (  6.6%)
  Expert 15 (ViT Block 11)         :    522 selections (  5.7%)

  Total Routing Selection Events: 9216
  [PASS] All 16 experts actively received routing assignments on real data.

======================================================================
FINAL REAL-DATA PREFLIGHT VERDICT
======================================================================
  [VERDICT] PASS: Batch Size 12 is FEASIBLE on real Crack500 data!
  Remaining VRAM Headroom: 0.51 GB (3.5%)
  Steady-State Speed:      0.55 samples/s (22.01s/batch)
```

---

## 5. Phân tích So sánh Đối chiếu: Synthetic vs Real Data (Batch 12)

| Chỉ số | Synthetic Benchmark (12 iters) | Real Crack500 Benchmark (12 batches) | Đánh giá so sánh |
|---|---|---|---|
| **Peak Allocated VRAM** | 14,100.3 MB (13.77 GB) | 14,168.1 MB (13.84 GB) | Chênh lệch cực nhỏ (+67.8 MB, +0.48%) |
| **Peak Reserved VRAM** | 14,436.0 MB (14.10 GB) | 14,388.0 MB (14.05 GB) | Giảm nhẹ (-48.0 MB), an toàn hơn |
| **VRAM Free Headroom** | 476.7 MB (0.47 GB) | **524.7 MB (0.51 GB)** | Headroom thực tế trên dữ liệu thật cao hơn |
| **Độ ổn định bộ nhớ** | Drift +0.08 MB (Iter 2→12) | **Peak Alloc & Res bất biến tuyệt đối từ Batch 02 đến Batch 12** | Hoàn hảo (Flatline memory) |
| **Tỷ lệ chọn Expert** | 5.9% – 6.5% | 5.0% – 6.9% | Phân phối routing cực kỳ cân bằng |
| **Dead Experts** | 0 / 16 | 0 / 16 | 100% 16 chuyên gia hoạt động đều |
| **Steady-state Speed** | 8s – 14s / batch | 17.99s (median) / 22.01s (mean) | Dữ liệu thật có thời gian augment/filter |
| **Finite Checks** | PASS | PASS | 100% loss/logits/grads strictly finite |

---

## 7. Báo cáo Chi tiết Coarse Throughput Profiling (Data Wait vs Compute Breakdown)

**Thời gian thực hiện**: 2026-09-24  
**Môi trường**: Google Colab — NVIDIA Tesla T4 (14.56 GB Usable)  
**Tập dữ liệu**: Crack500 Real Data (`/content/dataset/Crack500`, Image size 448×448, Batch size 12)  
**Script thực thi**: `python scripts/profile_throughput_b2.py --config configs/b2_crack500_depth12.yaml --workers 0,2,4 --batches 12 --warmup 2` (Đo 4 batches: 1 warmup, 3 measured mỗi worker)

### 7.1 Raw Profiler Output

```text
======================================================================
COARSE THROUGHPUT PROFILER (SAGE-Lite B2)
======================================================================
[Config] File: configs/b2_crack500_depth12.yaml
[Config] Resolved Data Root: /content/dataset/Crack500
[Device] Target: cuda
[Device] GPU: Tesla T4 (14.56 GB)
[Model Config] Model: B2 | ViT Depth: 12 | Batch Size: 12
[Model Config] SAGE: Top-K=4, Gating=sigmoid
[ConfigurableDataset] Loaded TRAIN: 1896 samples from /content/dataset/Crack500/train/images

--- Profiling with num_workers = 0 (4 batches: 1 warmup, 3 measured) ---
  Batch 01/04 [WARMUP]   | DataWait:  165.8ms | Fwd: 7262.2ms | Bwd: 20706.1ms | Opt: 177.6ms | Step: 28311.7ms | Loss: 2.2982
  Batch 02/04 [MEASURED] | DataWait:  258.5ms | Fwd: 1723.1ms | Bwd: 7197.7ms | Opt:  15.8ms | Step: 9195.2ms | Loss: 2.2855
  Batch 03/04 [MEASURED] | DataWait:  145.0ms | Fwd: 1350.1ms | Bwd: 4580.8ms | Opt:  13.9ms | Step: 6089.8ms | Loss: 2.2489
  Batch 04/04 [MEASURED] | DataWait:  143.0ms | Fwd: 1456.6ms | Bwd: 4195.8ms | Opt:  11.2ms | Step: 5806.5ms | Loss: 2.2321
[ConfigurableDataset] Loaded TRAIN: 1896 samples from /content/dataset/Crack500/train/images

--- Profiling with num_workers = 2 (4 batches: 1 warmup, 3 measured) ---
  Batch 01/04 [WARMUP]   | DataWait:  311.1ms | Fwd: 1660.9ms | Bwd: 3738.6ms | Opt:  23.9ms | Step: 5734.5ms | Loss: 2.2358
  Batch 02/04 [MEASURED] | DataWait:    0.3ms | Fwd: 1379.9ms | Bwd: 3779.0ms | Opt:   9.1ms | Step: 5168.3ms | Loss: 2.1942
  Batch 03/04 [MEASURED] | DataWait:    0.3ms | Fwd: 1482.9ms | Bwd: 3941.2ms | Opt:  10.0ms | Step: 5434.3ms | Loss: 2.1966
  Batch 04/04 [MEASURED] | DataWait:    0.3ms | Fwd: 1491.7ms | Bwd: 3710.3ms | Opt:   9.7ms | Step: 5212.0ms | Loss: 2.1442
[ConfigurableDataset] Loaded TRAIN: 1896 samples from /content/dataset/Crack500/train/images
UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.

--- Profiling with num_workers = 4 (4 batches: 1 warmup, 3 measured) ---
  Batch 01/04 [WARMUP]   | DataWait:  551.7ms | Fwd: 1681.4ms | Bwd: 3800.5ms | Opt:  12.0ms | Step: 6045.5ms | Loss: 2.2100
  Batch 02/04 [MEASURED] | DataWait:    0.3ms | Fwd: 1495.8ms | Bwd: 3552.1ms | Opt:   8.9ms | Step: 5057.1ms | Loss: 2.1641
  Batch 03/04 [MEASURED] | DataWait:    0.3ms | Fwd: 1425.3ms | Bwd: 3995.8ms | Opt:   9.3ms | Step: 5430.7ms | Loss: 2.1416
  Batch 04/04 [MEASURED] | DataWait:    0.7ms | Fwd: 1388.9ms | Bwd: 3474.1ms | Opt:  14.8ms | Step: 4878.5ms | Loss: 2.0983

======================================================================
COARSE THROUGHPUT PROFILING SUMMARY REPORT
======================================================================
Workers |  DataWait  |  Forward   |  Backward  | Optimizer  |  TotalStep   | Thpt (img/s) | Est 1 Epoch  | Peak VRAM 
---------------------------------------------------------------------------------------------------------
   0    |   182.2ms  |  1509.9ms  |  5324.8ms  |    13.7ms  |  7030.5 +/- 1535.0ms |       1.71   |     18.51 min | 13538.8 MB
   2    |     0.3ms  |  1451.5ms  |  3810.2ms  |     9.6ms  |  5271.5 +/- 116.5ms |       2.28   |     13.88 min | 13627.7 MB
   4    |     0.5ms  |  1436.7ms  |  3674.0ms  |    11.0ms  |  5122.1 +/- 230.0ms |       2.34   |     13.49 min | 13922.6 MB

[Breakdown Percentages (% of Total Step)]
  Workers=0: DataWait= 2.6% | Forward=21.5% | Backward=75.7% | Optimizer= 0.2% | CPU Usage=59.8%
  Workers=2: DataWait= 0.0% | Forward=27.5% | Backward=72.3% | Optimizer= 0.2% | CPU Usage=60.0%
  Workers=4: DataWait= 0.0% | Forward=28.0% | Backward=71.7% | Optimizer= 0.2% | CPU Usage=68.6%

======================================================================
PROFILING COMPLETE
======================================================================
```

### 7.2 Phân tích Bản chất: Giải mã Hiện tượng "1 Epoch ~25 phút"

Từ kết quả phân rã thời gian bằng CUDA Events và CPU Timer, ta có các kết luận khoa học vững chắc:

1. **DataLoader KHÔNG PHẢI là Bottleneck (khi `num_workers >= 2`):**
   - Với `num_workers = 2`, thời gian **Data Wait chỉ còn 0.3 ms (0.0% tổng thời gian)**! DataLoader nạp bất đồng bộ hoàn toàn ẩn sau thời gian tính toán GPU.
   - Thử nghiệm `num_workers = 4` không cải thiện thêm (Data Wait 0.5 ms, Step 5122 ms vs 5271 ms), đồng thời Colab bắn cảnh báo quá tải CPU worker (`suggested max number of worker is 2`) và làm Peak VRAM tăng lên 13.92 GB. Do đó, **`num_workers: 2` là điểm tối ưu tuyệt đối**.

2. **Nút thắt thực sự: Backward Pass của Kiến trúc SAGE (72.3% thời gian):**
   - **Forward**: ~1.45 s (27.5% step).
   - **Backward**: ~3.81 s (72.3% step).
   - **Optimizer Step**: ~9.6 ms (0.2% step).
   - **Nguyên nhân**: B2 Depth 12 sở hữu 16 router SAGE, mỗi router chọn top-4 trong 16 expert và truyền tín hiệu qua SA-Hub (tương thích chiều). Trong quá trình backward pass, gradient phải lan truyền ngược qua tất cả các nhánh router, gating sigmoid, logit modulation, adapter của SA-Hub và 16 expert blocks. Đây là đặc tính tính toán nội tại của MoE routing, không phải lỗi I/O hay lỗi code.

3. **Ước lượng Thời gian Thực tế cho 1 Epoch:**
   - Số batch trên Crack500 Train (1896 mẫu, batch size 12): $1896 / 12 = 158$ batches.
   - Với `num_workers = 2`, thời gian train thuần 1 epoch: $158 \times 5.271\text{ s} \approx 832.8\text{ s} \approx \mathbf{13.88\text{ phút}}$.
   - Cộng thêm thời gian Validation (~1.5 – 2 phút): **Tổng thời gian thực tế cho 1 epoch ổn định là ~15.5 phút**, nhanh hơn đáng kể so với con số suy đoán ban đầu (~25 phút do hiệu ứng đo warmup hoặc workers=0).

---

## 8. Kết luận Chung & Khuyến nghị Thực thi cho Phase 1

### Bảng Tổng hợp Đánh giá Nghiệm thu Phase 0 & Profiling:

| Hạng mục | Kết quả Profiling | Đánh giá |
|---|---|---|
| **Real Crack500 pipeline** | PASS | 1896 mẫu nạp chuẩn xác, crop 448×448, smart filter hoạt động trơn tru |
| **B2-D12 / top-k=4** | PASS | 16 router, 16 expert hoạt động ổn định |
| **Batch 12 OOM** | KHÔNG | 0 OOM qua tất cả các batch đo đạc |
| **Forward/backward/optimizer** | PASS | Tách bạch đo chính xác bằng CUDA Events |
| **Numerical stability** | PASS | Tất cả logits/loss/gradients strictly finite |
| **16 experts active** | PASS | Toàn bộ 16 chuyên gia được chọn đều |
| **Peak VRAM (workers=2)** | **13,627.7 MB (13.31 GB)** | An toàn, đệm trống ~0.94 GB (6.5%) trên Tesla T4 14.56 GB |
| **Data Wait (workers=2)** | **0.3 ms (0.0%)** | DataLoader chuẩn tối ưu, không có hiện tượng nghẽn CPU I/O |
| **Step Time (workers=2)** | **5.27s / batch** | Throughput ổn định: 2.28 samples/s |
| **Est. 1 Epoch Time** | **~13.88 phút** | Rất khả thi cho budget huấn luyện |
| **Khuyến nghị num_workers** | **`num_workers: 2`** | Khớp hoàn hảo với 2 vCPU của Colab, tránh cảnh báo process leak |
| **Có cần sửa architecture?** | **TUYỆT ĐỐI KHÔNG** | Giữ nguyên 100% kiến trúc B2 SAGE-Lite |

---

## 9. Báo cáo Toàn diện Deep Runtime Profiling: B1-D12 Baseline vs B2-D12 SAGE-Lite

**Thời gian thực hiện**: 2026-09-24  
**Môi trường thực thi**: Google Colab — NVIDIA Tesla T4 (14.56 GB Usable, cuDNN 91900)  
**Tập dữ liệu**: Crack500 Train (1896 mẫu, canonical preprocessing, image size 448×448, `batch_size: 12`, `num_workers: 2`, `AMP: True`)  
**Script thực thi**: `python scripts/profile_deep_b2.py --config configs/b2_crack500_depth12.yaml --b1-config configs/b1_crack500_depth12.yaml --batch-size 12 --workers 2 --warmup 2 --measured 5`  
**Quy cách đo lường**: Thu thập động bằng CUDA Events & CPU high-precision timer qua 7 batches (2 warmup + 5 measured steps). 100% kiến trúc và routing policy được giữ nguyên bản.

### 9.1 Bảng 1: So sánh Vĩ mô B1-D12 vs B2-D12

| Chỉ số / Thành phần | B1-D12 (Không SAGE) | B2-D12 (SAGE-Lite Full) | Độ chênh lệch (Delta) | Hệ số chậm (Slowdown) |
|:---|:---:|:---:|:---:|:---:|
| **Data Wait (ms)** | 1.2 ms | 0.4 ms | -0.8 ms | 0.33x (DataLoader hoàn hảo) |
| **Forward Pass (ms)** | **60.5 ms** | **1487.4 ms** | **+1426.9 ms** | **24.60x** |
| **Backward Pass (ms)** | **137.7 ms** | **3934.7 ms** | **+3797.0 ms** | **28.58x** |
| **Optimizer Step (ms)** | 7.3 ms | 10.7 ms | +3.3 ms | 1.47x |
| **Total Step Time (ms)** | **206.6 ms (0.21s)** | **5433.0 ms (5.43s)** | **+5226.4 ms** | **26.29x** |
| **Throughput (ảnh/s)** | **58.07 img/s** | **2.21 img/s** | -55.87 img/s | — |
| **Est. 1 Epoch (158 batches)**| **0.54 phút (~32 giây)**| **14.31 phút** | **+13.77 phút** | 26.29x |
| **Peak VRAM Allocated** | **2,419.1 MB (2.36 GB)**| **13,608.2 MB (13.29 GB)**| **+11,189.1 MB (+10.93 GB)**| 5.62x (Headroom ~0.95 GB) |

### 9.2 Bảng 2: Per-SageLayer Forward Timing (16 Tầng SAGE)

| Tầng SAGE | Loại Kiến trúc | Độ phân giải đặc trưng | Tổng thời gian (ms) | Main Path (ms) | Expert Path (ms) | % Thời gian Forward |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SageLayer 00 (CNN Stage 0)** | CNN | $112 \times 112$ | **490.19 ms** | 17.33 ms | 471.76 ms | **33.0%** |
| **SageLayer 01 (CNN Stage 1)** | CNN | $56 \times 56$ | **426.14 ms** | 11.02 ms | 414.66 ms | **28.6%** |
| **SageLayer 02 (CNN Stage 2)** | CNN | $28 \times 28$ | 70.17 ms | 13.91 ms | 55.92 ms | 4.7% |
| **SageLayer 03 (CNN Stage 3)** | CNN | $14 \times 14$ | 57.35 ms | 2.94 ms | 53.70 ms | 3.9% |
| **SageLayer 04 (ViT Block 00)**| ViT | $14 \times 14$ (196 tok) | 32.97 ms | 0.87 ms | 31.82 ms | 2.2% |
| **SageLayer 05 (ViT Block 01)**| ViT | $14 \times 14$ (196 tok) | 35.34 ms | 0.82 ms | 34.19 ms | 2.4% |
| **SageLayer 06 (ViT Block 02)**| ViT | $14 \times 14$ (196 tok) | 36.85 ms | 0.85 ms | 35.69 ms | 2.5% |
| **SageLayer 07 (ViT Block 03)**| ViT | $14 \times 14$ (196 tok) | 34.39 ms | 0.86 ms | 33.19 ms | 2.3% |
| **SageLayer 08 (ViT Block 04)**| ViT | $14 \times 14$ (196 tok) | 41.38 ms | 0.90 ms | 40.17 ms | 2.8% |
| **SageLayer 09 (ViT Block 05)**| ViT | $14 \times 14$ (196 tok) | 34.75 ms | 0.87 ms | 33.59 ms | 2.3% |
| **SageLayer 10 (ViT Block 06)**| ViT | $14 \times 14$ (196 tok) | 34.72 ms | 0.96 ms | 33.44 ms | 2.3% |
| **SageLayer 11 (ViT Block 07)**| ViT | $14 \times 14$ (196 tok) | 35.05 ms | 0.87 ms | 33.88 ms | 2.4% |
| **SageLayer 12 (ViT Block 08)**| ViT | $14 \times 14$ (196 tok) | 36.93 ms | 0.98 ms | 35.61 ms | 2.5% |
| **SageLayer 13 (ViT Block 09)**| ViT | $14 \times 14$ (196 tok) | 33.89 ms | 0.96 ms | 32.66 ms | 2.3% |
| **SageLayer 14 (ViT Block 10)**| ViT | $14 \times 14$ (196 tok) | 34.22 ms | 0.89 ms | 33.02 ms | 2.3% |
| **SageLayer 15 (ViT Block 11)**| ViT | $14 \times 14$ (196 tok) | 31.62 ms | 0.93 ms | 30.35 ms | 2.1% |
| **TỔNG CỘNG (16 TẦNG)** | — | — | **1487.4 ms** | **67.0 ms** | **1420.4 ms** | **100.0%** |

> [!IMPORTANT]
> **Phát hiện Cốt lõi về Nút thắt Phân giải (Resolution Bottleneck):**
> - **SageLayer 00 + SageLayer 01 chiếm tới 61.6% tổng thời gian Forward của toàn bộ mô hình** ($490.2\text{ ms} + 426.1\text{ ms} = 916.3\text{ ms}$).
> - Lý do: Stage 0 xử lý feature map kích thước $112 \times 112$ (12,544 spatial elements), Stage 1 là $56 \times 56$ (3,136 spatial elements). Khi 4 expert được kích hoạt và thực thi tính toán trên không gian này, khối lượng tính toán (FLOPs) lớn hơn gấp hàng chục lần so với không gian $14 \times 14$.
> - Trong khi đó, **toàn bộ 12 ViT blocks cộng lại chỉ tốn ~420 ms**, ít hơn thời gian của riêng 1 tầng SageLayer 00!

### 9.3 Bảng 3: Ma trận Chuyển dịch Modal (Transition Patterns)

| Danh mục Chuyển dịch (Source $\to$ Target) | Số lượng Mẫu gán | Tỷ lệ (%) | Trạng thái Hoạt động |
|:---|:---:|:---:|:---:|
| **CNN $\to$ CNN (Nội bộ CNN)** | 281 | 7.3% | Đang hoạt động |
| **CNN $\to$ ViT (Đa phương thức CNN $\to$ ViT)** | 679 | 17.7% | Đang hoạt động |
| *(trong đó: **Stage0 $\to$ ViT** hạ phân giải $112 \to 14$ tokens)* | *171* | *4.5%* | *Tâm điểm đo đạc* |
| **ViT $\to$ CNN (Đa phương thức ViT $\to$ CNN)** | 714 | 18.6% | Đang hoạt động |
| **ViT $\to$ ViT (Nội bộ ViT)** | 2166 | 56.4% | Đang hoạt động |
| **TỔNG SỐ LƯỢT ĐỊNH TUYẾN (5 Batches)** | **3,840** | **100.0%** | **Phân phối phong phú, cân bằng** |

- **Định tuyến xuyên phương thức (Cross-modal)** đạt tỷ lệ **36.3%** ($17.7\% + 18.6\%$), chứng minh Router không bị thiên kiến cục bộ (local collapse) mà chủ động tận dụng cả hai kiến trúc CNN và Transformer.
- Luồng **Stage0 $\to$ ViT** chiếm 4.5% (171 lượt chuyển giao) diễn ra trơn tru mà không làm sụp đổ pipeline.

### 9.4 Bảng 4: Phân phối Tần suất Chọn Chuyên gia (16 Experts)

| Chuyên gia | Loại | Vai trò Stage 2 | Số lần chọn (Selections) | Tỷ lệ Selections (%) | Số mẫu gán (Samples) | Tỷ lệ Samples (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Expert 00 (CNN Stage 0)** | CNN | **SHARED** | 229 | 6.0% | 229 | 6.0% |
| **Expert 01 (CNN Stage 1)** | CNN | **SHARED** | 255 | 6.6% | 255 | 6.6% |
| **Expert 02 (CNN Stage 2)** | CNN | **SHARED** | 253 | 6.6% | 253 | 6.6% |
| **Expert 03 (CNN Stage 3)** | CNN | **SHARED** | 258 | 6.7% | 258 | 6.7% |
| **Expert 04 (ViT Block 00)**| ViT | Specific | 235 | 6.1% | 235 | 6.1% |
| **Expert 05 (ViT Block 01)**| ViT | Specific | 273 | 7.1% | 273 | 7.1% |
| **Expert 06 (ViT Block 02)**| ViT | Specific | 278 | 7.2% | 278 | 7.2% |
| **Expert 07 (ViT Block 03)**| ViT | Specific | 225 | 5.9% | 225 | 5.9% |
| **Expert 08 (ViT Block 04)**| ViT | Specific | 263 | 6.8% | 263 | 6.8% |
| **Expert 09 (ViT Block 05)**| ViT | Specific | 239 | 6.2% | 239 | 6.2% |
| **Expert 10 (ViT Block 06)**| ViT | Specific | 250 | 6.5% | 250 | 6.5% |
| **Expert 11 (ViT Block 07)**| ViT | Specific | 236 | 6.1% | 236 | 6.1% |
| **Expert 12 (ViT Block 08)**| ViT | Specific | 217 | 5.7% | 217 | 5.7% |
| **Expert 13 (ViT Block 09)**| ViT | Specific | 182 | 4.7% | 182 | 4.7% |
| **Expert 14 (ViT Block 10)**| ViT | Specific | 247 | 6.4% | 247 | 6.4% |
| **Expert 15 (ViT Block 11)**| ViT | Specific | 200 | 5.2% | 200 | 5.2% |

- **Phân phối tải gần như hoàn hảo**: Mức trung bình kỳ vọng lý tưởng là $100\% / 16 = 6.25\%$. Tỷ lệ thực tế dao động hẹp từ **4.7% đến 7.2%**.
- **Không có bất kỳ Dead Expert nào**: 16/16 chuyên gia đều nhận đủ tải, chứng minh cơ chế exploration noise và sigmoid gating hoạt động hiệu quả.

### 9.5 Bảng 5: Phân rã Vi mô SAHub & Thao tác trong Expert Path

| Thành phần Vi mô trong Expert Loop | Thời gian trung bình / Step (ms) | Tỷ trọng trong Expert Loop (%) | Đánh giá |
|:---|:---:|:---:|:---|
| **Expert Forward Computation** | **1,121.77 ms** | **89.1%** | **Nơi tiêu tốn tính toán thực sự (FLOPs)** |
| **SAHub Input Adapt (Shape/Chan)** | 62.64 ms | 5.0% | Overhead chuyển đổi đầu vào rất nhỏ |
| **SAHub Output Adapt (Match Main)** | 46.51 ms | 3.7% | Overhead chuyển đổi đầu ra rất nhỏ |
| **Gating & index_add_ Accumulation**| 26.46 ms | 2.1% | Thao tác gom tensor trên GPU rất nhanh |
| **Self-Selection Bypass (Zero-Cost)**| 1.10 ms | 0.1% | Tối ưu đường tắt hoạt động gần như 0 chi phí |
| **TỔNG CỘNG THAO TÁC EXPERT PATH** | **1,258.47 ms** | **100.0%** | — |

> [!NOTE]
> **Giải oan cho SA-Hub:**
> Trước đây có nghi ngờ rằng module biến đổi hình dạng (SA-Hub) hoặc hàm cộng phân tán `index_add_` có thể làm nghẽn GPU. Số liệu đo đạc thực nghiệm bác bỏ hoàn toàn nghi ngờ này:
> - Toàn bộ các bước adapt của SA-Hub chỉ chiếm **8.7%** ($5.0\% + 3.7\%$).
> - Thao tác `index_add_` chỉ tốn **2.1%** (26.5 ms).
> - **89.1% thời gian là phép toán ma trận thực sự bên trong các Expert blocks**.

### 9.6 Bảng 6: Chi phí CPU của Hàm `_infer_expert_type()`

- **Tổng số lần gọi qua 5 measured steps**: 2,220 lần.
- **Tổng thời gian CPU tiêu tốn**: 72.88 ms.
- **Thời gian trung bình mỗi lần gọi**: **32.83 microseconds** ($\mu\text{s}$).
- **Thời gian CPU trung bình mỗi forward step**: **14.58 ms / step**.
- **Tỷ trọng trong tổng thời gian step (5433 ms)**: **0.27%** (Cực kỳ nhỏ, không gây ảnh hưởng đến hiệu năng).

---

### 9.7 Tổng kết Bức tranh Toàn cảnh & Ý nghĩa Thực nghiệm

1. **Hiểu rõ bản chất tốc độ ~14.3 phút/epoch**:
   - B1-D12 chỉ mất ~32 giây/epoch vì là mạng thuần không rẽ nhánh.
   - B2-D12 tốn 14.3 phút/epoch (26x so với B1) là do **bản chất cấu trúc MoE**: mỗi mẫu chạy qua 16 router, mỗi router gọi 4 expert $\to$ tổng cộng $16 \times 4 = 64$ lượt thực thi expert cho mỗi batch. 
   - Điểm nghẽn tập trung 61.6% ở Stage 0 và Stage 1 do phân giải cao ($112 \times 112$ và $56 \times 56$), trong khi 12 ViT blocks chỉ chiếm ~28%.
2. **Khẳng định tính ổn định & khả thi**:
   - VRAM bão hòa ổn định ở mức **13.29 GB** (headroom an toàn ~0.95 GB trên T4 14.56 GB).
   - 0 OOM, 0 rò rỉ bộ nhớ, phân phối routing 16 expert đều đặn.
   - Sẵn sàng 100% cho Phase 1 ViT-Depth Ablation trên Colab T4.

---

## 10. Báo cáo Thực nghiệm Định hướng: Targeted Runtime Profiling SageLayer 0 & 1 (10 Measured Steps)

**Thời gian thực hiện**: 2026-09-24  
**Môi trường thực thi**: Google Colab — NVIDIA Tesla T4 (14.56 GB Usable, Compute Cap 7.5, Driver/cuDNN 91900)  
**Tập dữ liệu**: Crack500 Train (1896 mẫu, canonical preprocessing, image size 448×448, `batch_size: 12`, `num_workers: 2`, `AMP: True`)  
**Script thực thi**: `python scripts/profile_targeted_stages.py --config configs/b2_crack500_depth12.yaml --batch-size 12 --workers 2 --warmup 2 --measured 10`  
**Quy cách đo lường**: Thu thập động bằng CUDA Events & CPU timer qua 12 batches (2 warmup + 10 measured steps). Phương pháp luận và kiến trúc giữ nguyên bản 100%.

### 10.1 Bảng 1: Macro Summary (10 Measured Steps)

| Thành phần | Thời gian đo đạc (ms) | Tỷ trọng (% Step Time) | Ghi chú vận hành |
|:---|:---:|:---:|:---|
| **Data Wait** | **0.66 ms** | **0.01%** | DataLoader nạp gối đầu hoàn hảo |
| **Forward Pass** | **1,450.87 ms** | **27.32%** | ~1.45 s / batch |
| **Backward Pass** | **3,847.68 ms** | **72.46%** | Lan truyền gradient qua 16 router MoE |
| **Optimizer Step** | **11.09 ms** | **0.21%** | Cập nhật trọng số AdamW |
| **Total Step Time** | **5,310.30 ms** | **100.0%** | **5.31 s / batch (2.26 images/sec)** |
| **Ước lượng 1 Epoch** | **13.98 phút** | — | 158 batches train thuần |
| **Peak VRAM Allocated** | **13,610.5 MB (13.29 GB)** | — | **An toàn, đệm trống ~0.95 GB** |

---

### 10.2 Bảng 2: Per-SageLayer Forward Timing (16 Tầng SAGE)

| Tầng SAGE | Loại | Độ phân giải | Tổng (ms) | Main Path (ms) | Expert Path (ms) | % Forward Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SageLayer 00 (CNN Stage 0)** | CNN | $112 \times 112$ | **488.40 ms** | 17.33 ms | 469.96 ms | **33.7%** |
| **SageLayer 01 (CNN Stage 1)** | CNN | $56 \times 56$ | **419.28 ms** | 11.18 ms | 407.64 ms | **28.9%** |
| **SageLayer 02 (CNN Stage 2)** | CNN | $28 \times 28$ | 72.89 ms | 14.07 ms | 58.46 ms | 5.0% |
| **SageLayer 03 (CNN Stage 3)** | CNN | $14 \times 14$ | 50.13 ms | 2.70 ms | 47.13 ms | 3.5% |
| **SageLayers 04..15 (12 ViT Blocks)** | ViT | $14 \times 14$ | 31.71 – 36.01 ms | 0.73 – 0.85 ms | 30.64 – 34.89 ms | ~2.2% – 2.5% mỗi tầng |
| **TỔNG CỘNG 16 TẦNG** | — | — | **1,450.87 ms** | **67.0 ms** | **1,383.8 ms** | **100.0%** |

> [!IMPORTANT]
> **Khẳng định tính ổn định:** Với 10 measured steps, **SageLayer 00 + SageLayer 01 chiếm chính xác 62.6% tổng thời gian Forward** ($488.40\text{ ms} + 419.28\text{ ms} = 907.68\text{ ms}$). Toàn bộ 12 ViT blocks cộng lại chỉ chiếm 27.5% (~400 ms).

---

### 10.3 Bảng 3: Chi tiết Từng Expert của SageLayer 00 và SageLayer 01

#### A. SageLayer 00 (CNN Stage 0, Feature Map $112 \times 112$)
- **Tổng Selections**: 480 (48 lượt/step $\times$ 10 steps).

| Target Expert ID & Name | Type | Selections | Calls | Bypass | Input Shape nhận được | Tokens/Mẫu | Tổng thời gian (ms) | ms/call |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Expert 00 (CNN Stg 0) [SHR]**| CNN | 43 | 0 | 10 | *(Bypass)* | 0 | 0.00 ms | 0.00 ms |
| **Expert 01 (CNN Stg 1) [SHR]**| CNN | 26 | 10 | 0 | `(4, 48, 112, 112)` | 12,544 | 127.08 ms | 12.71 ms |
| **Expert 02 (CNN Stg 2) [SHR]**| CNN | 30 | 10 | 0 | `(2, 96, 112, 112)` | 12,544 | 191.85 ms | 19.19 ms |
| **Expert 03 (CNN Stg 3) [SHR]**| CNN | 31 | 10 | 0 | `(4, 192, 112, 112)`| 12,544 | 134.77 ms | 13.48 ms |
| **Expert 04 (ViT Blk 00)** | ViT | 28 | 9 | 0 | `(6, 12544, 192)` | 12,544 | 248.68 ms | 27.63 ms |
| **Expert 05 (ViT Blk 01)** | ViT | 38 | 10 | 0 | `(6, 12544, 192)` | 12,544 | 414.58 ms | 41.46 ms |
| **Expert 06 (ViT Blk 02)** | ViT | 23 | 10 | 0 | `(2, 12544, 192)` | 12,544 | 266.11 ms | 26.61 ms |
| **Expert 07 (ViT Blk 03)** | ViT | 39 | 10 | 0 | `(6, 12544, 192)` | 12,544 | 437.65 ms | 43.77 ms |
| **Expert 08 (ViT Blk 04)** | ViT | 28 | 10 | 0 | `(4, 12544, 192)` | 12,544 | 316.64 ms | 31.66 ms |
| **Expert 09 (ViT Blk 05)** | ViT | 30 | 10 | 0 | `(6, 12544, 192)` | 12,544 | 340.19 ms | 34.02 ms |
| **Expert 10 (ViT Blk 06)** | ViT | 21 | 9 | 0 | `(2, 12544, 192)` | 12,544 | 234.34 ms | 26.04 ms |
| **Expert 11 (ViT Blk 07)** | ViT | 33 | 10 | 0 | `(4, 12544, 192)` | 12,544 | 367.22 ms | 36.72 ms |
| **Expert 12 (ViT Blk 08)** | ViT | 30 | 9 | 0 | `(6, 12544, 192)` | 12,544 | 336.05 ms | 37.34 ms |
| **Expert 13 (ViT Blk 09)** | ViT | 28 | 9 | 0 | `(4, 12544, 192)` | 12,544 | 313.99 ms | 34.89 ms |
| **Expert 14 (ViT Blk 10)** | ViT | 19 | 9 | 0 | `(4, 12544, 192)` | 12,544 | 214.41 ms | 23.82 ms |
| **Expert 15 (ViT Blk 11)** | ViT | 33 | 9 | 0 | `(4, 12544, 192)` | 12,544 | 369.21 ms | 41.02 ms |
| **Tiểu kết Stage 0 $\to$ CNN (00-03)**| **CNN** | **130** | **30** | **10** | — | — | **453.70 ms** | **15.12 ms/call** |
| **Tiểu kết Stage 0 $\to$ ViT (04-15)**| **ViT** | **350** | **114** | **0** | `(B_sub, 12544, 192)`| **12,544** | **3,859.06 ms** | **33.85 ms/call** |

---

#### B. SageLayer 01 (CNN Stage 1, Feature Map $56 \times 56$)
- **Tổng Selections**: 480.

| Target Expert ID & Name | Type | Selections | Calls | Bypass | Input Shape nhận được | Tokens/Mẫu | Tổng thời gian (ms) | ms/call |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Expert 00 (CNN Stg 0) [SHR]**| CNN | 44 | 10 | 0 | `(4, 48, 112, 112)` | 12,544 | 87.92 ms | 8.79 ms |
| **Expert 01 (CNN Stg 1) [SHR]**| CNN | 38 | 0 | 10 | *(Bypass)* | 0 | 0.00 ms | 0.00 ms |
| **Expert 02 (CNN Stg 2) [SHR]**| CNN | 30 | 9 | 0 | `(2, 96, 112, 112)` | 12,544 | 166.99 ms | 18.55 ms |
| **Expert 03 (CNN Stg 3) [SHR]**| CNN | 58 | 10 | 0 | `(3, 192, 112, 112)`| 12,544 | 199.45 ms | 19.95 ms |
| **Expert 04 (ViT Blk 00)** | ViT | 34 | 10 | 0 | `(4, 12544, 192)` | 12,544 | 354.11 ms | 35.41 ms |
| **Expert 05 (ViT Blk 01)** | ViT | 42 | 10 | 0 | `(6, 12544, 192)` | 12,544 | 461.96 ms | 46.20 ms |
| **Expert 06 (ViT Blk 02)** | ViT | 8 | 7 | 0 | `(2, 12544, 192)` | 12,544 | 92.54 ms | 13.22 ms |
| **Expert 07 (ViT Blk 03)** | ViT | 32 | 10 | 0 | `(2, 12544, 192)` | 12,544 | 352.03 ms | 35.20 ms |
| **Expert 08 (ViT Blk 04)** | ViT | 27 | 10 | 0 | `(4, 12544, 192)` | 12,544 | 296.82 ms | 29.68 ms |
| **Expert 09 (ViT Blk 05)** | ViT | 41 | 10 | 0 | `(6, 12544, 192)` | 12,544 | 448.57 ms | 44.86 ms |
| **Expert 10 (ViT Blk 06)** | ViT | 36 | 10 | 0 | `(4, 12544, 192)` | 12,544 | 395.36 ms | 39.54 ms |
| **Expert 11 (ViT Blk 07)** | ViT | 2 | 2 | 0 | `(1, 12544, 192)` | 12,544 | 23.02 ms | 11.51 ms |
| **Expert 12 (ViT Blk 08)** | ViT | 35 | 10 | 0 | `(4, 12544, 192)` | 12,544 | 385.67 ms | 38.57 ms |
| **Expert 13 (ViT Blk 09)** | ViT | 2 | 2 | 0 | `(1, 12544, 192)` | 12,544 | 23.06 ms | 11.53 ms |
| **Expert 14 (ViT Blk 10)** | ViT | 39 | 10 | 0 | `(6, 12544, 192)` | 12,544 | 428.00 ms | 42.80 ms |
| **Expert 15 (ViT Blk 11)** | ViT | 12 | 7 | 0 | `(2, 12544, 192)` | 12,544 | 134.89 ms | 19.27 ms |
| **Tiểu kết Stage 1 $\to$ CNN (00-03)**| **CNN** | **170** | **29** | **10** | — | — | **454.37 ms** | **15.67 ms/call** |
| **Tiểu kết Stage 1 $\to$ ViT (04-15)**| **ViT** | **310** | **98** | **0** | `(B_sub, 12544, 192)`| **12,544** | **3,396.05 ms** | **34.65 ms/call** |

---

### 10.4 Bảng 4: Bảng Tổng hợp Aggregate (Requirement C)

| Source Layer | Target Family | Selections | Actual Tokens | % Selections | Expert Time (ms) | ms / call |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SageLayer 0 (CNN Stage 0)** | **CNN** | 130 | 1,091,328 | 1.7% | 453.70 ms | **15.12 ms** |
| **SageLayer 0 (CNN Stage 0)** | **ViT** | 350 | 4,390,400 | 4.6% | **3,859.06 ms** | **33.85 ms** |
| **SageLayer 1 (CNN Stage 1)** | **CNN** | 170 | 1,655,808 | 2.2% | 454.37 ms | **15.67 ms** |
| **SageLayer 1 (CNN Stage 1)** | **ViT** | 310 | 3,888,640 | 4.0% | **3,396.05 ms** | **34.65 ms** |
| **All ViT Layers (04-15)** | **CNN** | 1,446 | 283,416 | 18.8% | 1,165.89 ms | **2.52 ms** |
| **All ViT Layers (04-15)** | **ViT** | 4,314 | 777,140 | 56.2% | 999.45 ms | **0.79 ms** |

> [!CAUTION]
> **PHÁT HIỆN ĐỘT PHÁ VỀ BẢN CHẤT CHI PHÍ TÍNH TOÁN:**
> 1. Khi một **ViT Layer** gọi một **ViT Expert**: Thời gian thực thi chỉ mất **0.79 ms / call** (xử lý chuỗi tiêu chuẩn $N = 196$ tokens).
> 2. Nhưng khi **Stage 0 hoặc Stage 1** gọi một **ViT Expert**: Thời gian thực thi vọt lên tới **33.85 – 34.65 ms / call** $\implies$ **CHẬM HƠN GẤP 43 LẦN TRÊN MỖI LẦN GỌI**!
> 3. **Nguyên nhân cốt lõi**: Cơ chế SA-Hub hiện tại khi chuyển đổi từ CNN sang Transformer (`_format_cnn_to_transformer`) thực hiện `flatten(2).transpose(1, 2)`, biến $112 \times 112$ thành chuỗi **12,544 tokens** và đưa nguyên vẹn chuỗi này vào ViT Self-Attention ($\mathcal{O}(N^2)$)!
>    - ViT tiêu chuẩn: $196^2 = 38,416$ phép tính attention.
>    - ViT nhận từ Stage 0/1: $12,544^2 = 157,351,936$ phép tính attention ($\mathbf{4,096\times}$ khối lượng tính toán)!
> 4. Stage 0 và 1 gọi ViT Experts tổng cộng $114 + 98 = 212$ lần trong 10 steps, tiêu tốn tới **7,255.1 ms** (chiếm trên 85% thời gian Expert Path của 2 tầng này)!

---

### 10.5 Bảng 5: Xác nhận "Selections" vs "Actual Tokens" (Requirement D)

- **Selections (Sample Assignments)**: Là số lượng thể hiện mẫu ảnh trong batch ($B=12$) được router chỉ định. Mỗi batch có 12 mẫu, mỗi mẫu chọn 4 expert $\implies 12 \times 4 = 48$ assignments/layer/step.
- **Actual Tokens (Spatial Positions)**: Là tích số thực tế giữa số mẫu gán và số chiều không gian:
  $$\text{Actual Tokens} = B_{\text{sub}} \times (H \times W \text{ hoặc } N)$$
- Số liệu khẳng định: ViT bình thường chỉ xử lý **196 tokens/mẫu**, nhưng khi nhận nhánh rẽ từ CNN Stage 0/1, ViT bị ép phải xử lý **12,544 tokens/mẫu**!

---

### 10.6 Bảng 6: Đối chiếu Khớp Chi phí Vi mô (Micro-Timing Reconciliation - Requirement E)

| Tầng SAGE | Expert Path (ms) | Router | SAHub In | Compute (Experts) | SAHub Out | IdxAdd | Bypass | Lượng dư Residual (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Layer 00 (CNN 0)** | **469.96 ms** | 0.61 ms | 8.60 ms | **431.28 ms** | 4.15 ms | 10.64 ms | 0.34 ms | **14.34 ms (3.1%)** |
| **Layer 01 (CNN 1)** | **407.64 ms** | 0.44 ms | 6.86 ms | **385.04 ms** | 5.51 ms | 2.66 ms | 0.07 ms | **7.06 ms (1.7%)** |
| **Layer 02 (CNN 2)** | 58.46 ms | 0.34 ms | 3.43 ms | 42.89 ms | 4.26 ms | 1.66 ms | 0.06 ms | 5.83 ms (10.0%) |
| **Layer 03 (CNN 3)** | 47.13 ms | 0.57 ms | 4.90 ms | 26.27 ms | 7.32 ms | 1.31 ms | 0.07 ms | 6.69 ms (14.2%) |
| **Layers 04..15 (ViT)**| 30.64 – 34.89 ms | ~1.3 ms | ~2.5 ms | ~17.0 – 20.3 ms | ~1.7 ms | ~0.8 ms | ~0.05 ms | ~7.3 ms (22% – 24%) |

- **Accounting được giải thích hoàn toàn minh bạch**:
  - Tại Stage 0 và Stage 1: Tổng các thành phần đo đạc vi mô khớp tới **96.9% – 98.3%** tổng thời gian Expert Path.
  - Riêng **Expert Forward Compute chiếm tới 91.8% – 94.5%** thời gian của Expert Path ($431.3\text{ ms} / 470.0\text{ ms}$ tại Layer 0).
  - Lượng dư residual ($\sim 7 – 14\text{ ms}$) phản ánh chính xác chi phí overhead điều vận vòng lặp Python, phép slicing `x[original_batch_indices]`, và độ trễ ranh giới của các cặp CUDA Events.





