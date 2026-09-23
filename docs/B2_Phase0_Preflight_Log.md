# B2 Phase 0 Runtime Preflight & OOM Probe Log (Tesla T4)

**Date**: 2026-09-23  
**Environment**: Google Colab — NVIDIA Tesla T4 (15.0 GB / 14.56 GB usable, Compute Capability 7.5, Driver/cuDNN 91900)  
**Target Model**: B2 (Full SAGE-Lite, ViT Depth 12, 16 Injected Routers, Top-K = 4, Sigmoid Gating, Logit Modulation = True)  
**Input Resolution**: 448×448, AMP FP16  
**Script**: `scripts/preflight_b2.py`  

---

## 1. Executive Summary & Hardware Boundary

| Batch Size | Status | Peak Alloc VRAM | Peak Reserved VRAM | Free VRAM | Note |
|---|---|---|---|---|---|
| **Batch 20** | ❌ **FAIL (OOM)** | ~14.56 GB | ~14.56 GB | 0 GB | OOM tại ViT Expert MLP & Decoder |
| **Batch 14+** | ❌ **FAIL (OOM)** | > 14.56 GB | — | 0 GB | Vượt ngưỡng dung lượng VRAM 14.56 GB của T4 |
| **Batch 12** | ✅ **PASS** | 14,333.5 MB (14.00 GB) | 14,506.0 MB (14.17 GB) | 406.7 MB (0.40 GB) | Vừa khít ngưỡng an toàn (~97.3% VRAM) |
| **Batch 10** | ✅ **PASS** | 13,919.6 MB (13.59 GB) | 14,294.0 MB (13.96 GB) | 618.7 MB (0.60 GB) | **Khuyến nghị chính thức** cho Full Training |

> [!IMPORTANT]
> **Quy tắc về Batch Size:**  
> Việc điều chỉnh `batch_size: 20 → 10` (hoặc 12) là **hardware-constrained runtime setting** (giới hạn phần cứng GPU T4 16GB), **KHÔNG PHẢI** là hyperparameter optimization (HPO) hay kiến trúc thay đổi.

---

## 2. Chi tiết Preflight Batch 12

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
STEP 1: OOM / VRAM PROBE (Batch Size = 12, 5 Iterations)
======================================================================
Instantiating B2 model for VRAM probe...
  Iter 1/5 (166.72s) | Loss: 2.3338 (Seg: 2.1622, LB: 0.1715) | Alloc: 275.8 MB | Peak Alloc: 14333.5 MB (14.00 GB) | Peak Res: 14466.0 MB (14.13 GB) | Free: 0.44 GB
  Iter 2/5 (58.31s) | Loss: 2.3289 (Seg: 2.1597, LB: 0.1692) | Alloc: 275.5 MB | Peak Alloc: 14333.5 MB (14.00 GB) | Peak Res: 14506.0 MB (14.17 GB) | Free: 0.40 GB
  Iter 3/5 (35.90s) | Loss: 2.3277 (Seg: 2.1573, LB: 0.1703) | Alloc: 276.0 MB | Peak Alloc: 14333.5 MB (14.00 GB) | Peak Res: 14506.0 MB (14.17 GB) | Free: 0.40 GB
  Iter 4/5 (29.79s) | Loss: 2.3227 (Seg: 2.1550, LB: 0.1677) | Alloc: 275.8 MB | Peak Alloc: 14333.5 MB (14.00 GB) | Peak Res: 14506.0 MB (14.17 GB) | Free: 0.40 GB
  Iter 5/5 (27.28s) | Loss: 2.3209 (Seg: 2.1529, LB: 0.1680) | Alloc: 276.1 MB | Peak Alloc: 14333.5 MB (14.00 GB) | Peak Res: 14506.0 MB (14.17 GB) | Free: 0.40 GB

[VRAM Memory Stability Check]
  Allocated Memory Iter 2: 275.50 MB
  Allocated Memory Iter 5: 276.09 MB
  Difference (Iter 5 - Iter 2): +0.60 MB
  [PASS] No monotonically increasing memory leak detected.

[VRAM Summary for Batch=12]
  - Peak Memory Allocated: 14333.5 MB (14.00 GB)
  - Peak Memory Reserved:  14506.0 MB (14.17 GB)
  - Free VRAM Remaining:   406.7 MB (0.40 GB) out of 14.56 GB
  [PASS] Step 1: OOM / VRAM Probe successfully passed without OOM.

======================================================================
STEP 2: MINI-TRAINING SMOKE (5 Training Batches)
======================================================================
  Batch 1/5 | Total Loss: 2.3253 (Seg: 2.1572, LB: 0.1681) | Logits & Grads: FINITE [OK]
  Batch 2/5 | Total Loss: 2.3242 (Seg: 2.1579, LB: 0.1663) | Logits & Grads: FINITE [OK]
  Batch 3/5 | Total Loss: 2.3241 (Seg: 2.1573, LB: 0.1667) | Logits & Grads: FINITE [OK]
  Batch 4/5 | Total Loss: 2.3253 (Seg: 2.1571, LB: 0.1682) | Logits & Grads: FINITE [OK]
  Batch 5/5 | Total Loss: 2.3288 (Seg: 2.1573, LB: 0.1716) | Logits & Grads: FINITE [OK]
  [PASS] Step 2: Mini-training smoke passed (all losses, logits, and gradients strictly finite).

======================================================================
STEP 3: ROUTING DIAGNOSTICS & EXPERT USAGE INSPECTION
======================================================================
  Total Injected Routers: 16 (4 CNN Stages + 12 ViT Blocks)
  Expert Pool Size:       16
  Target Top-K:           4

--- Per-Expert Selection Breakdown (Entire Mini-Run) ---
  Expert 00 (   CNN Stage) [SHARED]:    484 selections (  6.3%)
  Expert 01 (   CNN Stage) [SHARED]:    467 selections (  6.1%)
  Expert 02 (   CNN Stage) [SHARED]:    484 selections (  6.3%)
  Expert 03 (   CNN Stage) [SHARED]:    493 selections (  6.4%)
  Expert 04 (ViT Block 00)         :    467 selections (  6.1%)
  Expert 05 (ViT Block 01)         :    511 selections (  6.7%)
  Expert 06 (ViT Block 02)         :    484 selections (  6.3%)
  Expert 07 (ViT Block 03)         :    487 selections (  6.3%)
  Expert 08 (ViT Block 04)         :    467 selections (  6.1%)
  Expert 09 (ViT Block 05)         :    478 selections (  6.2%)
  Expert 10 (ViT Block 06)         :    501 selections (  6.5%)
  Expert 11 (ViT Block 07)         :    512 selections (  6.7%)
  Expert 12 (ViT Block 08)         :    455 selections (  5.9%)
  Expert 13 (ViT Block 09)         :    427 selections (  5.6%)
  Expert 14 (ViT Block 10)         :    436 selections (  5.7%)
  Expert 15 (ViT Block 11)         :    527 selections (  6.9%)

  [PASS] All 16 experts received at least 1 routing selection.
  Total routing selection events recorded: 7680
  [PASS] Step 3: Routing diagnostics verified (top_k=4 verified, expert usage recorded).

======================================================================
STEP 4: CHECKPOINT SAVE & LOAD INTEGRITY
======================================================================
  Saved test checkpoint to: /tmp/tmpck_98ovc.pth (158.93 MB)
  Loaded state_dict strictly: 0 missing keys, 0 unexpected keys.
  Max absolute difference between original and loaded model: 0.00000000e+00
  [PASS] Step 4: Checkpoint save & load integrity 100% verified.
```

---

## 3. Chi tiết Preflight Batch 10

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
STEP 1: OOM / VRAM PROBE (Batch Size = 10, 5 Iterations)
======================================================================
Instantiating B2 model for VRAM probe...
  Iter 1/5 (149.07s) | Loss: 2.3630 (Seg: 2.1935, LB: 0.1695) | Alloc: 268.0 MB | Peak Alloc: 13919.6 MB (13.59 GB) | Peak Res: 14294.0 MB (13.96 GB) | Free: 0.60 GB
  Iter 2/5 (54.28s) | Loss: 2.3589 (Seg: 2.1902, LB: 0.1687) | Alloc: 269.0 MB | Peak Alloc: 13919.6 MB (13.59 GB) | Peak Res: 14294.0 MB (13.96 GB) | Free: 0.60 GB
  Iter 3/5 (38.20s) | Loss: 2.3549 (Seg: 2.1871, LB: 0.1678) | Alloc: 268.4 MB | Peak Alloc: 13919.6 MB (13.59 GB) | Peak Res: 14294.0 MB (13.96 GB) | Free: 0.60 GB
  Iter 4/5 (26.67s) | Loss: 2.3552 (Seg: 2.1841, LB: 0.1710) | Alloc: 268.3 MB | Peak Alloc: 13919.6 MB (13.59 GB) | Peak Res: 14294.0 MB (13.96 GB) | Free: 0.60 GB
  Iter 5/5 (12.72s) | Loss: 2.3493 (Seg: 2.1815, LB: 0.1679) | Alloc: 269.3 MB | Peak Alloc: 13919.6 MB (13.59 GB) | Peak Res: 14294.0 MB (13.96 GB) | Free: 0.60 GB

[VRAM Memory Stability Check]
  Allocated Memory Iter 2: 268.96 MB
  Allocated Memory Iter 5: 269.35 MB
  Difference (Iter 5 - Iter 2): +0.39 MB
  [PASS] No monotonically increasing memory leak detected.

[VRAM Summary for Batch=10]
  - Peak Memory Allocated: 13919.6 MB (13.59 GB)
  - Peak Memory Reserved:  14294.0 MB (13.96 GB)
  - Free VRAM Remaining:   618.7 MB (0.60 GB) out of 14.56 GB
  [PASS] Step 1: OOM / VRAM Probe successfully passed without OOM.

======================================================================
STEP 2: MINI-TRAINING SMOKE (5 Training Batches)
======================================================================
  Batch 1/5 | Total Loss: 2.3562 (Seg: 2.1883, LB: 0.1679) | Logits & Grads: FINITE [OK]
  Batch 2/5 | Total Loss: 2.3569 (Seg: 2.1881, LB: 0.1688) | Logits & Grads: FINITE [OK]
  Batch 3/5 | Total Loss: 2.3558 (Seg: 2.1881, LB: 0.1677) | Logits & Grads: FINITE [OK]
  Batch 4/5 | Total Loss: 2.3586 (Seg: 2.1885, LB: 0.1702) | Logits & Grads: FINITE [OK]
  Batch 5/5 | Total Loss: 2.3562 (Seg: 2.1876, LB: 0.1686) | Logits & Grads: FINITE [OK]
  [PASS] Step 2: Mini-training smoke passed (all losses, logits, and gradients strictly finite).

======================================================================
STEP 3: ROUTING DIAGNOSTICS & EXPERT USAGE INSPECTION
======================================================================
  Total Injected Routers: 16 (4 CNN Stages + 12 ViT Blocks)
  Expert Pool Size:       16
  Target Top-K:           4

--- Per-Expert Selection Breakdown (Entire Mini-Run) ---
  Expert 00 (   CNN Stage) [SHARED]:    386 selections (  6.0%)
  Expert 01 (   CNN Stage) [SHARED]:    405 selections (  6.3%)
  Expert 02 (   CNN Stage) [SHARED]:    385 selections (  6.0%)
  Expert 03 (   CNN Stage) [SHARED]:    380 selections (  5.9%)
  Expert 04 (ViT Block 00)         :    387 selections (  6.0%)
  Expert 05 (ViT Block 01)         :    433 selections (  6.8%)
  Expert 06 (ViT Block 02)         :    420 selections (  6.6%)
  Expert 07 (ViT Block 03)         :    389 selections (  6.1%)
  Expert 08 (ViT Block 04)         :    401 selections (  6.3%)
  Expert 09 (ViT Block 05)         :    417 selections (  6.5%)
  Expert 10 (ViT Block 06)         :    387 selections (  6.0%)
  Expert 11 (ViT Block 07)         :    410 selections (  6.4%)
  Expert 12 (ViT Block 08)         :    397 selections (  6.2%)
  Expert 13 (ViT Block 09)         :    384 selections (  6.0%)
  Expert 14 (ViT Block 10)         :    406 selections (  6.3%)
  Expert 15 (ViT Block 11)         :    413 selections (  6.5%)

  [PASS] All 16 experts received at least 1 routing selection.
  Total routing selection events recorded: 6400
  [PASS] Step 3: Routing diagnostics verified (top_k=4 verified, expert usage recorded).

======================================================================
STEP 4: CHECKPOINT SAVE & LOAD INTEGRITY
======================================================================
  Saved test checkpoint to: /tmp/tmpgnefnvb_.pth (158.93 MB)
  Loaded state_dict strictly: 0 missing keys, 0 unexpected keys.
  Max absolute difference between original and loaded model: 0.00000000e+00
  [PASS] Step 4: Checkpoint save & load integrity 100% verified.
```

---

## 4. Phân tích Phân phối Routing & Cân bằng Tải

Cả ở Batch 12 và Batch 10, phân phối routing giữa 16 experts cực kỳ đồng đều:
- **Tỷ lệ chọn dao động lý tưởng**: Từ 5.6% đến 6.9% cho mỗi expert (mức trung bình kỳ vọng là $1/16 = 6.25\%$).
- **Không có dead expert**: Toàn bộ 16 experts (4 shared CNN + 12 ViT) đều được active thường xuyên.
- **Load Balancing Loss hoạt động chính xác**: Giữ độ phân tán nhỏ, chứng minh `load_balance_factor = 0.01` cùng cơ chế SAR + Logit modulation đạt trạng thái cân bằng hoàn hảo ngay từ các step đầu tiên.

---

## 5. Kết luận & Quyết định Thực thi

1. **Khóa Runtime Batch Size**:
   - Khuyến nghị sử dụng **`batch_size: 10`** cho full training trên T4 để có 0.60 GB đệm VRAM an toàn, hạn chế tối đa nguy cơ spike bộ nhớ đột xuất trong quá trình validation hoặc data caching.
   - Nếu muốn tối ưu tốc độ và kích thước batch gần nhất với baseline, **`batch_size: 12`** cũng hoàn toàn khả thi (đã pass 5 full iters và 5 mini-batches với 0.40 GB headroom).
2. **Phase 0 Runtime Preflight**: **CHÍNH THỨC HOÀN TẤT VÀ THÔNG QUA (PASS 100%)**.
3. Sẵn sàng bước vào **Phase 1: Full Training B2 Depth 12** trên Crack500.
