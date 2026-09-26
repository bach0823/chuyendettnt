# B0 Throughput Benchmark — Tesla T4 (2026-09-21)

**Notebook:** `notebooks/B0_Throughput_Benchmark_Colab_T4.ipynb`
**Script:** `scripts/benchmark_throughput.py`
**Repo Commit:** `51f82c3`
**Status:** ✅ DONE

---

## 1. Hardware & Env

| Field | Value |
|---|---|
| GPU Name | Tesla T4 |
| CUDA Version | 12.8 |
| Total VRAM | 14.56 GB |
| Batch Size | 20 |
| Image Size | 448×448 |

---

## 2. Pure Compute Benchmark

> ⚠️ **Lưu ý:** Đây là pure GPU compute (Forward+Backward) dùng synthetic tensors + BCE-only.
> **CHƯA bao gồm:** DataLoader I/O, Augmentation, Validation loop, Checkpoint save, SoftDice computation.

| Metric | Value |
|---|---|
| Compute Time per step | **0.309 s** |
| Compute Throughput | **64.7 images/s** |
| Peak VRAM (Compute) | **3.55 GB** |

---

## 3. Estimated Compute Time / Epoch

| Dataset Size | Compute Time | Compute Time (s) |
|---|---|---|
| 500 images | 0.13 min | 7.7 s |
| 1000 images | 0.26 min | 15.4 s |
| 1896 images | 0.49 min | 29.3 s |
| 3391 images | 0.87 min | 52.4 s |

---

## 4. Wall-Clock Analysis (Có tính Overhead)

Estimate overhead multiplier thực tế = **2.0× – 2.5×** trên pure compute
(JPEG disk read + Albumentations augmentation + forward-only validation loop + SoftDice term).

Lấy overhead factor = **2.0×** (conservative):

| Dataset Size | Est. Wall-Clock / Epoch |
|---|---|
| 500 images | ~0.26 min (~15 s) |
| 1000 images | ~0.52 min (~31 s) |
| 1896 images | ~0.98 min (~59 s) |
| 3391 images | ~1.74 min (~105 s) |

---

## 5. Estimated Total Training Time (Budget 15–30 Epochs)

Dùng dataset 1896 ảnh (Crack500 train split) với overhead 2×:
**≈ 1 min / epoch wall-clock.**

| Total Epochs | Estimated Wall-Clock | Fit trong Colab Session (12h)? |
|---|---|---|
| 15 | ~15 min | ✅ Rất thoải mái |
| 20 | ~20 min | ✅ Rất thoải mái |
| 25 | ~25 min | ✅ Rất thoải mái |
| 30 | ~30 min | ✅ Rất thoải mái |

> **Kết luận:** Toàn bộ ngân sách 15–30 epoch đều nằm trong vòng 30 phút wall-clock.
> T4 có đủ margin lớn để chạy. Quyết định split Stage 1/Stage 2 không bị ràng buộc bởi compute budget.
