# B0 Batch Size Sweep Benchmark — Tesla T4 (2026-09-21)

**Notebook:** `notebooks/B0_Throughput_Benchmark_Colab_T4.ipynb`
**Script:** `scripts/benchmark_throughput.py`
**Repo Commit:** `1e1cd42`
**Status:** ✅ ALL PASS

---

## 1. Hardware & Env

| Field | Value |
|---|---|
| GPU Name | Tesla T4 |
| CUDA Version | 12.8 |
| Total VRAM | 14.56 GB |
| Image Size | 448×448 |

---

## 2. Batch Size Sweep Results (Pure Compute)

> ⚠️ **Lưu ý:** Đây là pure GPU compute (Forward+Backward) dùng synthetic tensors, BCE-only.
> CHƯA bao gồm: DataLoader I/O, Augmentation, Validation loop, Checkpoint save, SoftDice.

| Batch | Time/step | Throughput (img/s) | Peak VRAM | Status |
|----:|--------:|---------:|--------:|:----:|
| 16 | 0.250 s | **64.0** | 2.87 GB | ✅ PASS |
| 20 | 0.312 s | **64.0** | 3.55 GB | ✅ PASS |
| 24 | 0.378 s | 63.5 | 4.38 GB | ✅ PASS |
| 28 | 0.443 s | 63.2 | 4.90 GB | ✅ PASS |

---

## 3. Phân tích

### Observation quan trọng: Throughput plateau

Throughput thực tế gần như phẳng hoàn toàn trên toàn dải BS=16 đến BS=28:
- BS=16: 64.0 img/s
- BS=20: 64.0 img/s  (plateau)
- BS=24: 63.5 img/s  (-0.5 img/s, -0.8%)
- BS=28: 63.2 img/s  (-0.8 img/s, -1.25%)

**Kết luận kỹ thuật:** T4 đã bão hòa (compute-bound) ngay từ BS=16 với model này (B0ConvNeXtViTUNet, img 448×448). Tăng batch size không mang lại lợi ích throughput; chỉ làm tăng VRAM và time/step.

### VRAM Safety Margin

Với overhead thực tế ước tính +1.0 GB (DataLoader pin_memory + Augmentation GPU buffer):

| Batch | Pure Compute VRAM | Est. Real VRAM | Margin (from 14.56 GB) |
|----:|---:|---:|---:|
| 16 | 2.87 GB | ~3.9 GB | **10.7 GB** |
| 20 | 3.55 GB | ~4.6 GB | **10.0 GB** |
| 24 | 4.38 GB | ~5.4 GB | **9.2 GB** |
| 28 | 4.90 GB | ~5.9 GB | **8.7 GB** |

Tất cả đều an toàn. Không có rủi ro OOM cho bất kỳ BS nào trong dải này.

---

## 4. Recommendation

**Giữ nguyên BS=20.**

Lý do:
- Throughput đồng nhất với BS=16 (64.0 img/s), không có lợi ích khi giảm batch.
- Gradient ổn định hơn BS=16 do batch lớn hơn một bước.
- Đã được validate bởi preflight B0 trước đó (Peak VRAM 3.49 GB khớp 3.55 GB đo lần này).
- VRAM margin còn ~10 GB — không cần phải tối ưu thêm.
- Tăng lên BS=24/28 không thu được throughput nhưng tốn thêm VRAM và time/step.

---

## 5. Estimated Wall-Clock Time / Epoch tại BS=20 (Crack500)

Dataset Crack500: ~1896 train images → **94.8 steps/epoch** ở BS=20.

| Metric | Giá trị |
|---|---|
| Time/step (pure compute) | 0.312 s |
| Pure compute time/epoch | 0.312 × 94.8 ≈ **29.6 s (~0.5 min)** |
| Est. wall-clock × 2.0 overhead | **~59 s (~1.0 min/epoch)** |

**Budget 15–30 epochs tương đương 15–30 phút wall-clock. Hoàn toàn fit trong 1 Colab session.**
