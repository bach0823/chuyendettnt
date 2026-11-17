# 10_config_diff.md — So Sánh Toàn Diện Cấu Hình và Mã Nguồn: b1_b2_end_to_end vs run1 vs run2 (Revision 4)

**Ngày:** 2026-11-17  
**Các lần chạy được đối chiếu:**
1. `phase7_s2g_b1_b2_end_to_end` (2026-10-10)
2. `phase7_s2g_new_pipeline_run1` (2026-10-17)
3. `phase7_s2g_new_pipeline_run2` (2026-10-17)  
**Tệp log và metadata:** `results/phase7_combination/*/train.log`, `stage2_completion.json`, `diagnostics/final_test_metrics.json`.

---

## 1. Đối Chiếu Tệp Cấu Hình và Siêu Tham Số

Trong trường `"config"` của `final_test_metrics.json`, cả ba run đều trỏ về cùng một tệp:
`/content/SAGE_LITE/configs/p3_ablation/phase7/phase7_b2_s2g_b1_b2_end_to_end.yaml`.

Bảng so sánh siêu tham số trích xuất từ log huấn luyện:

| Thuộc tính / Siêu tham số | `b1_b2_end_to_end` | `run1` | `run2` | Đánh giá so sánh |
|---|:---:|:---:|:---:|---|
| **Thời điểm chạy** | 2026-10-10 14:27:48 | 2026-10-17 07:32:57 | 2026-10-17 07:32:42 | Cách nhau 7 ngày |
| **Kiến trúc mô hình** | B2 (ConvNeXt-Femto + 4 ViT + SAGE) | B2 (ConvNeXt-Femto + 4 ViT + SAGE) | B2 (ConvNeXt-Femto + 4 ViT + SAGE) | Giống nhau bit-for-bit |
| **S2-Gate Module** | Block 1 (Conv 3×3) + Block 2 (Conv 3×3, Dilation 2) | Block 1 + Block 2 | Block 1 + Block 2 | Giống nhau |
| **Hàm mất mát (Loss)** | CrackBinaryLoss (BCE + Dice) | CrackBinaryLoss (BCE + Dice) | CrackBinaryLoss (BCE + Dice) | Giống nhau |
| **Bộ tối ưu & LR Stage 2** | AdamW: base 1e-4, shared 1e-4, s2_gate 1e-4 | AdamW: base 1e-4, shared 1e-4, s2_gate 1e-4 | AdamW: base 1e-4, shared 1e-4, s2_gate 1e-4 | Giống nhau |
| **Kích thước ảnh huấn luyện** | 448 × 448 | 448 × 448 | 448 × 448 | Giống nhau |
| **Giao thức đo Val trong train** | **setting_a** (tiling 448, stride 448) | **direct** (1-pass resize 448) | **direct** (1-pass resize 448) | **KHÁC BIỆT THEN CHỐT** |
| **Checkpoint được lưu tốt nhất** | Epoch 15 (Val: **0.7689**) | Epoch 18 (Val: **0.7783**) | Epoch 16 (Val: **0.7793**) | Dừng ở epoch khác nhau |

---

## 2. Giải Mã Sự Khác Biệt Val Dice (0.7689 vs 0.7783 vs 0.7793)

Tại sao điểm Val của `b1_b2_end_to_end` chỉ đạt 0.7689 trong khi `run1` đạt 0.7783 và `run2` đạt 0.7793?

Từ tệp `train.log`:
- `b1_b2_end_to_end` ghi:
  `2026-10-10 14:27:48 [INFO ] Evaluation protocol for validation: setting_a`
- `run1` và `run2` ghi:
  `2026-10-17 07:32:57 [INFO ] Evaluation protocol for validation: direct`

**Kết luận:** Điểm Val 0.7689 của `b1_b2_end_to_end` được đo bằng phương pháp **cắt tile Setting A (stride 448)** trong khi huấn luyện. Điểm Val 0.7783 và 0.7793 của `run1`/`run2` được đo bằng **direct 1-pass**.  
Do đó, việc so sánh điểm Val giữa `b1_b2_end_to_end` và `run1`/`run2` là so sánh khập khiễng khác giao thức. Nếu đối chiếu trên tập Test Setting A:
- `b1_b2_end_to_end`: Val Setting A = 0.7689 → Test Setting A = 0.7016 (**Gap = 6.73 điểm**)
- Cả Val và Test đều cùng mức hiệu năng logic.

---

## 3. So Sánh giữa run1 và run2 — Nhiễu Ngẫu nhiên

Giữa `run1` và `run2`:
- Chạy song song gần như cùng thời điểm ngày 2026-10-17.
- Sử dụng cùng một tệp mã nguồn, cùng cấu hình, cùng giao thức đo Val (direct) và Test (Setting A, Setting B, Direct).
- Điểm Val: `run1` = 0.7783 (Epoch 18) vs `run2` = 0.7793 (Epoch 16) → chênh lệch **+0.10 điểm**.
- Điểm Test Setting A: `run1` = 0.6881 vs `run2` = 0.6895 → chênh lệch **+0.14 điểm**.
- Điểm Test Setting B: `run1` = 0.6950 vs `run2` = 0.6950 → chênh lệch **0.00 điểm**.

**Kết luận:** Sự khác biệt giữa `run1` và `run2` thuần túy là **nhiễu thống kê giữa các lần chạy ngẫu nhiên (stochastic run-to-run noise)** do điểm dừng early stopping khác nhau (epoch 18 vs epoch 16) và thứ tự lấy mẫu batch.  
Bỏ hoàn toàn việc gán nhãn "pipeline mới / pipeline cũ" giữa run1 và run2.
