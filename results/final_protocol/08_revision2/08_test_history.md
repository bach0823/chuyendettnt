# 08_test_history.md — Lịch sử Test (Revision 2)

**Nguồn:** Truy xuất từ file JSON trong `results/`; không lấy từ trí nhớ.  
**Ngày viết:** 2026-11-16

---

## 1. Các lần xem kết quả test theo thứ tự thời gian

| # | Checkpoint (đường dẫn file) | Val best_dice (checkpoint) | Giao thức | Ngày (timestamp) | Test Dice | File nguồn |
|---|---|:---:|---|---|:---:|---|
| 1 | `results/baselines/50ep/B0_Crack500/` (B0, 50 epoch) | — | direct 1-pass, τ=0.5 | — | **0.7042** | `results/baselines/50ep/B0_Crack500/official_eval_test.json` |
| 2 | `results/baselines/100ep/B0_Crack500/` (B0, 100 epoch) | — | direct 1-pass, τ=0.5 | — | **0.7037** | `results/baselines/100ep/B0_Crack500/official_eval_all_splits.json` |
| 3 | `phase7_s2g_b1_b2_end_to_end/best_model_b2_stage2.pth` (B2, epoch 15, val_best_dice=0.7689) | 0.7689 | Setting A (tiling stride=448) | 2026-10-10T15:32:22 | **0.7016** | `results/phase7_combination/phase7_s2g_b1_b2_end_to_end/diagnostics/final_test_metrics.json` |
| 3b | (cùng checkpoint trên) | 0.7689 | Setting B (tiling stride=224) | 2026-10-10T15:32:22 | **0.7054** | (cùng file) |
| 4 | `phase7_s2g_new_pipeline/best_model_b2_stage2.pth` (Run 1, epoch 18, val_best_dice=0.7783) | 0.7783 | Setting A (tiling stride=448) | 2026-10-17T09:01:03 | **0.6881** | `results/phase7_combination/phase7_s2g_new_pipeline_run1/diagnostics/final_test_metrics.json` |
| 4b | (cùng checkpoint trên) | 0.7783 | Setting B (tiling stride=224) | 2026-10-17T09:01:03 | **0.6950** | (cùng file) |
| 5 | `phase7_s2g_new_pipeline/best_model_b2_stage2.pth` (Run 2, epoch 16, val_best_dice=0.7793) | 0.7793 | Setting A (tiling stride=448) | 2026-10-17T09:01:25 | **0.6895** | `results/phase7_combination/phase7_s2g_new_pipeline_run2/diagnostics/final_test_metrics.json` |
| 5b | (cùng checkpoint trên) | 0.7793 | Setting B (tiling stride=224) | 2026-10-17T09:01:25 | **0.6950** | (cùng file) |
| 6 | `phase7_s2g_new_pipeline/best_model_b2_stage2.pth` (Run 2, epoch 16) | 0.7793 | Direct 1-pass, τ=0.5, không TTA | — | **0.7149** | `results/phase7_combination/phase7_s2g_new_pipeline_run2/diagnostics/official_direct_evaluation.json` |
| **7 (LOCKED)** | Ensemble Phase 2 seed42/43/44 (3 checkpoint) | μ=0.778638 | Direct 1-pass + TTA flip_hv + ensemble | — | **0.7158** | `results/final_protocol/05_test_final_report.json` |

> **Lưu ý đặt tên checkpoint run1 vs run2:**  
> Cả run1 (`phase7_s2g_new_pipeline_run1`) và run2 (`phase7_s2g_new_pipeline_run2`) đều ghi checkpoint path là `phase7_s2g_new_pipeline/best_model_b2_stage2.pth` trong trường `metadata.checkpoint` của file JSON. Phân biệt qua `checkpoint_metadata.best_dice`: run1=0.7783, run2=0.7793, và timestamp.

---

## 2. Kiểm tra giả thuyết từ Revision 1

Báo cáo Revision 1 (§1) liệt kê 5 lần test với Dice: 0.6771, 0.6801, 0.6950, 0.7149, 0.7158.

### Giả thuyết 1: 0.6771 = B0 baseline
**BÁC BỎ.** B0 50ep test Dice = 0.7042, B0 100ep = 0.7037. Không tìm thấy giá trị 0.6771 trong file JSON kết quả B0.  
Truy xuất bổ sung: `05_test_final_report.json` line 840 chứa `0.6770914` — đây là **per_image_dice của 1 crop trong Phase 5**, không phải mean Dice của B0.

### Giả thuyết 2: 0.6801 = B2 early checkpoint
**KHÔNG XÁC NHẬN ĐƯỢC.** Giá trị 0.6801 không tìm thấy trong file tổng hợp nào. `05_test_final_report.json` line 275 chứa `0.6801572` — đây là **per_image_dice của 1 crop**, không phải mean của checkpoint nào.

### Giả thuyết 3: 0.6950 = run1 Setting B
**XÁC NHẬN.** Run1 Setting B = 0.6949654810 ≈ 0.6950. Run2 Setting B = 0.6950163 ≈ 0.6950. Cả hai đều xấp xỉ 0.6950.

### Giả thuyết 4: 0.7149 = canonical Phase 7 Run 2 direct 1-pass
**XÁC NHẬN.** `official_direct_evaluation.json`: `"dice": 0.7148690704104006`, checkpoint = Run2 (epoch 16, val_best_dice=0.7793, seed=42).

---

## 3. Kết luận bảng sửa lại

**Số checkpoint khác nhau đã từng xem test:** 5 (B0 50ep, B0 100ep, b1_b2_end_to_end, run1, run2 single + ensemble Phase2)  
**Số lần xem tổng:** ≥ 7 lần đo (tính cả Setting A/B riêng lẻ)

**Lỗi trong Revision 1:**
- Giá trị 0.6771 và 0.6801 **không có nguồn gốc xác định** trong các file JSON kết quả — không thể gán cho checkpoint nào. Có thể là số từ nguồn khác (log, bộ nhớ), hoặc per_image_dice bị hiểu nhầm.
- Bảng Revision 1 thiếu các lần đo Setting A/B (lần 3–5b).
- 0.7149 bị gán nhầm cho "seed 43" — thực tế là **seed 42, run2, direct 1-pass, không TTA**.

**Nguồn file xác nhận:**
- `results/baselines/50ep/B0_Crack500/official_eval_test.json` → dice=0.7042
- `results/baselines/100ep/B0_Crack500/official_eval_all_splits.json` → dice=0.7037
- `results/phase7_combination/phase7_s2g_b1_b2_end_to_end/diagnostics/final_test_metrics.json` → setting_a=0.7016, setting_b=0.7054
- `results/phase7_combination/phase7_s2g_new_pipeline_run1/diagnostics/final_test_metrics.json` → setting_a=0.6881, setting_b=0.6950
- `results/phase7_combination/phase7_s2g_new_pipeline_run2/diagnostics/final_test_metrics.json` → setting_a=0.6895, setting_b=0.6950
- `results/phase7_combination/phase7_s2g_new_pipeline_run2/diagnostics/official_direct_evaluation.json` → dice=0.7149
- `results/final_protocol/05_test_final_report.json` → dice=0.7158 (Phase 5 LOCKED)
