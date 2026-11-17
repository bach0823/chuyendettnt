# 09_test_history.md — Lịch sử Test (Revision 3)

**Ngày:** 2026-11-17  
**Nguồn chính:** các file JSON kết quả và `results/B0_Crack500_Run3_Results.md`

---

## 1. Bảng lịch sử test đầy đủ — ghép cặp Val + Test cùng giao thức

> [!IMPORTANT]
> Chỉ ghép cặp Val–Test khi cùng giao thức inference. Ô không có số liệu cùng giao thức ghi "—".

| # | Run/Checkpoint | Epoch | Giao thức | Val Dice | Test Dice | File nguồn |
|---|---|:---:|---|:---:|:---:|---|
| 1 | B0 Run3 (`best_model_b0.pth`) | 8 | Setting A (stride=448) | **0.7318** | **0.6771** | `results/B0_Crack500_Run3_Results.md` |
| 2 | B0 Run3 (`best_model_b0.pth`) | 8 | Setting B (stride=224) | **0.7360** | **0.6801** | `results/B0_Crack500_Run3_Results.md` |
| 3 | B0 50ep (`best_model`) | 41 | Direct 1-pass, τ=0.5 | **0.7789** | **0.7042** | `baselines/50ep/B0_Crack500/training_completion.json` (val) + `official_eval_test.json` (test) |
| 4 | B0 100ep (`best_model`) | 45 | Direct 1-pass, τ=0.5 | **0.7806** | **0.7037** | `baselines/100ep/B0_Crack500/training_completion.json` (val) + `official_eval_all_splits.json` (test) |
| 5 | `b1_b2_end_to_end` (epoch 15) | 15 | Setting A (stride=448) | — | **0.7016** | `phase7_combination/phase7_s2g_b1_b2_end_to_end/diagnostics/final_test_metrics.json` |
| 6 | `b1_b2_end_to_end` (epoch 15) | 15 | Setting B (stride=224) | — | **0.7054** | (cùng file) |
| 7 | `run1` (`phase7_s2g_new_pipeline`) epoch 18 | 18 | Setting A | — | **0.6881** | `phase7_combination/phase7_s2g_new_pipeline_run1/diagnostics/final_test_metrics.json` |
| 8 | `run1` epoch 18 | 18 | Setting B | — | **0.6950** | (cùng file) |
| 9 | `run2` (`phase7_s2g_new_pipeline`) epoch 16 | 16 | Setting A | — | **0.6895** | `phase7_combination/phase7_s2g_new_pipeline_run2/diagnostics/final_test_metrics.json` |
| 10 | `run2` epoch 16 | 16 | Setting B | — | **0.6950** | (cùng file) |
| 11 | `run2` epoch 16, seed=42 | 16 | Direct 1-pass, τ=0.5, không TTA | **0.7793** | **0.7149** | `run2/diagnostics/official_direct_evaluation.json` (test); `stage2_completion.json` (val) |
| **12 (LOCKED)** | Ensemble Phase 2 seed42/43/44 | — | Direct 1-pass + TTA flip_hv | μ=**0.7787** | **0.7158** | `final_protocol/05_test_final_report.json` |

**Ghi chú nguồn B0 50ep và 100ep:**  
- Val dice trong `training_completion.json` được đo bằng direct inline eval trong vòng lặp training (không phải tiling Setting A/B). Không có file JSON riêng đo val bằng Setting A hoặc B cho B0 50ep/100ep.  
- Test dice trong `official_eval_test.json` và `official_eval_all_splits.json` không ghi rõ giao thức trong file JSON (không có trường `protocol` hoặc `stride`). Từ tên file và cấu trúc JSON (chỉ có một dict metric đơn), suy đoán là direct 1-pass — **[Hypothesis - Unverified: giao thức val và test B0 50ep/100ep đều là direct]**. Nếu không cùng giao thức thì hàng 3 và 4 không phải cặp hợp lệ cho gap.

---

## 2. Gap Val → Test — chỉ các cặp cùng giao thức xác nhận được

| Cặp hợp lệ | Val Dice | Test Dice | Gap (điểm) | Ghi chú |
|---|:---:|:---:|:---:|---|
| B0 Run3 Setting A | 0.7318 | 0.6771 | **5.47 điểm** | Cùng checkpoint, cùng giao thức Setting A |
| B0 Run3 Setting B | 0.7360 | 0.6801 | **5.59 điểm** | Cùng checkpoint, cùng giao thức Setting B |
| run2 Direct 1-pass | 0.7793 | 0.7149 | **6.44 điểm** | Cùng checkpoint run2 epoch 16, cùng direct 1-pass |

**Bỏ dòng "B0: val 0.7318, test 0.7037, gap 2.81 điểm"** (xuất hiện trong Rev 2 §6.2): đây là ghép chéo giao thức (val Setting A + test direct 1-pass) → không hợp lệ.

---

## 3. Checkpoint và số lần đo

**Số checkpoint khác nhau đã xem test:** ≥ 6  
(B0 Run3; B0 50ep; B0 100ep; b1_b2_end_to_end; run1; run2 + ensemble Phase 2 từ cùng run2 weights)

**Số lần đo test (lượt chạy inference trên test):** ≥ 12  
(B0 Run3 ×2 giao thức; B0 50ep ×1; B0 100ep ×1; b1_b2_e2e ×2; run1 ×2; run2 ×3; ensemble ×1)

> [!WARNING]
> Test đã được xem ≥ 11 lần trước khi Phase 5 LOCK. Không thể loại trừ hoàn toàn ảnh hưởng đến các quyết định chọn hướng nghiên cứu.

---

## 4. Xác nhận/bác bỏ giả thuyết từ Rev 2

| Giả thuyết Rev 1 | Kết quả |
|---|---|
| "0.6771 và 0.6801 là cùng checkpoint B0 ở Setting A/B" | **XÁC NHẬN**: cùng `best_model_b0.pth` epoch 8, B0 Run3 |
| "0.6950 và 0.7149 là cùng checkpoint canonical Phase 7 Run 2 ở hai giao thức" | **XÁC NHẬN**: cùng run2 epoch 16, khác giao thức (Setting B tiling stride=224 vs direct 1-pass) |
| "0.7149 là seed 43" | **BÁC BỎ**: seed=42, run2 — nguồn: `official_direct_evaluation.json` không có trường seed=43 |

**Nguồn xác nhận:**
- `results/B0_Crack500_Run3_Results.md`: dòng `Official Test Dice = 0.6771 (Setting A), 0.6801 (Setting B)`
- `results/phase7_combination/phase7_s2g_new_pipeline_run2/diagnostics/official_direct_evaluation.json`: `"dice": 0.7148690704104006`
- `results/phase7_combination/phase7_s2g_new_pipeline_run2/diagnostics/final_test_metrics.json`: `checkpoint_metadata.best_dice = 0.7793468928939803` (run2 epoch 16)
