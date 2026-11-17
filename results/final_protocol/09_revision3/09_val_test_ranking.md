# 09_val_test_ranking.md — Val vs Test Ranking (Revision 3)

**Ngày:** 2026-11-17  
**Nguồn:** `results/phase7_combination/*/stage2_completion.json`, `diagnostics/final_test_metrics.json`, `diagnostics/official_direct_evaluation.json`

---

## 1. Bảng Val và Test cho b1_b2_e2e / run1 / run2

| Checkpoint | Epoch | Val Dice (direct inline) | Test Setting A | Test Setting B | Test Direct 1-pass | Nguồn val | Nguồn test |
|---|:---:|:---:|:---:|:---:|:---:|---|---|
| `b1_b2_end_to_end` | 15 | **0.7689** | **0.7016** | **0.7054** | — | `phase7_s2g_b1_b2_end_to_end/stage2_completion.json` → `best_dice` | `final_test_metrics.json` |
| `run1` | 18 | **0.7783** | **0.6881** | **0.6950** | — | `phase7_s2g_new_pipeline_run1/stage2_completion.json` → `best_dice` | `final_test_metrics.json` |
| `run2` | 16 | **0.7793** | **0.6895** | **0.6950** | **0.7149** | `phase7_s2g_new_pipeline_run2/stage2_completion.json` → `best_dice` | `final_test_metrics.json` + `official_direct_evaluation.json` |

**Giá trị val = 0.7689 hay 0.7623?**  
Từ `stage2_completion.json` của b1_b2_end_to_end:
- `"best_epoch": 15, "best_dice": 0.7689420548636025`

Giá trị 0.7623 không xuất hiện trong file này. Nếu có xuất hiện trong tài liệu tổng hợp cũ (synthesis), đó là số cũ không còn hợp lệ. **Số đúng: val = 0.7689** (nguồn: `stage2_completion.json`).

---

## 2. Thứ tự val và test

**Ranking theo val (direct inline, cao → thấp):**
1. run2 — 0.7793
2. run1 — 0.7783  
3. b1_b2_e2e — 0.7689

**Ranking theo test Setting A (cao → thấp):**
1. b1_b2_e2e — 0.7016
2. run2 — 0.6895
3. run1 — 0.6881

**Ranking theo test Setting B (cao → thấp):**
1. b1_b2_e2e — 0.7054
2. run2 = run1 — 0.6950 (cùng giá trị)

**Kết luận đo được:** Val và test Setting A xếp hạng **ngược chiều** giữa b1_b2_e2e và run2:
- b1_b2_e2e: val thấp nhất (0.7689) nhưng test Setting A cao nhất (0.7016)
- run2: val cao nhất (0.7793) nhưng test Setting A thấp hơn (0.6895)
- Mức chênh b1_b2_e2e vs run2 ở test Setting A: 0.7016 − 0.6895 = **1.21 điểm**

---

## 3. Nguyên nhân xếp hạng ngược

**Không gán nguyên nhân.** Lý do:

Cả ba checkpoint dùng **cùng config yaml:** `configs/p3_ablation/phase7/phase7_b2_s2g_b1_b2_end_to_end.yaml`  
(nguồn: trường `"config"` trong `final_test_metrics.json` của cả ba run đều trỏ về cùng file)

Không có sự khác nhau về cấu hình yaml giữa ba run. Khác biệt là ở training process (epoch dừng khác nhau: 15 / 18 / 16) và khởi tạo ngẫu nhiên (không ghi seed trong metadata). Với n=3 run và mức chênh ~1 điểm, không có đủ bằng chứng để gán nguyên nhân.

---

## 4. Mức tăng 0.6950 → 0.7149

Cùng checkpoint run2 epoch 16, hai giao thức khác nhau:
- Setting B (tiling stride=224): test = 0.6950
- Direct 1-pass (τ=0.5, không TTA): test = **0.7149**

Mức tăng = 0.7149 − 0.6950 = **0.0199 Dice = 1.99 điểm**

**Đây là do đổi giao thức inference, không phải cải thiện mô hình.**

Giải thích: Direct 1-pass resize ảnh về một kích thước cố định rồi chạy một lần, tránh được lỗi ghép tile ở ranh giới. Setting B tiling với stride=224 tạo nhiều tile có vùng chồng lấp — kết quả ghép tile có thể không tối ưu cho ảnh Crack500 (kích thước crop đã nhỏ).

> [!NOTE]
> Không có cặp hợp lệ Setting A/B cho run2 với val cùng giao thức. Val được đo bằng direct inline trong quá trình training (không phải tiling). Vì vậy gap "val direct 0.7793 → test Setting A 0.6895" là ghép chéo giao thức; gap hợp lệ duy nhất cho run2 là **direct: val 0.7793 → test 0.7149 = 6.44 điểm**.

---

## 5. Nguồn xác nhận

- `results/phase7_combination/phase7_s2g_b1_b2_end_to_end/diagnostics/final_test_metrics.json` → `checkpoint_metadata.best_dice = 0.7689420548636025`, `setting_a_official.mean_dice = 0.7015981323595685`
- `results/phase7_combination/phase7_s2g_new_pipeline_run1/diagnostics/final_test_metrics.json` → `checkpoint_metadata.best_dice = 0.7782917730246922`, `setting_a_official.mean_dice = 0.6881445266460968`
- `results/phase7_combination/phase7_s2g_new_pipeline_run2/diagnostics/final_test_metrics.json` → `checkpoint_metadata.best_dice = 0.7793468928939803`, `setting_a_official.mean_dice = 0.6895288240216444`
- `results/phase7_combination/phase7_s2g_new_pipeline_run2/diagnostics/official_direct_evaluation.json` → `"dice": 0.7148690704104006`
