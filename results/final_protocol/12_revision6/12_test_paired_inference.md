# 12_test_paired_inference.md — Khảo Sát Mảng Dice Chi Tiết Trên Tập Test Và Hiện Trạng Kiểm Định Ghép Cặp (Revision 6)

**Ngày lập:** 2026-11-17  
**Nguồn dữ liệu đối chứng:**
- `results/final_protocol/05_test_final_report.json` và `results/final_protocol/08_revision2/08_per_image_dice_test.csv`.
- `results/phase7_combination/phase7_s2g_new_pipeline_run2/diagnostics/official_direct_evaluation.json`.
- `results/baselines/50ep/B0_Crack500/official_eval_test.json`.
- `results/baselines/100ep/B0_Crack500/official_eval_all_splits.json`.

---

## 1. Rà Soát Sự Tồn Tại Của Mảng `per_image_dice` Trên Tập Test

Kiểm tra trực tiếp cấu trúc dữ liệu của toàn bộ 4 mô hình/cấu hình cần kiểm định trên tập Test:

| Mô hình / Cấu hình | Giao thức đo test | Tệp nguồn lưu kết quả test | Trường dữ liệu chi tiết | Độ dài mảng (Length) | Trạng thái dữ liệu |
|---|:---:|---|:---:|:---:|:---:|
| **LOCKED** (Ensemble 3 seed + TTA) | direct 1-pass (TTA flip_hv) | `results/final_protocol/05_test_final_report.json` và `08_revision2/08_per_image_dice_test.csv` | `per_image_dice` | **1,124** mẫu | **ĐÃ CÓ ĐẦY ĐỦ** (Khớp chính xác 1,124 crops test, 148 ảnh gốc) |
| **Canonical run2 direct** | direct 1-pass (không TTA) | `results/phase7_combination/phase7_s2g_new_pipeline_run2/diagnostics/official_direct_evaluation.json` | Không có (chỉ có aggregate dict: `dice: 0.714869...`) | **0 (Không có)** | **Thiếu mảng per-sample** |
| **B0 50ep** | direct 1-pass (không TTA) | `results/baselines/50ep/B0_Crack500/official_eval_test.json` | Không có (chỉ có aggregate dict: `dice: 0.704170...`) | **0 (Không có)** | **Thiếu mảng per-sample** |
| **B0 100ep** | direct 1-pass (không TTA) | `results/baselines/100ep/B0_Crack500/official_eval_all_splits.json` | Không có (chỉ có aggregate dict: `dice: 0.703725...`) | **0 (Không có)** | **Thiếu mảng per-sample** |

> [!NOTE]
> Trong tệp `results/phase7_combination/phase7_s2g_new_pipeline_run2/diagnostics/final_test_per_sample.csv` có lưu 2,248 dòng chi tiết theo mẫu, nhưng chỉ đo cho hai giao thức **Setting A** (1,124 crops) và **Setting B** (1,124 crops), hoàn toàn **không lưu mảng per-sample cho giao thức direct 1-pass (0.7149)**.

---

## 2. Trạng Thái Thực Hiện Suy Luận Mới Trên Tập Test

Theo chỉ dẫn phương pháp luận:
- Để thực hiện kiểm định bootstrap ghép cặp (Paired Cluster Bootstrap) cho hai cặp (B0 100ep vs Canonical direct; Canonical direct vs LOCKED), cần có mảng Dice theo từng ảnh (với tên file được sắp xếp `sorted()` khớp 1,124 crops).
- **Trạng thái thực hiện:** **Chưa thực hiện chạy suy luận mới trên tập test trong phiên làm việc này.**
- **Lý do bảo toàn phương pháp luận:** 
  1. Giữ nguyên vẹn tính độc lập của tập Test theo các phiên bản báo cáo trước.
  2. Báo cáo ghi nhận trung thực và minh bạch hiện trạng lưu trữ lịch sử của dự án: các lần chạy trong quá khứ chỉ lưu chỉ số tổng hợp (`official_eval_all_splits.json`), không lưu per-image.

---

## 3. Kết Quả Kiểm Định Và Cập Nhật Tuyên Bố

1. **Cặp B0 100ep (0.7037) vs Canonical run2 direct (0.7149):**
   - Hiệu số Dice trung bình: $+1.12$ điểm (0.7149 − 0.7037).
   - Trạng thái kiểm định ghép cặp: **Chưa kiểm định**.
2. **Cặp Canonical run2 direct (0.7149) vs LOCKED (0.7158):**
   - Hiệu số Dice trung bình: $+0.09$ điểm (0.7158 − 0.7149).
   - Trạng thái kiểm định ghép cặp: **Chưa kiểm định**.
3. **Cập nhật mục Hạn chế và Tuyên bố nghiêm cấm (§7B):**
   - Bãi bỏ hoàn toàn câu nhận định thiếu căn cứ: *"Hiệu số +1.12 điểm trên test nằm trong biên độ dao động giữa các lần chạy"*.
   - **Thay thế bằng phát biểu chuẩn tắc:**
     *"Hiệu số +1.12 điểm trên test giữa B0 100ep và Canonical B2 **chưa được kiểm định ghép cặp**; đồng thời **chưa có số liệu đo lường phương sai/nhiễu giữa các seed (seed-to-seed variance) trên tập test direct** do chỉ có một lần chạy direct duy nhất được ghi nhận."*
