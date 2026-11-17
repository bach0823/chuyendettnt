# 11_test_paired_test.md — Kiểm Định Ghép Cặp Trên Tập Test và Đánh Giá Khả Dụng Dữ Liệu (Revision 5)

**Ngày lập:** 2026-11-17  
**Quy tắc cứng số 1:** KHÔNG chạy suy luận mới trên tập test. Chỉ sử dụng số liệu Dice theo ảnh (per-image / per-crop) đã được lưu sẵn trong kho lưu trữ trước thời điểm lập báo cáo.

---

## 1. Khảo Sát Hiện Trạng Dữ Liệu Dice Theo Ảnh (Per-image / Per-sample Dice) Trên Tập Test

Rà soát toàn bộ các tệp kết quả trong kho lưu trữ đối với 4 mô hình/cấu hình cần kiểm định:

| Mô hình / Checkpoint | Giao thức đo test | Tệp nguồn lưu kết quả test | Dữ liệu Dice theo ảnh (per-sample / per-image) | Trạng thái khả dụng dữ liệu ghép cặp |
|---|:---:|---|:---:|:---:|
| **Canonical run2 direct** (0.7149) | direct 1-pass (τ=0.5, không TTA) | `results/phase7_combination/phase7_s2g_new_pipeline_run2/diagnostics/official_direct_evaluation.json` | **Không có** (Tệp chỉ chứa các chỉ số tổng hợp: `precision: 0.6841`, `recall: 0.7711`, `dice: 0.714869`, `iou: 0.5735`, `boundary_iou: 0.4449`, `hd95: 19.38`) | **Thiếu dữ liệu per-image direct** |
| **B0 50ep** (0.6974) | direct 1-pass | `results/baselines/50ep/B0_Crack500/official_eval_test.json` | **Không có** (Tệp chỉ lưu aggregate metrics, không có mảng per-image) | **Thiếu dữ liệu per-image** |
| **B0 100ep** (0.7037) | direct 1-pass | `results/baselines/100ep/B0_Crack500/official_eval_all_splits.json` | **Không có** (Tệp chỉ lưu aggregate metrics: `dice: 0.703714`, không có mảng per-image) | **Thiếu dữ liệu per-image** |
| **LOCKED** (0.7158) | Ensemble 3 seed + TTA flip_hv | `results/final_protocol/05_test_final_report.json` và `results/final_protocol/08_revision2/08_per_image_dice_test.csv` | **ĐÃ CÓ** (1,124 crops gộp thành 148 ảnh gốc độc lập, trung bình: 0.7158) | **Đã có dữ liệu per-image** |

> [!NOTE]
> Trong tệp `results/phase7_combination/phase7_s2g_new_pipeline_run2/diagnostics/final_test_per_sample.csv` có lưu 2,248 dòng chi tiết theo mẫu, nhưng chỉ đo cho hai giao thức **Setting A** (1,124 crops, Dice = 0.6895) và **Setting B** (1,124 crops, Dice = 0.6950), hoàn toàn **không lưu mảng per-sample cho giao thức direct 1-pass (0.7149)**.

---

## 2. Kết Quả Kiểm Định Ghép Cặp (Paired Cluster Bootstrap Test)

Theo Quy tắc cứng số 1, agent **tuyệt đối không chạy suy luận mới trên test** để tự tạo ra dữ liệu per-image cho Canonical run2 direct hay B0 100ep. Do đó:

### Cặp 1: B0 100ep direct (0.7037) vs Canonical run2 direct (0.7149)
- **Hiệu số Dice trung bình ($\Delta$):** $+1.12$ điểm (0.7149 − 0.7037).
- **Trạng thái kiểm định:** **Chưa kiểm định được**.
- **Nguyên nhân:** Cả `official_eval_all_splits.json` của B0 100ep và `official_direct_evaluation.json` của Canonical run2 direct đều không lưu mảng Dice theo từng ảnh của tập Test; quy tắc nghiên cứu cấm chạy suy luận mới trên test.

### Cặp 2: Canonical run2 direct (0.7149) vs LOCKED (0.7158)
- **Hiệu số Dice trung bình ($\Delta$):** $+0.09$ điểm (0.7158 − 0.7149).
- **Trạng thái kiểm định:** **Chưa kiểm định được**.
- **Nguyên nhân:** Mô hình LOCKED có lưu mảng 1,124 crops (`08_per_image_dice_test.csv`), nhưng Canonical run2 direct không có mảng Dice per-sample tương ứng để thực hiện trừ cặp (pairwise difference) theo từng ảnh gốc.

---

## 3. Thay Thế Lập Luận Thống Kê Trong Báo Cáo

1. **Bãi bỏ hoàn toàn các câu suy diễn định tính cũ:**
   - Bỏ câu: *"Chênh lệch giữa B0 100ep và B2 canonical (1.12 điểm) nằm trong khoảng tin cậy của test"*.
   - Bỏ câu: *"LOCKED tăng 0.09 điểm nằm trong CI của test"*.
2. **Cập nhật phát biểu chuẩn xác theo dữ liệu thực tế:**
   - *"Hiệu số giữa B0 100ep và B2 canonical trên cùng giao thức direct là +1.12 điểm (0.7037 vs 0.7149); hiệu số giữa canonical direct và LOCKED là +0.09 điểm (0.7149 vs 0.7158). Cả hai phép so sánh này **chưa kiểm định được** bằng bootstrap ghép cặp do log lịch sử của B0 100ep và Canonical run2 direct chỉ lưu số liệu tổng hợp mà không lưu mảng Dice theo từng ảnh, đồng thời quy tắc cứng cấm chạy suy luận mới trên test."*
3. **Giả định phương pháp luận:**
   - Khi có đủ dữ liệu per-image trong các nghiên cứu tương lai, phép kiểm định bootstrap ghép cặp theo cụm ảnh gốc (cluster bootstrap 10,000 lần) **chỉ xử lý và kiểm soát sai số lấy mẫu ảnh (image sampling noise)**, hoàn toàn **không xử lý được sai số ngẫu nhiên giữa các lần chạy huấn luyện (seed-to-seed variance)**.
