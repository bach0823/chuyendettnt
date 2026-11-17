# 12_phase3_baseline.md — Khôi Phục Baseline Chuẩn Cho Thử Nghiệm Bóc Tách Phase 3 (Revision 6)

**Ngày lập:** 2026-11-17  
**Nguồn dữ liệu đối chứng:**
- Tệp metadata Phase 2 canonical: `results/final_protocol/phase2_seed42/run_metadata.json`, `phase2_seed43/run_metadata.json`, `phase2_seed44/run_metadata.json`.
- Tệp metadata Phase 3 ablation: `results/final_protocol/phase3_variant*/run_metadata.json`.
- Tệp log chạy Run1/Run2 cũ: `results/phase7_combination/phase7_s2g_new_pipeline_run1/train.log` và `run2/train.log`.

---

## 1. Làm Rõ Bản Chất Dữ Liệu Baseline Canonical

### A. Sự nhầm lẫn giữa Run1/Run2 và Hai Seed Độc Lập Trong Các Báo Cáo Trước
- Trong các báo cáo Revision 1–5 trước đây, hai số liệu **0.7783** (`run1`) và **0.7793** (`run2`) từng được gán nhầm làm "hai seed canonical đối chứng" (seed 42 và seed 43).
- **Thực tế mã nguồn và log:** Cả `run1` và `run2` trong thư mục `phase7_combination` đều được khởi chạy với **cùng `seed: 42`** (chạy song song cách nhau vài giây vào ngày 2026-10-17). Sự chênh lệch giữa 0.7783 và 0.7793 (+0.10 điểm) thuần túy phản ánh **nhiễu thống kê trong cùng một seed (intra-seed stochastic noise)** do xáo trộn batch và epoch dừng early stopping (epoch 18 vs epoch 16).
- Hai giá trị này **không phải là hai seed độc lập** và không được dùng làm baseline đối chứng chuẩn cho các biến thể seed 42 và 43 của Phase 3.

### B. Baseline Chuẩn Từ Thử Nghiệm Đa Seed Phase 2 (Cùng Mã Nguồn và Môi Trường)
Trong thử nghiệm Phase 2 (`results/final_protocol/phase2_seed*/run_metadata.json`), mô hình Canonical B2 được huấn luyện độc lập trên 3 seed khác nhau:
- **Seed 42:** `val_dice = 0.780409` (78.0409%)
- **Seed 43:** `val_dice = 0.782932` (78.2932%)
- **Seed 44:** `val_dice = 0.772572` (77.2572%)

Hai giá trị baseline trung bình chuẩn tắc:
1. **Baseline cặp 2 seed (Seed 42 và 43) — Khớp chính xác với 2 seed của Phase 3:**
   $$\bar{Y}_{\text{Phase2 (42,43)}} = \frac{0.780409 + 0.782932}{2} = \mathbf{0.781670} \quad (78.1670\%)$$
2. **Baseline toàn thể 3 seed (Seed 42, 43, 44):**
   $$\bar{Y}_{\text{Phase2 (3 seeds)}} = \frac{0.780409 + 0.782932 + 0.772572}{3} = \mathbf{0.778638} \quad (77.8638\%)$$

---

## 2. Bảng Đối Chiếu Hiệu Năng Phase 3 Từng Seed Với Cả Hai Baseline

Trích xuất trực tiếp từ các tệp `run_metadata.json` của Phase 3:

| Cấu hình / Biến thể | Tham số | Seed 42 Val Dice | Seed 43 Val Dice | Trung bình 2 seed | $\Delta$ vs Baseline 2 seed (0.781670) | $\Delta$ vs Baseline 3 seed (0.778638) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Canonical Phase 2 (Seed 42, 43)** | 10,205,755 | 0.780409 | 0.782932 | **0.781670** | **Chuẩn đối chứng 2 seed** | $+0.30$ điểm |
| **Canonical Phase 2 (3 seeds: 42, 43, 44)** | 10,205,755 | — | — | **0.778638** | $-0.30$ điểm | **Chuẩn đối chứng 3 seed** |
| **Variant D** (DenseAdapter MLP) | 16,918,915 | 0.773668 | 0.782093 | **0.777881** | **-0.38 điểm** (-0.3790đ) | **-0.08 điểm** (-0.0757đ) |
| **Variant R** (RandomRouter) | 9,824,315 | 0.779086 | 0.777913 | **0.778499** | **-0.32 điểm** (-0.3171đ) | **-0.01 điểm** (-0.0138đ) |
| **Variant N** (`residual_scale=0`) | 10,205,755 | 0.778212 | 0.777812 | **0.778012** | **-0.37 điểm** (-0.3658đ) | **-0.06 điểm** (-0.0625đ) |

---

## 3. Phân Tích Thống Kê Và Đánh Giá Khoa Học

1. **Hiệu chỉnh toàn diện số liệu chênh lệch ($\Delta$):**
   - So với **baseline 2 seed (42, 43)** cùng cặp seed: Cả ba biến thể D, R, N đều có điểm Val trung bình **thấp hơn baseline từ −0.32 đến −0.38 điểm** (D: −0.38 điểm, R: −0.32 điểm, N: −0.37 điểm).
   - So với **baseline 3 seed toàn thể (0.778638)**: Cả ba biến thể dao động quanh mức baseline từ **−0.01 đến −0.08 điểm** (D: −0.08 điểm, R: −0.01 điểm, N: −0.06 điểm).
   - **Sửa chữa:** Bãi bỏ các tóm tắt sơ sài ghi "chênh lệch chỉ từ −0.03 đến −0.09 điểm" (vốn chỉ so sánh với baseline 3 seed hoặc so với run2 đơn lẻ).
2. **Quan sát then chốt về chênh lệch 0.3 – 0.4 điểm:**
   - Khi so sánh trực tiếp trên cùng cặp seed (42 và 43), cả ba biến thể bóc tách (D, R, N) đều thấp hơn Canonical B2 khoảng **0.3 – 0.4 điểm**.
   - Tuy nhiên, với kích thước mẫu chỉ vỏn vẹn **$n = 2$ seed** cho mỗi biến thể, cùng với việc độ phân tán giữa các seed lên tới **0.84 điểm** (như quan sát thấy ở Variant D: seed 42 = 0.7737 vs seed 43 = 0.7821), cỡ mẫu $n=2$ **hoàn toàn không có đủ công suất thống kê để khẳng định mức chênh lệch 0.3 – 0.4 điểm này là hiệu ứng thực sự của kiến trúc hay chỉ là nhiễu ngẫu nhiên giữa các lần chạy**.
