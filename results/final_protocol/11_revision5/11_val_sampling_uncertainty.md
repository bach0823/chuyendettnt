# 11_val_sampling_uncertainty.md — Sai Số Lấy Mẫu Cụm Tập Val, Nhiễu Seed và Hệ Quả Đối Với MDE Phase 3 (Revision 5)

**Ngày lập:** 2026-11-17  
**Nguồn dữ liệu:**
- Val per-image Dice Single-seed 42: `results/final_protocol/09_revision3/val_per_image_dice.csv` (348 crops, 50 ảnh gốc).
- Val per-image Dice LOCKED: `results/final_protocol/10_revision4/val_locked_per_image_dice.csv` (348 crops, 50 ảnh gốc).
- Log & Metadata Phase 3: `results/final_protocol/phase3_variant*/run_metadata.json` và `results/final_protocol/10_revision4/10_phase3_reconcile.md`.

---

## 1. Khoảng Tin Cậy Cụm (Cluster Bootstrap 95% CI) Của Điểm Val Dice

Tập Validation Crack500 gồm 348 ảnh cắt (crops) được sinh ra từ **50 ảnh gốc độc lập**. Khi tính toán khoảng tin cậy phi tham số bằng phương pháp Cluster Bootstrap (10,000 lần resample ở cấp độ ảnh gốc độc lập):

| Cấu hình mô hình trên tập Val | Val Mean Dice | 95% CI Bootstrap cụm [Low, High] | Độ rộng khoảng tin cậy 95% |
|---|:---:|:---:|:---:|
| **Canonical Single-seed 42** (direct 1-pass) | **0.7793** | **[0.7489, 0.8063]** | **5.74 điểm** |
| **LOCKED Ensemble** (3 seeds + TTA flip_hv) | **0.7853** | **[0.7557, 0.8113]** | **5.56 điểm** |

> [!IMPORTANT]
> Do kích thước tập Val chỉ có vỏn vẹn **50 ảnh gốc**, sai số lấy mẫu cụm (cluster sampling uncertainty) là cực kỳ lớn: khoảng tin cậy 95% của điểm Val có độ rộng lên tới **~5.6 – 5.7 điểm Dice**!

---

## 2. Đối Chiếu Độ Phân Tán Seed Và Chênh Lệch Giữa Các Biến Thể Phase 3

### A. Độ phân tán giữa các seed (Seed-to-seed variance)
Từ tệp `run_metadata.json` gốc:
- **Canonical B2:** Seed 42 = 0.7793, Seed 43 = 0.7783 $\to$ Chênh lệch giữa 2 seed: **0.10 điểm**.
- **Variant D (DenseAdapter):** Seed 42 = 0.7737, Seed 43 = 0.7821 $\to$ Chênh lệch giữa 2 seed: **0.84 điểm**.
- **Variant R (RandomRouter):** Seed 42 = 0.7791, Seed 43 = 0.7779 $\to$ Chênh lệch giữa 2 seed: **0.12 điểm**.
- **Variant N (Residual scale = 0):** Seed 42 = 0.7782, Seed 43 = 0.7778 $\to$ Chênh lệch giữa 2 seed: **0.04 điểm**.

### B. So sánh hiệu số trung bình giữa các biến thể với Canonical B2

| Biến thể | Điểm Val trung bình (2 seed) | Chênh lệch so với Canonical B2 ($\Delta$) | So với độ phân tán seed (0.84đ) | So với độ rộng CI Val (5.74đ) |
|---|:---:|:---:|:---:|:---:|
| **Canonical B2** | 0.7788 | Chuẩn đối chứng | — | — |
| **Variant D** | 0.7779 | **-0.09 điểm** | Nhỏ hơn 9.3 lần | Nhỏ hơn 63.8 lần |
| **Variant R** | 0.7785 | **-0.03 điểm** | Nhỏ hơn 28.0 lần | Nhỏ hơn 191.3 lần |
| **Variant N** | 0.7780 | **-0.08 điểm** | Nhỏ hơn 10.5 lần | Nhỏ hơn 71.8 lần |

---

## 3. Hệ Quả Kép Đối Với MDE (Minimum Detectable Effect) Và Kết Luận Phase 3

1. **Hiệu ứng kép chèn ép tín hiệu (Double confounding):**
   Mọi chênh lệch hiệu năng quan sát được giữa các biến thể Phase 3 ($\Delta < 0.10$ điểm trung bình, hoặc tối đa 0.56 điểm ở từng seed đơn lẻ) đều bị chìm ngập hoàn toàn bởi hai tầng nhiễu ngẫu nhiên:
   - **Tầng 1 (Nhiễu ngẫu nhiên giữa các lần chạy - Seed variance):** Độ phân tán ngẫu nhiên do khởi tạo trọng số và xáo trộn dữ liệu giữa các lần chạy lên tới **0.84 điểm** (như đã thấy ở Variant D).
   - **Tầng 2 (Sai số lấy mẫu cụm của tập Val - Cluster sampling error):** Độ rộng khoảng tin cậy do cỡ mẫu chỉ có 50 ảnh gốc lên tới **5.74 điểm**.
2. **Hiệu chỉnh kết luận về MDE (Minimum Detectable Effect):**
   - Với chỉ 2 seed huấn luyện và tập kiểm định gồm 50 ảnh gốc, hiệu ứng tối thiểu có thể phát hiện được (MDE) với độ tin cậy thống kê chấp nhận được ($\alpha=0.05$, power $80\%$) phải đạt ít nhất **$\ge 1.0 – 1.5$ điểm**.
   - Vì toàn bộ chênh lệch của Phase 3 đều $< 0.1$ điểm (trung bình), toàn bộ các biến thể D, R, N hoàn toàn không thể phân biệt được với Canonical B2 về mặt thống kê.
3. **Kết luận về câu hỏi nghiên cứu Phase 3:**
   - Thử nghiệm Phase 3 **chưa đủ công suất thống kê (underpowered)** để chứng minh Router có học được biểu diễn chuyên biệt hay không, hoặc nhánh expert có đóng góp gia tăng hay không.
   - Tuy nhiên, việc tắt hoàn toàn nhánh expert (Variant N) hay thay bằng MLP dày 6.71M tham số (Variant D) vẫn giữ nguyên điểm Val trong biên độ $\pm 0.1$ điểm cho thấy: **phần lõi tạo nên hiệu năng 0.778+ trên Crack500 nằm ở nhánh chính Backbone pretrain + Decoder + S2-Gate (chiếm 92.77% tham số), chứ không phụ thuộc vào nhánh SAGE riêng biệt.**
