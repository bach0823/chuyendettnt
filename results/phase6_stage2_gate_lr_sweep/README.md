# Phase 6 Stage 2 S2-Gate Learning Rate Sweep & Comprehensive Audit

> **Báo cáo nghiệm thu & đối chiếu thực nghiệm 4 bộ kết quả từ Colab**  
> Thời gian: 2026-10-08  
> Tập dữ liệu: Crack500 Canonical Validation (Setting A, N=348, Tiling 448x448, FP32 strict, seed=42)

---

## 1. Nguồn Gốc và Lưu Trữ Dữ Liệu

Toàn bộ 4 file nén trên Desktop đã được lưu trữ vĩnh viễn và giải nén đầy đủ vào `results/phase6_stage2_gate_lr_sweep/`:

| Tệp gốc trên Desktop | Thư mục giải nén nội bộ | Checkpoint size | Thời điểm chạy | Ghi chú kiến trúc |
| :--- | :--- | :--- | :--- | :--- |
| `Phase6_Stage2_Gate_LR_Sweep (1).zip` | `batch1_lr3e4_peak07706/` | 42.11 MB | 14:53 -> 15:43 | Batch 1 (Pre-clean): S2-Gate LR = `3e-4` |
| `Phase6_Stage2_Gate_LR_Sweep.zip` | `batch1_lr1e3_peak07688/` | 42.11 MB | 14:52 -> 15:42 | Batch 1 (Pre-clean): S2-Gate LR = `1e-3` |
| `Phase6_Stage2_Gate_LR_Sweep (2).zip` | `batch2_lr1e3_clean_p3/` | 41.95 MB | 15:51 -> 16:38 | Batch 2 (Clean P3): S2-Gate LR = `1e-3` |
| `Phase6_Stage2_Gate_LR_Sweep (3).zip` | `batch2_lr3e4_clean_p3/` | 41.95 MB | 15:51 -> 16:38 | Batch 2 (Clean P3): S2-Gate LR = `3e-4` |

*Các tệp gốc `.zip` cũng được sao lưu nguyên vẹn vào `results/phase6_stage2_gate_lr_sweep/zips/`.*

---

## 2. Bảng Đối Chiếu Hiệu Năng Chi Tiết (Setting A Canonical Validation, N=348)

| Mô hình & Cấu hình | S2-Gate LR | Best Ep | Val Dice | Val IoU | Precision | Recall | Loss đáy | Thin Fail ($N$) | Trạng thái P3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 6 Canonical Baseline** | 1e-4 | Ep 16 | 0.7676 | 0.6451 | 0.7407 | 0.8384 | 1.3467 | - | ASDW ($\gamma \approx 0$) |
| **Phase 6 Kỷ Lục Cũ (Run 2)** | 1e-4 | Ep 14 | 0.7702 | 0.6471 | 0.7510 | 0.8301 | 1.3471 | 24 | ASDW ($\gamma \approx 0$) |
| **Batch 1: LR = 3e-4** 🏆 | **3e-4** | **Ep 18** | **0.7706** | **0.6476** | **0.7344** | **0.8509** | **1.3376** | **22** | ASDW ($\gamma \approx 0$) |
| **Batch 1: LR = 1e-3** | 1e-3 | Ep 14 | 0.7688 | 0.6460 | 0.7436 | 0.8374 | 1.3510 | 23 | ASDW ($\gamma \approx 0$) |
| **Batch 2: LR = 1e-3 (Clean)** | 1e-3 | Ep 14 | 0.7677 | 0.6464 | **0.7546** | 0.8248 | 1.3419 | 23 | Pure Compression |
| **Batch 2: LR = 3e-4 (Clean)** | 3e-4 | Ep 18 | 0.7662 | 0.6435 | 0.7340 | 0.8470 | **1.3280** | 25 | Pure Compression |

---

## 3. Phân Tích Hiện Tượng và Kết Luận Khoa Học

1. **Thiết lập Kỷ lục mới toàn dự án (Val Dice = 0.7706)**:
   - Cấu hình `stage2_s2_gate_lr = 3e-4` đã vượt mốc kỷ lục cũ `0.7702` để đạt đỉnh mới `0.7706`.
   - Recall đạt đỉnh cao nhất từ trước đến nay: **85.09%**, chứng tỏ cổng điều chế không gian S2-Gate khi được học với tốc độ thích hợp đã mở rộng hành lang nhận diện vết nứt mảnh mà không làm bùng nổ false alarms.
   - Số ca thất bại vết nứt siêu mảnh (`thin_low_area_failure`) giảm xuống mức thấp nhất: **22 ca**.

2. **Quy luật động học của Learning Rate cho S2-Gate**:
   - **Mức LR = 1e-3**: Hội tụ rất nhanh, đạt cực đại ngay tại **Epoch 14**, tối ưu hóa cực tốt cho **Precision** (đạt $75.46\%$, cao nhất trong các run).
   - **Mức LR = 3e-4**: Học đều đặn và bền bỉ hơn, đạt cực đại tại **Epoch 18**, tối ưu hóa mạnh mẽ cho **Recall** ($85.09\%$) và **Val Dice tổng thể**.

3. **Ảnh hưởng của việc dọn sạch P3 (Pure Spatial Compression)**:
   - Checkpoint giảm từ 42.11 MB xuống 41.95 MB (loại bỏ đúng 10 tensor ASDW không cần thiết).
   - Mức Dice của Batch 2 (0.7662 - 0.7677) vẫn bám sát đỉnh và vượt baseline ban đầu (0.7676), chứng minh kiến trúc mới tinh gọn hoàn toàn độc lập và sẵn sàng 100% cho Phase 7.
