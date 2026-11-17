# 11_b1_b2_vs_run2.md — Đối Chiếu b1_b2_end_to_end vs run1/run2 Cùng Giao Thức Setting A (Revision 5)

**Ngày lập:** 2026-11-17  
**Nguồn dữ liệu:**
- `results/final_protocol/10_revision4/10_config_diff.md`
- Log huấn luyện: `results/phase7_combination/phase7_s2g_b1_b2_end_to_end/train.log` và `results/phase7_combination/phase7_s2g_new_pipeline_run2/train.log`.
- Kết quả test: `results/phase7_combination/phase7_s2g_b1_b2_end_to_end/diagnostics/final_test_metrics.json` và `results/phase7_combination/phase7_s2g_new_pipeline_run2/diagnostics/final_test_metrics.json`.

---

## 1. Đối Chiếu Kết Quả Test Cùng Giao Thức Setting A

Khi đánh giá trên cùng tập Test Crack500 theo **cùng một giao thức Setting A** (cắt patch 448×448, bước trượt stride 448, τ=0.5, không TTA):

| Lần chạy (Run) | Checkpoint tốt nhất | Điểm Test Setting A | Hiệu số so với run2 |
|---|:---:|:---:|:---:|
| **`phase7_s2g_b1_b2_end_to_end`** | Epoch 15 | **0.7016** | **+1.21 điểm** |
| **`phase7_s2g_new_pipeline_run1`** | Epoch 18 | **0.6881** | **-0.14 điểm** |
| **`phase7_s2g_new_pipeline_run2`** | Epoch 16 | **0.6895** | **0.00 điểm (Chuẩn)** |

**Quan sát thực nghiệm:** Cùng đo bằng Setting A trên test, `b1_b2_end_to_end` đạt **0.7016**, cao hơn `run2` (**0.6895**) là **1.21 điểm Dice** (và cao hơn `run1` là 1.35 điểm).

---

## 2. Bảng Liệt Kê Toàn Bộ Khác Biệt Cấu Hình Giữa Hai Bên

Trích xuất từ tệp cấu hình và nhật ký huấn luyện chi tiết:

| Thuộc tính | `b1_b2_end_to_end` | `phase7_s2g_new_pipeline_run2` | Bản chất khác biệt |
|---|:---:|:---:|---|
| **Thời điểm chạy** | 2026-10-10 14:27:48 | 2026-10-17 07:32:42 | Cách nhau 7 ngày |
| **Kiến trúc mô hình** | B2 UNet (Femto + 4 ViT + SAGE) | B2 UNet (Femto + 4 ViT + SAGE) | Hoàn toàn giống nhau bit-for-bit |
| **Hàm mất mát & LR** | CrackBinaryLoss, AdamW stage 2 lr 1e-4 | CrackBinaryLoss, AdamW stage 2 lr 1e-4 | Hoàn toàn giống nhau |
| **S2-Gate Module** | Block 1 + Block 2 (dilation 2) | Block 1 + Block 2 (dilation 2) | Hoàn toàn giống nhau |
| **Giao thức đo Val trong huấn luyện** | **`setting_a`** (tiling 448, stride 448) | **`direct`** (resize 1-pass 448) | **Khác biệt then chốt về tiêu chí chọn checkpoint** |
| **Epoch chọn checkpoint tốt nhất** | Epoch 15 (Val: **0.7689**) | Epoch 16 (Val: **0.7793**) | Dừng ở epoch khác nhau |

---

## 3. Kết Luận Khoa Học & Giới Hạn Diễn Giải

1. **Chưa thể gán nguyên nhân cho mức chênh lệch 1.21 điểm:**
   - Mặc dù `b1_b2_end_to_end` đạt Test Setting A cao hơn `run2` 1.21 điểm, trong lịch sử nghiên cứu **chưa từng có thí nghiệm kiểm chứng độc lập (ablation study)** để cô lập riêng rẽ các yếu tố: ảnh hưởng của tiêu chí chọn checkpoint (Val Setting A vs Val Direct), ảnh hưởng của epoch dừng (15 vs 16), hay ảnh hưởng của seed khởi tạo và xáo trộn batch.
   - Do đó, việc gán nguyên nhân chênh lệch này cho bất kỳ yếu tố cụ thể nào (như "pipeline mới làm giảm hiệu năng" hay "overfitting") đều là **suy diễn chưa được kiểm chứng**.
2. **Val và Test ranking không thể so sánh được:**
   - Điểm Val của `b1_b2_end_to_end` là **0.7689** (đo bằng Setting A).
   - Điểm Val của `run2` là **0.7793** (đo bằng Direct 1-pass).
   - Vì hai điểm Val này được sinh ra từ hai phương pháp đo lường hoàn toàn khác nhau, mọi phát biểu so sánh thứ tự xếp hạng (ranking) giữa Val và Test giữa hai lần chạy này đều vô nghĩa về mặt khoa học.
