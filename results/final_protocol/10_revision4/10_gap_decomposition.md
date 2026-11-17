# 10_gap_decomposition.md — Phân rã Oaxaca-Blinder và Khoảng Tin Cậy Cụm Cho Cấu Hình LOCKED (Revision 4)

**Ngày:** 2026-11-17  
**Cấu hình chính (LOCKED):** Ensemble 3 seed (42, 43, 44) + TTA flip_hv, $\tau=0.5$  
**Cấu hình đối chiếu:** Single-seed canonical (Seed 42, không TTA, direct 1-pass)  
**Tập dữ liệu:** 348 crops Val (50 ảnh gốc) và 1,124 crops Test (148 ảnh gốc)  

---

## 1. Kết quả suy luận và Ước lượng Khoảng tin cậy dạng Cụm (Cluster Bootstrap 10,000 lần)

Ước lượng khoảng tin cậy 95% bằng cách tái lấy mẫu độc lập có hoàn lại trên các cụm ảnh gốc (50 cụm Val và 148 cụm Test):

| Cấu hình mô hình | Phân tách | Dice trung bình | 95% CI Cluster | Độ rộng CI |
|---|:---:|:---:|:---:|:---:|
| **LOCKED (Ensemble + TTA)** | **VAL** | **0.7853** | **[0.7557, 0.8113]** | 5.56 điểm |
| **LOCKED (Ensemble + TTA)** | **TEST** | **0.7158** | **[0.7002, 0.7307]** | 3.06 điểm |
| **LOCKED Gap (Val − Test)** | — | **+6.95 điểm** | **[3.64, 10.00] điểm** | 6.36 điểm |
| *Single-seed (Seed 42 no TTA)* | *VAL* | *0.7793* | *[0.7489, 0.8063]* | 5.73 điểm |
| *Single-seed Gap (Val − Test)* | — | *+6.35 điểm* | *[2.94, 9.46] điểm* | 6.52 điểm |

> [!IMPORTANT]
> Khoảng cách suy giảm hiệu năng Val → Test của cấu hình LOCKED là **+6.95 điểm**, với 95% CI cluster là **[3.64, 10.00] điểm**.

---

## 2. Bảng Phân rã Oaxaca-Blinder theo Nhóm Độ rộng

Phân rã chênh lệch $\Delta = \bar{Y}_{val} - \bar{Y}_{test} = \Delta_{mix} + \Delta_{within}$:

| Thành phần phân rã | Cấu hình LOCKED (Ensemble+TTA) | Cấu hình Đối chiếu (Single seed 42) |
|---|:---:|:---:|
| **Val Dice tổng thể** | **0.7853** | **0.7793** |
| **Test Dice tổng thể** | **0.7158** | **0.7158** |
| **Tổng khoảng cách (Gap)** | **+6.95 điểm (100.0%)** | **+6.35 điểm (100.0%)** |
| **Within effect (chênh lệch trong cùng nhóm)** | **+6.72 điểm (96.6%)** | **+6.12 điểm (96.3%)** |
| **Mix effect (do lệch phân bố tỉ lệ nhóm)** | **+0.24 điểm (3.4%)** | **+0.23 điểm (3.7%)** |

### Chi tiết các nhóm độ rộng (Cấu hình LOCKED):
- **Nhóm hẹp ($\le 3$ px):** 0 crop ở cả hai tập.
- **Nhóm vừa ($3–10$ px):** 
  - Val: 113 crops (32.5%), Dice = **0.7192**
  - Test: 392 crops (34.9%), Dice = **0.6557**
  - Chênh lệch trong nhóm: **+6.35 điểm**
  - Đóng góp: Within = 2.21 điểm, Mix = -1.73 điểm.
- **Nhóm rộng ($>10$ px):**
  - Val: 235 crops (67.5%), Dice = **0.8171**
  - Test: 732 crops (65.1%), Dice = **0.7479**
  - Chênh lệch trong nhóm: **+6.91 điểm**
  - Đóng góp: Within = 4.50 điểm, Mix = 1.96 điểm.

---

## 3. Diễn giải Khoa học Đúng mực (Tránh Kết luận Vượt quá Dữ liệu)

1. **Ý nghĩa thống kê của "Within effect":**
   - Thuật ngữ "within-group effect" (chiếm ~96%) trong phân rã Oaxaca thuần túy phản ánh mặt số học: điểm Dice trung bình của tập Test thấp hơn tập Val ở cả hai nhóm độ rộng (vết vừa tụt ~5.8 điểm, vết rộng tụt ~6.3 điểm).
   - **Không được kết luận** rằng "mô hình dự đoán kém hơn" do lỗi kiến trúc, và **không được khẳng định** "tập Test không khó hơn". Sự chênh lệch này gồm nhiều nguyên nhân chưa thể tách rời:
     - Các đặc tính hình ảnh/bề mặt chưa được đo lường (độ tương phản, độ nhám mặt đường, ánh sáng, nhiễu bóng râm).
     - Thiên lệch chọn lựa checkpoint và siêu tham số được tối ưu trên tập Val (Selection bias: ~0.22 điểm quanh vùng hội tụ).
     - Nhiễu lấy mẫu xuất phát từ cỡ mẫu Val nhỏ (chỉ có 50 ảnh gốc độc lập).
2. **Thiên lệch chọn epoch (Selection bias):**
   - Sử dụng con số **0.22 điểm** (mức chênh giữa epoch tốt nhất và trung bình 5 epoch cuối quanh vùng hội tụ) làm ước lượng đại diện cho thiên lệch chọn epoch.
   - Loại bỏ con số 1.60 điểm (tính trên toàn bộ 18 epoch) vì con số này gộp cả các epoch đầu khi mạng chưa hội tụ, không phản ánh đúng thiên lệch chọn mô hình.
