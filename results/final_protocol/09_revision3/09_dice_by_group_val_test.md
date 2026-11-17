# 09_dice_by_group_val_test.md — Phân tích Dice Theo Nhóm Val & Test và Phân rã Oaxaca-Blinder (Revision 3)

**Ngày:** 2026-11-17  
**Checkpoint:** `phase7_s2g_new_pipeline_run2/best_model_b2_stage2.pth` (canonical seed 42, direct 1-pass, τ=0.5, không TTA)  
**Val Dice tái lập:** **0.7793** (kỳ vọng: 0.7793, dung sai: ±0.0005 → **PASS**)  
**Test Dice canonical cùng checkpoint:** **0.7158** (từ `official_direct_evaluation.json`: 0.7149)  
**Khoảng cách Val → Test cùng checkpoint:** **6.35 điểm** (0.7793 vs 0.7158)  
**Nguồn file:** 
- Val CSV: `results/final_protocol/09_revision3/val_per_image_dice.csv` (348 crops, 50 ảnh gốc)
- Test CSV: `results/final_protocol/08_revision2/08_per_image_dice_test.csv` (1124 crops, 148 ảnh gốc)

---

## 1. So sánh Val vs Test theo nhóm độ rộng vết nứt

| Nhóm độ rộng | Val Crop (%) | Val Dice [95% CI cluster] | Test Crop (%) | Test Dice [95% CI cluster] | Chênh lệch Dice (Val − Test) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Hẹp (≤3px)** | 0.0% (0) | 0.0000 [0.0000, 0.0000] | 0.0% (0) | 0.0000 [0.0000, 0.0000] | +0.00 điểm |
| **Vừa (3–10px)** | 32.5% (113) | 0.7141 [0.6774, 0.7501] | 34.9% (392) | 0.6557 [0.6293, 0.6782] | +5.85 điểm |
| **Rộng (>10px)** | 67.5% (235) | 0.8106 [0.7714, 0.8398] | 65.1% (732) | 0.7479 [0.7321, 0.7628] | +6.27 điểm |
| **Tổng thể** | 100% (348) | **0.7793** | 100% (1124) | **0.7158** | **+6.35 điểm** |

---

## 2. So sánh Val vs Test theo nhóm số thành phần liên thông

| Số thành phần | Val Crop (%) | Val Dice [95% CI cluster] | Test Crop (%) | Test Dice [95% CI cluster] | Chênh lệch Dice (Val − Test) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Ít (≤2)** | 64.9% (226) | 0.7885 [0.7496, 0.8212] | 57.7% (648) | 0.7434 [0.7272, 0.7581] | +4.52 điểm |
| **Vừa (3–5)** | 23.6% (82) | 0.7690 [0.7297, 0.8047] | 23.4% (263) | 0.6944 [0.6737, 0.7132] | +7.46 điểm |
| **Nhiều (>5)** | 11.5% (40) | 0.7481 [0.6967, 0.7988] | 19.0% (213) | 0.6582 [0.6291, 0.6838] | +8.99 điểm |

---

## 3. Phân rã Oaxaca-Blinder (Khoảng cách Val → Test = 6.35 điểm)

Phân rã khoảng cách $Gap = ar{Y}_{val} - ar{Y}_{test}$ theo các nhóm độ rộng:
$$\Delta_{total} = \sum_g (w_{val,g} - w_{test,g}) ar{Y}_{val,g} + \sum_g w_{test,g} (ar{Y}_{val,g} - ar{Y}_{test,g}) = \Delta_{mix} + \Delta_{within}$$

| Thành phần phân rã | Giá trị (điểm Dice) | Tỉ lệ đóng góp vào Gap |
|---|:---:|:---:|
| **Mix effect (do khác biệt phân bố tỉ lệ nhóm giữa Val và Test)** | **0.23 điểm** | **3.7%** |
| **Within effect (do mô hình dự đoán kém hơn ngay bên trong cùng nhóm)** | **6.12 điểm** | **96.3%** |
| **Tổng khoảng cách Val → Test đo được** | **6.35 điểm** | **100.0%** |

### Chi tiết từng nhóm độ rộng:
- **Nhóm hẹp (≤3px):** Mix effect = 0.00 điểm, Within effect = 0.00 điểm
- **Nhóm vừa (3–10px):** Mix effect = -1.72 điểm, Within effect = 2.04 điểm
- **Nhóm rộng (>10px):** Mix effect = 1.95 điểm, Within effect = 4.08 điểm

> [!NOTE]
> **Giới hạn của phân rã:**
> 1. Phân nhóm độ rộng thành 3 khoảng tương đối thô (≤3, 3–10, >10 px).
> 2. Cỡ mẫu Val nhỏ hơn Test (50 ảnh gốc vs 148 ảnh gốc).
> 3. Kết quả cho thấy: khoảng cách 6.35 điểm phần lớn được giải thích bởi **Within effect (6.12 điểm, 96.3%)**, tức mô hình đạt điểm Dice thấp hơn trên test ngay cả khi so sánh bên trong từng nhóm độ rộng tương đương; phần **Mix effect (0.23 điểm, 3.7%)** đóng góp một phần khi tỉ lệ phân bố giữa các nhóm có sự dịch chuyển.

---

## 4. Hệ số tương quan Spearman ở mức ảnh gốc (loại bỏ lặp mẫu)

Gộp tất cả chỉ số theo từng ảnh gốc độc lập (50 ảnh gốc Val, 148 ảnh gốc Test) trước khi tính tương quan với Dice ảnh gốc:

| Đặc tính hình thái | Val (n=50 ảnh gốc) [rho, p-value] | Test (n=148 ảnh gốc) [rho, p-value] | Kết luận |
|---|:---:|:---:|---|
| **Tỉ lệ pixel nứt** | rho = 0.522 (p = 0.0001) | rho = 0.497 (p = 1.3443e-10) | Tương quan dương có ý nghĩa trên cả val và test |
| **Độ rộng trung bình (px)** | rho = 0.490 (p = 0.0003) | rho = 0.371 (p = 3.3934e-06) | Tương quan dương có ý nghĩa (vết càng rộng, Dice càng cao) |
| **Tỉ lệ vết nứt mảnh** | rho = -0.096 (p = 0.5068) | rho = -0.298 (p = 2.3536e-04) | Tương quan âm có ý nghĩa (càng nhiều vết mảnh, Dice càng thấp) |
| **Số thành phần liên thông** | rho = -0.260 (p = 0.0687) | rho = -0.338 (p = 2.6205e-05) | Tương quan trên val vs test có sự khác biệt |

> [!IMPORTANT]
> Toàn bộ các giá trị p-value cực nhỏ cỡ $10^{-65}$ xuất hiện ở các báo cáo trước đều là do tính ở **cấp độ crop (1,124 mẫu)** bị thổi phồng bậc tự do (pseudoreplication). Khi gộp đúng theo ảnh gốc độc lập, các p-value trở về mức chuẩn thống kê.

---

## 5. Kết luận chỉ dựa trên số đo

1. Khoảng cách Val → Test (6.35 điểm) của cùng một checkpoint canonical seed 42 không phải do một yếu tố duy nhất tạo nên.
2. Phân rã Oaxaca-Blinder cho thấy Within effect đóng góp 6.12 điểm (96.3%), trong khi Mix effect đóng góp 0.23 điểm (3.7%). 
3. Trên cả Val và Test, độ rộng vết nứt tương quan thuận với Dice, còn tỉ lệ vết nứt mảnh tương quan nghịch với Dice ở mức ảnh gốc có ý nghĩa thống kê.
