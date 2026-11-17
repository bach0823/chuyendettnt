# 11_spearman_ci.md — Khoảng Tin Cậy 95% Của Tương Quan Spearman và Hiệu Chỉnh Lập Luận Vết Nứt Mảnh (Revision 5)

**Ngày lập:** 2026-11-17  
**Nguồn dữ liệu:**
- Val Dice theo ảnh: `results/final_protocol/09_revision3/val_per_image_dice.csv` (348 crops gộp thành 50 ảnh gốc độc lập).
- Test Dice theo ảnh: `results/final_protocol/08_revision2/08_per_image_dice_test.csv` (1,124 crops gộp thành 148 ảnh gốc độc lập).
- Mặt nạ nhãn ground truth: `datasets/Crack500_ready/val/masks/` và `datasets/Crack500_ready/test/masks/`.
- Tệp kết quả JSON: `results/final_protocol/11_revision5/spearman_results.json`.

---

## 1. Bảng Khoảng Tin Cậy 95% Của Hệ Số Tương Quan Spearman ($\rho$) Ở Mức Ảnh Gốc

Tính toán được thực hiện sau khi gộp trung bình theo từng ảnh gốc độc lập (loại bỏ lặp mẫu do chia crop): $n_{\text{val}} = 50$ ảnh gốc, $n_{\text{test}} = 148$ ảnh gốc. Khoảng tin cậy 95% được tính bằng cả hai phương pháp:
- **Fisher $z$-transformation:** Với rank correlation Spearman, $SE \approx \frac{1.06}{\sqrt{n - 3}}$.
- **Bootstrap phi tham số (10,000 lần resample ở cấp ảnh gốc).**

| Đặc tính hình thái | Tập Val ($n=50$ ảnh gốc)<br>$\rho$ ($p$-value) | Val CI 95% [Fisher]<br>Val CI 95% [Bootstrap] | Tập Test ($n=148$ ảnh gốc)<br>$\rho$ ($p$-value) | Test CI 95% [Fisher]<br>Test CI 95% [Bootstrap] | Khoảng CI 95% có chồng lấp (Overlap)? |
|---|:---:|:---:|:---:|:---:|:---:|
| **Tỉ lệ pixel nứt** (`crack_ratio`) | $\rho = 0.522$<br>($p = 0.0001$) | [0.270, 0.708]<br>[0.258, 0.726] | $\rho = 0.497$<br>($p = 1.34\times 10^{-10}$) | [0.356, 0.615]<br>[0.351, 0.627] | **Chồng lấp hoàn toàn**<br>(Overlap: True) |
| **Độ rộng trung bình** (`mean_width`) | $\rho = 0.490$<br>($p = 0.0003$) | [0.229, 0.685]<br>[0.242, 0.681] | $\rho = 0.371$<br>($p = 3.39\times 10^{-6}$) | [0.214, 0.510]<br>[0.216, 0.509] | **Chồng lấp hoàn toàn**<br>(Overlap: True) |
| **Tỉ lệ vết nứt mảnh** (`thin_crack_ratio`) | $\rho = -0.096$<br>($p = 0.5068$) | [-0.379, 0.204]<br>[-0.381, 0.204] | $\rho = -0.298$<br>($p = 2.32\times 10^{-4}$) | [-0.446, -0.134]<br>[-0.447, -0.133] | **Chồng lấp**<br>(Overlap: True) |
| **Số thành phần** (`n_components`) | $\rho = -0.260$<br>($p = 0.0687$) | [-0.514, 0.037]<br>[-0.526, 0.041] | $\rho = -0.338$<br>($p = 2.62\times 10^{-5}$) | [-0.481, -0.178]<br>[-0.477, -0.185] | **Chồng lấp hoàn toàn**<br>(Overlap: True) |

---

## 2. Phân Tích Thống Kê & Bác Bỏ Lập Luận Cũ

### A. Về tính chất chồng lấp khoảng tin cậy của `thin_crack_ratio`
- Trên tập Val ($n=50$), $\rho = -0.096$ có $p = 0.5068$ (không đạt ý nghĩa thống kê ở ngưỡng $\alpha=0.05$). Tuy nhiên, khoảng tin cậy 95% của Val là $[-0.379, 0.204]$.
- Trên tập Test ($n=148$), $\rho = -0.298$ có $p = 2.32\times 10^{-4}$ (có ý nghĩa thống kê), CI 95% là $[-0.446, -0.134]$.
- **Quan sát then chốt:** Đoạn $[-0.379, -0.134]$ là vùng giao nhau giữa CI của Val và CI của Test. Vì cỡ mẫu tập Val rất nhỏ ($n=50$ ảnh gốc), ước lượng $\rho$ trên Val có phương sai lấy mẫu lớn dẫn đến khoảng tin cậy rất rộng bao trùm cả 0 lẫn phần lớn khoảng tin cậy của Test.
- Do đó, việc Val có $p = 0.5068$ là hệ quả của **cỡ mẫu nhỏ dẫn đến công suất thống kê yếu (low statistical power)**, không thể dùng làm bằng chứng để tuyên bố hai tập có tương quan khác nhau.

### B. Bỏ lập luận sai: "Val không có tương quan nên test cũng không"
- Báo cáo Revision 3 và 4 trước đây từng lập luận rằng vì tương quan trên Val không có ý nghĩa thống kê ($p = 0.51$) nên vết nứt mảnh không ảnh hưởng tới kết quả kiểm định. Đây là lập luận sai lầm về mặt logic thống kê (*absence of evidence is not evidence of absence*).
- **Lập luận này bị loại bỏ hoàn toàn khỏi Revision 5.**

### C. Căn cứ khoa học thực tế để bác bỏ "Khoảng cách Val→Test là do vết nứt mảnh"
Căn cứ duy nhất đúng và có cơ sở dữ liệu thực nghiệm để bác bỏ giả thuyết cho rằng sự suy giảm Dice từ Val sang Test là do phân bố vết nứt mảnh gây ra gồm hai bằng chứng:
1. **Tỉ lệ vết nứt mảnh giữa hai tập không có sự khác biệt có ý nghĩa thống kê:**
   - Ở mức crop: Val = $0.1470$, Test = $0.1455$ ($p = 0.7830$ theo kiểm định Mann-Whitney U cụm điều chỉnh lặp mẫu).
   - Ở mức ảnh gốc độc lập: Val = $0.1719$, Test = $0.2063$ ($p = 0.4139$ theo Mann-Whitney U hai phía).
   - Sự phân bố của vết nứt mảnh giữa tập Val và Test là hoàn toàn tương đương.
2. **Kết quả phân rã Oaxaca-Blinder:**
   - Khoảng cách Val $\to$ Test là 6.35 điểm (hoặc 6.96 điểm ở LOCKED).
   - **Mix effect** (phần chênh lệch do sai khác tỉ lệ phân bố giữa các nhóm vết nứt) chỉ chiếm **0.23 – 0.24 điểm (3.4% – 3.7%)**.
   - **Within effect** (phần suy giảm do mô hình dự đoán kém hơn ngay bên trong từng nhóm vết nứt cùng độ rộng) chiếm tới **6.12 – 6.72 điểm (96.3% – 96.6%)**.
3. **Kết luận:** Mô hình bị tụt điểm đồng đều trên mọi nhóm độ rộng vết nứt (cả vết vừa 3–10px lẫn vết rộng >10px), chứ không phải do tập Test tập trung nhiều vết mảnh hơn tập Val.
