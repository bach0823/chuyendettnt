# 12_thin_crack.md — Phân Tích Chi Tiết Tỉ Lệ Vết Nứt Mảnh Theo Hai Định Nghĩa Ở Mức Ảnh Gốc (Revision 6)

**Ngày lập:** 2026-11-17  
**Nguồn dữ liệu:**
- Mặt nạ nhãn ground truth: `datasets/Crack500_ready/val/masks/` (348 crops, 50 ảnh gốc) và `datasets/Crack500_ready/test/masks/` (1,124 crops, 148 ảnh gốc).
- Tệp tính toán chi tiết: `results/final_protocol/12_revision6/thin_crack_analysis.json`.

---

## 1. Hai Định Nghĩa Đo Lường Tỉ Lệ Vết Nứt Mảnh Ở Mức Ảnh Gốc

Định nghĩa thành phần vết nứt mảnh: Một thành phần liên thông (connected component) được phân loại là **vết nứt mảnh** nếu đường kính trung bình $\bar{d} \le 3.0 \text{ pixels}$ (tính thông qua Euclidean Distance Transform: $\bar{d} = 2 \times \text{mean}(EDT)$).

Ở mức ảnh gốc (loại bỏ lặp mẫu do chia crop), tỉ lệ vết nứt mảnh được tính toán theo hai phương pháp thống kê chuẩn:

1. **Cách 1: Macro-average (Trung bình của các tỉ lệ theo từng ảnh gốc độc lập):**
   - Với mỗi ảnh gốc $i$, tính tỉ lệ vết mảnh riêng biệt:
     $$r_i = \frac{\text{Số lượng thành phần vết mảnh trong ảnh } i}{\text{Tổng số thành phần trong ảnh } i}$$
   - Tỉ lệ của toàn bộ tập dữ liệu là trung bình số học: $\bar{r} = \frac{1}{N} \sum_{i=1}^N r_i$.
   - Phương pháp này gán trọng số đồng đều cho từng ảnh gốc, phản ánh kỳ vọng phân bố trên một ảnh chụp độc lập ngẫu nhiên.
2. **Cách 2: Micro-average / Pooled totals (Gộp toàn bộ pixel và thành phần trên toàn tập):**
   - Gộp tổng số lượng trên toàn bộ tập dữ liệu rồi mới chia tỉ lệ:
     $$R_{\text{pooled}} = \frac{\sum_{i=1}^N \text{Số thành phần mảnh}_i}{\sum_{i=1}^N \text{Tổng số thành phần}_i} \quad \text{hoặc} \quad R_{\text{pixel}} = \frac{\sum_{i=1}^N \text{Số pixel nứt mảnh}_i}{\sum_{i=1}^N \text{Tổng pixel nứt}_i}$$
   - Phương pháp này nhạy cảm hơn với các ảnh lớn chứa nhiều vết nứt phức tạp.

---

## 2. Bảng Kết Quả Định Lượng Cho Cả Hai Cách Tính

Toàn bộ khoảng tin cậy 95% được ước lượng bằng phương pháp Bootstrap phi tham số (10,000 lần resample ở cấp ảnh gốc độc lập):

| Phương pháp đo lường | Tập Val ($n=50$ ảnh gốc)<br>[Giá trị & 95% CI] | Tập Test ($n=148$ ảnh gốc)<br>[Giá trị & 95% CI] | Hiệu số ($\text{Test} - \text{Val}$) | Kiểm định thống kê ($p$-value) | Độ chênh nhỏ nhất phát hiện được (MDD, power 80%) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Cách 1: Macro-average (Theo thành phần liên thông)** | **27.69%**<br>[22.78%, 32.76%] | **33.88%**<br>[30.18%, 37.61%] | **+6.19%** | Mann-Whitney U: $p = 0.1408$<br>Welch's t-test: $p = 0.0557$ | **8.96 điểm %** |
| **Cách 1: Macro-average (Theo pixel nứt mảnh)** | **0.136%**<br>[0.049%, 0.256%] | **0.205%**<br>[0.144%, 0.277%] | **+0.069%** | Mann-Whitney U: $p = 0.0519$<br>Welch's t-test: $p = 0.2824$ | **0.179 điểm %** |
| **Cách 2: Micro-average (Gộp thành phần liên thông)** | **32.89%**<br>[27.07%, 38.44%] | **51.67%**<br>[45.04%, 56.63%] | **+18.79%** | Pooled two-proportion $z$-test:<br>$p < 0.0001$ | — |
| **Cách 2: Micro-average (Gộp pixel nứt mảnh)** | **0.068%**<br>[0.036%, 0.113%] | **0.205%**<br>[0.131%, 0.298%] | **+0.138%** | Gộp trên toàn bộ pixel | — |

---

## 3. Hiệu Chỉnh Kết Luận Thống Kê & Giới Hạn Công Suất

1. **Thay thế cách diễn đạt cũ:**
   - Bãi bỏ hoàn toàn các câu khẳng định chủ quan như *"bác bỏ hoàn toàn giả thuyết vết nứt mảnh"*.
   - **Cách diễn đạt chuẩn xác mới:**
     *"Ở mức độ ảnh gốc độc lập (Macro-average), kiểm định phi tham số **không phát hiện sự khác biệt có ý nghĩa thống kê** về tỉ lệ vết nứt mảnh giữa tập Val và Test ($p = 0.1408$ đối với tỉ lệ thành phần và $p = 0.0519$ đối với tỉ lệ pixel). Tuy nhiên, cần lưu ý rằng phép thử này có **công suất thống kê thấp** do kích thước tập dữ liệu hạn chế ($n_{\text{val}}=50$ và $n_{\text{test}}=148$ ảnh gốc). Độ chênh lệch nhỏ nhất có thể phát hiện được với độ tin cậy 80% (MDD) lên tới **8.96 điểm %** đối với thành phần và **0.18 điểm %** đối với pixel — lớn hơn độ chênh lệch quan sát được giữa hai tập."*
2. **Quan sát khi gộp mẫu (Micro-average):**
   - Khi tính theo phương pháp gộp toàn bộ thành phần liên thông trên toàn tập (Micro-average), tập Test có tỉ lệ thành phần mảnh cao hơn tập Val ($51.67\%$ vs $32.89\%$).
   - Tuy nhiên, trong phân rã Oaxaca-Blinder đã được kiểm chứng, hiệu ứng dịch chuyển phân bố nhóm (**Mix effect**) chỉ giải thích **0.23 – 0.24 điểm (3.4% – 3.7%)** trong tổng khoảng cách Val $\to$ Test (6.35 – 6.96 điểm), trong khi **Within effect giải thích tới 96.3% – 96.6%** khoảng cách. Do đó, dù có sự gia tăng vết mảnh ở cấp độ gộp, đây vẫn không phải là nhân tố chủ đạo tạo nên khoảng cách suy giảm hiệu năng giữa Val và Test.
