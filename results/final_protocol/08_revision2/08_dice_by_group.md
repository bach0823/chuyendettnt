# 08_dice_by_group.md — Phân tích Dice theo Nhóm Đặc tính Nhãn (Revision 2)

**Nguồn:** `08_per_image_dice_test.csv`, `08_dice_by_group.json`, `05_test_final_report.json`  
**Giao thức đánh giá:** Phase 5 LOCKED ensemble (seed 42, 43, 44, τ=0.5, TTA flip_hv, direct 1-pass)  
**Ngày thực hiện:** 2026-11-16

---

## 1. Phương pháp ghép nối

- Dữ liệu `per_image_dice` (1,124 phần tử) trong `05_test_final_report.json` được sinh bởi vòng lặp tuần tự qua `test_pairs` trong `05_run_test_official.py`.
- `test_pairs` được xây dựng bằng `sorted(glob.glob(...))` trong `evaluate_crack_official.py` (dòng 34), đảm bảo thứ tự tất định tuyệt đối theo bảng chữ cái tên file.
- Thứ tự 1,124 phần tử khớp 1-đối-1 với danh sách tên file trong `datasets/Crack500_ready/test/images/`.
- File kết quả ánh xạ chi tiết được lưu tại: [`08_per_image_dice_test.csv`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/08_revision2/08_per_image_dice_test.csv).

---

## 2. Tương quan giữa Test Dice và Đặc tính Nhãn Ground Truth

Tính trên toàn bộ 1,124 crops của tập test:

| Đặc tính nhãn | Spearman $\rho$ | $p$-value Spearman | Pearson $r$ | $p$-value Pearson | Diễn giải xu hướng |
|---|:---:|:---:|:---:|:---:|---|
| **Tỉ lệ pixel mảnh ($\le 3$ px)** | **$-0.4793$** | $1.24 \times 10^{-65}$ | **$-0.4121$** | $2.66 \times 10^{-47}$ | **Tương quan âm mạnh**: Càng nhiều vết mảnh, Dice càng suy giảm mạnh. |
| **Độ rộng crack trung bình (px)** | **$+0.4455$** | $6.63 \times 10^{-56}$ | **$+0.3889$** | $6.77 \times 10^{-42}$ | **Tương quan dương mạnh**: Vết càng rộng, mô hình nhận diện càng chuẩn xác. |
| **Tỉ lệ pixel nứt (crack ratio)** | **$+0.4209$** | $1.75 \times 10^{-49}$ | **$+0.3927$** | $9.21 \times 10^{-43}$ | **Tương quan dương**: Vết nứt càng chiếm diện tích lớn, Dice càng cao. |
| **Số thành phần liên thông** | **$-0.2992$** | $1.13 \times 10^{-24}$ | **$-0.1800$** | $1.21 \times 10^{-9}$ | **Tương quan âm**: Càng nhiều mảnh nứt rời rạc (cấu trúc phức tạp), Dice càng thấp. |

---

## 3. Phân nhóm Test Dice

### 3.1 Theo Độ rộng vết nứt trung bình (Crack Width)

| Nhóm độ rộng | Tiêu chí | Số crop ($n$) | Mean Dice | Median Dice | Std Dice | Khoảng tứ phân vị [Q25, Q75] |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Mảnh** | $\le 3$ px | 0* | — | — | — | — |
| **Vừa** | $3\text{ px} < \text{width} \le 10\text{ px}$ | 394 | **0.6559** (65.59 điểm) | 0.6964 | 0.1601 | [0.5725, 0.7731] |
| **Rộng** | $> 10$ px | 730 | **0.7481** (74.81 điểm) | 0.7787 | 0.1274 | [0.6935, 0.8356] |

*\*Ghi chú:* Không có crop nào trong tập test có độ rộng trung bình toàn crop $\le 3$ px vì các crop đều chứa ít nhất một phần thân vết nứt dày hơn 3 px.

> **Chênh lệch giữa vết vừa và vết rộng:** $0.7481 - 0.6559 = \mathbf{+0.0922}$ (**9.22 điểm phần trăm Dice**). Mô hình hoạt động vượt trội trên các vết nứt bề rộng lớn.

---

### 3.2 Theo Độ phức tạp hình thái (Số thành phần liên thông - Connected Components)

| Nhóm phức tạp | Tiêu chí ($n_{\text{comp}}$) | Số crop ($n$) | Mean Dice | Median Dice | Std Dice | Khoảng tứ phân vị [Q25, Q75] |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Đơn giản** | $\le 2$ | 711 | **0.7380** (73.80 điểm) | 0.7772 | 0.1448 | [0.6838, 0.8366] |
| **Trung bình** | $3 \le n_{\text{comp}} \le 5$ | 270 | **0.6915** (69.15 điểm) | 0.7287 | 0.1438 | [0.6168, 0.7951] |
| **Phức tạp** | $> 5$ | 143 | **0.6511** (65.11 điểm) | 0.6799 | 0.1329 | [0.5927, 0.7398] |

> **Chênh lệch giữa cấu trúc đơn giản và phức tạp:** $0.7380 - 0.6511 = \mathbf{+0.0869}$ (**8.69 điểm phần trăm Dice**). Càng phân mảnh thành nhiều nhánh nhỏ rời rạc, mô hình càng khó phát hiện đầy đủ.

---

## 4. Kết luận khoa học

1. **Vết mảnh và phân mảnh là hai rào cản suy giảm hiệu năng chính:** 
   - Hệ số tương quan Spearman của tỉ lệ pixel mảnh đạt $\rho = -0.4793$ ($p < 10^{-64}$).
   - Crops có vết bề rộng $\le 10$ px bị tụt $9.22$ điểm Dice so với vết $> 10$ px.
   - Crops có cấu trúc phân mảnh $> 5$ thành phần bị tụt $8.69$ điểm Dice so với vết đơn giản $\le 2$ thành phần.

2. **Liên hệ với khoảng cách Val $\to$ Test:**
   - Dù ở mức ảnh gốc (50 val vs 148 test), các đặc tính trung bình không khác biệt có ý nghĩa thống kê ($p > 0.10$), các crops rơi vào nhóm bất lợi (vết mảnh, phân mảnh cao) trong tập test chịu tổn thất Dice nặng nề hơn rất nhiều.
