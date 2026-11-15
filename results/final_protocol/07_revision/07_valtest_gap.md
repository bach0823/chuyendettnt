# Việc 2 — Phân tích Khoảng cách Val→Test qua Nhãn (Masks)

**Ngày thực hiện:** 2026-11-16  
**Script:** `results/final_protocol/07_revision/run_work2_valtest_gap.py`  
**Dữ liệu đầu vào:** Nhãn ground-truth (masks) từ `datasets/Crack500_ready/{val,test}/masks/`.  
Không dùng bất kỳ dự đoán của mô hình. Quy tắc cứng 1 được tuân thủ.

---

## 1. Phương pháp Đo độ rộng Vết Nứt

- **Distance transform (DT):** Dùng `cv2.distanceTransform(binary_mask, DIST_L2, 5)` trên mặt nạ nhị phân.  
  Với mỗi pixel crack, giá trị DT ≈ khoảng cách Euclid đến biên nứt gần nhất (tương đương nửa chiều rộng cục bộ).  
  Độ rộng ước tính: `width ≈ 2 × DT`.
- **Vết mảnh:** Pixel có `DT ≤ 1.5` → chiều rộng ≤ 3 px.
- **Thành phần liên thông:** `cv2.connectedComponents()` trên mask nhị phân, trừ nền.

---

## 2. Thống kê Mô tả theo Split

| Chỉ số | Val (348 crops) | Test (1124 crops) |
|---|:---:|:---:|
| **Tỉ lệ pixel nứt** (mean ± std) | 0.0570 ± 0.0480 | 0.0559 ± 0.0410 |
| **Tỉ lệ pixel nứt** (median / P25–P75) | 0.0431 / 0.022–0.078 | 0.0460 / 0.026–0.079 |
| **Độ rộng crack trung bình (px)** (mean ± std) | 14.40 ± 7.38 | 13.94 ± 7.06 |
| **Độ rộng crack** (median / P25–P75) | 12.83 / 8.99–18.13 | 12.66 / 8.70–17.49 |
| **Tỉ lệ crack pixel mảnh ≤ 3 px** (mean) | 0.1437 | 0.1499 |
| **Số thành phần liên thông** (mean ± std) | 2.24 ± 1.91 | **3.32 ± 4.87** |
| **Số thành phần liên thông** (median / max) | 2.0 / 14 | 2.0 / **64** |
| **Crops với < 1000 crack px** | 0/348 (0%) | 0/1124 (0%) |

---

## 3. Kiểm định Thống kê So sánh Val vs Test (Mann-Whitney U, two-sided)

| Chỉ số | Δ median (test − val) | p-value | Kết luận |
|---|:---:|:---:|:---|
| Tỉ lệ pixel nứt | +0.0029 | 0.3357 | Không khác biệt có ý nghĩa |
| Độ rộng crack trung bình (px) | −0.17 | 0.4143 | Không khác biệt có ý nghĩa |
| Tỉ lệ crack mảnh ≤ 3 px | −0.0005 | 0.2234 | Không khác biệt có ý nghĩa |
| **Số thành phần liên thông** | **0.0 (median)** | **0.0002** | **Khác biệt có ý nghĩa** |

> [!NOTE]
> **Chú ý về chỉ số thành phần liên thông:** Trung vị bằng nhau (cả hai = 2.0), nhưng phân phối đuôi phải của test rộng hơn rõ rệt (std = 4.87 vs 1.91, max = 64 vs 14). Kiểm định MWU phát hiện sự khác biệt này (p = 0.0002) ở mức phân phối toàn bộ, không phải ở trung vị.

---

## 4. Khoảng cách Val→Test theo Dice

| So sánh | Val Dice | Test Dice | Khoảng cách (điểm) |
|---|:---:|:---:|:---:|
| Canonical đơn vs Locked Test | 0.7786 (mean 3 seed) | 0.7158 | **6.3 điểm** |
| Ensemble 3-seed val vs Locked Test | 0.7850 | 0.7158 | **6.9 điểm** |
| B0 baseline (tham khảo) | 0.7318 | 0.6771 | **5.5 điểm** |

B2 val→test gap (6.3 điểm) lớn hơn B0 (5.5 điểm) nhưng cùng bậc độ lớn.

---

## 5. Hạn chế Phân tích và Những gì Không làm Được

**CHƯA LÀM ĐƯỢC:**
- Phân tích Dice theo nhóm (crack width, crack ratio) cho test: `05_test_final_report.json` lưu `per_image_dice` như một mảng số (1124 phần tử) **không kèm filenames**. Không thể map chỉ số → tên file → đặc tính nhãn mà không suy ra thứ tự sắp xếp từ dataset loader (cần chạy lại loader, không có GPU).
- Phân tích Dice theo nhóm cho val: `val_result.json` chỉ lưu aggregate metrics, không có per-image dice.

---

## 6. Kết luận

**Số đo ủng hộ việc nói:**
- Test có độ phân tán số thành phần liên thông cao hơn val có ý nghĩa thống kê (p = 0.0002), với đuôi phải dài hơn nhiều (max = 64 vs 14).
- Ba chỉ số còn lại (tỉ lệ nứt, độ rộng vết, tỉ lệ mảnh) không khác biệt có ý nghĩa.

**Không được nói do chưa đo:**
- Không thể khẳng định val→test gap 6.3–6.9 điểm đến từ độ phức tạp topo (nhiều thành phần hơn) hay nguồn gốc khác, vì thiếu phân tích Dice theo nhóm.
- Khoảng cách B2 (6.3 điểm) không khác xa khoảng cách B0 (5.5 điểm) → val→test gap có thể là đặc tính chung của dataset, không riêng B2.
