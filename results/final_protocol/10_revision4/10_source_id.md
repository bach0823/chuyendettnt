# 10_source_id.md — Khảo Sát và Xác Định Mã Ảnh Gốc Crack500 (Revision 4)

**Ngày:** 2026-11-17  
**Nguồn dữ liệu:** Cấu trúc tệp thực tế trong `datasets/Crack500_ready/{train,val,test}/images/` (1,896 train, 348 val, 1,124 test).  
**Script thực nghiệm:** `results/final_protocol/10_revision4/scratch/test_source_id_definitions.py`.

---

## 1. Khảo Sát Các Định Nghĩa Mã Ảnh Gốc

Tên file crop trong tập dữ liệu có dạng:
`YYYYMMDD_HHMMSS_X_Y.jpg` hoặc `IMG_XXXX_X_Y.jpg` (trong đó X, Y là tọa độ cắt crop).

Thực nghiệm đã chạy thử nghiệm 5 phương thức phân tách mã ảnh gốc trên toàn bộ 3,368 tệp:

| Định nghĩa mã ảnh gốc | Phương pháp trích xuất | Train ($n=1,896$) | Val ($n=348$) | Test ($n=1,124$) | Khớp quy chuẩn bài báo gốc (250/50/200)? |
|---|---|:---:|:---:|:---:|:---:|
| **Định nghĩa 1** | Tách theo dấu `_`, lấy 2 phần đầu (`date_time` hoặc `IMG_XXXX`) | **250** | **50** | **148** | Khớp Train & Val; Test thiếu 52 ảnh |
| **Định nghĩa 2** | Tách theo dấu `_`, lấy 3 phần đầu (`date_time_X`) | 912 | 181 | 544 | Không khớp (bị chia nhỏ theo dải crop) |
| **Định nghĩa 3** | Chỉ lấy phần ngày chụp (`YYYYMMDD`) | 21 | 16 | 16 | Không khớp (gộp nhiều bức ảnh cùng ngày) |
| **Định nghĩa 4** | Biểu thức chính quy Regex `\d{8}_\d{6}` | 296 | 64 | 148 | Bị sót các ảnh có tiền tố `IMG_` |
| **Định nghĩa 5** | Bỏ 2 tọa độ crop cuối (X, Y) | **250** | **50** | **148** | Khớp Train & Val; Test thiếu 52 ảnh |

---

## 2. Kết Luận về Quy Chuẩn Mã Ảnh Gốc

- **Định nghĩa 1 và 5 là chuẩn xác nhất:**  
  Bằng cách tách lấy phần tiền tố trước tọa độ cắt crop (`YYYYMMDD_HHMMSS` hoặc `IMG_XXXX`), số lượng ảnh gốc độc lập thu được là:
  - **Tập Train:** Đúng chính xác **250 ảnh gốc** (khớp 100% với con số 250 trong bài báo Crack500 gốc).
  - **Tập Val:** Đúng chính xác **50 ảnh gốc** (khớp 100% với con số 50 trong bài báo Crack500 gốc).
  - **Tập Test:** Thu được **148 ảnh gốc** (không có định nghĩa nào cho ra 200 ảnh).

---

## 3. Vấn đề 52 Ảnh Gốc Thiếu ở Tập Test

- **Hiện trạng:** Bài báo Crack500 công bố tập test có 200 ảnh gốc. Tuy nhiên, thư mục test local chỉ chứa các crop xuất phát từ đúng **148 ảnh gốc độc lập**.
- **Nguyên nhân:** **Chưa giải thích được** từ các tệp hiện có trong kho lưu trữ (không có nhật ký tải về, không có tệp manifest ghi lại danh sách phân chia split ban đầu). Loại bỏ giả định không có căn cứ ở phiên bản trước cho rằng "do cách thức đóng gói local".
- **Hệ quả thống kê:**  
  Các phép kiểm định cụm (Cluster Bootstrap và Mann-Whitney U theo ảnh gốc) được thực hiện trên 148 cụm test thực tế là **ước lượng khách quan và đúng đắn nhất đối với dữ liệu local hiện tại**. Khoảng tin cậy cluster của Test Dice là **[0.7005, 0.7307]**. Cần ghi nhận độ bao phủ test đạt 74% (148/200) làm một hạn chế dữ liệu trong báo cáo.
