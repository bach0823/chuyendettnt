# 09_source_id.md — Mã Ảnh Gốc Crack500 (Revision 3)

**Ngày:** 2026-11-17  
**Nguồn:** Phân tích tên file từ dataset local; `07a_cluster_bootstrap_results.json`

---

## 1. Định nghĩa mã ảnh gốc

Tên file crop: `YYYYMMDD_HHMMSS_X_Y.jpg`  
Mã ảnh gốc (source ID) = **2 phần đầu** phân tách bởi `_`: `YYYYMMDD_HHMMSS`

**Ví dụ 10 tên file và mã tách ra (từ dataset val):**

| Tên file | Source ID |
|---|---|
| `20160222_080850_1281_361.jpg` | `20160222_080850` |
| `20160222_080933_361_1.jpg` | `20160222_080933` |
| `20160222_081005_721_361.jpg` | `20160222_081005` |
| `20160222_081035_1_1.jpg` | `20160222_081035` |
| `20160222_081118_1_1.jpg` | `20160222_081118` |
| `20160222_081149_1_1.jpg` | `20160222_081149` |
| `20160222_081222_1_1.jpg` | `20160222_081222` |
| `20160222_081253_1_1.jpg` | `20160222_081253` |
| `20160222_081326_1_1.jpg` | `20160222_081326` |
| `20160222_081400_1_1.jpg` | `20160222_081400` |

*(Nguồn: `08_revision2/08_cluster_recheck.md` §3)*

---

## 2. Số ảnh gốc duy nhất theo split

Phân tích tên file với các định nghĩa khác nhau cho số phần phân tách:

| Số phần (sep=`_`) | Train | Val | Test |
|:---:|:---:|:---:|:---:|
| 2 phần (`YYYYMMDD_HHMMSS`) | 250 | **50** | **148** |
| 3 phần | 912 | 181 | 544 |
| 4 phần (= số crops) | 1896 | 348 | 1124 |

**Nguồn xác nhận:** `07a_cluster_bootstrap_results.json`:
- `val_cluster_stats.n_original = 50`
- `test_cluster_stats.n_original = 148`

---

## 3. Giải thích test < 200

Paper Crack500 (Zhang et al.) nêu tập test có 200 ảnh gốc. Dataset local chỉ có **148 ảnh gốc test** (1,124 crops).

Chênh lệch: 200 − 148 = **52 ảnh gốc thiếu**.

Lý do không xác định được từ file dataset local:
- Không có file manifest hay log tải về lưu trữ.
- Không có ảnh nào trong tập test thiếu hoàn toàn crop (không có ảnh gốc nào có 0 crop trong 1,124).
- Có thể dataset được tải về một subset hoặc split khác với paper gốc.

**Hệ quả:** Cluster bootstrap hiện tại dùng **148 ảnh gốc test** là đúng với dataset local. CI cluster `[0.7005, 0.7307]` phản ánh 148 nguồn ảnh. Không thể suy rộng sang 200 ảnh gốc theo paper.

---

## 4. Hạn chế cần nêu trong báo cáo

- Coverage test: 148/200 = 74% ảnh gốc theo paper. 52 ảnh gốc thiếu (và số crop tương ứng) chưa được đánh giá.
- CI cluster bootstrap được tính trên 148 cụm, không phải 200. Với n nhỏ hơn, CI có thể rộng hơn hoặc hẹp hơn tùy phân bố.
- Không ảnh hưởng đến điểm Dice 0.7158 (tính trên 1,124 crops của dataset local).
