# Việc 1 — Bootstrap theo Ảnh Gốc (Cluster Bootstrap)

**Ngày thực hiện:** 2026-11-16  
**Script:** `results/final_protocol/07_revision/run_work1_cluster_bootstrap.py`

---

## 1. Trích Mã Ảnh Gốc từ Tên File

Tên file có dạng `YYYYMMDD_HHMMSS_X_Y.jpg` trong đó `YYYYMMDD_HHMMSS` là timestamp của ảnh gốc, còn `X_Y` là offset crop. Mã gốc được trích bằng cách lấy 2 phần đầu (ngăn cách bởi `_`).

| Tập | Số crops | Số ảnh gốc | TB crop/ảnh gốc | Min/Median/Max |
|---|:---:|:---:|:---:|:---:|
| **Val** | 348 | **50** | 6.96 | 4 / 7 / 12 |
| **Test** | 1,124 | **148** | 7.59 | 1 / 7 / 15 |

> [!WARNING]
> Val chỉ có **50 ảnh gốc** (không phải 348 ảnh độc lập). Các crop trong cùng ảnh gốc có nhãn tương quan cao → bootstrap độc lập theo crop (đã dùng ở Phase 1, 2) **bị phồng CI** (underestimate variance).

---

## 2. CI Test Dice — Cluster Bootstrap vs Crop Bootstrap

Cluster bootstrap: resample 148 ảnh gốc (có thay thế) × 10,000 lần. Mỗi lần lấy tất cả crops của ảnh gốc được chọn.

| Phương pháp Bootstrap | CI 95% dưới | CI 95% trên | Độ rộng CI | So với Crop |
|---|:---:|:---:|:---:|:---:|
| **Crop-level (cũ)** | 0.7072 | 0.7243 | 0.0171 | 1.00× |
| **Cluster (ảnh gốc)** | **0.7005** | **0.7307** | **0.0302** | **1.76×** |

Test Dice mean = **0.7158**. CI cluster rộng hơn 1.76 lần CI crop.

---

## 3. CI Val Dice và MDE

Val `val_result.json` chỉ lưu aggregate metrics → **không thể thực hiện cluster bootstrap cho val** mà không chạy lại suy luận trên GPU.

**CI cho MDE (dựa trên 3 seed):**

Với $s_{\text{seed}} = 0.005402$ và $n = 3$, khoảng tin cậy 95% cho $s$ qua $\chi^2(df=2)$:

$$s_{\text{95\% CI}} = [0.00281, 0.03396]$$
$$\text{MDE} = 2s \implies \text{MDE}_{95\% \text{CI}} = [0.0056, 0.0679]$$

MDE danh nghĩa $= 0.0108$ chỉ là điểm ước lượng. Khoảng thực là từ **0.6 điểm đến 6.8 điểm Dice** ở mức tin cậy 95%.

---

## 4. Những gì KHÔNG làm được

**CHƯA LÀM ĐƯỢC:**
- **CI cluster cho Val Dice** (canonical, 3-seed mean, ensemble): thiếu per_image_dice cho val.
- **CI cluster cho TTA delta**: phase1 TTA bootstrap dùng per_image_dice trong RAM (không lưu ra file), tái tính cần GPU.
- **CI cluster cho Δ ensemble vs single-seed trên val**: cùng lý do trên.

---

## 5. Tóm tắt dữ liệu đầu ra

- [`07a_cluster_bootstrap_results.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/07_revision/07a_cluster_bootstrap_results.json)
- [`07a_val_cluster_summary.csv`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/07_revision/07a_val_cluster_summary.csv) — 50 ảnh gốc val
- [`07a_test_cluster_summary.csv`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/07_revision/07a_test_cluster_summary.csv) — 148 ảnh gốc test
