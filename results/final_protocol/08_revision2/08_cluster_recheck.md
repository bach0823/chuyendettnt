# 08_cluster_recheck.md — Kiểm tra lại Cluster Bootstrap (Revision 2)

**Nguồn:** `results/final_protocol/07_revision/07a_cluster_bootstrap_results.json`, dataset val/test  
**Ngày viết:** 2026-11-16

---

## 1. Mã ảnh gốc trong tên file

Tên file format: `YYYYMMDD_HHMMSS_X_Y.jpg`  
Mã ảnh gốc = `YYYYMMDD_HHMMSS` (8+6 ký tự đầu, tách bởi `_`).

**Ví dụ:**
- `20160222_080850_1281_361.jpg` → source `20160222_080850`
- `20160222_080933_361_1.jpg` → source `20160222_080933`

Tên file **có chứa** mã ảnh gốc → cluster bootstrap theo ảnh gốc khả thi.

---

## 2. Thống kê số ảnh gốc

Chạy từ `datasets/Crack500_ready/{val,test}/images/` (kết quả từ script Python):

| Tập | Số crops | Số ảnh gốc | Avg crops/ảnh | Min | Max | Median |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Val | 348 | **50** | 6.96 | 4 | 12 | 7.0 |
| Test | 1,124 | **148** | 7.59 | 1 | 15 | 7.0 |

**Nguồn xác nhận:** `07a_cluster_bootstrap_results.json` → `val_cluster_stats.n_original=50`, `test_cluster_stats.n_original=148`.

**Ghi chú test < 200:** Crack500 paper nêu 200 ảnh gốc test, nhưng dataset local chỉ có 148 ảnh gốc (tương ứng 1,124 crops). Chênh lệch 52 ảnh gốc không giải thích được từ file dataset hiện tại — có thể do một subset được dùng hoặc do cách split khác nhau. Hạn chế: không thể xác nhận coverage.

---

## 3. Cluster bootstrap Test Dice (đã có từ Việc 1 cũ)

Từ `07a_cluster_bootstrap_results.json` (`test_dice_cluster_bootstrap`):

| Phương pháp | CI 95% dưới | CI 95% trên | Độ rộng |
|---|:---:|:---:|:---:|
| Crop-level (cũ) | 0.7072 | 0.7243 | 0.0171 |
| **Cluster theo ảnh gốc** | **0.7005** | **0.7307** | **0.0302** |

CI cluster rộng hơn CI crop: **0.0302 / 0.0171 = 1.76×**.  
(Lưu ý: Revision 1 §4 ghi "rộng hơn 12×" — đây là sai; "12×" mô tả CI MDE, không phải CI cluster. Xem §4 bên dưới.)

---

## 4. Sửa cách diễn đạt "12×" trong Revision 1

**Revision 1 §4 viết:** "Khoảng này rộng hơn 12× so với ước lượng điểm."

Từ `07a_cluster_bootstrap_results.json` → `mde_analysis`:
- MDE nominal = 0.010804
- CI 95%: lo = 0.005625, hi = 0.067924

Tỉ lệ: 0.067924 / 0.005625 ≈ **12.1×** (cận trên / cận dưới).

**Cách diễn đạt đúng:** "Cận trên CI 95% của MDE gấp khoảng 12 lần cận dưới (hi/lo ≈ 12.1×), phản ánh độ bất định cao khi n=3 seed."  
Không phải "CI rộng hơn 12× so với ước lượng điểm" (ước lượng điểm = 0.0108, cận trên = 0.0679, tỉ lệ ≈ 6.3×, nhưng cách trình bày "cận trên / cận dưới = 12.1×" có nguồn rõ ràng hơn).

---

## 5. Val cluster bootstrap và TTA delta cluster bootstrap

**Không tính được.** Lý do từ `07a_cluster_bootstrap_results.json`:
- `val_cluster_bootstrap_possible: false` → "`val_result.json` chỉ lưu aggregate dice, không có per_image_dice. Cluster bootstrap cho val không khả thi mà không chạy lại suy luận."
- `tta_cluster_bootstrap_possible: false` → "phase1 TTA bootstrap dùng per_image_dice trong RAM, không lưu ra file. Không thể tái tính cluster bootstrap TTA mà không GPU."

**Hạn chế cần nêu trong báo cáo:** CI crop-level cho val Dice và TTA delta (đã báo cáo trong Revision 1) đều là **underestimate** vì bỏ qua tương quan trong ảnh gốc. Val có 50 ảnh gốc, 348 crops → trung bình 6.96 crops/ảnh. CI thực cho val và TTA delta có thể rộng hơn.

---

## 6. MWU test theo ảnh gốc (Cluster-level Mann-Whitney U Test)

**Phương pháp:** Gộp trung bình các chỉ số ground truth mask của từng ảnh gốc (50 ảnh gốc Val vs 148 ảnh gốc Test) và chạy kiểm định phi tham số Mann-Whitney U hai phía:

| Chỉ số hình thái | Val trung bình ảnh gốc ($n=50$) | Test trung bình ảnh gốc ($n=148$) | $U$-statistic | $p$-value MWU | Kết luận ($p < 0.05$) |
|---|:---:|:---:|:---:|:---:|---|
| **Tỉ lệ pixel nứt (crack ratio)** | $0.0557 \pm 0.0317$ | $0.0558 \pm 0.0254$ | 3,452.0 | **0.4799** | Không khác biệt |
| **Độ rộng trung bình vết (px)** | $14.12 \pm 6.25$ | $14.13 \pm 5.66$ | 3,620.0 | **0.8205** | Không khác biệt |
| **Tỉ lệ crack pixel mảnh $\le 3$ px** | $0.1470 \pm 0.0777$ | $0.1455 \pm 0.0676$ | 3,603.0 | **0.7830** | Không khác biệt |
| **Số thành phần liên thông** | $2.28 \pm 1.13$ | $2.97 \pm 2.63$ | 3,132.0 | **0.1052** | **Không khác biệt ở mức ảnh gốc** |

> [!IMPORTANT]
> **Phát hiện quan trọng:** Ở cấp độ crop (348 val vs 1,124 test), số thành phần liên thông từng cho $p = 0.0002$ do hiện tượng giả lặp mẫu (pseudoreplication) làm thổi phồng bậc tự do. Khi gộp đúng về **mức ảnh gốc độc lập (50 val vs 148 test)**, kiểm định Mann-Whitney U cho $p = 0.1052 > 0.05$.
> 
> **Kết luận:** **Không có bất kỳ đặc tính nhãn nào khác biệt có ý nghĩa thống kê giữa tập Val và tập Test ở mức ảnh gốc.** Khoảng cách hiệu năng 6.44 điểm Dice không bắt nguồn từ sự lệch phân phối hình thái vết nứt bề mặt.
