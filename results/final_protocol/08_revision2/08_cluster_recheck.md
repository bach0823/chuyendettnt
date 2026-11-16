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

## 6. MWU test theo ảnh gốc

CHƯA LÀM ĐƯỢC — lý do: các chỉ số đặc tính nhãn (crack_ratio, width, n_components...) trong `07b_valtest_gap_results.json` lưu ở mức crop, không có per-image aggregation theo ảnh gốc. Tái tính MWU theo ảnh gốc cần chạy lại script `run_work2_valtest_gap.py` với aggregation theo source ID — không cần GPU nhưng cần thời gian. Hạn chế này ghi vào báo cáo.
