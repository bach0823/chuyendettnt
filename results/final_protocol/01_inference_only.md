# Phase 1 — Khai thác chỉ ở suy luận

**Thực hiện:** 2026-11-14 | **Script:** `results/final_protocol/01_inference_only/`  
**Checkpoint:** `phase7_s2g_new_pipeline_run2/best_model_b2_stage2.pth` (tau=0.5 baseline Val Dice = **0.779294**)

---

## 1. Tau Sweep (0.30 → 0.70, bước 0.025)

| tau | Dice | Precision | Recall | Thin Dice | BIoU | HD95 (px) |
|-----|------|-----------|--------|-----------|------|-----------|
| 0.300 | 0.7718 | 0.7170 | 0.8778 | 0.3386 | 0.1140 | 40.28 |
| 0.400 | 0.7766 | 0.7360 | 0.8610 | 0.3472 | 0.1241 | 39.38 |
| 0.450 | 0.7781 | 0.7445 | 0.8527 | 0.3504 | 0.1281 | 39.41 |
| **0.500** | **0.7793** | **0.7525** | **0.8446** | **0.3532** | **0.1318** | **38.69** |
| 0.525 | 0.7797 | 0.7564 | 0.8405 | 0.3543 | 0.1337 | 38.71 |
| 0.550 | 0.7799 | 0.7601 | 0.8363 | 0.3552 | 0.1351 | 38.89 |
| 0.575 | 0.7801 | 0.7638 | 0.8320 | 0.3559 | 0.1366 | 38.89 |
| **0.600** | **0.7801** | **0.7675** | **0.8276** | **0.3567** | **0.1380** | **38.90** |
| 0.625 | 0.7801 | 0.7713 | 0.8230 | 0.3573 | 0.1392 | 38.77 |
| 0.650 | 0.7799 | 0.7751 | 0.8182 | 0.3580 | 0.1408 | 38.54 |
| 0.700 | 0.7792 | 0.7830 | 0.8079 | 0.3586 | 0.1431 | 38.61 |

**Nhận xét:**
- Dice plateau rộng từ tau=0.575 đến 0.650 (biên độ < 0.0002).
- Tau cao hơn → Precision tăng, Recall giảm, Thin Dice tăng nhẹ, Dice không đổi đáng kể.
- Peak tại **tau=0.600** (Dice=0.780141, Δ=+0.000847 so với tau=0.5).

### Bootstrap CI — tau=0.6 vs tau=0.5 (10,000 lần, ghép cặp theo ảnh)

| Δ trung bình | 95% CI | Chấp nhận? |
|---|---|---|
| +0.000847 | [−0.000234, +0.001985] | ❌ **KHÔNG** (CI chứa 0) |

**Quyết định tau:** Giữ **tau=0.5**. Thay đổi sang tau=0.6 không có bằng chứng thống kê.

---

## 2. TTA Flip

| TTA | Dice @ tau=0.5 | Dice @ tau=0.6 | Thin Dice @ tau=0.6 |
|-----|----------------|----------------|----------------------|
| none | 0.779294 | 0.780141 | 0.3567 |
| flip_h | 0.781055 | 0.781944 | 0.3590 |
| flip_v | 0.781611 | 0.782318 | 0.3596 |
| **flip_hv** | **0.782452** | **0.783404** | **0.3607** |

### Bootstrap CI — flip_hv vs none (10,000 lần, ghép cặp theo ảnh)

| tau | Δ trung bình | 95% CI | Chấp nhận? |
|-----|---|---|---|
| 0.5 | +0.003158 | [+0.001021, +0.005301] | ✅ **CÓ** |
| 0.6 | +0.003263 | [+0.001073, +0.005360] | ✅ **CÓ** |

> [!NOTE]
> Kiểm tra ổn định TTA qua 2 nửa val chưa chạy (stability script chỉ bao gồm tau selection). Tuy nhiên, lợi ích flip_hv có CI dương rõ (+0.001 ÷ +0.005) và nhất quán ở cả hai tau.

**Quyết định TTA:** Chấp nhận **flip_hv** tại tau=0.5.

---

## 3. Kiểm tra ổn định tau (20 lần lặp, 2 nửa val)

| | Giá trị |
|---|---|
| Mean cross-eval Dice | 0.779867 |
| Std cross-eval Dice | 0.006081 |
| Tau được chọn trên nửa A | thường rơi vào 0.550–0.650 |
| Tau được chọn trên nửa B | thường rơi vào 0.575–0.650 |

Std = 0.0061 trên nửa tập 174 ảnh. Cho thấy tau chọn trên val đầy đủ (348 ảnh) ổn định hơn — plateau 0.575–0.650 nhất quán qua các lần split.

---

## 4. Thin Crack Dice

| Cấu hình | Thin Dice |
|---|---|
| tau=0.5, no TTA | 0.3532 |
| tau=0.6, no TTA | 0.3567 |
| tau=0.5, flip_hv | — (xem mục 2) |
| tau=0.6, flip_hv | 0.3607 |

Thin Crack Dice vẫn rất thấp (~0.35) so với Dice tổng (~0.78). Tăng tau và TTA chỉ cải thiện nhẹ. Đây là giới hạn của kiến trúc/training, không phải ngưỡng quyết định.

---

## 5. Quyết định Phase 1

| Thay đổi | Δ Val Dice | CI dưới | Ổn định | **Quyết định** |
|---|---|---|---|---|
| tau: 0.5 → 0.6 | +0.000847 | −0.000234 (< 0) | — | ❌ Không nhận |
| TTA: flip_hv @ tau=0.5 | +0.003158 | +0.001021 (> 0) | Chưa xác minh nửa-nửa | ✅ Nhận (CI dương rõ) |
| TTA: flip_hv @ tau=0.6 | +0.003263 | +0.001073 (> 0) | Chưa xác minh nửa-nửa | ✅ Nhận nếu kết hợp |

**Cấu hình tốt nhất từ Phase 1 (chỉ dựa val):**
- Checkpoint: `best_model_b2_stage2.pth`
- tau = **0.5** (tau=0.6 không vượt CI)
- TTA: **flip_hv**
- Val Dice dự kiến: **~0.7825** (+0.003 so với baseline)

> [!IMPORTANT]
> Cấu hình này CHƯA được khóa. Phải chờ Phase 2 xác lập MDE trước khi kết luận flip_hv thật sự vượt ngưỡng có nghĩa.

---

## 6. File đầu ra

| File | Nội dung |
|---|---|
| [`tau_sweep.csv`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/01_inference_only/tau_sweep.csv) | 17 giá trị tau × 9 chỉ số |
| [`tta_sweep.csv`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/01_inference_only/tta_sweep.csv) | 4 TTA × 2 tau |
| [`tta_bootstrap.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/01_inference_only/tta_bootstrap.json) | Bootstrap CI flip_hv |
| [`stability.csv`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/01_inference_only/stability.csv) | 20 × 2 half-split, tau stability |
| [`phase1_summary.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/01_inference_only/phase1_summary.json) | Tổng hợp |
