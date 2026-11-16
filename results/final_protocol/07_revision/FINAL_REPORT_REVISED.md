# BÁO CÁO TỔNG HỢP SỬA ĐỔI — SAGE-Lite trên Crack500

**Ngày:** 2026-11-16  
**Phiên bản:** Revision 1 (Phase 7)  
**Thay thế:** Báo cáo tổng hợp trước (trước khi có phân tích cluster bootstrap và val→test gap)  

---

## TÓM TẮT THAY ĐỔI SO VỚI BÁO CÁO CŨ

| Mục | Kết luận cũ | Kết luận mới | Bằng chứng |
|---|---|---|---|
| **Test không bị ô nhiễm** | "Test hoàn toàn held-out, không nhìn" | "τ, TTA, ensemble đóng băng trước Phase 5; test đã được xem ở 4 lần trước" | Lịch sử checkpoint mục §1 |
| **TTA/Ensemble hiệu quả ở test** | "TTA và ensemble cải thiện rõ rệt" | "Val: +0.0064 ensemble, +0.0032 TTA. Test: +0.0009 ensemble vs canonical đơn — nằm trong CI" | 05_test_final_report.json, 05_frozen_config.json |
| **MDE áp dụng mọi so sánh** | MDE áp dụng để so sánh TTA delta | "MDE đo nhiễu giữa các lần huấn luyện; so sánh ghép cặp TTA/τ dùng bootstrap ghép cặp, không dùng MDE" | Mục §3.3 |
| **Phase 3 "tương đương tuyệt đối"** | "Canonical, D, R tương đương tuyệt đối" | "Không phát hiện chênh lệch ≥ ~1 điểm Dice với 2 seed mỗi biến thể" | 03_capacity_control.md |
| **Val→test gap do vết mảnh** | "Test khó hơn vì nhiều vết mảnh hơn" | "Ba chỉ số crack (ratio, width, thin-crack fraction) không khác biệt có ý nghĩa (p > 0.22); chỉ n_components khác biệt (p=0.0002)" | 07_valtest_gap.md |
| **CI test Dice** | CI crop [0.7072, 0.7243] | **CI cluster [0.7005, 0.7307]** (rộng hơn 1.76×) | 07_cluster_bootstrap.md |
| **Vai trò nhánh expert** | Chưa thử nghiệm trực tiếp | "Không phát hiện đóng góp đo được với 2 seed (Δ = −0.000625, không vượt MDE)" — Variant N đã chạy 2026-11-16 | 07_variant_N.md |

**Câu trả lời thẳng cho 4 câu hỏi chính:**

| Câu hỏi | Câu trả lời |
|---|---|
| Router có đóng góp? | Không phát hiện đóng góp trong tầm MDE (Δ = −0.000138 điểm vs canonical; |Δ| ≪ MDE = 0.0108) |
| Nhánh expert có đóng góp? | **Không phát hiện** — Variant N (residual_scale=0) cho Δ = −0.000625 điểm vs canonical, không vượt MDE. "Không phát hiện" ≠ "không có đóng góp" |
| Ensemble+TTA có giúp ở test? | **Không xác nhận** — test +0.0009 điểm so với canonical đơn, nằm trong CI cluster |
| Khoảng cách val→test 6.3–6.9 điểm đến từ đâu? | Không rõ — chỉ n_components có ý nghĩa thống kê; không phân tích được Dice theo nhóm (thiếu filenames) |

**CHƯA LÀM/CHƯA XÁC MINH:**
- CI cluster cho val Dice và TTA delta (thiếu per_image_dice trong file)
- Phân tích Dice theo nhóm cho test (per_image_dice không kèm filenames)
- Phase 4 (domain adaptation) và Phase 6 (inference optimization) chưa thực hiện

---

## §1. TÍNH TRONG SÁNG CỦA DỮ LIỆU TEST

**τ, TTA và ensemble được đóng băng trước khi chạy test** (commit `05_frozen_config.json`). Kiến trúc, siêu tham số và pipeline được chọn dựa hoàn toàn trên val.

**Test đã được dùng để so sánh checkpoint ở các giai đoạn trước.** Các lần test đã xem (theo thứ tự):

| Lần | Checkpoint | Test Dice |
|---|---|---|
| 1 | B0 baseline | 0.6771 |
| 2 | B2 early checkpoint | 0.6801 |
| 3 | B2 intermediate | 0.6950 |
| 4 | B2 pre-Phase2 canonical | 0.7149 |
| **5 (LOCKED — Phase 5)** | **Ensemble 3-seed Phase 2** | **0.7158** |

> [!WARNING]
> Vì test đã được xem ở 4 lần trước khi Phase 5 LOCK, không thể khẳng định test hoàn toàn không bị ảnh hưởng bởi các quyết định chọn hướng nghiên cứu. Mức độ ảnh hưởng không đo được.

**Kết quả Phase 5 (LOCKED — không thay đổi):**
- Test Dice = **0.7158** (mean crop-level, 1,124 crops)
- Precision = 0.6724 | Recall = 0.8123 | IoU = 0.5755
- Thin-crack Dice = 0.2883 | HD95 = 50.93 px
- Checkpoint: ensemble 3 seed (42, 43, 44), τ = 0.5, TTA = flip_hv
- SHA-256: seed42 = `4B409B...`, seed43 = `7379C7...`, seed44 = `F5F7DC...`

---

## §2. KHOẢNG TIN CẬY TEST DICE — CLUSTER BOOTSTRAP

Dữ liệu: 1,124 crops từ **148 ảnh gốc**. Các crops của cùng ảnh gốc có nhãn tương quan cao. Bootstrap theo crop bỏ qua tương quan này → underestimate variance.

**Cluster bootstrap (10,000 lần resample 148 ảnh gốc):**

| Phương pháp | CI 95% dưới | CI 95% trên | Độ rộng |
|---|:---:|:---:|:---:|
| **Crop-level (đã báo cáo trước)** | 0.7072 | 0.7243 | 0.0171 |
| **Cluster theo ảnh gốc (mới)** | **0.7005** | **0.7307** | **0.0302** |

CI cluster rộng hơn 1.76× CI crop. **CI cluster [0.7005, 0.7307] là ước lượng đúng hơn cho test Dice.**

**CI val và TTA delta:** Không tính được cluster bootstrap vì `val_result.json` chỉ lưu aggregate metrics. CI crop val và TTA delta đã báo cáo trước đây là underestimate do không tính tương quan trong ảnh gốc (val chỉ có 50 ảnh gốc, 348 crops).

---

## §3. PHASE 1 — TTA VÀ NGƯỠNG τ

### 3.1 Kết quả số (val)

| Cấu hình | Val Dice | Δ vs no-TTA |
|---|:---:|:---:|
| No TTA, τ=0.5 | 0.779294 | — |
| TTA flip_hv, τ=0.5 | 0.782452 | **+0.003158** |

TTA delta bootstrap (crop-level): CI 95% = **[+0.001021, +0.005301]** — dương hoàn toàn trên val.

> [!IMPORTANT]
> CI này là crop-level bootstrap. Val có 50 ảnh gốc (348 crops). CI cluster không tính được (thiếu per_image_dice trong file). CI thực cho TTA delta trên val có thể rộng hơn.

### 3.2 Kết quả TTA ở test

| So sánh | Dice |
|---|:---:|
| Canonical đơn-seed val tốt nhất (seed 43) | 0.7149 (đã xem ở lần 4) |
| Ensemble 3-seed + TTA (Phase 5 LOCKED) | **0.7158** |
| Δ | +0.0009 |

**+0.0009 điểm nằm trong CI cluster [0.7005, 0.7307] của chính ensemble.**  
Không thể xác nhận TTA/ensemble cải thiện đáng kể ở test từ số liệu này.

**Giả thuyết (chưa kiểm chứng):** Val có thể lạc quan do chọn epoch tốt nhất trên val (best checkpoint chọn theo val dice) dẫn đến TTA delta bị overestimate trên val.

### 3.3 MDE và phạm vi áp dụng

> [!NOTE]
> MDE = 2×s_seed đo ngưỡng phát hiện chênh lệch **giữa hai cấu hình huấn luyện khác nhau** khi lấy trung bình qua nhiều seed. MDE **không áp dụng** cho so sánh ghép cặp trên cùng checkpoint (TTA delta, τ sensitivity). Các so sánh đó cần bootstrap ghép cặp như đã làm ở Phase 1.

τ sensitivity: test không được dùng để khảo sát τ — không có số liệu từ test. Quyết định τ=0.5 được LOCK trên val.

---

## §4. PHASE 2 — MULTI-SEED CANONICAL

| Seed | Val Dice | Δ vs μ |
|---|:---:|:---:|
| 42 | 0.780409 | +0.001771 |
| 43 | 0.782932 | +0.004294 |
| 44 | 0.772572 | −0.006066 |
| **μ = 0.778638** | σ = **0.005402** | — |

MDE = 2σ = **0.010804** (ước lượng điểm). CI 95% cho MDE qua chi-squared (df=2):

$$\text{MDE}_{95\%\text{ CI}} = [0.0056, \ 0.0679]$$

Khoảng này rộng hơn 12× so với ước lượng điểm. Mọi so sánh dùng MDE cần kèm cảnh báo này.

Ensemble val Dice = **0.7850** (Δ vs μ single = +0.0064). Chưa được xác nhận ở test (+0.0009 điểm, không có ý nghĩa).

---

## §5. PHASE 3 — ĐỐI CHỨNG DUNG LƯỢNG

### 5.1 Thiết kế

- **Variant D:** Thay expert_pool + router bằng DenseAdapter (tham số tương đương: 6,713,160 vs mục tiêu 6,713,256, sai lệch −96 params = −0.0014%).
- **Variant R:** Giữ expert_pool, thay SageRouter bằng RandomFixedRouter (0 trainable params, chọn top-2 ngẫu nhiên theo seed cố định).
- **Variant N:** Giữ toàn bộ kiến trúc canonical (10,205,755 params), set `residual_scale = 0.0` trên tất cả 8 SageLayer — expert pool tồn tại nhưng đóng góp = 0 ở forward pass. **ĐÃ CHẠY** (commit `110d652`, Colab T4, 2026-11-16).

### 5.2 Kết quả (Variant D, R, N — mỗi biến thể 2 seed)

| Cấu hình | Mean Val Dice | Δ vs Canonical μ | Vượt MDE (0.0108)? |
|---|:---:|:---:|:---:|
| Canonical (Phase 2 μ) | 0.778638 | — | — |
| **Variant R mean** | 0.778500 | −0.000138 | Không |
| **Variant N mean** | **0.778012** | **−0.000625** | **Không** |
| **Variant D mean** | 0.777881 | −0.000757 | Không |

Chi tiết Variant N theo seed:

| Seed | Stage 1 Val Dice | Stage 2 Val Dice | Global Best | Δ vs Canonical μ |
|---|:---:|:---:|:---:|:---:|
| N-42 | 0.758173 | 0.778212 | 0.778212 | −0.000425 |
| N-43 | 0.759447 | 0.777812 | 0.777812 | −0.000825 |
| **N mean** | — | — | **0.778012** | **−0.000625** |

**Diễn giải đúng (thay "tương đương tuyệt đối"):**

> Không phát hiện chênh lệch ≥ ~1 điểm Dice giữa canonical và bất kỳ biến thể nào (D, R, N), với 2 seed mỗi biến thể và MDE = 0.0108 điểm (CI MDE 95%: [0.0056, 0.0679]).

**Về router (Variant R):** Không phát hiện đóng góp đo được của cơ chế định tuyến có học (Δ = −0.000138 điểm vs random router).

**Về nhánh expert (Variant N):** Không phát hiện đóng góp đo được của expert pool contribution (Δ = −0.000625 điểm vs residual_scale=0). Chuỗi R → N → D đều không phân biệt được với Canonical trong tầm MDE.

> [!CAUTION]
> Routing diagnostics (Variant R, seed 42): Phân phối chọn 8 expert gần đều tuyệt đối (11.66%–13.04%, kỳ vọng 12.50%), xác nhận RandomFixedRouter hoạt động đúng. Điều này đồng thời cho thấy canonical router cũng chọn expert gần đều — tuy nhiên hành vi của canonical router chưa được đo đối chiếu.

---

## §6. VAL→TEST GAP

### 6.1 So sánh đặc tính nhãn

Đo trên ground-truth masks (`datasets/Crack500_ready/{val,test}/masks/`). Không dùng dự đoán mô hình.

| Chỉ số | Val (348 crops) | Test (1124 crops) | p-value MWU | Kết luận |
|---|:---:|:---:|:---:|---|
| Tỉ lệ pixel nứt | 0.0570 ± 0.0480 | 0.0559 ± 0.0410 | 0.3357 | Không khác biệt |
| Độ rộng crack trung bình (px) | 14.40 ± 7.38 | 13.94 ± 7.06 | 0.4143 | Không khác biệt |
| Tỉ lệ crack pixel mảnh ≤ 3px | 0.1437 | 0.1499 | 0.2234 | Không khác biệt |
| **Số thành phần liên thông** | **2.24 ± 1.91 (max=14)** | **3.32 ± 4.87 (max=64)** | **0.0002** | **Khác biệt có ý nghĩa** |

### 6.2 Khoảng cách Dice

| So sánh | Val Dice | Test Dice | Khoảng cách |
|---|:---:|:---:|:---:|
| Canonical đơn vs Locked Test | 0.7786 (mean 3 seed) | 0.7158 | **6.3 điểm** |
| Ensemble val vs Locked Test | 0.7850 | 0.7158 | **6.9 điểm** |
| B0 baseline (tham khảo) | 0.7318 | 0.6771 | **5.5 điểm** |

> [!WARNING]
> Không được nói: "val→test gap do test có nhiều vết mảnh hơn" — thin crack fraction p=0.22, không khác biệt. Chỉ n_components khác biệt có ý nghĩa. Val→test gap 6.3 điểm không khác xa gap B0 (5.5 điểm) → gap có thể là đặc tính chung của dataset, không riêng B2.

---

## §7. HẠN CHẾ

1. **Val chỉ có 50 ảnh gốc** (348 crops), không phải 348 ảnh độc lập. Mọi CI crop-level đã báo cáo cho val là underestimate variance.

2. **Test đã được xem ở 4 lần trước Phase 5** (Dice: 0.6771, 0.6801, 0.6950, 0.7149). Không thể loại trừ hoàn toàn ảnh hưởng đến các quyết định chọn hướng nghiên cứu.

3. **n=3 seed cho MDE** — CI 95% thực của MDE là [0.0056, 0.0679], rộng hơn 12× ước lượng điểm 0.0108. Mọi kết luận Phase 3 phụ thuộc vào MDE này đều có độ chắc chắn thấp.

4. **Val lạc quan do chọn checkpoint tốt nhất trên val.** Hiệu quả của TTA (+0.0032 điểm val) có thể bị overestimate so với test.

5. **per_image_dice không lưu kèm filenames** (cả val lẫn test). Không thể phân tích Dice theo nhóm (crack width, topology complexity) mà không chạy lại inference trên GPU.

6. **Variant N đã hoàn thành** (2026-11-16). Không phát hiện đóng góp đo được của nhánh expert pool (Δ = −0.000625 điểm, không vượt MDE). "Không phát hiện" ≠ "không có đóng góp" — CI MDE rộng [0.0056, 0.0679] giới hạn kết luận.

7. **Phase 4** (domain adaptation/augmentation) và **Phase 6** (inference optimization/quantization) chưa thực hiện. Báo cáo này không bao gồm hai phase đó.

---

## §8. NHỮNG ĐIỀU KHÔNG ĐƯỢC TUYÊN BỐ

| Tuyên bố | Lý do không được nói |
|---|---|
| "Test không bị data snooping" | Test đã xem 4 lần trước Phase 5 |
| "TTA/ensemble giúp ích ở test" | Δ test = +0.0009 điểm, nằm trong CI cluster; không xác nhận được |
| "Val→test gap do vết mảnh" | thin crack fraction p=0.22, không khác biệt; không có số đo ủng hộ |
| "SAGE router vô ích hoàn toàn" | Chỉ đo được 2 seed; CI MDE rộng; không đo hành vi canonical router |
| "Nhánh expert không đóng góp" | Chỉ đo được 2 seed; CI MDE rộng; "không phát hiện" ≠ "không có đóng góp" |
| "B2 tốt hơn B0 về gap val→test" | Gap B2 = 6.3 điểm, Gap B0 = 5.5 điểm — không khác xa, kết luận vô căn cứ |

---

## §9. TÀI LIỆU THAM CHIẾU

| File | Mô tả |
|---|---|
| [`05_frozen_config.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/05_frozen_config.json) | Cấu hình LOCKED trước Phase 5 |
| [`05_test_final_report.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/05_test_final_report.json) | Kết quả test chính thức (Phase 5) |
| [`02_phase2_summary.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/02_phase2_summary.json) | Multi-seed canonical val results |
| [`02_ensemble_val_result.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/02_ensemble_val_result.json) | Ensemble val dice |
| [`03_capacity_control.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/03_capacity_control.md) | Phase 3 Variant D & R |
| [`07_cluster_bootstrap.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/07_revision/07_cluster_bootstrap.md) | Cluster bootstrap CI (Việc 1) |
| [`07a_cluster_bootstrap_results.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/07_revision/07a_cluster_bootstrap_results.json) | Dữ liệu raw cluster bootstrap |
| [`07_valtest_gap.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/07_revision/07_valtest_gap.md) | Val→test gap analysis (Việc 2) |
| [`07b_valtest_gap_results.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/07_revision/07b_valtest_gap_results.json) | Dữ liệu raw val→test gap |
| [`07_variant_N.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/07_revision/07_variant_N.md) | Variant N — chuẩn bị, chờ GPU |
| [`phase3_variant_N_colab.ipynb`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/07_revision/phase3_variant_N_colab.ipynb) | Notebook Colab để chạy Variant N |
