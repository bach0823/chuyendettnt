# BÁO CÁO TỔNG HỢP SỬA ĐỔI — SAGE-Lite trên Crack500

**Ngày:** 2026-11-16  
**Phiên bản:** Revision 2  
**Thay thế:** Revision 1 (`07_revision/FINAL_REPORT_REVISED.md`)  
**Không ghi đè:** Revision 1 được giữ nguyên tại đường dẫn cũ.

---

## BẢNG THAY ĐỔI SO VỚI REVISION 1

| Lỗi Rev 1 | Nội dung lỗi | Sửa trong Rev 2 | File 08_* tương ứng |
|---|---|---|---|
| §1: Bảng lịch sử test | 0.6771/0.6801 không truy xuất được; "seed 43" sai | Bảng đầy đủ với checkpoint/giao thức/ngày rõ ràng; bỏ 0.6771/0.6801 vì nguồn không xác định | `08_test_history.md` |
| §3.2: "seed 43" | "Canonical đơn-seed val tốt nhất (seed 43)" | 0.7149 là seed=42, run2, direct 1-pass, không phải seed 43 | `08_test_comparison.md` |
| §3.2: Δ=+0.0009 | Giải thích như hiệu quả TTA/ensemble | Δ giữa hai đối tượng khác giao thức; không rút ra kết luận | `08_test_comparison.md` |
| §6.2: So sánh gap | "val mean 3-seed vs test ensemble" | So sánh đúng: seed=42 no-TTA val=0.7793 vs test=0.7149 → 6.44 điểm | `08_test_comparison.md` |
| §4: "12×" | "CI rộng hơn 12× so với ước lượng điểm" | "Cận trên gấp 12.1 lần cận dưới (hi/lo)" | `08_cluster_recheck.md` |
| §5: Baseline Variant D/R/N | Chỉ dùng μ 3-seed | Thêm μ 2-seed khớp (seed42/43) làm baseline bổ sung | `08_phase3_recheck.md` |
| §7: Phase 4/6 | "domain adaptation" / "inference optimization" | Phase 4 = LR screening; Phase 6 = cross-dataset | `08_phase3_description.md` |
| Toàn bộ: đơn vị | Lẫn lộn "điểm" cho cả scale 0–1 và điểm phần trăm | Quy ước: "0.0009 Dice = 0.09 điểm" nhất quán toàn bộ | (sửa trong báo cáo này) |

**Giả định khi thiếu thông tin:** Giá trị 0.6771 và 0.6801 trong Revision 1 không tìm thấy nguồn gốc rõ ràng trong các file JSON kết quả — có thể là per_image_dice của 1 crop hoặc số từ nguồn khác. Báo cáo này bỏ hai giá trị này khỏi bảng lịch sử test và ghi rõ lý do.

---

## §1. TÍNH TRONG SÁNG CỦA DỮ LIỆU TEST

**τ, TTA và ensemble được đóng băng trước khi chạy test** (commit `05_frozen_config.json`). Kiến trúc, siêu tham số và pipeline được chọn hoàn toàn trên val.

**Test đã được xem nhiều lần trước Phase 5.** Các lần đo xác nhận được từ file nguồn:

| # | Checkpoint | Val best_dice | Giao thức | Test Dice | File nguồn |
|---|---|:---:|---|:---:|---|
| 1 | B0 baseline (50 epoch) | — | Direct 1-pass, τ=0.5 | 0.7042 | `baselines/50ep/B0_Crack500/official_eval_test.json` |
| 2 | B0 baseline (100 epoch) | — | Direct 1-pass, τ=0.5 | 0.7037 | `baselines/100ep/B0_Crack500/official_eval_all_splits.json` |
| 3 | `phase7_s2g_b1_b2_end_to_end` (epoch 15) | 0.7689 | Setting A (stride=448) | 0.7016 | `phase7_combination/.../final_test_metrics.json` |
| 4 | `phase7_s2g_b1_b2_end_to_end` (epoch 15) | 0.7689 | Setting B (stride=224) | 0.7054 | (cùng file) |
| 5 | `phase7_s2g_new_pipeline` run1 (epoch 18) | 0.7783 | Setting A | 0.6881 | `phase7_combination/phase7_s2g_new_pipeline_run1/.../final_test_metrics.json` |
| 6 | `phase7_s2g_new_pipeline` run1 (epoch 18) | 0.7783 | Setting B | 0.6950 | (cùng file) |
| 7 | `phase7_s2g_new_pipeline` run2 (epoch 16) | 0.7793 | Setting A | 0.6895 | `phase7_combination/phase7_s2g_new_pipeline_run2/.../final_test_metrics.json` |
| 8 | `phase7_s2g_new_pipeline` run2 (epoch 16) | 0.7793 | Setting B | 0.6950 | (cùng file) |
| 9 | `phase7_s2g_new_pipeline` run2 (epoch 16, **seed=42**) | 0.7793 | Direct 1-pass, τ=0.5, không TTA | **0.7149** | `phase7_combination/phase7_s2g_new_pipeline_run2/diagnostics/official_direct_evaluation.json` |
| **10 (LOCKED — Phase 5)** | Ensemble Phase 2 seed42/43/44 | μ=0.7787 | Direct 1-pass + TTA flip_hv | **0.7158** | `final_protocol/05_test_final_report.json` |

**Số checkpoint khác nhau đã xem test:** 5 (B0 50ep, B0 100ep, b1_b2_end_to_end, run1, run2+ensemble Phase2)  
**Số lần đo:** ≥ 10

> [!WARNING]
> Test đã được xem ≥ 9 lần trước khi Phase 5 LOCK. Không thể loại trừ hoàn toàn ảnh hưởng đến các quyết định chọn hướng nghiên cứu.

**Kết quả Phase 5 (LOCKED — không thay đổi):**
- Test Dice = **0.7158** (mean crop-level, 1,124 crops)
- Precision = 0.6724 | Recall = 0.8123 | IoU = 0.5755
- Thin-crack Dice = 0.2883 | HD95 = 50.93 px
- Checkpoint: ensemble 3 seed (42, 43, 44), τ = 0.5, TTA = flip_hv
- SHA-256: seed42 = `4B409B...`, seed43 = `7379C7...`, seed44 = `F5F7DC...`

---

## §2. KHOẢNG TIN CẬY TEST DICE — CLUSTER BOOTSTRAP

Dữ liệu: 1,124 crops từ **148 ảnh gốc** (tên file format `YYYYMMDD_HHMMSS_X_Y.jpg`, source ID = `YYYYMMDD_HHMMSS`). Val: 348 crops từ **50 ảnh gốc**.

| Phương pháp | CI 95% dưới | CI 95% trên | Độ rộng | Tỉ lệ |
|---|:---:|:---:|:---:|:---:|
| **Crop-level (đã báo cáo trước)** | 0.7072 | 0.7243 | 0.0171 | 1.00× |
| **Cluster theo ảnh gốc (10,000 lần)** | **0.7005** | **0.7307** | **0.0302** | **1.76×** |

CI cluster rộng hơn 1.76× CI crop. **CI cluster [0.7005, 0.7307] là ước lượng đúng hơn cho test Dice.**

**CI val và TTA delta:** Không tính được cluster bootstrap (thiếu per_image_dice trong file val_result.json và tta bootstrap). CI crop-level đã báo cáo là underestimate.

> **Nguồn:** `results/final_protocol/07_revision/07a_cluster_bootstrap_results.json`

---

## §3. PHASE 1 — TTA VÀ NGƯỠNG τ

### 3.1 Kết quả số (val)

| Cấu hình | Val Dice | Δ vs no-TTA |
|---|:---:|:---:|
| No TTA, τ=0.5 | 0.779294 | — |
| TTA flip_hv, τ=0.5 | 0.782452 | **+0.003158 (0.32 điểm)** |

TTA delta bootstrap (crop-level): CI 95% = **[+0.001021, +0.005301]** — dương hoàn toàn trên val.

> [!IMPORTANT]
> CI này là crop-level bootstrap. Val có 50 ảnh gốc (348 crops). CI cluster không tính được (thiếu per_image_dice trong file). CI thực cho TTA delta trên val có thể rộng hơn.

### 3.2 TTA/Ensemble tại test — không so sánh trực tiếp được

| Đối tượng | Checkpoint | Giao thức | Val Dice | Test Dice |
|---|---|---|:---:|:---:|
| Canonical run2 | 1 ckpt (seed=42) | Direct, τ=0.5, không TTA | 0.7793 | 0.7149 |
| Phase 5 LOCKED | 3 ckpt ensemble (seed42/43/44) | Direct + TTA flip_hv | 0.7850 | 0.7158 |
| Δ | — | Khác nhau: checkpoint set, TTA, ensemble | — | **+0.0009 (0.09 điểm)** |

Δ test = **+0.09 điểm** phản ánh đồng thời tác động của thay đổi checkpoint set + TTA + ensemble. Không thể phân tách. Không xác nhận TTA/ensemble cải thiện đáng kể ở test.

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

Nguồn: `results/final_protocol/02_phase2_summary.json`

MDE = 2σ = **0.010804 Dice (1.08 điểm)**. CI 95% cho MDE qua chi-squared (df=2):

$$\text{MDE}_{95\%\text{ CI}} = [0.0056, \ 0.0679]$$

**Cận trên gấp khoảng 12 lần cận dưới (0.0679 / 0.0056 ≈ 12.1×)**, phản ánh độ bất định cao khi n=3 seed. Mọi so sánh dùng MDE cần kèm cảnh báo này.

Ensemble val Dice = **0.7850** (Δ vs μ single = +0.64 điểm). Chưa được xác nhận ở test (+0.09 điểm, nằm trong CI cluster).

---

## §5. PHASE 3 — ĐỐI CHỨNG DUNG LƯỢNG

### 5.1 Thiết kế

- **Variant D:** Thay expert_pool + router bằng DenseAdapter (tham số tương đương: 6,713,160 vs mục tiêu 6,713,256, sai lệch −96 params = −0.0014%).
- **Variant R:** Giữ expert_pool, thay SageRouter bằng RandomFixedRouter (0 trainable params, chọn top-2 ngẫu nhiên theo seed cố định).
- **Variant N:** Giữ toàn bộ kiến trúc canonical (10,205,755 params), set `residual_scale = 0.0` trên tất cả 8 SageLayer. Expert pool tồn tại nhưng đóng góp = 0 ở forward pass và gradient = 0 về expert params trong train. **ĐÃ CHẠY** (commit `110d652`, Colab T4, 2026-11-16).

Nguồn: `sage/components/sage_layer.py` dòng 81: `self.residual_scale = float(residual_scale)` — là Python `float`, không phải `nn.Parameter`.

### 5.2 Kết quả (mỗi biến thể 2 seed: 42/43)

| Cấu hình | Mean Val Dice | Δ vs Canonical μ (3-seed) | Δ vs Canonical μ (2-seed 42/43) | Vượt MDE (1.08 điểm)? |
|---|:---:|:---:|:---:|:---:|
| Canonical μ (3 seed: 42/43/44) | 0.778638 | — | — | — |
| Canonical μ (2 seed: 42/43) | 0.781670 | — | — | — |
| **Variant R mean** | 0.778500 | −0.000138 (−0.01 điểm) | −0.003170 (−0.32 điểm) | Không |
| **Variant N mean** | **0.778012** | **−0.000625 (−0.06 điểm)** | **−0.003658 (−0.37 điểm)** | **Không** |
| **Variant D mean** | 0.777881 | −0.000757 (−0.08 điểm) | −0.003789 (−0.38 điểm) | Không |

Nguồn: `03_capacity_control.md`, `phase3_variantN_seed42/run_metadata.json`, `phase3_variantN_seed43/run_metadata.json`

**Diễn giải đúng:**

> Không phát hiện chênh lệch ≥ 1.08 điểm Dice giữa canonical và bất kỳ biến thể nào (D, R, N), với 2 seed mỗi biến thể và MDE = 1.08 điểm (CI MDE 95%: [0.56, 6.79] điểm). Điều này đúng với cả baseline 3-seed và baseline 2-seed khớp seed.

**Về router (Variant R):** Không phát hiện đóng góp đo được của cơ chế định tuyến có học (Δ = −0.01 điểm vs random router, không vượt MDE).

**Về nhánh expert (Variant N):** Không phát hiện đóng góp đo được của expert pool contribution (Δ = −0.06 điểm vs residual_scale=0, không vượt MDE). "Không phát hiện" ≠ "không có đóng góp".

> [!CAUTION]
> Routing diagnostics (Variant R, seed 42): Phân phối chọn 8 expert gần đều tuyệt đối (11.66%–13.04%, kỳ vọng 12.50%), xác nhận RandomFixedRouter hoạt động đúng. Hành vi của canonical router **chưa được đo đối chiếu** — không thể so sánh phân phối canonical vs random.

---

## §6. VAL→TEST GAP

### 6.1 So sánh đặc tính nhãn (Ground Truth Masks)

Đo trên ground-truth masks (`datasets/Crack500_ready/{val,test}/masks/`). Không dùng dự đoán mô hình.

| Chỉ số hình thái | Val trung bình crop ($n=348$) | Test trung bình crop ($n=1,124$) | $p$-value crop | Val trung bình ảnh gốc ($n=50$) | Test trung bình ảnh gốc ($n=148$) | $p$-value ảnh gốc (Cluster MWU) | Kết luận |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Tỉ lệ pixel nứt (crack ratio)** | $0.0570 \pm 0.0480$ | $0.0559 \pm 0.0410$ | 0.3357 | $0.0557 \pm 0.0317$ | $0.0558 \pm 0.0254$ | **0.4799** | Không khác biệt |
| **Độ rộng crack trung bình (px)** | $14.40 \pm 7.38$ | $13.94 \pm 7.06$ | 0.4143 | $14.12 \pm 6.25$ | $14.13 \pm 5.66$ | **0.8205** | Không khác biệt |
| **Tỉ lệ pixel mảnh $\le 3$ px** | 0.1437 | 0.1499 | 0.2234 | $0.1470 \pm 0.0777$ | $0.1455 \pm 0.0676$ | **0.7830** | Không khác biệt |
| **Số thành phần liên thông** | $2.24 \pm 1.91$ (max 14) | $3.32 \pm 4.87$ (max 64) | 0.0002* | $2.28 \pm 1.13$ | $2.97 \pm 2.63$ | **0.1052** | **Không khác biệt ở mức ảnh gốc** |

*\*Lưu ý:* $p = 0.0002$ ở mức crop là do hiện tượng giả lặp mẫu (pseudoreplication). Khi gộp đúng về mức ảnh gốc độc lập, $p = 0.1052 > 0.05$.

> [!IMPORTANT]
> **Không có bất kỳ đặc tính nhãn nào khác biệt có ý nghĩa thống kê giữa tập Val và Test ở mức ảnh gốc.** Khoảng cách hiệu năng 6.44 điểm không bắt nguồn từ sự lệch phân bố hình thái bề mặt.

### 6.2 Khoảng cách Dice

| So sánh | Val Dice | Test Dice | Khoảng cách |
|---|:---:|:---:|:---:|
| **Seed=42, không TTA, direct 1-pass (cùng checkpoint + giao thức)** | **0.7793** | **0.7149** | **6.44 điểm** |
| Canonical μ (3 seed) vs Locked Test (ensemble) | 0.7787 | 0.7158 | 6.29 điểm |
| Ensemble val vs Locked Test | 0.7850 | 0.7158 | 6.92 điểm |
| B0 baseline (tham khảo) | 0.7318 | 0.7037 | 2.81 điểm |

**So sánh đúng nhất:** seed=42, không TTA, cùng giao thức direct 1-pass → **gap = 6.44 điểm** (nguồn: `official_direct_evaluation.json` + `00_audit/audit_result.json`).

### 6.3 Phân tích Test Dice theo nhóm đặc tính nhãn (Việc 6)

Ánh xạ 1-đối-1 giữa `per_image_dice` (1,124 crop) và đặc tính mask tương ứng (`08_per_image_dice_test.csv`):

- **Tương quan:**
  * Tỉ lệ vết mảnh ($\le 3$ px): Spearman $\rho = \mathbf{-0.4793}$ ($p = 1.2 \times 10^{-65}$) — tương quan âm mạnh nhất.
  * Độ rộng trung bình vết: Spearman $\rho = \mathbf{+0.4455}$ ($p = 6.6 \times 10^{-56}$) — vết càng rộng Dice càng cao.
  * Số thành phần liên thông: Spearman $\rho = \mathbf{-0.2992}$ ($p = 1.1 \times 10^{-24}$) — càng phân mảnh Dice càng giảm.
- **Phân nhóm độ rộng:**
  * Vết vừa ($3 < \text{width} \le 10$ px, $n=394$): **Mean Dice = 0.6559 (65.59 điểm)**
  * Vết rộng ($\text{width} > 10$ px, $n=730$): **Mean Dice = 0.7481 (74.81 điểm)**
  * Chênh lệch: **+9.22 điểm Dice** cho vết rộng.
- **Phân nhóm độ phức tạp liên thông:**
  * Đơn giản ($\le 2$ thành phần, $n=711$): **Mean Dice = 0.7380 (73.80 điểm)**
  * Phức tạp ($> 5$ thành phần, $n=143$): **Mean Dice = 0.6511 (65.11 điểm)**
  * Chênh lệch: **+8.69 điểm Dice** cho cấu trúc đơn giản.

---

## §7. HẠN CHẾ

1. **Val chỉ có 50 ảnh gốc** (348 crops). CI crop-level cho val và TTA delta là underestimate.

2. **Test đã được xem ≥ 9 lần trước Phase 5 LOCK.** Không thể loại trừ hoàn toàn ảnh hưởng đến quyết định chọn hướng nghiên cứu.

3. **n=3 seed cho MDE** — CI 95% thực của MDE là [0.56, 6.79] điểm. Mọi kết luận Phase 3 phụ thuộc vào MDE này đều có độ chắc chắn thấp.

4. **Val lạc quan do chọn checkpoint tốt nhất trên val.** Hiệu quả của TTA (+0.32 điểm val) có thể bị overestimate.

5. **Variant N đã hoàn thành** (2026-11-16). Không phát hiện đóng góp đo được của nhánh expert pool (Δ = −0.06 điểm, không vượt MDE). "Không phát hiện" ≠ "không có đóng góp".

6. **Phase 4** (sàng lọc tốc độ học — LR screening, tùy chọn) và **Phase 6** (đánh giá cross-dataset — chờ nhãn) chưa thực hiện.

7. **Test coverage:** Dataset local có 148 ảnh gốc test (1,124 crops), thấp hơn 200 ảnh gốc nêu trong paper Crack500. Chênh lệch 52 ảnh gốc không giải thích được từ file hiện tại.

---

## §8. NHỮNG ĐIỀU KHÔNG ĐƯỢC TUYÊN BỐ

| Tuyên bố | Lý do không được nói |
|---|---|
| "Test không bị data snooping" | Test đã xem ≥ 9 lần trước Phase 5 |
| "TTA/ensemble giúp ích ở test" | Δ test = +0.09 điểm giữa hai đối tượng khác giao thức; không xác nhận được |
| "0.7149 là kết quả của seed 43" | 0.7149 là seed=42, run2, direct 1-pass |
| "Val→test gap do phân phối vết mảnh lệch nhau" | Tỉ lệ vết mảnh giữa Val và Test không khác biệt ($p = 0.78$ ở mức ảnh gốc) |
| "SAGE router vô ích hoàn toàn" | Chỉ đo được 2 seed; CI MDE rộng; không đo hành vi canonical router |
| "Nhánh expert không đóng góp" | Chỉ đo được 2 seed; CI MDE rộng; "không phát hiện" ≠ "không có đóng góp" |
| "B2 tốt hơn B0 về gap val→test" | Gap B2 = 6.44 điểm, B0 = 2.81 điểm — B2 gap lớn hơn, không nhỏ hơn |

---

## §9. CÂU TRẢ LỜI CHO 4 CÂU HỎI CHÍNH

| Câu hỏi | Câu trả lời |
|---|---|
| Router có đóng góp? | Không phát hiện đóng góp trong tầm MDE (Δ = −0.01 điểm vs canonical; \|Δ\| ≪ MDE = 1.08 điểm) |
| Nhánh expert có đóng góp? | **Không phát hiện** — Variant N (residual_scale=0) cho Δ = −0.06 điểm vs canonical, không vượt MDE. "Không phát hiện" ≠ "không có đóng góp" |
| Ensemble+TTA có giúp ở test? | **Không xác nhận** — Δ test = +0.09 điểm giữa hai đối tượng khác giao thức; nằm trong CI cluster |
| Khoảng cách val→test 6.44 điểm đến từ đâu? | Phân phối nhãn Val và Test không lệch nhau ở mức ảnh gốc ($p > 0.10$ mọi chỉ số). Nhưng ở mức vi mô, Test Dice bị kéo tụt chủ yếu bởi crops có nhiều vết mảnh ($\rho = -0.48$) và cấu trúc phân mảnh ($\rho = -0.30$) |

---

## §10. TÀI LIỆU THAM CHIẾU

| File | Mô tả |
|---|---|
| [`05_frozen_config.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/05_frozen_config.json) | Cấu hình LOCKED trước Phase 5 |
| [`05_test_final_report.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/05_test_final_report.json) | Kết quả test chính thức (Phase 5) |
| [`02_phase2_summary.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/02_phase2_summary.json) | Multi-seed canonical val results |
| [`03_capacity_control.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/03_capacity_control.md) | Phase 3 Variant D & R |
| [`07_cluster_bootstrap.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/07_revision/07_cluster_bootstrap.md) | Cluster bootstrap CI |
| [`07a_cluster_bootstrap_results.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/07_revision/07a_cluster_bootstrap_results.json) | Dữ liệu raw cluster bootstrap |
| [`07_valtest_gap.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/07_revision/07_valtest_gap.md) | Val→test gap analysis |
| [`07_variant_N.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/07_revision/07_variant_N.md) | Variant N — chuẩn bị |
| [`phase3_variantN_seed42/run_metadata.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/phase3_variantN_seed42/run_metadata.json) | Variant N seed42 |
| [`phase3_variantN_seed43/run_metadata.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/phase3_variantN_seed43/run_metadata.json) | Variant N seed43 |
| [`08_revision2/08_test_history.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/08_revision2/08_test_history.md) | Lịch sử test đầy đủ |
| [`08_revision2/08_test_comparison.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/08_revision2/08_test_comparison.md) | So sánh 0.7149 vs 0.7158 |
| [`08_revision2/08_phase3_description.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/08_revision2/08_phase3_description.md) | Mô tả Phase 4 & 6 |
| [`08_revision2/08_cluster_recheck.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/08_revision2/08_cluster_recheck.md) | Kiểm tra cluster bootstrap & MWU ảnh gốc |
| [`08_revision2/08_phase3_recheck.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/08_revision2/08_phase3_recheck.md) | Kiểm tra Phase 3 — baseline khớp seed, gradient Variant N |
| [`08_revision2/08_dice_by_group.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/08_revision2/08_dice_by_group.md) | Phân tích Dice theo nhóm độ rộng và độ phức tạp |
| [`08_revision2/08_per_image_dice_test.csv`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/08_revision2/08_per_image_dice_test.csv) | Ánh xạ chi tiết 1,124 test crop với Dice và đặc tính mask |
| [`07_cluster_bootstrap.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/07_revision/07_cluster_bootstrap.md) | Cluster bootstrap CI |
| [`07a_cluster_bootstrap_results.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/07_revision/07a_cluster_bootstrap_results.json) | Dữ liệu raw cluster bootstrap |
| [`07_valtest_gap.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/07_revision/07_valtest_gap.md) | Val→test gap analysis |
| [`07_variant_N.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/07_revision/07_variant_N.md) | Variant N — chuẩn bị |
| [`phase3_variantN_seed42/run_metadata.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/phase3_variantN_seed42/run_metadata.json) | Variant N seed42 |
| [`phase3_variantN_seed43/run_metadata.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/phase3_variantN_seed43/run_metadata.json) | Variant N seed43 |
| [`08_revision2/08_test_history.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/08_revision2/08_test_history.md) | Lịch sử test đầy đủ |
| [`08_revision2/08_test_comparison.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/08_revision2/08_test_comparison.md) | So sánh 0.7149 vs 0.7158 |
| [`08_revision2/08_phase3_description.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/08_revision2/08_phase3_description.md) | Mô tả Phase 4 & 6 |
| [`08_revision2/08_cluster_recheck.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/08_revision2/08_cluster_recheck.md) | Kiểm tra cluster bootstrap |
| [`08_revision2/08_phase3_recheck.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/08_revision2/08_phase3_recheck.md) | Kiểm tra Phase 3 — baseline khớp seed, gradient Variant N |
