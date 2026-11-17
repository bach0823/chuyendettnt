# FINAL_REPORT_REVISION6.md — Báo Cáo Nghiên Cứu SAGE-Lite Trên Crack500 (Revision 6 - Bản Chốt Cuối)

**Ngày lập:** 2026-11-17  
**Phiên bản:** Revision 6 (Bản chốt khoa học cuối cùng)  
**Quy ước đo lường:** Đơn vị đo điểm phần trăm ($0.0009 \text{ Dice} = 0.09 \text{ điểm}$).  
**Nguyên tắc kỹ thuật:** Báo cáo bằng tiếng Việt, ngắn gọn, số liệu có căn cứ mã nguồn và tệp nguồn đối chứng, không giải thích số mới bằng số cũ nếu chưa đối chiếu tệp, "không phát hiện" không có nghĩa là "bác bỏ".

---

## 1. Tóm Tắt Kết Luận Cuối Cùng (Executive Summary)

1. **Hiệu năng kiểm định trên tập Test:** Mô hình SAGE-Lite B2 đạt điểm Test Direct là **0.7149** (`official_direct_evaluation.json`) và khi áp dụng cấu hình LOCKED (Ensemble 3 seed + TTA flip_hv) đạt **0.7158** (`05_test_final_report.json`), trong khi mô hình baseline B0 huấn luyện 100 epoch đạt **0.7037** (`results/baselines/100ep/B0_Crack500/official_eval_all_splits.json`). Hiệu số giữa B0 100ep và B2 canonical (+1.12 điểm) cũng như mức tăng của LOCKED (+0.09 điểm) hiện **chưa được kiểm định ghép cặp** và **chưa có số liệu đo lường phương sai giữa các seed (seed-to-seed variance) trên tập test direct** do các bản ghi lịch sử của B0 và Canonical direct chỉ lưu chỉ số tổng hợp mà không lưu mảng Dice theo từng ảnh, đồng thời quy tắc nghiên cứu nghiêm cấm suy luận mới trên test.
2. **Cấu trúc tham số thực tế:** Toàn bộ `expert_pool` trong mô hình thực chất là con trỏ tham chiếu (view/reference) trỏ trực tiếp vào các khối `main_block` của Backbone. Mô hình B2 gồm **10,205,755 tham số** (`sage_lite/scripts/train_phase3_ablation.py`), trong đó **92.77% tham số** (Backbone 64.93% + Decoder/S2-Gate 27.83%) hoạt động bình thường và nhận gradient đầy đủ khi huấn luyện; chỉ có **7.23% (738,216 tham số)** thuộc về các thành phần riêng của SAGE (gồm 8 Router chiếm 88,776 tham số và SA-Hub adapters chiếm 649,440 tham số; mô-đun P3 bằng 0 tham số và không kích hoạt).
3. **Kết quả thử nghiệm bóc tách (Phase 3 Ablation):** Khi đối chiếu với baseline chuẩn đa seed của Phase 2 (Seed 42 đạt 0.780409, Seed 43 đạt 0.782932, trung bình 2 seed là **0.781670**; trung bình 3 seed là **0.778638**), cả ba biến thể bóc tách D, R, N đều có điểm Val trung bình **thấp hơn baseline 2 seed từ −0.32 đến −0.38 điểm** (D: −0.38 điểm, R: −0.32 điểm, N: −0.37 điểm; hoặc thấp hơn −0.01 đến −0.08 điểm so với baseline 3 seed). Tuy nhiên, với kích thước mẫu chỉ $n = 2$ seed cho mỗi biến thể, cùng với việc khoảng tin cậy 95% của tập Val có độ rộng lên tới **5.74 điểm** (do chỉ có 50 ảnh gốc), cỡ mẫu $n=2$ **hoàn toàn không có đủ công suất thống kê để khẳng định mức chênh lệch 0.3 – 0.4 điểm này là hiệu ứng thực sự của kiến trúc hay chỉ là nhiễu ngẫu nhiên giữa các lần chạy**.
4. **Bản chất của Variant D và Variant R:** Variant D là **"Canonical B2 cộng thêm một nhánh dày DenseAdapter gồm 6.71M tham số"** (tổng 16,918,915 tham số, tăng +65.78% dung lượng, không phải đối chứng thay thế cùng dung lượng) nhưng không cải thiện được điểm Val (đạt 0.7779 trung bình). Variant R thay thế 8 `SageRouter` bằng `RandomFixedRouter` (loại bỏ đúng **88,776 tham số** của 8 Router, giữ nguyên SA-Hub, tổng còn 10,116,979 tham số) vẫn đạt điểm Val trung bình 0.7785 (chỉ chênh −0.03 điểm so với canonical 2 seed).
5. **Bản chất khoảng cách Val $\to$ Test và Vết nứt mảnh:** Ở mức độ ảnh gốc độc lập (Macro-average), kiểm định phi tham số **không phát hiện sự khác biệt có ý nghĩa thống kê** về tỉ lệ vết nứt mảnh giữa tập Val và Test ($p = 0.1408$ đối với tỉ lệ thành phần và $p = 0.0519$ đối với tỉ lệ pixel) với **công suất thống kê thấp** (độ chênh nhỏ nhất phát hiện được MDD ở power 80% là 8.96 điểm % đối với thành phần và 0.18 điểm % đối với pixel). Phân rã Oaxaca-Blinder chứng minh rằng **Within effect chiếm 96.3% – 96.6%** khoảng cách Val $\to$ Test (mô hình suy giảm hiệu năng đồng đều trên mọi nhóm độ rộng), trong khi **Mix effect chỉ đóng góp 3.4% – 3.7%**.
6. **Đánh giá tổng thể & Hướng đi tiếp theo:** Mô hình SAGE-Lite không chứng minh được sự vượt trội có ý nghĩa thống kê so với baseline B0 khi B0 được huấn luyện đủ ngân sách (100 epoch). Giai đoạn tiếp theo (Phase 6) khuyến nghị đánh giá trực tiếp trên **dataset tường riêng của người dùng** (zero-shot, fine-tune, huấn luyện từ ImageNet) và tuân thủ **nguyên tắc Pareto**: ưu tiên lựa chọn Baseline B0 nếu đạt hiệu năng tương đương để tối ưu hóa chi phí tính toán.

---

## 2. Bảng Theo Dõi Thay Đổi So Với Revision 5 Và Danh Mục Tệp Chi Tiết

| Nội dung thay đổi | Bản chất hiệu chỉnh trong Revision 6 | Tệp tài liệu chi tiết tương ứng |
|---|---|---|
| **Khôi phục baseline Phase 3** | Dùng baseline Phase 2 seed 42 (0.780409), seed 43 (0.782932), trung bình 2 seed (0.781670) và 3 seed (0.778638); làm rõ Run1/Run2 cũ là cùng seed 42 (nhiễu intra-seed); tính lại $\Delta$ của D, R, N so với cả hai baseline (−0.32 đến −0.38đ vs 2 seed; −0.01 đến −0.08đ vs 3 seed); chỉ ra $n=2$ không loại được chênh lệch cỡ 0.3–0.4đ. | [12_phase3_baseline.md](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/12_revision6/12_phase3_baseline.md) |
| **Đối chiếu B0 với file gốc** | Lập bảng chi tiết epoch tối đa, epoch tốt nhất, giao thức; đính chính B0 50ep là **0.7042** (khớp tệp `official_eval_test.json`, giải thích số 0.6974 ở Rev 5 là lỗi gõ nhầm); chỉ ghép cặp val/test cùng giao thức. | [12_b0_reconciliation.md](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/12_revision6/12_b0_reconciliation.md) |
| **Tỉ lệ vết nứt mảnh** | Phân tích hai định nghĩa ở mức ảnh gốc (Macro-average vs Micro-average gộp pixel), báo cáo 95% CI và $p$-value; thay "bác bỏ hoàn toàn" bằng "không phát hiện khác biệt (công suất thấp với 50 và 148 ảnh gốc, MDD = 8.96%)". | [12_thin_crack.md](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/12_revision6/12_thin_crack.md) |
| **Cơ chế D, R và P3** | Mô tả chính xác forward của D (SAGE bị bypass nhưng router/adapters vẫn nằm trong bộ nhớ; D = canonical + nhánh dày 6.71M tham số); giải thích R giảm đúng 88,776 tham số do thay 8 Router; in trạng thái P3 = None (0 tham số). | [12_variant_d_r_p3.md](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/12_revision6/12_variant_d_r_p3.md) |
| **Kiểm định ghép cặp trên test** | Rà soát độ dài mảng per_image_dice (chỉ LOCKED có 1,124; ba mô hình còn lại bằng 0); xác nhận chưa chạy suy luận mới trên test; đổi tuyên bố ở §7B thành "chưa kiểm định ghép cặp; chưa đo nhiễu seed trên test". | [12_test_paired_inference.md](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/12_revision6/12_test_paired_inference.md) |
| **Thiết kế Phase 6 chuẩn hóa** | Xóa tên dataset công khai không kiểm chứng, thay bằng "dataset tường riêng của người dùng"; giữ quy tắc Pareto ưu tiên B0; yêu cầu bắt buộc chạy ít nhất 2 seed hoặc cluster bootstrap theo tường (`wall_id`). | [12_phase6_design.md](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/12_revision6/12_phase6_design.md) |

---

## 3. Lịch Sử Đánh Giá Baseline B0 Và Ghép Cặp Val / Test Chuẩn Giao Thức

Được đối chiếu và xác minh trực tiếp từ các tệp nguồn gốc (`results/B0_Crack500_Run3_Results.md`, `results/baselines/50ep/B0_Crack500/official_eval_test.json`, `results/baselines/100ep/B0_Crack500/official_eval_all_splits.json`):

| Mô hình / Lần chạy | Checkpoint | Ngân sách Epoch tối đa (Patience) | Epoch tốt nhất (Best Epoch) | Giao thức đo Val | Giao thức đo Test | Val Dice | Test Dice | Khoảng cách Val $\to$ Test | Đường dẫn tệp nguồn đối chứng |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **B0 Run 3** (Setting A) | `best_model_b0.pth` | 30 epochs (Patience = 6) | Epoch 8 (dừng ở ep 14) | **Setting A** (stride 448) | **Setting A** (stride 448) | **0.7318** | **0.6771** | **+5.47 điểm** | `results/B0_Crack500_Run3_Results.md` |
| **B0 Run 3** (Setting B) | `best_model_b0.pth` | 30 epochs (Patience = 6) | Epoch 8 (dừng ở ep 14) | **Setting B** (stride 224 overlap) | **Setting B** (stride 224 overlap) | **0.7360** | **0.6801** | **+5.59 điểm** | `results/B0_Crack500_Run3_Results.md` |
| **B0 50ep** | `best_model_b0.pth` | 50 epochs | Epoch 41 (chạy đủ 50 ep) | **direct** (1-pass resize 448) | **direct** (1-pass resize 448) | **0.7789** | **0.7042** | **+7.47 điểm** | `results/baselines/50ep/B0_Crack500/official_eval_test.json` |
| **B0 100ep** | `best_model_b0.pth` | 100 epochs | Epoch 45 (dừng ở ep 60) | **direct** (1-pass resize 448) | **direct** (1-pass resize 448) | **0.7806** | **0.7037** | **+7.69 điểm** | `results/baselines/100ep/B0_Crack500/official_eval_all_splits.json` |
| **B2 Canonical run2** | Epoch 16 | Stage 2 (16 ep) | Epoch 16 | **direct** (1-pass resize 448) | **direct** (1-pass resize 448) | **0.7793** | **0.7149** | **+6.44 điểm** | `results/phase7_combination/phase7_s2g_new_pipeline_run2/` |
| **B2 LOCKED** | Ensemble 3 seed | Stage 2 | — | **TTA flip_hv** | **TTA flip_hv** | **0.7853** | **0.7158** | **+6.96 điểm** | `results/final_protocol/05_test_final_report.json` |

---

## 4. Cấu Trúc Tham Số Thực Tế Và Đo Lường Gradient

Trích xuất từ kiểm tra đối tượng bộ nhớ `id(p)` và phân rã quyền sở hữu trong mã nguồn:

```
TỔNG THAM SỐ TOÀN MÔ HÌNH B2 CANONICAL: 10,205,755 tham số
├── (a) Backbone Pretrain (ConvNeXt-Femto + 4 khối ViT) : 6,626,928 tham số (64.93%)
│       └── expert_pool: 100% trỏ vào các khối này (0 tham số độc lập)
├── (b) Thành phần riêng của SAGE (8 Router + SA-Hub)   :   738,216 tham số ( 7.23%)
│       ├── 8 SageRouter (query/noise/shared gates)     :    88,776 tham số ( 0.87%)
│       ├── SA-Hub Channel & Spatial Adapters           :   649,440 tham số ( 6.36%)
│       └── P3 Refinement                               :         0 tham số ( 0.00% - Không kích hoạt)
└── (c) Decoder + Head + S2-Gate + Projections           : 2,840,611 tham số (27.83%)
```

- **Đo lường Gradient Norm trên Variant N (`residual_scale=0.0`):**
  - Nhóm (a) Backbone: Grad Norm = **0.0246** (Nhận gradient bình thường qua luồng chính `main_block`).
  - Nhóm (c) Decoder + Head + S2-Gate: Grad Norm = **0.0683** (Nhận gradient bình thường).
  - Nhóm (b) SAGE riêng biệt: Grad Norm = **0.0000** (Hoàn toàn không nhận gradient do đầu ra nhánh chuyên gia bị triệt tiêu bằng 0).
- **Kết luận kiến trúc:** Mô hình Variant N khi tắt SAGE vẫn tối ưu hóa bình thường trên **9,467,539 tham số (92.77% mô hình)**. Do đó, việc Variant N đạt Val Dice 0.7780 là do 92.77% kiến trúc chính đảm nhiệm, không phải do mô hình bị đóng băng dung lượng.

---

## 5. Đối Chiếu Kết Quả Bóc Tách Phase 3 Với Cả Hai Baseline Chuẩn

Trích xuất trực tiếp từ các tệp `run_metadata.json` của Phase 2 và Phase 3:

| Cấu hình / Biến thể | Tham số | Seed 42 Val Dice | Seed 43 Val Dice | Trung bình 2 seed | $\Delta$ vs Baseline 2 seed (0.781670) | $\Delta$ vs Baseline 3 seed (0.778638) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Canonical Phase 2 (Seed 42, 43)** | 10,205,755 | 0.780409 | 0.782932 | **0.781670** | **Chuẩn đối chứng 2 seed** | $+0.30$ điểm |
| **Canonical Phase 2 (3 seeds: 42, 43, 44)** | 10,205,755 | — | — | **0.778638** | $-0.30$ điểm | **Chuẩn đối chứng 3 seed** |
| **Variant D** (DenseAdapter MLP) | 16,918,915 | 0.773668 | 0.782093 | **0.777881** | **-0.38 điểm** (-0.3790đ) | **-0.08 điểm** (-0.0757đ) |
| **Variant R** (RandomRouter) | 10,116,979 | 0.779086 | 0.777913 | **0.778499** | **-0.32 điểm** (-0.3171đ) | **-0.01 điểm** (-0.0138đ) |
| **Variant N** (`residual_scale=0`) | 10,205,755 | 0.778212 | 0.777812 | **0.778012** | **-0.37 điểm** (-0.3658đ) | **-0.06 điểm** (-0.0625đ) |

> [!NOTE]
> - Cả ba biến thể D, R, N đều có điểm Val trung bình thấp hơn baseline 2 seed từ **−0.32 đến −0.38 điểm**.
> - Tuy nhiên, do độ phân tán giữa các seed lên tới 0.84 điểm (ở Variant D) và kích thước mẫu chỉ có $n=2$ seed, phương sai lấy mẫu seed không cho phép kết luận mức chênh lệch 0.3–0.4 điểm này là hiệu ứng thực sự của kiến trúc.

---

## 6. Phân Rã Khoảng Cách Val $\to$ Test Và Phân Tích Vết Nứt Mảnh

1. **Phân rã Oaxaca-Blinder trên khoảng cách Val $\to$ Test (6.35 điểm):**
   - **Within effect:** **6.12 điểm (96.3%)** — Mô hình bị suy giảm điểm số đồng đều ngay bên trong từng khoảng độ rộng vết nứt (vết vừa 3–10px giảm 5.85 điểm; vết rộng >10px giảm 6.27 điểm).
   - **Mix effect:** **0.23 điểm (3.7%)** — Phần chênh lệch do sai khác tỉ lệ phân bố giữa các nhóm vết nứt.
2. **Kiểm định tỉ lệ vết nứt mảnh ở mức ảnh gốc độc lập:**
   - **Macro-average (theo thành phần liên thông):** Val = **27.69%** [22.78%, 32.76%] vs Test = **33.88%** [30.18%, 37.61%] (chênh lệch $+6.19\%$, Mann-Whitney U: $p = 0.1408$, Welch's t-test: $p = 0.0557$).
   - **Macro-average (theo pixel):** Val = **0.136%** vs Test = **0.205%** (Mann-Whitney U: $p = 0.0519$, t-test: $p = 0.2824$).
   - **Đánh giá công suất:** Độ chênh nhỏ nhất phát hiện được (MDD ở power 80%) là **8.96 điểm %** đối với thành phần và **0.18 điểm %** đối với pixel. Vì chênh lệch thực tế nhỏ hơn MDD, kiểm định **không phát hiện sự khác biệt có ý nghĩa thống kê** do hạn chế về công suất của cỡ mẫu ($n_{\text{val}}=50, n_{\text{test}}=148$).

---

## 7. Các Hạn Chế Của Nghiên Cứu Và Tuyên Bố Bị Nghiêm Cấm

### A. Hạn chế nghiên cứu đã được xác minh
1. **Dữ liệu kiểm tra lịch sử không đầy đủ:** Các tệp kết quả test của B0 50ep, B0 100ep và Canonical direct không lưu mảng Dice theo từng ảnh, khiến việc thực hiện kiểm định ghép cặp phi tham số trên test chưa thể tiến hành được nếu không chạy lại suy luận trên test.
2. **Thiếu công suất thống kê trên tập Val:** Kích thước tập Val chỉ có 50 ảnh gốc khiến khoảng tin cậy 95% có độ rộng tới ~5.7 điểm, không có khả năng phân giải các cải tiến kiến trúc nhỏ ($< 1.0$ điểm).
3. **Số lượng seed hạn chế:** Các biến thể Phase 3 chỉ được huấn luyện trên 2 seed, dẫn đến phương sai lấy mẫu seed cao (tới 0.84 điểm), không đủ để đo lường Minimum Detectable Effect ở mức dưới 1 điểm.

### B. Các tuyên bố bị nghiêm cấm đưa ra
- **KHÔNG ĐƯỢC TUYÊN BỐ:** *"SAGE-Lite vượt trội hơn Baseline B0 một cách có ý nghĩa thống kê"* (Hiệu số +1.12 điểm trên test **chưa được kiểm định ghép cặp**; đồng thời **chưa có số liệu đo lường phương sai giữa các seed trên test direct** do chỉ có một lần chạy direct duy nhất).
- **KHÔNG ĐƯỢC TUYÊN BỐ:** *"Router học được cơ chế phân công chuyên gia chuyên biệt"* (Variant R định tuyến ngẫu nhiên đạt kết quả tương đương).
- **KHÔNG ĐƯỢC TUYÊN BỐ:** *"Variant D là kiến trúc cùng dung lượng với Canonical"* (Variant D có 16.92M tham số, là canonical cộng thêm nhánh dày DenseAdapter 6.71M tham số).
- **KHÔNG ĐƯỢC TUYÊN BỐ:** *"Bác bỏ hoàn toàn giả thuyết vết nứt mảnh"* (Phải diễn đạt đúng: không phát hiện khác biệt có ý nghĩa thống kê ở mức ảnh gốc do công suất thống kê thấp với 50 và 148 ảnh gốc, MDD = 8.96%).
- **KHÔNG ĐƯỢC TUYÊN BỐ:** *"Thứ tự Val và Test giữa b1_b2_end_to_end và run2 bị đảo ngược do mô hình suy giảm"* (Val của hai run đo bằng hai giao thức khác nhau Setting A vs Direct).

---

## 8. Khung Thực Nghiệm Cho Giai Đoạn Đánh Giá Trên Dataset Tường Riêng (Phase 6)

Để đánh giá thực chất năng lực tổng quát hóa của kiến trúc, giai đoạn tiếp theo (Phase 6) được thiết lập như sau:

1. **Dữ liệu mục tiêu:** Đánh giá trên **dataset tường riêng của người dùng** (gắn mã định danh theo từng bức tường `wall_id`).
2. **Ba mô hình so sánh đối đầu:**
   - **Baseline B0 (100 epochs):** Mô hình mạng cơ sở nhẹ, không có SAGE router hay chuyên gia.
   - **Canonical B2:** Mô hình đầy đủ gồm Backbone Femto + 4 ViT + SAGE router + S2-Gate.
   - **Variant N (B2 không SAGE):** Mô hình giữ nguyên Backbone Femto + 4 ViT + S2-Gate nhưng triệt tiêu hoàn toàn nhánh SAGE (`residual_scale = 0.0`).
3. **Ba kịch bản chuyển giao:**
   - **Zero-shot Transfer:** Nạp trực tiếp trọng số từ Crack500 để đo đạc trực tiếp trên dataset tường riêng.
   - **Fine-tuning:** Tinh chỉnh mô hình trên một tập huấn luyện nhỏ thuộc dataset tường riêng với cùng lịch trình siêu tham số.
   - **Huấn luyện từ ImageNet Pretrain:** Huấn luyện từ đầu trực tiếp trên dataset tường riêng với trọng số khởi tạo từ ImageNet.
4. **Tiêu chuẩn thống kê và Nguyên tắc Pareto:**
   - Mọi so sánh bắt buộc phải dùng **ít nhất 2 seed độc lập** HOẶC áp dụng **Cluster Bootstrap theo từng bức tường (`wall_id`)** để kiểm soát tương quan cụm đối với cỡ mẫu nhỏ.
   - **Nguyên tắc quyết định Pareto:** Nếu Baseline B0 đạt hiệu năng tương đương Canonical B2 trên dataset tường riêng của người dùng, **Baseline B0 bắt buộc phải được ưu tiên lựa chọn làm kiến trúc tối ưu cho triển khai thực tế** nhằm tiết kiệm tối đa tài nguyên tính toán và đơn giản hóa hệ thống.
