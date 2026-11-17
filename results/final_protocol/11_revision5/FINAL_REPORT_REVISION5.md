# FINAL_REPORT_REVISION5.md — Báo Cáo Nghiên Cứu SAGE-Lite Trên Crack500 (Revision 5 - Bản Chốt)

**Ngày lập:** 2026-11-17  
**Phiên bản:** Revision 5 (Bản chốt toàn diện)  
**Quy ước đo lường:** Đơn vị đo điểm phần trăm ($0.0009 \text{ Dice} = 0.09 \text{ điểm}$).  
**Nguyên tắc cốt lõi:** Báo cáo bằng tiếng Việt, ngắn gọn, số liệu có căn cứ mã nguồn và tệp log đối chứng, loại bỏ hoàn toàn các đánh giá định tính chủ quan.

---

## 1. Tóm Tắt Kết Luận Cuối Cùng (Executive Summary)

1. **Hiệu năng kiểm định trên tập Test:** Mô hình SAGE-Lite B2 đạt điểm Test Direct là **0.7149** (`official_direct_evaluation.json`) và khi áp dụng cấu hình LOCKED (Ensemble 3 seed + TTA flip_hv) đạt **0.7158** (`05_test_final_report.json`), trong khi mô hình baseline B0 huấn luyện 100 epoch đạt **0.7037** (`results/baselines/100ep/B0_Crack500/official_eval_all_splits.json`). Hiệu số giữa B0 100ep và B2 canonical (+1.12 điểm) cũng như mức tăng của LOCKED (+0.09 điểm) hiện **chưa thể kiểm định ghép cặp (paired cluster bootstrap)** do các bản ghi lịch sử của B0 và Canonical direct chỉ lưu chỉ số tổng hợp mà không lưu mảng Dice theo từng ảnh của tập Test, đồng thời quy tắc nghiên cứu nghiêm cấm suy luận mới trên test.
2. **Cấu trúc tham số thực tế:** Toàn bộ `expert_pool` trong mô hình thực chất là con trỏ tham chiếu (view/reference) trỏ trực tiếp vào các khối `main_block` của Backbone. Mô hình B2 gồm **10,205,755 tham số** (`sage_lite/scripts/train_phase3_ablation.py`), trong đó **92.77% tham số** (Backbone 64.93% + Decoder/S2-Gate 27.83%) hoạt động bình thường và nhận gradient đầy đủ khi huấn luyện; chỉ có **7.23% (738,216 tham số)** thuộc về các thành phần riêng của SAGE (8 Router và SA-Hub adapters).
3. **Kết quả thử nghiệm bóc tách (Phase 3 Ablation):** Việc triệt tiêu hoàn toàn nhánh expert (Variant N, `residual_scale=0.0`) hay bổ sung thêm một nhánh dày DenseAdapter gồm 6.71M tham số (Variant D, tổng 16.92M tham số) đều cho kết quả Val Dice quanh mức **0.778 – 0.779** (chênh lệch so với Canonical B2 chỉ từ $-0.09$ đến $-0.03$ điểm trung bình). Mọi chênh lệch giữa các biến thể này đều bị chìm ngập hoàn toàn bởi độ phân tán ngẫu nhiên giữa các seed (lên tới 0.84 điểm ở Variant D) và sai số lấy mẫu cụm của tập Val (khoảng tin cậy 95% có độ rộng tới **5.74 điểm** do tập Val chỉ có 50 ảnh gốc độc lập).
4. **Bản chất khoảng cách Val $\to$ Test (6.35 – 6.96 điểm):** Phân rã Oaxaca-Blinder chứng minh rằng **Within effect chiếm 96.3% – 96.6%** khoảng cách (do mô hình suy giảm hiệu năng đồng đều trên tất cả các nhóm vết nứt), trong khi **Mix effect chỉ đóng góp 3.4% – 3.7%**. Tỉ lệ vết nứt mảnh giữa Val và Test là hoàn toàn tương đương ($p = 0.7830$ ở mức crop, $p = 0.4139$ ở mức ảnh gốc), bác bỏ hoàn toàn giả thuyết cho rằng khoảng cách Val $\to$ Test là do phân bố vết nứt mảnh gây ra.
5. **Đánh giá tổng thể:** Mô hình SAGE-Lite không chứng minh được sự vượt trội có ý nghĩa thống kê so với baseline B0 khi B0 được huấn luyện đủ ngân sách (100 epoch), và các thành phần định tuyến riêng biệt của SAGE không mang lại đóng góp gia tăng đo lường được trên Crack500.

---

## 2. Bảng Theo Dõi Thay Đổi So Với Revision 4 Và Danh Mục Tệp Chi Tiết

| Nội dung thay đổi | Bản chất hiệu chỉnh trong Revision 5 | Tệp tài liệu chi tiết tương ứng |
|---|---|---|
| **Kiểm định ghép cặp trên test** | Xác nhận B0 100ep và Canonical direct không lưu per-image test dice; đổi từ "nằm trong CI" thành "chưa kiểm định được". | [11_test_paired_test.md](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/11_revision5/11_test_paired_test.md) |
| **Khoảng tin cậy Spearman & Vết mảnh** | Tính CI 95% Spearman cho Val ($n=50$) và Test ($n=148$); chứng minh CI chồng lấp; bác bỏ khoảng cách do vết mảnh bằng tỉ lệ phân bố ($p=0.78$) và Oaxaca-Blinder; bỏ lập luận "val không tương quan nên test cũng không". | [11_spearman_ci.md](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/11_revision5/11_spearman_ci.md) |
| **Cấu trúc Variant D & P3** | Xác định Variant D là Canonical cộng thêm nhánh DenseAdapter 6.71M tham số (tổng 16.92M params); xác nhận P3 có 0 tham số và không được kích hoạt trên mọi biến thể. | [11_variant_d_and_p3.md](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/11_revision5/11_variant_d_and_p3.md) |
| **Sai số lấy mẫu cụm tập Val** | Báo cáo khoảng tin cậy 95% của Val Dice (độ rộng 5.74 điểm); chứng minh chênh lệch biến thể Phase 3 bị triệt tiêu bởi sai số lấy mẫu Val và nhiễu seed. | [11_val_sampling_uncertainty.md](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/11_revision5/11_val_sampling_uncertainty.md) |
| **`b1_b2_end_to_end` vs `run2`** | Liệt kê khác biệt cấu hình giữa hai bên; giải thích Test Setting A chênh 1.21 điểm nhưng chưa thể gán nguyên nhân; xác nhận Val và Test ranking không thể so sánh vì Val đo khác giao thức. | [11_b1_b2_vs_run2.md](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/11_revision5/11_b1_b2_vs_run2.md) |

---

## 3. Lịch Sử Đánh Giá Baseline B0 Và Ghép Cặp Val / Test Đúng Giao Thức

Được khôi phục từ các bản ghi nguồn gốc (`results/B0_Crack500_Run3_Results.md`, `results/baselines/50ep/B0_Crack500/official_eval_test.json`, `results/baselines/100ep/B0_Crack500/official_eval_all_splits.json`):

| Mô hình / Lần chạy | Checkpoint | Ngân sách huấn luyện | Val Dice (Giao thức) | Test Dice (Giao thức) | Khoảng cách Val $\to$ Test | Nguồn xác minh |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **B0 Run 3** | Epoch 8 | 50 epochs | **0.7318** (Setting A) | **0.6771** (Setting A)<br>**0.6801** (Setting B) | **+5.47 điểm** (Setting A)<br>**Không ghép chéo** | `results/B0_Crack500_Run3_Results.md` |
| **B0 50ep** | Best epoch | 50 epochs | Không có | **0.6974** (direct 1-pass) | Không có cặp cùng giao thức | `results/baselines/50ep/B0_Crack500/official_eval_test.json` |
| **B0 100ep** | Best epoch | 100 epochs | **0.7806** (direct 1-pass) | **0.7037** (direct 1-pass) | **+7.69 điểm** (direct) | `results/baselines/100ep/B0_Crack500/official_eval_all_splits.json` |
| **B2 Canonical run2** | Epoch 16 | Stage 2 (16 ep) | **0.7793** (direct 1-pass) | **0.7149** (direct 1-pass)<br>0.6895 (Setting A)<br>0.6950 (Setting B) | **+6.44 điểm** (direct) | `results/phase7_combination/phase7_s2g_new_pipeline_run2/` |
| **B2 LOCKED** | Ensemble 3 seed | Stage 2 | **0.7853** (TTA flip_hv) | **0.7158** (TTA flip_hv) | **+6.96 điểm** (LOCKED) | `results/final_protocol/05_test_final_report.json` |

### Phân tích so sánh B0 100ep vs B2 Canonical:
- Khi so sánh trên **cùng giao thức direct 1-pass**:
  - Điểm Val: B0 100ep đạt **0.7806**, cao hơn B2 Canonical (**0.7793**) là $+0.13$ điểm.
  - Điểm Test: B0 100ep đạt **0.7037**, B2 Canonical đạt **0.7149** (chênh lệch $+1.12$ điểm).
  - Khoảng cách Val $\to$ Test: B0 100ep có khoảng cách là **7.69 điểm**, trong khi B2 Canonical là **6.44 điểm** (chênh lệch gap là 1.25 điểm).

---

## 4. Cấu Trúc Tham Số Thực Tế Và Đo Lường Gradient

Trích xuất từ kiểm tra đối tượng bộ nhớ `id(p)` và đo lường trực tiếp trên mã nguồn:

```
TỔNG THAM SỐ TOÀN MÔ HÌNH B2 CANONICAL: 10,205,755 tham số
├── (a) Backbone Pretrain (ConvNeXt-Femto + 4 khối ViT) : 6,626,928 tham số (64.93%)
│       └── expert_pool: 100% trỏ vào các khối này (0 tham số độc lập)
├── (b) Thành phần riêng của SAGE (8 Router + SA-Hub)   :   738,216 tham số ( 7.23%)
│       └── P3 Refinement                               :         0 tham số ( 0.00% - Không kích hoạt)
└── (c) Decoder + Head + S2-Gate + Projections           : 2,840,611 tham số (27.83%)
```

- **Đo lường Gradient Norm trên Variant N (`residual_scale=0.0`):**
  - Nhóm (a) Backbone: Grad Norm trung bình = **0.0246** (Nhận gradient lan truyền ngược đầy đủ qua luồng chính `main_block`).
  - Nhóm (c) Decoder + Head + S2-Gate: Grad Norm trung bình = **0.0683** (Nhận gradient bình thường).
  - Nhóm (b) SAGE riêng biệt (Router + SA-Hub): Grad Norm = **0.0000** (Hoàn toàn không nhận gradient do đầu ra expert bị triệt tiêu).
- **Kết luận kiến trúc:** Mô hình Variant N khi tắt SAGE vẫn tối ưu hóa bình thường trên **9,467,539 tham số (92.77% mô hình)**. Do đó, việc Variant N đạt Val Dice 0.7780 là do 92.77% kiến trúc chính đảm nhiệm, không phải mô hình bị đóng băng dung lượng.

---

## 5. Đối Chiếu Kết Quả Bóc Tách Phase 3 Theo Từng Seed

Trích xuất trực tiếp từ các tệp `run_metadata.json` gốc:

| Cấu hình / Biến thể | Tổng số tham số | Seed 42 Val Dice | Seed 43 Val Dice | Trung bình (Mean) | Chênh lệch 2 seed | Hiệu số so với Canonical |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Canonical B2** | 10,205,755 | 0.7793 | 0.7783 | **0.7788** | 0.10 điểm | Chuẩn đối chứng |
| **Variant D** (DenseAdapter) | **16,918,915** (+65.8%) | 0.7737 | 0.7821 | **0.7779** | **0.84 điểm** | **-0.09 điểm** |
| **Variant R** (RandomRouter) | 9,824,315 (-3.7%) | 0.7791 | 0.7779 | **0.7785** | 0.12 điểm | **-0.03 điểm** |
| **Variant N** (`residual_scale=0`) | 10,205,755 | 0.7782 | 0.7778 | **0.7780** | 0.04 điểm | **-0.08 điểm** |

> [!NOTE]
> - Variant D được bổ sung thêm **6.71M tham số** (tổng 16.92M params) nhưng điểm Val trung bình (0.7779) không vượt qua Canonical B2 (0.7788).
> - Mọi chênh lệch giữa các biến thể ($\le 0.09$ điểm trung bình) đều nhỏ hơn nhiều so với độ phân tán giữa các seed (0.84 điểm) và độ rộng khoảng tin cậy của tập Val (5.74 điểm). Thử nghiệm Phase 3 hoàn toàn không đủ công suất thống kê để tách biệt hiệu ứng kiến trúc khỏi nhiễu ngẫu nhiên.

---

## 6. Phân Rã Khoảng Cách Val $\to$ Test Và Kiểm Định Tương Quan

1. **Phân rã Oaxaca-Blinder trên khoảng cách Val $\to$ Test (6.35 điểm của Canonical direct):**
   - **Within effect:** **6.12 điểm (96.3%)** — Mô hình bị suy giảm điểm số đồng đều ngay bên trong từng khoảng độ rộng vết nứt (vết vừa 3–10px giảm 5.85 điểm; vết rộng >10px giảm 6.27 điểm).
   - **Mix effect:** **0.23 điểm (3.7%)** — Phần chênh lệch do sai khác tỉ lệ phân bố giữa các nhóm vết nứt.
2. **Kiểm định phân bố vết nứt mảnh giữa Val và Test:**
   - Tỉ lệ vết nứt mảnh ở cấp độ crop: Val = $0.1470$ vs Test = $0.1455$ ($p = 0.7830$ theo Mann-Whitney U cụm).
   - Tỉ lệ vết nứt mảnh ở cấp độ ảnh gốc: Val = $0.1719$ vs Test = $0.2063$ ($p = 0.4139$ theo Mann-Whitney U hai phía).
   - **Bác bỏ:** Khoảng cách Val $\to$ Test không bắt nguồn từ việc tập Test có nhiều vết mảnh hơn tập Val.
3. **Khoảng tin cậy tương quan rank Spearman ở cấp ảnh gốc:**
   - Tỉ lệ pixel nứt (`crack_ratio`): Val $\rho = 0.522$ [95% CI: 0.270, 0.708] vs Test $\rho = 0.497$ [95% CI: 0.356, 0.615] $\to$ **Chồng lấp hoàn toàn**.
   - Độ rộng trung bình (`mean_width`): Val $\rho = 0.490$ [95% CI: 0.229, 0.685] vs Test $\rho = 0.371$ [95% CI: 0.214, 0.510] $\to$ **Chồng lấp hoàn toàn**.
   - Tỉ lệ vết mảnh (`thin_crack_ratio`): Val $\rho = -0.096$ [95% CI: -0.379, 0.204] vs Test $\rho = -0.298$ [95% CI: -0.446, -0.134] $\to$ **Chồng lấp trong đoạn [-0.379, -0.134]**.

---

## 7. Các Hạn Chế Của Nghiên Cứu Và Tuyên Bố Bị Nghiêm Cấm

### A. Hạn chế nghiên cứu đã được xác minh
1. **Dữ liệu kiểm tra lịch sử không đầy đủ:** Các tệp kết quả test của B0 50ep, B0 100ep và Canonical direct không lưu mảng Dice theo từng ảnh, khiến việc thực hiện kiểm định ghép cặp phi tham số trên test không thể tiến hành được nếu không vi phạm quy tắc cấm suy luận mới trên test.
2. **Thiếu công suất thống kê trên tập Val:** Kích thước tập Val chỉ có 50 ảnh gốc khiến khoảng tin cậy 95% có độ rộng tới ~5.7 điểm, khiến tập Val không có khả năng phân giải các cải tiến kiến trúc nhỏ ($< 1.0$ điểm).
3. **Số lượng seed hạn chế:** Các biến thể Phase 3 chỉ được huấn luyện trên 2 seed, dẫn đến phương sai lấy mẫu seed cao (tới 0.84 điểm), không đủ để đo lường Minimum Detectable Effect ở mức dưới 1 điểm.

### B. Các tuyên bố bị nghiêm cấm đưa ra
- **KHÔNG ĐƯỢC TUYÊN BỐ:** *"SAGE-Lite vượt trội hơn Baseline B0 một cách có ý nghĩa thống kê"* (chưa kiểm định ghép cặp được; hiệu số +1.12 điểm trên test nằm trong biên độ dao động giữa các lần chạy).
- **KHÔNG ĐƯỢC TUYÊN BỐ:** *"Router học được cơ chế phân công chuyên gia chuyên biệt"* (Variant R định tuyến ngẫu nhiên và Variant N triệt tiêu router đạt kết quả tương đương).
- **KHÔNG ĐƯỢC TUYÊN BỐ:** *"Variant D là kiến trúc cùng dung lượng với Canonical"* (Variant D có 16.92M tham số, tăng thêm 6.71M tham số so với Canonical).
- **KHÔNG ĐƯỢC TUYÊN BỐ:** *"Khoảng cách Val $\to$ Test là do tập test chứa nhiều vết nứt mảnh hơn"* (đã bị bác bỏ bởi phân rã Oaxaca-Blinder 96.3% Within effect và kiểm định tỉ lệ $p = 0.78$).
- **KHÔNG ĐƯỢC TUYÊN BỐ:** *"Thứ tự Val và Test giữa b1_b2_end_to_end và run2 bị đảo ngược do mô hình suy giảm"* (Val của hai run đo bằng hai giao thức khác nhau Setting A vs Direct).

---

## 8. Gợi Ý Thiết Kế Thử Nghiệm Cho Giai Đoạn Tiếp Theo (Phase 6 Recommendation)

Để đánh giá thực chất năng lực tổng quát hóa của kiến trúc ngoài tập dữ liệu Crack500, giai đoạn tiếp theo (Phase 6) nên được thiết kế như sau:

1. **Tập dữ liệu mục tiêu:** Đánh giá trên tập dữ liệu ảnh chụp nứt tường thực tế độc lập (chẳng hạn như Concrete Crack Conglomerate hoặc tập dữ liệu tường tự thu thập), phản ánh đúng bài toán kiểm định công trình.
2. **Các mô hình đưa vào so sánh đối đầu:**
   - **Baseline B0 (100 epochs):** Kiến trúc chuẩn nhẹ, không có SAGE.
   - **Canonical B2:** Kiến trúc đầy đủ gồm Backbone + SAGE router + S2-Gate.
   - **Variant N (B2 không SAGE):** Giữ nguyên Backbone + S2-Gate nhưng tắt hoàn toàn nhánh SAGE (`residual_scale=0.0`).
3. **Ba kịch bản đánh giá bắt buộc:**
   - **Zero-shot Transfer:** Nạp trực tiếp trọng số đã huấn luyện trên Crack500 để đo đạc trực tiếp trên tập dữ liệu tường mới mà không cập nhật trọng số.
   - **Fine-tuning:** Tinh chỉnh mô hình trên một tập huấn luyện nhỏ của ảnh tường mới (với cố định cùng số epoch và cùng learning rate scheduler trên cả 3 mô hình).
   - **Huấn luyện từ đầu (Train from ImageNet pretrain):** Huấn luyện cả 3 mô hình từ đầu trực tiếp trên tập dữ liệu tường mới.
4. **Nguyên tắc Pareto về chi phí tính toán:**
   - Baseline B0 có chi phí tính toán (FLOPs/VRAM) và số lượng tham số thấp hơn đáng kể so với B2.
   - **Quy tắc quyết định:** Nếu Baseline B0 đạt hiệu năng tương đương hoặc chỉ chênh lệch không đáng kể so với Canonical B2 trên tập dữ liệu tường thực tế, **Baseline B0 phải được chọn làm kiến trúc tối ưu cho triển khai thực tế** nhằm tiết kiệm tài nguyên tính toán và đơn giản hóa hệ thống.
