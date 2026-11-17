# BÁO CÁO TỔNG HỢP SỬA ĐỔI — SAGE-Lite trên Crack500

**Ngày:** 2026-11-17  
**Phiên bản:** Revision 4  
**Thay thế:** Revision 3 (`09_revision3/FINAL_REPORT_REVISION3.md`)  
**Thư mục lưu trữ:** `results/final_protocol/10_revision4/`  
**Nguyên tắc:** Mọi con số đều trích từ file nguồn hoặc log thực nghiệm, kèm đường dẫn xác thực. Đơn vị đo độ chênh Dice: điểm phần trăm (0.0009 Dice = 0.09 điểm). Không khen ngợi.

---

## BẢNG SO SÁNH KẾT LUẬN REVISION 3 VÀ REVISION 4

| Mục / Luận điểm | Kết luận trong Revision 3 | Sửa đổi & Chuẩn hóa trong Revision 4 | Căn cứ & Bằng chứng kiểm tra thực tế |
|---|---|---|---|
| **Cấu trúc tham số SAGE (§5.1)** | "Expert pool chiếm 6.62M (64.91%), phần còn lại 3.49M" | Expert pool không có tham số riêng mà trỏ trực tiếp đến `main_block` của Backbone. Backbone chiếm **6,626,928** (64.93%), SAGE riêng chiếm **738,216** (7.23%), Decoder/Head/Projections chiếm **2,840,611** (27.83%). Tổng khớp **10,205,755** | [`10_param_structure.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/10_revision4/10_param_structure.md), `sage/networks/sage_injection.py:102-120` |
| **Bản chất Variant N (§5.2)** | "Variant N đóng băng 65.78% tham số, chỉ tối ưu trên 3.49M params" | Backbone (6.63M) và Decoder (2.84M) vẫn nhận gradient đầy đủ (**92.77% mô hình được train**). Chỉ có 738,216 params (7.23%) của Router và SA-Hub bị đóng băng. Variant N là **"B2 không SAGE"** trong pipeline mới | [`10_param_structure.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/10_revision4/10_param_structure.md), thực nghiệm đo Grad Norm sau 1 batch |
| **Số liệu từng seed Phase 3 (§5.3)** | Đưa vào số liệu làm tròn/điều chỉnh (D: 0.7775/0.7783; R: 0.7788/0.7782) | Khôi phục 100% số liệu gốc: Variant D (0.773668 / 0.782093, độ lệch seed **0.84 điểm**); Variant R (0.779086 / 0.777913); Variant N (0.778212 / 0.777812) | [`10_phase3_reconcile.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/10_revision4/10_phase3_reconcile.md), `phase3_variant*/run_metadata.json` |
| **So sánh B0 vs B2 (§1, §3)** | Nêu B0 100ep nhưng chưa phân tích ngân sách và tính tương đồng giao thức | Xác minh B0 50ep/100ep dùng cùng giao thức Direct. B0 100ep đạt Val 0.7806 (> B2 canonical 0.7793). Chênh lệch Test (+1.12 điểm) nằm trọn trong khoảng tin cậy cluster của Test [0.7005, 0.7307] | [`10_b0_vs_b2.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/10_revision4/10_b0_vs_b2.md), `SAGE_LITE/scripts/evaluate_crack_official.py:113-118` |
| **Phân rã Oaxaca-Blinder (§6.3)** | Dùng cặp Single-seed (Val 0.7793 vs Test 0.7158) | Chạy suy luận Val cho đúng cấu hình **LOCKED** (Ensemble 3 seed + TTA flip_hv, Val = **0.7853**). Phân rã cho thấy Within effect chiếm **96.6% (6.72 điểm)**, Mix effect chiếm **3.4% (0.24 điểm)** | [`10_gap_decomposition.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/10_revision4/10_gap_decomposition.md), `val_locked_per_image_dice.csv` |
| **Thiên lệch chọn epoch (§4)** | Dùng con số 1.60 điểm (trung bình toàn bộ 18 epoch) | Con số 1.60 điểm gộp cả các epoch đầu chưa hội tụ. Con số chuẩn phản ánh thiên lệch chọn epoch là **0.22 điểm** (so với trung bình 5 epoch cuối quanh vùng hội tụ) | [`09_selection_bias.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/09_revision3/09_selection_bias.md), `10_gap_decomposition.md` |
| **Cấu hình run1 vs run2 vs b1_b2_e2e (§1)** | Cho rằng b1_b2_e2e và run1/run2 xếp hạng ngược chiều cùng cấu hình | Điểm Val 0.7689 của b1_b2_e2e đo bằng Setting A, còn run1/run2 đo bằng Direct. run1 và run2 dùng cùng pipeline, chênh lệch 0.10–0.14 điểm thuần túy là nhiễu ngẫu nhiên giữa các lần chạy | [`10_config_diff.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/10_revision4/10_config_diff.md), `results/phase7_combination/*/train.log` |
| **Mã ảnh gốc Crack500 (§2)** | Nêu "do cách thức đóng gói local" làm thiếu 52 ảnh test | Khảo sát 5 định nghĩa mã ảnh gốc: Train (250) và Val (50) khớp 100% bài báo gốc; Test chỉ có 148 ảnh gốc độc lập. Nguyên nhân thiếu 52 ảnh là "chưa giải thích được" | [`10_source_id.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/10_revision4/10_source_id.md) |
| **Thuật ngữ kiến trúc Decoder (§5.1)** | "Transformer Decoder" | Đính chính: **UNet decoder (conv 3×3) với ViT bottleneck/blocks trong encoder** | `SAGE_LITE/sage/networks/convnext_transformer_unet.py:135` |

---

## TRẢ LỜI THẲNG 5 CÂU HỎI LỚN CỦA DỰ ÁN

1. **Phần riêng của SAGE có đóng góp đo được không và nó gồm những gì?**  
   **KHÔNG PHÁT HIỆN ĐÓNG GÓP ĐO ĐƯỢC.**  
   Phần riêng của SAGE gồm **8 Routers (88,776 tham số)** và **hệ thống chuyển đổi định dạng SA-Hub / Format Adapters (649,440 tham số)**, tổng cộng **738,216 tham số (chiếm 7.23% mô hình)**; mô-đun P3 refinement không được kích hoạt (0 tham số). Khi tắt hoàn toàn phần riêng của SAGE bằng `residual_scale = 0.0` (Variant N, biến mô hình thành "B2 không SAGE"), điểm Val Dice chỉ giảm **0.06 điểm** so với canonical 3-seed ($\mu = 0.7786$) và **0.37 điểm** so với canonical 2-seed ($\mu = 0.7817$), nằm sâu bên dưới ngưỡng phát hiện tối thiểu $\text{MDE} = 1.08$ điểm.
2. **B2 có vượt B0 được huấn luyện đủ ngân sách không?**  
   **CHƯA ĐƯỢC CHỨNG MINH.**  
   Khi được huấn luyện đủ ngân sách (45–60 epoch ở single-stage), mô hình đường cơ sở thuần CNN B0 (100ep) đạt Val Dice **0.7806**, cao hơn B2 Canonical (**0.7793**). Trên tập Test, B2 đạt **0.7149** so với B0 là **0.7037** (chênh lệch **+1.12 điểm**). Tuy nhiên, khoảng tin cậy dạng cụm (cluster CI 95%) của tập Test là **[0.7005, 0.7307]** (độ rộng 3.02 điểm). Mức chênh lệch +1.12 điểm nằm trọn bên trong khoảng tin cậy thống kê này.
3. **Khoảng cách Val→Test: bao nhiêu chưa giải thích được?**  
   **PHẦN LỚN KHOẢNG CÁCH (~6.7 ĐIỂM) CHƯA ĐƯỢC GIẢI THÍCH BỞI HÌNH THÁI VẾT NỨT.**  
   Khoảng cách Val→Test của cấu hình LOCKED là **6.95 điểm** (Val 0.7853 vs Test 0.7158). Phân rã Oaxaca-Blinder cho thấy sự khác biệt về phân bố tỉ lệ nhóm độ rộng (Mix effect) chỉ chiếm **0.24 điểm (3.4%)**. Khoảng cách còn lại **6.72 điểm (96.6%)** là do hiệu năng trong từng nhóm độ rộng đều sụt giảm (Within effect). Khoảng sụt giảm này gồm các yếu tố chưa tách rời được: các đặc tính hình ảnh chưa đo (độ tương phản, độ nhám mặt đường, chiếu sáng), thiên lệch chọn epoch/siêu tham số trên val (~0.22 điểm), và sai số lấy mẫu từ cỡ mẫu nhỏ 50 ảnh gốc val.
4. **Ensemble và TTA có giúp ích ở tập Test không?**  
   **KHÔNG XÁC NHẬN CẢI THIỆN ĐO ĐƯỢC.**  
   So sánh giữa mô hình đơn seed 42 không TTA (**0.7149**) và mô hình Phase 5 LOCKED ensemble 3 seed kèm TTA flip_hv (**0.7158**) chỉ ghi nhận mức tăng **+0.0009 (+0.09 điểm)**. Mức tăng này nằm trọn trong khoảng tin cậy của test [0.7005, 0.7307].
5. **Đã có số liệu đánh giá trên tập dữ liệu tường riêng chưa?**  
   **CHƯA CÓ BẤT KỲ SỐ LIỆU NÀO.**  
   Phase 6 (đánh giá cross-dataset trên ảnh tường ngoài thực tế) chưa được thực hiện do đang chờ thu thập và dán nhãn dữ liệu. Hiện chưa có bất kỳ kiểm chứng thực nghiệm nào ngoài tập Crack500.

---

## §1. TÍNH TRONG SÁNG CỦA DỮ LIỆU TEST VÀ LỊCH SỬ ĐÁNH GIÁ

Cấu hình chính thức được đóng băng tại commit `05_frozen_config.json`. Việc kiểm tra lịch sử lưu trữ xác nhận tập Test đã được truy cập nhiều lần qua các giai đoạn.

### Bảng lịch sử các lần đo Test và ghép cặp Val/Test:

| # | Checkpoint | Epoch | Giao thức suy luận | Val Dice | Test Dice | Khoảng cách (Val − Test) | File nguồn xác thực |
|---|---|:---:|---|:---:|:---:|:---:|---|
| 1 | B0 Run3 (`best_model_b0.pth`) | 8 | Setting A (stride 448) | **0.7318** | **0.6771** | **+5.47 điểm** | `results/B0_Crack500_Run3_Results.md` |
| 2 | B0 Run3 (`best_model_b0.pth`) | 8 | Setting B (stride 224) | **0.7360** | **0.6801** | **+5.59 điểm** | `results/B0_Crack500_Run3_Results.md` |
| 3 | B0 50ep (`best_model`) | 41 | Direct 1-pass, $\tau=0.5$ | **0.7789** | **0.7042** | **+7.47 điểm** | `baselines/50ep/B0_Crack500/` |
| 4 | B0 100ep (`best_model`) | 45 | Direct 1-pass, $\tau=0.5$ | **0.7806** | **0.7037** | **+7.69 điểm** | `baselines/100ep/B0_Crack500/` |
| 5 | `b1_b2_end_to_end` | 15 | Setting A (stride 448) | **0.7689** | **0.7016** | **+6.73 điểm** | `phase7_s2g_b1_b2_end_to_end/` |
| 6 | `b1_b2_end_to_end` | 15 | Setting B (stride 224) | —* | **0.7054** | — | (cùng file metadata) |
| 7 | `run1` (`phase7_new_pipeline`) | 18 | Setting A | —* | **0.6881** | — | `phase7_s2g_new_pipeline_run1/` |
| 8 | `run1` (`phase7_new_pipeline`) | 18 | Setting B | —* | **0.6950** | — | (cùng file metadata) |
| 9 | `run2` (`phase7_new_pipeline`) | 16 | Setting A | —* | **0.6895** | — | `phase7_s2g_new_pipeline_run2/` |
| 10 | `run2` (`phase7_new_pipeline`) | 16 | Setting B | —* | **0.6950** | — | (cùng file metadata) |
| 11 | `run2` (Canonical seed 42) | 16 | Direct 1-pass, không TTA | **0.7793** | **0.7149** | **+6.44 điểm** | `run2/diagnostics/official_direct_evaluation.json` |
| **12** | **Ensemble Phase 2 (LOCKED)** | — | Direct 1-pass + TTA flip_hv | **0.7853** | **0.7158** | **+6.95 điểm** | `results/final_protocol/05_test_final_report.json` |

*\*Ghi chú:* Hàng 6–10 không có giá trị Val cùng giao thức Setting B hoặc Setting A do vòng lặp train của run1/run2 chỉ đo Val bằng Direct 1-pass (Val Direct của run1 là 0.7783, run2 là 0.7793).

**Kết luận đánh giá:**
- Tập Test đã được đo lường $\ge 12$ lượt với $\ge 6$ checkpoint khác nhau.
- **Bản chất mức tăng 0.6950 → 0.7149:** Đây là phép đo trên cùng checkpoint `run2` epoch 16; mức tăng **+1.99 điểm** hoàn toàn do đổi giao thức suy luận từ tiling Setting B (stride 224) sang direct 1-pass, không phải sự cải thiện của mô hình.
- Khoảng cách Val→Test của mọi mô hình đều nằm trong khoảng **5.5 đến 7.7 điểm**.

---

## §2. KHOẢNG TIN CẬY VÀ TẬP DỮ LIỆU TEST (CLUSTER BOOTSTRAP)

Khảo sát 5 định nghĩa mã ảnh gốc trên 3,368 tệp dữ liệu xác nhận định dạng chuẩn `YYYYMMDD_HHMMSS` hoặc `IMG_XXXX`:
- **Tập Train:** 250 ảnh gốc độc lập (1,896 crops) — khớp 100% bài báo Crack500 gốc.
- **Tập Val:** 50 ảnh gốc độc lập (348 crops, trung bình 6.96 crops/ảnh) — khớp 100% bài báo gốc.
- **Tập Test:** **148 ảnh gốc độc lập** (1,124 crops, trung bình 7.59 crops/ảnh). Thiếu 52 ảnh gốc so với con số 200 ảnh của bài báo gốc; nguyên nhân thiếu là chưa giải thích được từ các tệp hiện có.

### Khoảng tin cậy Test Dice (148 cụm ảnh gốc):

| Phương pháp ước lượng | CI 95% dưới | CI 95% trên | Độ rộng CI | Tỉ lệ so với crop CI |
|---|:---:|:---:|:---:|:---:|
| Bootstrap độc lập theo Crop | 0.7072 | 0.7243 | 0.0171 | 1.00× |
| **Cluster Bootstrap theo Ảnh gốc (10,000 lần)** | **0.7005** | **0.7307** | **0.0302** | **1.76×** |

Ước lượng theo crop bị thổi phồng độ chính xác do lặp mẫu cụm. **Khoảng tin cậy chuẩn xác của Test Dice là [0.7005, 0.7307]**.

---

## §3. ĐÁNH GIÁ TTA VÀ NGƯỠNG $\tau$ (PHASE 1)

1. **Trên tập Val:** TTA flip_hv đạt **0.782452** so với no-TTA **0.779294**, tăng **+0.32 điểm** (95% CI bootstrap: $[+0.10, +0.53]$ điểm).
2. **Trên tập Test:** Chênh lệch giữa canonical đơn seed 42 không TTA (0.7149) và LOCKED ensemble 3 seed kèm TTA (0.7158) chỉ là **+0.09 điểm**, nằm trọn trong khoảng sai số cluster bootstrap. Không thể xác nhận TTA mang lại lợi ích thực tế ở tập test.

---

## §4. ĐỘ BIẾN THIÊN ĐA SEED VÀ THIÊN LỆCH CHỌN EPOCH (PHASE 2)

### 4.1 Kết quả đa seed canonical trên Val:

| Seed | SHA-256 Checkpoint (`best_model_b2_global.pth`) | Best Val Dice | Độ lệch vs $\mu$ |
|---|---|:---:|:---:|
| 42 | `4B409B60A7DF788AE48DECE727751E1B8D904CD95EE1102E53D78563F2556CB7` | 0.780409 | +0.001771 |
| 43 | `7379C7346616ACE3569AE2D9D640D838098638AD70F17443A34166AC89D4A271` | 0.782932 | +0.004294 |
| 44 | `F5F7DC55C9D0CBEB606D5EEA396E6B8983ADC64391F23FE96395A19C5EB75F09` | 0.772572 | −0.006066 |
| **Trung bình ($\mu$)** | — | **0.778638** | $\sigma = \mathbf{0.005402}$ |

Ngưỡng phát hiện chênh lệch tối thiểu: $\text{MDE} = 2\sigma = \mathbf{0.010804 \text{ Dice (1.08 điểm)}}$.  
Khoảng tin cậy 95% của MDE ($df=2$) là **$[0.56, 6.79]$ điểm** (cận trên gấp 12.1 lần cận dưới), thể hiện độ bất định lớn khi $n=3$ seed.

### 4.2 Thiên lệch chọn Epoch (Selection Bias):
Trích xuất từ 18 epoch trong `train.log` của 3 seed:
- **Ước lượng quanh vùng hội tụ (5 epoch cuối 14–18):**
  - Seed 42: Best ep 15 (0.7804) vs mean 5 ep cuối (0.7784) → thiên lệch **+0.20 điểm**.
  - Seed 43: Best ep 18 (0.7829) vs mean 5 ep cuối (0.7802) → thiên lệch **+0.27 điểm**.
  - Seed 44: Best ep 14 (0.7726) vs mean 5 ep cuối (0.7707) → thiên lệch **+0.19 điểm**.
  - **Mức thiên lệch chọn epoch đại diện: +0.22 điểm.**
- *(Ghi chú: Con số +1.60 điểm ở các bản báo cáo cũ gộp cả các epoch khởi động ban đầu khi mạng chưa học xong, không phản ánh đúng thiên lệch chọn lựa).*

---

## §5. CẤU TRÚC THAM SỐ THẬT VÀ ĐỐI CHỨNG DUNG LƯỢNG (PHASE 3)

### 5.1 Cấu trúc tham số thực tế (Không trùng lặp):
Mã nguồn `sage_injection.py:102-120` xác nhận các khối trong `expert_pool` trỏ cùng vùng nhớ (`id(p)`) với `main_block` của Backbone. Phân loại theo quyền sở hữu độc quyền:
- **(a) Backbone Pretrain:** **6,626,928 tham số (64.93%)** — gồm ConvNeXt Stem (2,448) + 4 ConvNeXt Stages (4,845,024) + 4 ViT Blocks (1,779,456).
- **(b) Thành phần riêng của SAGE:** **738,216 tham số (7.23%)** — gồm 8 Routers SAGE (88,776) + SA-Hubs / Format Adapters (649,440); P3 refinement chiếm 0 tham số.
- **(c) UNet Decoder, Head & Projections:** **2,840,611 tham số (27.83%)** — gồm UNet Decoder conv 3×3 (2,653,795) + các tầng chiếu và chuẩn hóa (186,816).
- **Tổng tham số mô hình:** **10,205,755** (khớp 100% với `model.parameters()`).

### 5.2 Cơ chế thực nghiệm Variant N:
Khi đặt `residual_scale = float(0.0)`:
- Đạo hàm truyền qua nhánh chính giữ nguyên: Grad Norm của Backbone pretrain là **0.024597**; Grad Norm của Decoder là **0.068322**. **92.77% tham số mô hình vẫn được tối ưu hóa bình thường.**
- Nhánh phụ SAGE bị ngắt hoàn toàn: Grad Norm của nhóm SAGE riêng là **0.000000** (toàn bộ 738,216 tham số có đạo hàm bằng 0).
- Do đó, Variant N đại diện cho mô hình **"B2 không SAGE"**.

### 5.3 Đối chiếu số liệu gốc Phase 3:

| Cấu hình | Seed 42 | Seed 43 | Trung bình ($\mu$) | Độ lệch vs Canonical 2-seed | Vượt MDE (1.08 điểm)? |
|---|:---:|:---:|:---:|:---:|:---:|
| **Canonical** | 0.780409 | 0.782932 | **0.781670** | — | — |
| **Variant R** (Random Top-2 Router) | 0.779086 | 0.777913 | **0.778500** | **−0.32 điểm** | Không |
| **Variant N** (B2 không SAGE, `scale=0`) | 0.778212 | 0.777812 | **0.778012** | **−0.37 điểm** | Không |
| **Variant D** (Dense Adapter thay SAGE) | 0.773668 | 0.782093 | **0.777881** | **−0.38 điểm** | Không |

*(Ghi chú: So với canonical 3-seed $\mu = 0.778638$, độ lệch của Variant R là **−0.01 điểm**, Variant N là **−0.06 điểm**, Variant D là **−0.08 điểm**).*

> [!IMPORTANT]
> **Kết luận:** Tắt bỏ hoàn toàn cơ chế SAGE (Variant N) chỉ làm suy giảm ~0.06 đến 0.37 điểm Val Dice, nằm sâu bên dưới sai số thống kê MDE. Độ biến thiên nội tại giữa các seed của Variant D lên tới **0.84 điểm**. Chưa có bằng chứng cho thấy cơ chế định tuyến SAGE đóng góp giá trị độc lập so với kiến trúc kết hợp ConvNeXt + ViT + S2-Gate.

---

## §6. KHOẢNG CÁCH VAL→TEST VÀ PHÂN RÃ OAXACA-BLINDER

### 6.1 Đặc tính nhãn Ground Truth (Mức ảnh gốc độc lập):
Đo trên 50 ảnh gốc Val và 148 ảnh gốc Test:
- Tỉ lệ pixel vết nứt: Val $0.0557 \pm 0.0317$ vs Test $0.0558 \pm 0.0254$ ($p = 0.48$).
- Độ rộng trung bình: Val $14.12 \pm 6.25$ px vs Test $14.13 \pm 5.66$ px ($p = 0.82$).
- Tỉ lệ vết nứt mảnh ($\le 3$ px): Val $0.1470 \pm 0.0777$ vs Test $0.1455 \pm 0.0676$ ($p = 0.78$).
- Số thành phần liên thông: Val $2.28 \pm 1.13$ vs Test $2.97 \pm 2.63$ ($p = 0.1052$).  
**Không có bất kỳ đặc tính hình thái nào khác biệt có ý nghĩa thống kê ở mức ảnh gốc.**

### 6.2 Phân rã Oaxaca-Blinder cho Cấu hình LOCKED:
Chạy suy luận trên tập Val với cấu hình LOCKED (Ensemble 3 seed + TTA flip_hv) đạt Val Dice **0.785278** (95% CI cluster: $[0.7557, 0.8113]$). Đối chiếu với Test LOCKED (**0.7158**, 95% CI cluster: $[0.7002, 0.7307]$):
- **Khoảng cách tổng thể (LOCKED Gap):** **+6.95 điểm** (95% CI cluster: **[3.64, 10.00] điểm**).

#### Kết quả phân rã toán học theo nhóm độ rộng:
$$\Delta_{total} = \Delta_{mix} \text{ (lệch phân bố nhóm)} + \Delta_{within} \text{ (chênh lệch trong cùng nhóm)}$$

| Thành phần | Giá trị (điểm Dice) | Tỉ lệ đóng góp | Diễn giải |
|---|:---:|:---:|---|
| **Within effect** | **+6.72 điểm** | **96.6%** | Cả vết vừa ($3–10$ px) lẫn vết rộng ($>10$ px) trên tập Test đều đạt điểm Dice thấp hơn trên Val khoảng ~6.0–6.5 điểm |
| **Mix effect** | **+0.24 điểm** | **3.4%** | Sự dịch chuyển tỉ lệ crop giữa các nhóm độ rộng chỉ giải thích được 0.24 điểm khoảng cách |
| **Tổng Gap LOCKED** | **+6.95 điểm** | **100.0%** | — |

*(Đối chiếu với cấu hình Single-seed 42: Gap = **6.35 điểm**, trong đó Within effect = **6.12 điểm (96.3%)**, Mix effect = **0.23 điểm (3.7%)**).*

### 6.3 Tương quan Spearman mức ảnh gốc:
- Tỉ lệ vết nứt mảnh trên tập Val **không có tương quan ý nghĩa thống kê** với Dice ($\rho = -0.096, p = 0.5068$), trong khi trên Test có tương quan nghịch ($\rho = -0.298, p = 0.0002$).
- Độ rộng vết nứt tương quan thuận có ý nghĩa trên cả Val ($\rho = +0.490, p = 0.0003$) và Test ($\rho = +0.371, p = 3.4 \times 10^{-6}$).

---

## §7. CÁC GIAI ĐOẠN ROADMAP VÀ HẠN CHẾ CÒN LẠI

1. **Phase 4 (Sàng lọc tốc độ học — LR Screening):** Chưa thực hiện.
2. **Phase 6 (Đánh giá cross-dataset trên tập ảnh tường ngoài):** Chưa thực hiện do đang chờ thu thập và gán nhãn dữ liệu. Mô hình chưa từng được kiểm chứng trên ảnh bề mặt tường ngoài thực tế.
3. **Cỡ mẫu Val và Test:** Val chỉ có 50 ảnh gốc độc lập. Tập Test local thiếu 52 ảnh gốc so với thiết kế bài báo Crack500 gốc (chưa giải thích được nguyên nhân).
4. **Độ bất định của Within effect:** Thuật ngữ "within effect" chỉ mô tả về mặt số học rằng điểm Dice trong từng nhóm độ rộng đều thấp hơn trên test; không thể tách rời nguyên nhân giữa việc tập test có điều kiện bề mặt khó hơn hay do mô hình bị overfit/thiên lệch chọn lựa trên val.

---

## §8. BẢNG DANH MỤC TUYÊN BỐ CẤM VÀ ĐIỀU CHỈNH

| Tuyên bố bị cấm / Diễn giải sai | Lý do bác bỏ và Căn cứ thực tế |
|---|---|
| *"Tập Test hoàn toàn khách quan, không bị data snooping"* | Tập Test đã được xem $\ge 12$ lượt với $\ge 6$ checkpoint trước khi khóa cấu hình Phase 5. |
| *"TTA hoặc Ensemble mang lại cải thiện đo được trên Test"* | Chênh lệch giữa single seed 42 và LOCKED ensemble+TTA chỉ là +0.09 điểm, nằm trọn trong sai số cluster bootstrap. |
| *"Variant N đóng băng 65.8% tham số mô hình"* | Variant N vẫn huấn luyện 92.77% tham số (Backbone + Decoder); chỉ đóng băng 7.23% tham số của Router và SA-Hub. |
| *"Dung lượng quyết định hoàn toàn hiệu năng SAGE"* | Không dùng cụm từ này vì cấu trúc tham số là chia sẻ trọng số từ Backbone, không phải kho dung lượng độc lập. |
| *"Khoảng cách Val→Test xuất phát từ việc tập Test có nhiều vết mảnh hơn"* | Tỉ lệ vết mảnh giữa Val và Test không khác biệt ($p = 0.78$ ở mức ảnh gốc). Phân rã Oaxaca chứng minh lệch phân bố chỉ chiếm 3.4% khoảng cách. |
| *"Mô hình bị kéo tụt chủ yếu bởi vết nứt mảnh"* | Tương quan vết mảnh trên Val không có ý nghĩa thống kê ($p = 0.51$). Khoảng cách sụt giảm xuất hiện đồng đều cả ở vết nứt rộng. |
| *"B2 vượt trội so với B0"* | B0 100ep đạt Val 0.7806 (> B2 canonical 0.7793). Chênh lệch Test (+1.12 điểm) nằm trong khoảng tin cậy của test. Mọi mô hình đều có khoảng cách suy giảm 5.5–7.7 điểm. |

---

## §9. TÀI LIỆU MINH CHỨNG VÀ TRÍCH XUẤT NGUỒN

Toàn bộ các phân tích thực nghiệm chi tiết được lưu trữ tại thư mục [`results/final_protocol/10_revision4/`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/10_revision4/):
- [`10_param_structure.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/10_revision4/10_param_structure.md) — Phân tích chia sẻ trọng số `expert_pool` và đo lường Grad Norm của Variant N.
- [`10_phase3_reconcile.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/10_revision4/10_phase3_reconcile.md) — Đối chiếu số liệu gốc từng seed của Phase 3 kèm mã băm SHA-256.
- [`10_b0_vs_b2.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/10_revision4/10_b0_vs_b2.md) — So sánh B0 và B2 theo đúng giao thức Direct và ngân sách huấn luyện.
- [`10_gap_decomposition.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/10_revision4/10_gap_decomposition.md) — Phân rã Oaxaca-Blinder cho cấu hình LOCKED và khoảng tin cậy cluster 10,000 lần.
- [`val_locked_per_image_dice.csv`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/10_revision4/val_locked_per_image_dice.csv) — Kết quả suy luận từng ảnh Val của cấu hình LOCKED (Mean Dice = 0.7853).
- [`10_config_diff.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/10_revision4/10_config_diff.md) — So sánh chi tiết cấu hình b1_b2_end_to_end, run1 và run2.
- [`10_source_id.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/10_revision4/10_source_id.md) — Khảo sát 5 định nghĩa mã ảnh gốc và kiểm kê 148 ảnh test local.
- [`05_frozen_config.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/05_frozen_config.json) — Cấu hình chính thức đóng băng trước Phase 5.
- [`05_test_final_report.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/05_test_final_report.json) — Báo cáo kết quả kiểm thử Phase 5 LOCKED (Test Dice = 0.7158).
