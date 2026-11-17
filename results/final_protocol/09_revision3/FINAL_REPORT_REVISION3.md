# BÁO CÁO TỔNG HỢP SỬA ĐỔI — SAGE-Lite trên Crack500

**Ngày:** 2026-11-17  
**Phiên bản:** Revision 3  
**Thay thế:** Revision 2 (`08_revision2/FINAL_REPORT_REVISION2.md`) và Revision 1 (`07_revision/FINAL_REPORT_REVISED.md`)  
**Thư mục lưu trữ:** `results/final_protocol/09_revision3/`  
**Nguyên tắc:** Mọi con số đều trích từ file nguồn hoặc log thực nghiệm, kèm đường dẫn xác thực. Đơn vị đo độ chênh Dice: điểm phần trăm (0.0009 Dice = 0.09 điểm). Không khen ngợi.

---

## BẢNG THAY ĐỔI SO VỚI REVISION 2

| Mục / Vấn đề | Nội dung trong Revision 2 | Sửa đổi & Chuẩn hóa trong Revision 3 | Căn cứ & File nguồn |
|---|---|---|---|
| **Bảng lịch sử test (§1)** | Bỏ 0.6771 và 0.6801 vì coi là "không rõ nguồn gốc"; thiếu B0 Setting A/B | Khôi phục 0.6771 (Setting A) và 0.6801 (Setting B) từ checkpoint B0 Run3 epoch 8; cập nhật ≥6 checkpoint, ≥12 lần xem test | [`09_test_history.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/09_revision3/09_test_history.md), `results/B0_Crack500_Run3_Results.md` |
| **Xếp hạng Val vs Test (§1 & §6)** | Chưa làm rõ nghịch lý xếp hạng giữa các checkpoint Setting A/B | Lập bảng chi tiết cho b1_b2_e2e, run1, run2: Val và Test Setting A xếp hạng ngược nhau (~1.21 điểm) dù cùng config yaml | [`09_val_test_ranking.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/09_revision3/09_val_test_ranking.md) |
| **Giao thức 0.6950 → 0.7149** | Chưa phân biệt rõ mức tăng do mô hình hay do giao thức | Xác nhận 0.6950 → 0.7149 (+1.99 điểm) trên cùng checkpoint run2 hoàn toàn do đổi giao thức (tiling Setting B sang direct 1-pass) | [`09_val_test_ranking.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/09_revision3/09_val_test_ranking.md), `diagnostics/official_direct_evaluation.json` |
| **So sánh khoảng cách Val→Test (§6.2)** | Ghép chéo giao thức: "B0 val 0.7318, test 0.7037, gap 2.81" | Bỏ dòng ghép chéo; chỉ dùng các cặp cùng giao thức: B0 Run3 Setting A gap 5.47 điểm, Setting B gap 5.59 điểm; B2 direct gap 6.44 điểm | [`09_test_history.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/09_revision3/09_test_history.md) |
| **So sánh khoảng cách B2 vs B0 (§8)** | Ghi "B2 gap 6.44 điểm lớn hơn B0 gap 2.81 điểm" (dựa trên số ghép chéo) | Sửa theo cặp hợp lệ: B2 direct (gap 6.44 điểm) lớn hơn B0 Setting A/B (gap 5.47–5.59 điểm) khoảng ~0.85–0.97 điểm | [`09_test_history.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/09_revision3/09_test_history.md) |
| **Phân tích nhóm trên Val & Oaxaca (§6.3)** | Chưa chạy suy luận trên Val theo nhóm; dùng p-value crop $10^{-65}$ | Tái lập Val Dice 0.7793 (delta 0.000006), phân rã Oaxaca-Blinder: Within effect chiếm 96.3% (6.12 điểm), Mix effect chỉ chiếm 3.7% (0.23 điểm) | [`09_dice_by_group_val_test.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/09_revision3/09_dice_by_group_val_test.md), `val_per_image_dice.csv` |
| **Tương quan hình thái (§6.1 & §6.3)** | Cho rằng vết mảnh kéo tụt hiệu năng (dựa trên tương quan crop $10^{-65}$) | Gộp theo ảnh gốc: tương quan vết mảnh trên Val không có ý nghĩa thống kê ($\rho=-0.096, p=0.51$). Cấm quy kết gap do vết mảnh | [`09_dice_by_group_val_test.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/09_revision3/09_dice_by_group_val_test.md) |
| **Mã ảnh gốc Crack500 (§2)** | Nêu test=148 ảnh gốc nhưng chưa đối chiếu cấu trúc tên file | Xác nhận quy tắc tách 2 phần `YYYYMMDD_HHMMSS` cho train=250, val=50, test=148 (thiếu 52 ảnh gốc so với 200 ảnh của bài báo) | [`09_source_id.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/09_revision3/09_source_id.md) |
| **Selection bias chọn epoch (§4)** | Chưa có số liệu cụ thể cho Phase 2, chỉ đánh dấu là hạn chế | Phân tích toàn bộ 18 epoch từ `train.log` của 3 seed Phase 2: Selection bias trung bình là 1.60 điểm (toàn bộ epoch) và 0.22 điểm (5 epoch cuối) | [`09_selection_bias.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/09_revision3/09_selection_bias.md) |
| **Đếm tham số & Variant N (§5)** | Chỉ nêu tổng 10.2M; chưa phân rã module và cơ chế gradient | Phân rã: Expert pool 6.62M (64.9%), Router 88.8K (0.9%), phần còn lại 3.49M (34.2%). `residual_scale=0` làm 6.71M params bị frozen hoàn toàn | [`09_phase3_params.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/09_revision3/09_phase3_params.md) |
| **Mô tả Phase 4 & Phase 6 (§7)** | Từng bị nhầm thành "cosine-restart" hoặc "domain adaptation" | Xác nhận theo roadmap/PROGRESS: Phase 4 là sàng lọc learning rate (LR screening); Phase 6 là đánh giá trên dataset tường riêng | [`08_phase3_description.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/08_revision2/08_phase3_description.md) |
| **Tham chiếu §10** | Bị lặp một khối danh sách tham chiếu lớn | Loại bỏ hoàn toàn khối lặp, bổ sung liên kết đến toàn bộ tài liệu `09_*` mới | [`FINAL_REPORT_REVISION3.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/09_revision3/FINAL_REPORT_REVISION3.md) |

---

## TRẢ LỜI THẲNG 4 CÂU HỎI LỚN CỦA DỰ ÁN

1. **SAGE-Lite có hoạt động trên Crack500 không?**  
   **CÓ, NHƯNG SUY GIẢM ĐÁNG KỂ KHI SANG TẬP TEST.**  
   Mô hình đạt Val Dice canonical là **0.7793** (3-seed trung bình: 0.7786) và Test Dice Phase 5 LOCKED là **0.7158** (canonical đơn-seed test: **0.7149**). Tuy nhiên, khoảng cách suy giảm từ Val sang Test là **6.35 – 6.44 điểm**, lớn hơn mức suy giảm của mô hình đường cơ sở B0 (5.47 – 5.59 điểm).
2. **Router và Expert pool có đóng góp ngoài dung lượng tham số không?**  
   **KHÔNG PHÁT HIỆN BẰNG CHỨNG.**  
   Trong thử nghiệm triệt tiêu dung lượng (Phase 3), khi thay router động bằng router ngẫu nhiên (Variant R), Val Dice chỉ lệch **−0.01 điểm**. Khi tắt hoàn toàn nhánh expert pool và router bằng `residual_scale = 0.0` (Variant N, đóng băng 65.8% tham số), Val Dice chỉ lệch **−0.06 điểm** so với canonical. Mọi chênh lệch đều nằm sâu bên dưới ngưỡng phát hiện tối thiểu $\text{MDE} = 1.08$ điểm.
3. **Mô hình có sẵn sàng triển khai thực tế trên bề mặt tường ngoài không?**  
   **CHƯA.**  
   Phase 6 (đánh giá chuyển giao tập dữ liệu sang ảnh tường thực tế) chưa được thực hiện do chưa có nhãn. Mô hình chưa từng được kiểm chứng trên bất kỳ ảnh tường ngoài nào ngoài tập Crack500 (mặt đường nhựa/bê tông). Đồng thời, trên vết nứt mảnh (đặc trưng phổ biến ở tường ngoài), mô hình chỉ đạt Dice **0.2883**.
4. **Đánh giá trên tập test có hoàn toàn khách quan và đáng tin cậy không?**  
   **ĐỘ TIN CẬY BỊ GIỚI HẠN BỞI HAI YẾU TỐ.**  
   Thứ nhất, tập test đã được chạy đánh giá **≥ 12 lần** qua các giai đoạn trước khi đóng băng cấu hình Phase 5, tiềm ẩn rủi ro thiên lệch thông tin (data snooping). Thứ hai, tập test local chỉ chứa **148 ảnh gốc** (1,124 crops), thiếu 52 ảnh gốc so với quy chuẩn 200 ảnh của bài báo Crack500 gốc. Khoảng tin cậy dạng cụm (cluster CI 95%) thực tế là **[0.7005, 0.7307]**, rộng hơn 1.76 lần so với tính toán thông thường theo từng crop.

---

## §1. TÍNH TRONG SÁNG CỦA DỮ LIỆU TEST VÀ LỊCH SỬ ĐÁNH GIÁ

Kiến trúc, siêu tham số và ngưỡng $\tau = 0.5$ được đóng băng tại commit `05_frozen_config.json`. Tuy nhiên, lịch sử mã nguồn và báo cáo kết quả cho thấy tập Test đã được truy cập nhiều lần trước thời điểm này.

### Bảng lịch sử các lần đo Test và ghép cặp Val/Test cùng giao thức:

| # | Checkpoint | Epoch | Giao thức suy luận | Val Dice | Test Dice | Khoảng cách (Val − Test) | File nguồn |
|---|---|:---:|---|:---:|:---:|:---:|---|
| 1 | B0 Run3 (`best_model_b0.pth`) | 8 | Setting A (stride 448) | **0.7318** | **0.6771** | **+5.47 điểm** | `results/B0_Crack500_Run3_Results.md` |
| 2 | B0 Run3 (`best_model_b0.pth`) | 8 | Setting B (stride 224) | **0.7360** | **0.6801** | **+5.59 điểm** | `results/B0_Crack500_Run3_Results.md` |
| 3 | B0 50ep (`best_model`) | 41 | Direct 1-pass, $\tau=0.5$ | 0.7789 | 0.7042 | +7.47 điểm* | `baselines/50ep/B0_Crack500/` (val & test json) |
| 4 | B0 100ep (`best_model`) | 45 | Direct 1-pass, $\tau=0.5$ | 0.7806 | 0.7037 | +7.69 điểm* | `baselines/100ep/B0_Crack500/` (val & test json) |
| 5 | `b1_b2_end_to_end` | 15 | Setting A (stride 448) | — | **0.7016** | — | `phase7_combination/.../final_test_metrics.json` |
| 6 | `b1_b2_end_to_end` | 15 | Setting B (stride 224) | — | **0.7054** | — | (cùng file) |
| 7 | `run1` (`phase7_new_pipeline`) | 18 | Setting A | — | **0.6881** | — | `phase7_combination/.../final_test_metrics.json` |
| 8 | `run1` (`phase7_new_pipeline`) | 18 | Setting B | — | **0.6950** | — | (cùng file) |
| 9 | `run2` (`phase7_new_pipeline`) | 16 | Setting A | — | **0.6895** | — | `phase7_combination/.../final_test_metrics.json` |
| 10 | `run2` (`phase7_new_pipeline`) | 16 | Setting B | — | **0.6950** | — | (cùng file) |
| 11 | `run2` (seed 42, canonical) | 16 | Direct 1-pass, không TTA | **0.7793** | **0.7149** | **+6.44 điểm** | `official_direct_evaluation.json`, `audit_result.json` |
| **12** | **Ensemble Phase 2 (LOCKED)** | — | Direct 1-pass + TTA flip_hv | **0.7850** | **0.7158** | **+6.92 điểm** | `results/final_protocol/05_test_final_report.json` |

*\*Ghi chú:* Hàng 3 và 4 đo Val bằng direct inline trong quá trình huấn luyện, test json không ghi rõ trường protocol nên được phân loại là giả thuyết chưa kiểm chứng độc lập.

**Tổng kết:**
- **Số checkpoint khác nhau từng được chạy Test:** $\ge 6$ (B0 Run3, B0 50ep, B0 100ep, b1_b2_end_to_end, run1, run2).
- **Số lượt chạy suy luận trên Test:** $\ge 12$ lượt.
- **Nghịch lý xếp hạng (Ranking Inversion):** Ở Setting A, `b1_b2_end_to_end` có Val thấp nhất (0.7689) nhưng Test cao nhất (0.7016); trong khi `run2` có Val cao nhất (0.7793) nhưng Test thấp hơn (0.6895) — chênh lệch **1.21 điểm**. Cả hai run đều dùng chung tệp cấu hình YAML `phase7_b2_s2g_b1_b2_end_to_end.yaml`.
- **Bản chất mức tăng 0.6950 → 0.7149:** Đây là hai phép đo trên **cùng một checkpoint run2 epoch 16**. Mức tăng **+1.99 điểm** hoàn toàn do đổi giao thức suy luận từ tiling Setting B (stride 224) sang direct 1-pass (ảnh co về kích thước cố định), không phải là sự cải thiện về năng lực mô hình.

---

## §2. KHOẢNG TIN CẬY VÀ TẬP DỮ LIỆU TEST (CLUSTER BOOTSTRAP)

Bài báo Crack500 gốc chia tập dữ liệu thành 250 ảnh gốc huấn luyện, 50 ảnh gốc thẩm định (val) và 200 ảnh gốc kiểm thử (test).  
Đối chiếu mã nguồn và phân tích tên file crop (`YYYYMMDD_HHMMSS_X_Y.jpg`) cho thấy:
- **Tập huấn luyện (Train):** 250 ảnh gốc (1,896 crops).
- **Tập thẩm định (Val):** 50 ảnh gốc (348 crops, trung bình 6.96 crops/ảnh).
- **Tập kiểm thử (Test):** **148 ảnh gốc** (1,124 crops, trung bình 7.59 crops/ảnh). Thiếu 52 ảnh gốc so với bài báo gốc do cách thức đóng gói tập dữ liệu local.

### Khoảng tin cậy Test Dice (1,124 crops từ 148 ảnh gốc):

| Phương pháp ước lượng | CI 95% dưới | CI 95% trên | Độ rộng khoảng CI | Hệ số mở rộng |
|---|:---:|:---:|:---:|:---:|
| Bootstrap độc lập theo Crop | 0.7072 | 0.7243 | 0.0171 | 1.00× |
| **Cluster Bootstrap theo Ảnh gốc (10,000 lần)** | **0.7005** | **0.7307** | **0.0302** | **1.76×** |

Do các crop cắt ra từ cùng một bức ảnh có sự phụ thuộc lẫn nhau, ước lượng theo từng crop bị hiện tượng giả lặp mẫu (pseudoreplication) làm thu hẹp giả tạo khoảng tin cậy. **Khoảng tin cậy chuẩn xác của Test Dice là [0.7005, 0.7307]**.

---

## §3. ĐÁNH GIÁ TTA VÀ NGƯỠNG $\tau$ (PHASE 1)

1. **Hiệu quả TTA trên tập Val:**
   - No TTA, $\tau = 0.5$: Val Dice = **0.779294**
   - TTA flip_hv, $\tau = 0.5$: Val Dice = **0.782452**
   - Chênh lệch trên Val: **+0.003158 (+0.32 điểm)**. Khoảng tin cậy bootstrap ghép cặp theo crop: $[+0.10, +0.53]$ điểm.
2. **Hiệu quả tại Test — Không xác nhận được:**
   - So sánh giữa canonical run2 đơn-seed không TTA (0.7149) và mô hình Phase 5 LOCKED ensemble 3 seed kèm TTA (0.7158) ghi nhận mức chênh lệch **+0.0009 (+0.09 điểm)**.
   - Khoảng chênh lệch +0.09 điểm này gộp đồng thời cả tác động đổi tập checkpoint, ensemble đa seed và phép biến đổi TTA. Do nằm trọn bên trong khoảng tin cậy cluster của test ([0.7005, 0.7307]), **không thể khẳng định TTA mang lại cải thiện có ý nghĩa trên tập test**.

---

## §4. ĐỘ BIẾN THIÊN ĐA SEED VÀ THIÊN LỆCH CHỌN EPOCH (PHASE 2)

### 4.1 Kết quả đa seed canonical trên Val:

| Seed | Checkpoint SHA-256 | Best Val Dice | Độ lệch vs $\mu$ |
|---|---|:---:|:---:|
| 42 | `4B409B60...` | 0.780409 | +0.001771 |
| 43 | `7379C7A1...` | 0.782932 | +0.004294 |
| 44 | `F5F7DCF8...` | 0.772572 | −0.006066 |
| **Trung bình ($\mu$)** | — | **0.778638** | $\sigma = \mathbf{0.005402}$ |

Ngưỡng phát hiện chênh lệch tối thiểu định trước:  
$$\text{MDE} = 2\sigma = \mathbf{0.010804 \text{ Dice (1.08 điểm)}}$$
Với $n=3$ seed, khoảng tin cậy 95% của MDE theo phân phối Chi-bình phương ($df=2$) là **$[0.56, 6.79]$ điểm**. Cận trên gấp 12.1 lần cận dưới, cho thấy độ bất định thống kê rất cao khi kiểm định các giả thuyết suy diễn.

### 4.2 Thiên lệch chọn lựa do lấy Epoch tốt nhất trên Val (Selection Bias):
Trích xuất từ log huấn luyện đầy đủ 18 epoch của 3 seed Phase 2:
- **Seed 42:** Đạt đỉnh tại epoch 15 (0.7804). Trung bình 18 epoch: 0.7668. Thiên lệch: **+1.36 điểm**. (So với trung bình 5 epoch cuối 14–18: +0.20 điểm).
- **Seed 43:** Đạt đỉnh tại epoch 18 (0.7829). Trung bình 18 epoch: 0.7687. Thiên lệch: **+1.42 điểm**. (So với trung bình 5 epoch cuối: +0.27 điểm).
- **Seed 44:** Đạt đỉnh tại epoch 14 (0.7726). Trung bình 18 epoch: 0.7525. Thiên lệch: **+2.01 điểm**. (So với trung bình 5 epoch cuối: +0.19 điểm).
- **Mức thiên lệch chọn lựa trung bình:** **+1.60 điểm** khi tính trên toàn bộ tiến trình và **+0.22 điểm** khi tính quanh vùng hội tụ cuối. Điều này giải thích một phần vì sao Val Dice luôn đạt mức lạc quan hơn so với năng lực tổng quát hóa thực tế.

---

## §5. PHÂN RÃ THAM SỐ VÀ KIỂM CHỨNG DUNG LƯỢNG (PHASE 3)

### 5.1 Phân rã tham số mô hình SAGE B2:
Từ việc đếm tham số trên đồ thị tính toán không trùng lặp (`model.parameters()`) và đối chiếu với checkpoint:
- **Tổng tham số mô hình:** **10,205,755** (10.21M).
- **Expert Pool:** **6,624,480 tham số (64.91%)** — dùng chung qua 8 tầng `SageLayer`.
- **Router (8 tầng SageRouter):** **88,776 tham số (0.87%)**.
- **Khung chính (Backbone ConvNeXt + Decoder + Head + Projections):** **3,492,499 tham số (34.22%)**.

### 5.2 Cơ chế thực thi và kết quả Variant N:
Variant N can thiệp trực tiếp vào mã nguồn `sage/components/sage_layer.py:81`:
$$\text{self.residual\_scale} = \text{float}(0.0)$$
Vì `residual_scale` là số thực thông thường (không phải `nn.Parameter`), việc nhân $0.0$ với đầu ra của expert khiến đạo hàm ngược truyền về Expert Pool và Router bằng $0$ tuyệt đối trong suốt quá trình huấn luyện. **65.78% tham số của mô hình bị đóng băng hoàn toàn**, mô hình chỉ thực sự tối ưu hóa trên 3.49M tham số còn lại.

### 5.3 So sánh kết quả đối chứng Phase 3 (Val Dice):

| Biến thể | Cơ chế can thiệp | Val Dice (Seed 42) | Val Dice (Seed 43) | Trung bình 2 seed | Chênh lệch vs Canonical 2-seed | Vượt MDE (1.08 điểm)? |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Canonical** | Chuẩn (Router học + Expert pool) | 0.7804 | 0.7829 | **0.781670** | — | — |
| **Variant R** | Router chọn ngẫu nhiên Top-2 | 0.7788 | 0.7782 | **0.778500** | **−0.32 điểm** | Không |
| **Variant N** | Đóng băng Expert + Router (`scale=0`) | 0.7782 | 0.7778 | **0.778012** | **−0.37 điểm** | Không |
| **Variant D** | Thay bằng Dense Adapter tương đương | 0.7775 | 0.7783 | **0.777881** | **−0.38 điểm** | Không |

*(Ghi chú: So với trung bình 3-seed canonical $\mu = 0.7786$, chênh lệch của Variant R là **−0.01 điểm**, Variant N là **−0.06 điểm**, Variant D là **−0.08 điểm**).*

> [!IMPORTANT]
> **Kết luận chuẩn xác:** Không phát hiện thấy sự sụt giảm hiệu năng vượt quá ngưỡng sai số thống kê MDE khi vô hiệu hóa Router hoặc đóng băng toàn bộ Expert Pool. Hiệu năng của SAGE-Lite trên tập dữ liệu này được quyết định bởi dung lượng và kiến trúc nền tảng (Backbone ConvNeXt + Transformer Decoder), chưa ghi nhận bằng chứng thực nghiệm cho thấy cơ chế định tuyến chuyên biệt hóa đóng góp giá trị gia tăng độc lập.

---

## §6. KHOẢNG CÁCH VAL→TEST VÀ PHÂN RÃ OAXACA-BLINDER

### 6.1 So sánh đặc tính nhãn hình thái (Ground Truth Masks):
Đo đạc trên 348 mask Val (50 ảnh gốc) và 1,124 mask Test (148 ảnh gốc):

| Đặc tính hình thái | Val (50 ảnh gốc) | Test (148 ảnh gốc) | $p$-value kiểm định cụm Mann-Whitney U | Nhận xét |
|---|:---:|:---:|:---:|---|
| **Tỉ lệ pixel vết nứt** | $0.0557 \pm 0.0317$ | $0.0558 \pm 0.0254$ | **0.4799** | Không có sự khác biệt |
| **Độ rộng trung bình vết nứt** | $14.12 \pm 6.25$ px | $14.13 \pm 5.66$ px | **0.8205** | Không có sự khác biệt |
| **Tỉ lệ vết nứt mảnh ($\le 3$ px)** | $0.1470 \pm 0.0777$ | $0.1455 \pm 0.0676$ | **0.7830** | Không có sự khác biệt |
| **Số thành phần liên thông** | $2.28 \pm 1.13$ | $2.97 \pm 2.63$ | **0.1052** | Không có sự khác biệt ở mức ảnh gốc |

*(Ở mức crop riêng lẻ, số thành phần từng cho $p = 0.0002$, nhưng đây là sai lệch do thổi phồng bậc tự do lặp mẫu; khi gom về mức ảnh gốc độc lập, $p = 0.1052 > 0.05$).*

### 6.2 So sánh nhóm và Phân rã Oaxaca-Blinder trên cùng Checkpoint Canonical Seed 42:
Chạy suy luận trực tiếp trên 348 ảnh Val thu được đúng Val Dice **0.779294** (khớp kỳ vọng 0.7793 với sai số $6 \times 10^{-6}$). Đối chiếu với 1,124 ảnh Test (Test Dice 0.7158), khoảng cách suy giảm là **6.35 điểm** (nếu so với canonical direct test 0.7149, khoảng cách là **6.44 điểm**).

#### So sánh hiệu năng theo nhóm độ rộng:
- **Nhóm vừa (3–10 px):** Val chiếm 32.5% (Dice: **0.7141** [0.6774, 0.7501]) vs Test chiếm 34.9% (Dice: **0.6557** [0.6293, 0.6782]) → Khoảng cách: **+5.85 điểm**.
- **Nhóm rộng (>10 px):** Val chiếm 67.5% (Dice: **0.8106** [0.7714, 0.8398]) vs Test chiếm 65.1% (Dice: **0.7479** [0.7321, 0.7628]) → Khoảng cách: **+6.27 điểm**.

#### Phân rã Oaxaca-Blinder:
Khoảng cách tổng thể được tách thành hai thành phần toán học:
$$\Delta_{total} = \Delta_{mix} \text{ (do lệch phân bố tỉ lệ nhóm)} + \Delta_{within} \text{ (do mô hình kém hơn bên trong từng nhóm)}$$

| Thành phần phân rã | Giá trị (điểm Dice) | Tỉ lệ đóng góp vào Gap |
|---|:---:|:---:|
| **Within effect (Suy giảm nội tại trong từng nhóm)** | **+6.12 điểm** | **96.3%** |
| **Mix effect (Khác biệt về tỉ lệ phân bố mẫu)** | **+0.23 điểm** | **3.7%** |
| **Tổng khoảng cách phân rã đo được** | **+6.35 điểm** | **100.0%** |

**Ý nghĩa:** 96.3% khoảng cách suy giảm xuất phát từ việc mô hình dự đoán kém hơn đồng loạt trên tập Test đối với mọi loại vết nứt (cả vết vừa lẫn vết rộng đều tụt khoảng 6 điểm), chứ không phải do tập Test chứa nhiều ảnh khó hơn tập Val.

### 6.3 Tương quan Spearman ở mức ảnh gốc độc lập:
Loại bỏ hoàn toàn các giá trị $p \approx 10^{-65}$ do lặp mẫu ở cấp crop. Khi tính trung bình theo từng ảnh gốc:
- **Tỉ lệ vết nứt mảnh ($\le 3$ px):** Trên Test có tương quan nghịch ($\rho = -0.298, p = 0.0002$), nhưng trên Val **không có tương quan ý nghĩa thống kê** ($\rho = -0.096, p = 0.5068$).
- **Độ rộng trung bình:** Tương quan thuận có ý nghĩa trên cả Val ($\rho = +0.490, p = 0.0003$) và Test ($\rho = +0.371, p = 3.4 \times 10^{-6}$).

---

## §7. TÌNH TRẠNG CÁC GIAI ĐOẠN ROADMAP VÀ HẠN CHẾ

1. **Phase 4 (Sàng lọc tốc độ học — LR Screening):**  
   Mục tiêu là thử nghiệm 3 mức LR ($5\times 10^{-5}, 10^{-4}, 2\times 10^{-4}$) trong 8–10 epoch để tìm cấu hình hội tụ tối ưu. Giai đoạn này **chưa thực hiện**.
2. **Phase 6 (Đánh giá chuyển giao tập dữ liệu — Cross-dataset Evaluation):**  
   Mục tiêu là đánh giá mô hình trên tập ảnh tường ngoài thực tế để kiểm tra tính tổng quát hóa ra ngoài miền Crack500. Giai đoạn này **chưa thực hiện** do đang chờ thu thập và dán nhãn dữ liệu. Hiện chưa có bất kỳ số liệu nào về năng lực phân đoạn vết nứt tường nhà.
3. **Cỡ mẫu Val và Test:**  
   Val chỉ có 50 ảnh gốc, tiềm ẩn phương sai lớn. Test local chỉ có 148 ảnh gốc thay vì 200 ảnh theo thiết kế ban đầu của tập dữ liệu.

---

## §8. BẢNG DANH MỤC TUYÊN BỐ CẤM VÀ ĐIỀU CHỈNH

| Tuyên bố bị cấm / Diễn giải sai | Lý do bác bỏ và Căn cứ thực tế |
|---|---|
| *"Tập Test hoàn toàn khách quan, không bị data snooping"* | Tập Test đã được xem $\ge 12$ lần với $\ge 6$ checkpoint trước khi khóa cấu hình Phase 5. |
| *"TTA hoặc Ensemble mang lại cải thiện rõ rệt trên tập Test"* | Chênh lệch giữa canonical đơn-seed và Phase 5 LOCKED chỉ là +0.09 điểm, nằm trọn trong sai số cluster bootstrap. |
| *"0.7149 là kết quả của seed 43"* | 0.7149 là kết quả của canonical seed 42 run2 direct 1-pass; seed 43 chưa từng được đánh giá đơn lẻ trên test. |
| *"Khoảng cách Val→Test xuất phát từ việc tập Test có nhiều vết mảnh hơn"* | Tỉ lệ vết mảnh giữa Val và Test không khác biệt ($p = 0.78$ ở mức ảnh gốc). Phân rã Oaxaca chứng minh lệch phân bố chỉ chiếm 3.7% khoảng cách. |
| *"Mô hình bị kéo tụt chủ yếu bởi vết nứt mảnh"* | Tương quan vết mảnh trên Val không có ý nghĩa thống kê ($p = 0.51$). Khoảng cách sụt giảm xuất hiện đồng đều cả ở vết nứt rộng (+6.27 điểm). |
| *"B2 thu hẹp khoảng cách Val→Test tốt hơn B0"* | Khoảng cách của B2 là 6.44 điểm (direct), trong khi B0 Run3 chỉ là 5.47 điểm (Setting A) và 5.59 điểm (Setting B). B2 có độ suy giảm lớn hơn B0 gần 1 điểm. |
| *"SAGE Router và Expert pool mang lại giá trị gia tăng vượt trội"* | Cả 3 biến thể triệt tiêu dung lượng (D, R, N) đều cho hiệu năng tương đương canonical trong phạm vi sai số MDE. |

---

## §9. TỔNG KẾT VÀ TÀI LIỆU MINH CHỨNG

Bản báo cáo Revision 3 đã chuẩn hóa toàn bộ các mâu thuẫn dữ kiện lịch sử, xác lập cơ sở toán học cho khoảng tin cậy dạng cụm, và chỉ ra bằng chứng thực nghiệm về việc năng lực mô hình đến từ kiến trúc nền tảng hơn là cơ chế định tuyến SAGE.

### Danh mục tài liệu minh chứng:
- [`09_test_history.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/09_revision3/09_test_history.md) — Khôi phục lịch sử test B0 Run3 và bảng ghép cặp cùng giao thức.
- [`09_val_test_ranking.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/09_revision3/09_val_test_ranking.md) — Phân tích hiện tượng đảo ngược xếp hạng giữa Val và Test Setting A/B.
- [`09_dice_by_group_val_test.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/09_revision3/09_dice_by_group_val_test.md) — Tái lập suy luận Val, phân tích nhóm và phân rã Oaxaca-Blinder.
- [`val_per_image_dice.csv`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/09_revision3/val_per_image_dice.csv) — Bảng điểm Dice và mã ảnh gốc của 348 ảnh thẩm định.
- [`09_selection_bias.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/09_revision3/09_selection_bias.md) — Đo lường thiên lệch chọn epoch từ 18 epoch huấn luyện của Phase 2.
- [`09_source_id.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/09_revision3/09_source_id.md) — Xác thực định dạng mã ảnh gốc và kiểm kê 148 ảnh test local.
- [`09_phase3_params.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/09_revision3/09_phase3_params.md) — Đếm tham số theo module và chứng minh đóng băng 65.8% tham số ở Variant N.
- [`05_frozen_config.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/05_frozen_config.json) — Cấu hình chính thức đóng băng trước Phase 5.
- [`05_test_final_report.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/05_test_final_report.json) — Báo cáo kết quả kiểm thử Phase 5 LOCKED.
