# Báo Cáo Thực Nghiệm Phase 6D: P3 — Feature/Detail Enhancement → Spatial Compression

**Thời gian thực hiện:** 03/10/2026  
**Chuyên đề:** Nghiệm thu & Ablation đối đầu P3-ASDW trên tập dữ liệu Crack500  
**Target Model:** SAGE-Lite Candidate B Baseline (`B2ConvNeXtViTUNet`, Setting A, Tile 448)  
**Tập dữ liệu:** Crack500 Validation Set ($N = 348$ cặp ảnh/nhãn kích thước đầy đủ)  
**File kết quả chi tiết:** `results/diagnostics/phase6d_p3_ablation/metrics_p3_with_asdw_per_sample.csv` & `metrics_p3_without_asdw_per_sample.csv`  
**File tổng hợp JSON:** `results/diagnostics/phase6d_p3_ablation/p3_asdw_ablation_summary.json`

---

## 1. Phán Quyết Thực Nghiệm (Executive Verdict)

> [!IMPORTANT]
> **KẾT LUẬN QUYẾT ĐỊNH (DECISION GATE VERDICT): ĐÓNG PHASE 6D VÀ BỎ HƯỚNG NÀY.**
> 
> Dữ liệu thực nghiệm trên toàn bộ 348 ảnh validation cho thấy:
> Việc bổ sung ASDW Refinement trước nén không gian $28 \times 28$ **hoàn toàn không mang lại bất kỳ cải thiện nào** đối với hiện tượng False Bridge (118 vs 118 events, $\Delta = 0$), Breakage (35 vs 35 events, $\Delta = 0$), Boundary IoU ($0.2418$ vs $0.2418$, $\Delta = 0$), hay Thin-Crack Dice ($0.4230$ vs $0.4230$, $\Delta = 0$).
> 
> Tuân thủ nghiêm ngặt chỉ thị của người dùng:
> *"Nếu P3 không cho evidence rằng high-frequency enhancement trước compression cải thiện trade-off detail/connectivity mà không gây breakage đáng kể, đóng Phase 6D và bỏ hướng này, không tiếp tục patch thêm module."*
> $\implies$ **Chính thức khóa và đóng toàn bộ Phase 6D.** Không phát triển thêm bất kỳ adapter, gate, loss hay post-processing nào cho nhánh P3.

---

## 2. Kết Quả Kiểm Chứng Zero-Training Trước Training (Zero-Training Verification)

Script kiểm định [`scripts/tests/test_phase6_p3_asdw_verification.py`](file:///d:/truong/SpecialSubjectTTNT/scripts/tests/test_phase6_p3_asdw_verification.py) đã chạy và đạt **PASS 100% (4/4 checkpoints)**:

```text
================================================================================
PHASE 6D: P3-ASDW ZERO-TRAINING VERIFICATION SUITE
================================================================================
[Test 1] Tensor Shapes & Interface Contracts:
  - Stage 0: (2, 48, 112, 112) -> ASDW -> Pool(28,28) -> Tokens (2, 784, 48) [VERIFIED]
  - Stage 1: (2, 96, 56, 56)   -> ASDW -> Pool(28,28) -> Tokens (2, 784, 96) [VERIFIED]
  --> PASS: Khớp 100% shape contract và không làm thay đổi kích thước không gian.

[Test 2] High-Frequency Detail Enhancement Trước Nén (Vật lý & Phổ):
  - Năng lượng Laplacian cao tần (Input thô):      172,263.77
  - Năng lượng Laplacian cao tần (Sau ASDW):      172,844.02
  - Tỉ số tăng cường cao tần (HF Boost Ratio):     1.0034x (> 1.0)
  - Độ tương phản vết nứt sau Pooling 28x28 (Thô): 0.3456
  - Độ tương phản sau Pooling 28x28 (Có ASDW):     0.3459
  - Tỉ số bảo toàn tương phản:                     1.0009x
  --> PASS: Module ASDW thực sự tạo ra đáp ứng tần số cao vật lý trước khi nén.

[Test 3] Strict Branch Localization (Cô lập nhánh can thiệp):
  - Stem Output Max Diff:                          0.000000e+00 (Bitwise Identical)
  - Stage 0 Main Path Max Diff:                    0.000000e+00 (Bitwise Identical)
  - Stage 1 Main Path Max Diff:                    0.000000e+00 (Bitwise Identical)
  - Stage 0 P3 Refined Tokens Max Diff:            2.929613e-03 (Chỉ khác biệt tại tokens đưa vào ViT)
  --> PASS: Nhánh Main CNN và Skip connections hoàn toàn không bị ảnh hưởng.

[Test 4] Zero Dynamic Parameter Creation (Bất biến tham số trong forward):
  - Tổng số tham số:                              10,118,955 (Cố định, 0 allocation động)
  - Tổng số buffers:                               62 (Bao gồm backbone.pe28_fixed)
  --> PASS: Không có tham số mới nào được sinh ra trong forward pass.
```

---

## 3. Bảng Kết Quả Thực Nghiệm Ablation Đối Đầu (N=348 Validation Set)

Chạy đối đầu độc lập giữa **Model C (With ASDW - Candidate B Baseline)** và **Model A (Without ASDW - Identity Compression Control)** trên toàn bộ 348 ảnh validation của tập Crack500 theo Setting A chuẩn (Tile 448):

| Chỉ số Đánh giá (Metric) | Có ASDW (Run C) | Không ASDW (Run A) | Độ Lệch $\Delta$ (C - A) | Đánh Giá Ý Nghĩa Thống Kê |
| :--- | :---: | :---: | :---: | :--- |
| **Val Dice** | **0.7641** | **0.7641** | $+0.0000$ ($+1.3 \times 10^{-8}$) | Không đổi (Sai số máy tính) |
| **Recall** | **0.8477** | **0.8477** | $-0.0000$ | Không đổi |
| **Precision** | **0.7337** | **0.7337** | $+0.0000$ | Không đổi |
| **Centerline Dice (clDice)** | **0.8499** | **0.8499** | $-0.0000$ | Không đổi |
| **Boundary IoU (d=2)** | **0.2418** | **0.2418** | $-0.0000$ | Không đổi |
| **HD95 (px, thấp hơn là tốt hơn)** | **53.65 px** | **53.64 px** | $+0.01\text{ px}$ | Lệch không đáng kể ($0.01\text{ px}$) |
| **Boundary F1 (BF1)** | **0.3820** | **0.3820** | $-0.0000$ | Không đổi |
| **Thin-Crack Dice ($\le 3\text{ px}$)** | **0.4230** | **0.4230** | $-0.0000$ | Không đổi |
| **False Bridge Events** | **118** | **118** | **0** | **100% Cố định (Không đổi)** |
| **False Bridge Images** | **110** | **110** | **0** | **100% Cố định (Không đổi)** |
| **Break Events (Đứt đoạn)** | **35** | **35** | **0** | **100% Cố định (Không đổi)** |
| **Break Images** | **34** | **34** | **0** | **100% Cố định (Không đổi)** |
| **Spurious Islands** | **111** | **111** | **0** | **100% Cố định (Không đổi)** |
| **Thời gian suy luận (Runtime)** | **147.9 ms/ảnh** | **147.0 ms/ảnh** | $+0.9\text{ ms}$ | ASDW tốn thêm $\sim 0.6\%$ thời gian |
| **Peak VRAM** | **239.3 MB** | **234.3 MB** | $+5.0\text{ MB}$ | ASDW tốn thêm $5\text{ MB}$ VRAM |

---

## 4. Phân Tích Bản Chất Cơ Chế (Mechanistic Explanation)

Tại sao ASDW Refinement trước nén không gian không thể giải quyết lỗi False Bridge hay cải thiện Boundary / Thin-Crack metrics?

### 4.1. Hệ số khuếch đại $\gamma$ quá nhỏ và bị suy giảm lũy thừa
- Khi kiểm tra checkpoint thực tế của Candidate B:
  - $\gamma_{\text{S0}} = 0.0147$
  - $\gamma_{\text{S1}} = 0.0158$
- Tín hiệu tinh chỉnh $F$ chỉ đóng góp khoảng **$1.5\%$** vào feature map đưa vào ViT expert ($x + 0.015 F$).
- Chưa dừng lại ở đó, tại tầng dung hợp SAGE layer, đầu ra của expert lại bị nhân tiếp với hệ số tỷ lệ phần dư `residual_scale = 0.1`:
  $$\text{Contribution} \approx 0.015 \times 0.1 = 0.0015 \ (0.15\%)$$
- Một độ lệch $0.15\%$ ở bottleneck $28 \times 28$ là hoàn toàn không đủ để tạo ra bất kỳ sự lật ngược phân loại (logit sign flip) nào tại đầu ra 448x448 của decoder.

### 4.2. Mâu thuẫn với Causal Pathway đã chứng minh trong Phase 6
- Trong thí nghiệm **2x2 Factorial Causal Pathway Ablation** (commit `1923590` / `80ccb22`):
  $$\text{Decoder}_{56}: \quad \Delta_S = 0.6427 \gg \Delta_U = 0.2307$$
  Đường dẫn nhân quả chi phối trực tiếp và áp đảo nhất gây ra False Bridge là **Stage-1 Skip Connection** đi thẳng vào Decoder Block 1.
- Nhánh P3 nằm ở đâu? P3 chỉ nằm trên đường dẫn đi vào **ViT Expert Bottleneck** (sau đó được giải mã qua Upsample path). Nhánh này có tỷ trọng nhân quả thứ cấp ($\Delta_U = 0.2307$), trong khi Skip Connection mang biểu diễn ConvNeXt thô trực tiếp đi vòng qua toàn bộ khối ViT và P3.
- Do đó, dù có làm sắc nét hay tăng cường tần số cao tại đầu vào ViT expert, nhánh Skip Connection thô vẫn bơm nguyên vẹn đặc trưng liên tục cấu trúc mức thấp vào Decoder, khiến False Bridge hoàn toàn bất biến.

---

## 5. Kết Luận & Hành Động Tiếp Theo

1. **Khép lại Phase 6D:** Đã hoàn thành đầy đủ nghĩa vụ chứng minh theo đúng protocol yêu cầu của người dùng. P3-ASDW không có tác động cải thiện connectivity hay trade-off bridge/breakage.
2. **Không patch thêm:** Tuyệt đối không mở thêm adapter, loss hay post-processing cho nhánh P3.
3. **Bài học hệ thống:** Hướng giải quyết tận gốc hiện tượng False Bridge không thể nằm ở nhánh phụ ViT Bottleneck, mà bắt buộc phải nằm ở cội nguồn nơi biểu diễn skip được sinh ra: **Upstream Genesis / Stem Anti-Aliased Downsampling** (như đã được khẳng định trong báo cáo thẩm định lý thuyết Nyquist-Shannon).
