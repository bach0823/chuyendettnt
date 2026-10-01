# Phase 6-D.2 Specification: Pure Inter-Component Separation Isolation Probe

**Status**: FROZEN / READY FOR EXECUTION  
**Role**: Single-Variable Mechanism Probe (Explicit Inter-Component Negative Moat Supervision)  
**Parent Lineage**: Candidate B Stage-1 Checkpoint (`P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_stage1.pth`)  
**Design Paradigm**: Pure Separation Isolation Probe on Baseline (No PLU, No AB-BPL, No clDice, No BoundaryIoU)  

---

## 1. Mục Tiêu & Câu Hỏi Nghiên Cứu (Research Question)

### 1.1. Câu hỏi khoa học cốt lõi
> **"Can explicit inter-component separation supervision ($G_{\max} = 8.0\text{ px}, \lambda_{\text{sep}} = 0.010$) reduce false bridges that persist after boundary tightening, without sacrificing crack continuity?"**

### 1.2. Định vị so với Phase 6-D.1
- **Bài học từ D.1**: Phase 6-D.1 đã chứng minh rằng việc co nhỏ diện tích mặt nạ bằng penalty biên ngoài (thinning/boundary penalty) là không đủ để chữa lành False Bridge ($105/110$ ca vẫn dai dẳng). Hiện tượng này chịu sự chi phối của một **Residual Separation Bottleneck** tại các khe nứt hẹp kẹp giữa các nhánh.
- **Sứ mệnh của D.2**: Tấn công trực diện vào khe nền phân cách giữa các nhánh nứt (negative moat), áp đặt lực đẩy xác suất về 0 ($p \to 0$) tại hành lang giữa các GT connected components riêng biệt.
- **Tính chất cô lập (Counterfactual Isolation)**: D.2 được thiết kế như một **Pure Separation Isolation Probe** trên nền Candidate B gốc. Bằng cách cố định hoàn toàn kiến trúc và chỉ bổ sung duy nhất $L_{\text{sep}}$, mọi sự biến động của False Bridge và Breakage đều được quy về một nguyên nhân duy nhất (causal attribution).

---

## 2. Công Thức Toán Học Của Inter-Component Separation Loss ($L_{\text{sep}}$)

Hàm mất mát phân tách $L_{\text{sep}}$ hoạt động theo quy trình 3 bước hình học:

### Bước 1: Sàng lọc cặp components nguy cơ cao ($d_{\min} \le G_{\max}$)
Từ Ground Truth nhị phân $GT$, trích xuất tập các connected components (8-connectivity, diện tích $\ge 5\text{ px}$):
$$\mathcal{C} = \{CC_1, CC_2, \dots, CC_K\}$$
- Nếu $K \le 1 \implies M_{\text{sep}} = \emptyset$, $L_{\text{sep}} = 0$.
- Nếu $K \ge 2$, tính khoảng cách Euclidean tối thiểu giữa từng cặp:
$$d_{\min}(CC_i, CC_j) = \min_{u \in CC_i, v \in CC_j} \|u - v\|_2$$
Tập các cặp hợp lệ:
$$\mathcal{P}_{\text{valid}} = \left\{ (i, j) \;\middle|\; 1 \le i < j \le K \;\text{ và }\; d_{\min}(CC_i, CC_j) \le G_{\max} \right\} \quad \text{với } G_{\max} = 8.0\text{ px}$$

*Data-driven primary radius*: $G_{\max} = 8.0\text{ px}$ được chọn vì bao phủ trọn vẹn **$79.1\%$ tổng số ca False Bridge của Candidate B ($87/110$)** và **$81.0\%$ các ca persistent under D.1 ($85/105$)** theo kết quả thực nghiệm D.2-Preflight Geometry Diagnostic.

### Bước 2: Xác định hành lang phân tách cục bộ (Local Inter-Component Moat)
Với mỗi cặp $(i, j) \in \mathcal{P}_{\text{valid}}$ có khoảng cách $g_{ij} = d_{\min}(CC_i, CC_j)$, bán kính mở rộng cục bộ là:
$$r_{ij} = \left\lceil \frac{g_{ij}}{2} \right\rceil + 1$$
Vùng moat của cặp $(i, j)$ là phần giao thoa giữa hai vùng mở rộng nhưng thuộc về background:
$$M_{ij} = \Big( \text{dilate}(CC_i, r_{ij}) \cap \text{dilate}(CC_j, r_{ij}) \Big) \cap \text{Background}$$
Toàn bộ mặt nạ moat:
$$M_{\text{sep}} = \bigcup_{(i, j) \in \mathcal{P}_{\text{valid}}} M_{ij}$$

### Bước 3: Trọng số nghịch đảo khoảng cách & Hàm Loss
Mỗi pixel $x \in M_{ij}$ nhận trọng số:
$$w(x) = \max\left(0.0, \; 1.0 - \frac{g_{ij}}{G_{\max}}\right) = \max\left(0.0, \; 1.0 - \frac{d_{\min}(CC_i, CC_j)}{8.0}\right)$$
- Khe siêu hẹp ($g_{ij} = 2\text{ px}$) $\implies w(x) = 0.75$ (phạt mạnh).
- Khe chạm ngưỡng ($g_{ij} = 8\text{ px}$) $\implies w(x) = 0.00$ (inclusive cutoff, zero-weight endpoint).

Hàm mất mát phân tách hoàn chỉnh:
$$L_{\text{sep}} = \begin{cases} 
0, & \text{nếu } M_{\text{sep}} = \emptyset \\
\displaystyle\frac{\sum_{x \in M_{\text{sep}}} w(x) \cdot \text{softplus}(\text{logits}_x)}{\sum_{x \in M_{\text{sep}}} w(x)}, & \text{nếu } M_{\text{sep}} \ne \emptyset
\end{cases}$$

---

## 3. Hiệu Chuẩn Gradient & Khóa Trọng Số $\lambda_{\text{sep}}$

Đo đạc thực tế trên **batch 8 ca False Bridge kinh điển từ tập Validation Crack500** tại checkpoint Candidate B Stage-1 (Preflight 7):
- $\|dW_{\text{Base}}\|_2 = 0.668809$
- $\|dW_{\text{Sep}}\|_2 = 3.776147$
- Tỷ số gradient quan sát:
  
$$\frac{\|g_{\text{sep}}\|}{\|g_{\text{Base}}\|} = \mathbf{5.6461\times}$$

Với ngân sách gradient mục tiêu bảo thủ $\rho_{\text{target}} = 0.05$ ($5.0\%$ Base gradient nhằm bảo vệ tối đa tính liên tục):
$$\lambda_{\text{sep}} = \frac{0.05}{5.6461} = 0.00886 \approx \mathbf{0.010}$$

Khóa trọng số chính thức:
$$\boxed{\lambda_{\text{sep}} = 0.010}$$
Đóng góp auxiliary gradient: $0.010 \times 5.6461 = \mathbf{5.65\%}$ Base gradient.

---

## 4. Kiến Trúc & Cấu Hình Huấn Luyện (Protocol Freeze)

### 4.1. Thành phần Loss
$$L_{\text{total}} = 1.0 \times L_{\text{BCE}} + 1.5 \times L_{\text{Dice}} + 0.010 \times L_{\text{sep}}$$

### 4.2. Các biến số bị khóa tắt (Strictly OFF)
- `use_plu_head: false` (Decoder 1x1 conv + 4x bilinear upsample chuẩn của Candidate B)
- `ab_bpl_weight: 0.0` (Tắt hoàn toàn AB-BPL)
- `boundary_iou_weight: 0.0` (Tắt hoàn toàn Boundary-IoU)
- `cldice_weight: 0.0` (Tắt hoàn toàn clDice)

### 4.3. Lineage & Quy trình huấn luyện
- **Stage 1 Checkpoint**: `results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_stage1.pth` (Model) và `last_model_b2_stage1.pth` (RNG/Scaler).
- **Stage 2 Protocol**:
  * Epochs: $18$ epochs (tương đương toàn bộ Phase 5 và Phase 6 Stage 2).
  * Batch size: $14$, Image size: $448 \times 448$, Seed: $42$.
  * Patience: $8$ epochs.
  * Optimizer / Scheduler: Khởi tạo fresh Stage 2, `lr = 1e-4`, CosineAnnealing.

---

## 5. Bảng Điểm Nghiệm Thu & Cây Quyết Định (Decision Logic)

### 5.1. Primary Readout (False Bridge Metrics)
- **Baseline Candidate B anchor**: $110 / 348 = 31.61\%$
- **Target Guidepost**:
  
$$\boxed{N_{\text{bridge}} \le 105 \quad (\text{False Bridge Rate} < 30.2\%, \quad \text{NetChange} \le -5)}$$

### 5.2. Guardrail Metrics (Bảo toàn chất lượng phân đoạn)
- **Breakage Rate**: $\le 9.5\%$ (không được tăng so với Base $9.5\%$).
- **Thin Crack Dice ($n=66$)**: $\ge 0.6400$ (Base: $0.6404$).
- **Global Dice ($N=348$)**: $\ge 0.7640$ (Base: $0.7641$).
- **Global Recall**: $\ge 0.8400$ (Base: $0.8477$).

### 5.3. Cây Quyết Định (Decision Tree)
1. **Branch 1 (Mechanism Confirmed - Thành công)**:
   - $N_{\text{bridge}} \le 105$ ($\text{NetChange} \le -5$) VÀ Breakage $\le 9.5\%$.
   - *Hành động*: Xác nhận $L_{\text{sep}}$ có lực triệt tiêu False Bridge thực sự mà không làm đứt gãy vết nứt. Chuyển sang bước tiếp theo: kết hợp $L_{\text{sep}}$ với PLU để giải quyết toàn diện residual bridge.
2. **Branch 2 (Clean Negative Result - Bác bỏ cơ chế Moat)**:
   - $N_{\text{bridge}} \in [108, 114]$ ($\text{NetChange} \approx 0$).
   - *Hành động*: Bằng chứng phủ định trực tiếp rằng việc phạt xác suất nền ở khe hẹp là không đủ để mạng neuron tách rời hai nhánh nứt. Cần đánh giá lại receptive field hoặc cơ chế kiến trúc.
3. **Branch 3 (Unacceptable Trade-off - Đứt gãy quá mức)**:
   - $N_{\text{bridge}} \le 105$ NHƯNG Breakage $> 10.5\%$.
   - *Hành động*: Gradient phân tách quá thô bạo làm gãy các nhánh nứt mảnh hợp lệ. Cần hạ $\lambda_{\text{sep}}$ hoặc kết hợp với cơ chế giữ liên tục (continuity regularizer).
