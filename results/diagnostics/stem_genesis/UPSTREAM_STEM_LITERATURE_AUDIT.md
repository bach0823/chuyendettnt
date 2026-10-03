# BÁO CÁO THẨM ĐỊNH ĐỘC LẬP (INDEPENDENT AUDIT REPORT)
**Chuyên đề:** Upstream Genesis / Stem Information Preservation cho SAGE-Lite trên Crack500  
**Vai trò:** Cross-Checker / Literature & Architectural Auditor độc lập  
**Mã đối chiếu:** SAGE-LITE-AUDIT-STEM-01  

---

## PHẦN 1: KHẢO SÁT TÀI LIỆU KHOA HỌC (LITERATURE RESEARCH)

### 1.1. Anti-aliased Downsampling & Định lý Lấy mẫu Nyquist-Shannon trong CNN
* **Tài liệu tham chiếu:** 
  - Richard Zhang (Adobe Research), *"Making Convolutional Networks Shift-Invariant Again"*, ICML 2019.
  - Zou et al., *"Delving into Anti-aliasing in Convolutional Neural Networks"*, BMVC 2020.
  - Karras et al., *"Alias-Free Generative Adversarial Networks (StyleGAN3)"*, NeurIPS 2021.

* **Bản chất lý thuyết & Cơ chế:**
  - Định lý Nyquist-Shannon chỉ ra rằng để tái tạo hoàn hảo một tín hiệu liên tục có tần số cực đại $f_{max}$, tần số lấy mẫu $f_s$ phải thỏa mãn:
    $$f_s \ge 2 f_{max} \iff f_{max} \le f_{Nyquist} = \frac{f_s}{2}$$
  - Trong CNN tiêu chuẩn, các thao tác giảm mẫu rời rạc (như MaxPool stride 2, Strided Conv stride 2 hoặc stride 4) vi phạm trực tiếp định lý Nyquist-Shannon: Các thành phần phổ tần số cao (edges, fine textures) vượt ngưỡng $f_{Nyquist}$ bị "gập" (spectral folding/aliasing) vào dải tần số thấp. Điều này phá vỡ tính bất biến dịch chuyển (shift-invariance) — một dịch chuyển 1 pixel ở ảnh đầu vào có thể làm đảo lộn hoàn toàn vector đặc trưng.
  - **Cơ chế BlurPool (Zhang, 2019):** Phân rã thao tác strided-downsampling thành 2 bước tách biệt:
    $$\text{Conv/Pool}(\text{stride}=1) \longrightarrow \text{Low-Pass Filter (LPF)} \longrightarrow \text{Subsampling}(\text{stride}=S)$$
    Bộ lọc LPF thường là các nhân nhị thức (binomial kernels) xấp xỉ Gaussian, ví dụ: $3\times3$ filter $[1, 2, 1]^T [1, 2, 1] / 16$ hoặc $5\times5$ filter $[1, 4, 6, 4, 1]^T [1, 4, 6, 4, 1] / 256$.

* **Thẩm định: Có giải quyết "Information Loss" hay chỉ "Smoothing"?**
  - **Sự thật toán học:** LPF về bản chất là bộ lọc triệt tiêu tần số cao ($H(f) \to 0$ khi $f \to f_s/2$). Do đó, **BlurPool KHÔNG phục hồi thông tin vi mô bị mất**, mà nó **chủ động làm mịn/làm mờ (low-pass smoothing)** để ngăn chặn hiện tượng gập tần số (aliasing artifacts) và ổn định gradient.
  - **Hệ quả đối với vết nứt mảnh:** Đối với cấu trúc siêu mỏng (1-2 pixel), việc làm mờ bằng LPF trước khi downsample làm giảm biên độ tương phản cục bộ (Signal-to-Noise Ratio - SNR) của vết nứt so với nền bê tông gồ ghề. BlurPool giúp mạng ổn định trước nhiễu dịch chuyển, nhưng **không làm tăng thông tin hình học vi mô** của vết nứt.

* **Chi phí tính toán & Tham số:**
  - **Tham số:** 0 tham số học được (nếu dùng fixed binomial kernel) hoặc $K^2$ tham số per-channel (nếu dùng learnable depthwise LPF).
  - **FLOPs:** Tăng không đáng kể (thêm 1 phép Depthwise Conv cố định trước khi subsample, chiếm $< 0.5\%$ tổng FLOPs của backbone).

---

### 1.2. Overlapping vs Non-overlapping Patch Embedding
* **Tài liệu tham chiếu:**
  - Dosovitskiy et al., *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"*, ICLR 2021 (ViT non-overlapping patchify).
  - Liu et al., *"A ConvNet for the 2020s"*, CVPR 2022 (ConvNeXt patchify stem $4\times4$, stride 4).
  - Xiao et al. (FAIR), *"Early Convolutions Help Transformers See Better"*, NeurIPS 2021.
  - Wang et al., *"PVT v2: Improved Baselines with Linear SVD for Vision Transformer"*, CVM 2022.

* **Phân tích cơ chế:**
  - **Non-overlapping Stem ($4\times4$, stride 4, padding 0):**
    * ConvNeXt vay mượn thiết kế này từ ViT nhằm mô phỏng patchification token. Mỗi khối $4\times4$ pixel được ánh xạ độc lập thành 1 vector đặc trưng 48 chiều.
    * **Khuyết điểm nghiêm trọng (Boundary Discontinuity / Phase Dependency):** Hai pixel nằm sát nhau ở hai bên ranh giới patch $4\times4$ bị gán vào hai token hoàn toàn tách biệt mà không có sự tương tác receptive field cục bộ nào ở tầng đầu tiên. Khi ảnh bị dịch chuyển 1-2 pixel, một cấu trúc vi mô có thể bị "cắt đôi" qua ranh giới patch, gây biến dạng feature map nghiêm trọng.
  - **Overlapping Stem (Conv $7\times7$ stride 2 + Pool, hoặc Micro-Stem xếp chồng $3\times3$ stride 2):**
    * Xiao et al. (NeurIPS 2021) đã chứng minh thực nghiệm rằng việc thay thế non-overlapping patchify bằng một chuỗi tích chập nhẹ (conv stem) giúp cải thiện độ ổn định tối ưu hóa (loss landscape phẳng hơn), tăng khả năng hội tụ và duy trì inductive bias không gian tốt hơn nhiều lần.
    * Trong overlapping stem, mỗi receptive field của điểm ảnh đầu ra bao phủ một vùng chồng lấn với láng giềng. Thông tin tại biên patch được chia sẻ liên tục (continuous spatial transitions), loại bỏ hoàn toàn hiện tượng "rách ranh giới" (grid artifacts).

* **Chi phí tính toán & Tham số:**
  - Non-overlapping Conv $4\times4$ ($C_{in}=3 \to C_{out}=48$): $3 \times 4 \times 4 \times 48 + 48 = 2,352$ tham số.
  - Micro-Stem 2 tầng $3\times3$ stride 2 ($3 \to 24 \to 48$):
    - Conv1 ($3 \to 24$, $3\times3$, s2, p1): $3 \times 3 \times 3 \times 24 + 24 = 672$ params.
    - Conv2 ($24 \to 48$, $3\times3$, s2, p1): $24 \times 3 \times 3 \times 48 + 48 = 10,416$ params.
    - Tổng: $\sim 11,088$ params (tăng $\approx 4.7\times$ tham số của stem, nhưng xét trên toàn mạng 6M params của ConvNeXtV2-Femto thì độ tăng $< 0.15\%$).
  - FLOPs tại $448\times448$: Tầng Conv1 chạy trên độ phân giải đầy đủ $448\times448$, sinh ra $\approx 135$ MFLOPs (hoàn toàn chấp nhận được trên GPU/Colab).

---

### 1.3. Downsampling trong Phân đoạn Vết nứt Mảnh (Thin / Curvilinear / Crack Segmentation)
* **Tài liệu tham chiếu:**
  - Yang et al., *"Feature Pyramid and Hierarchical Feature Fusion for Crack Detection"*, IEEE T-ITS 2021.
  - Badrinarayanan et al., *"SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation"*, IEEE TPAMI 2017.
  - Cheng et al., *"Boundary-preserving Dense Residual Network for Crack Detection"*, CACAIE 2021.
  - Alom et al., *"Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation"*, 2018.

* **Tác động của Stride-4 đối với vết nứt có bề rộng $< 4$ pixel trên Crack500:**
  1. **Hiệu ứng Thể tích Cục bộ (Partial Volume Effect):**
     Vết nứt trên Crack500 thường có bề rộng thực tế chỉ $1 - 3$ pixel. Khi một patch $4\times4$ (16 pixel) chỉ chứa 1-2 pixel vết nứt, diện tích vết nứt chỉ chiếm $6.25\% - 12.5\%$ diện tích patch. Phép tích chập không chồng lấn $4\times4$ stride 4 tính tổng có trọng số của cả 16 pixel; độ tương phản của vết nứt bị "hòa tan" hoàn toàn vào nhiễu kết cấu bê tông (aggregate texture, pores, shadows).
  2. **Đứt gãy Tô pô (Topological Disconnectivity & Skeleton Fracturing):**
     Một vết nứt liên tục dạng sợi (1D curvilinear manifold) nếu chạy chéo qua lưới patch $4\times4$ sẽ bị ngắt quãng tại những điểm mà năng lượng vết nứt bị suy giảm dưới ngưỡng kích hoạt của kernel. Kết quả là tại Stage 0, vết nứt liên tục bị biến thành chuỗi các điểm rời rạc (punctate activations). Mất mát tô pô ở tầng Stem là **không thể khôi phục (irreversible)**: UNet Decoder ở các tầng sau dù có skip connections cũng chỉ lấy lại được đặc trưng đã bị đứt gãy từ Stage 0.
  3. **Nhạy cảm với Pha Lấy mẫu (Sampling Phase Dependency):**
     Nếu vết nứt rơi vào tâm kernel $4\times4$, tín hiệu được ghi nhận; nếu rơi vào đúng ranh giới chuyển tiếp giữa 2 patch không chồng lấn, tín hiệu bị chia đôi và tiêu biến. Đây chính là nguyên nhân khiến mô hình dao động dự đoán mạnh khi ảnh chỉ bị rung rinh nhẹ.

---

## PHẦN 2: THẨM ĐỊNH CÁC YẾU TỐ GÂY NHIỄU (CAUSAL CONFOUNDING FACTORS)

### 2.1. Bản chất Hiện trạng Mã nguồn (Source Code Proof)
Trong codebase SAGE-Lite (`sage/networks/convnextv2_vit_hybrid.py`):
```python
# Minh chứng từ file: convnextv2_vit_hybrid.py dòng 75-97 & kiểm tra thực tế:
self.convnext = timm.create_model("convnextv2_femto.fcmae", pretrained=True)
# Cấu trúc Stem thực tế:
# Sequential(
#   (0): Conv2d(3, 48, kernel_size=(4, 4), stride=(4, 4), bias=True)
#   (1): LayerNorm2d((48,), eps=1e-06, elementwise_affine=True)
# )
```

### 2.2. Danh sách Các Nhân tố Bị Confounded khi Thay thế bằng Micro-Stem 2 tầng $3\times3$ stride 2
Nếu chỉ đơn giản tháo Stem cũ ra và lắp Micro-Stem 2 tầng vào rồi đo Dice/mIoU, thí nghiệm sẽ **vô giá trị về mặt khoa học** vì xảy ra hiện tượng **đa nhân tố đồng biến (conjoint confounding)**:

1. **Confounder 1: Overlap vs Non-overlap (Spatial Continuity):**
   Kernel $3\times3$ stride 2 có bước nhảy chồng lấn (overlap 1 pixel), trong khi $4\times4$ stride 4 không chồng lấn.
2. **Confounder 2: Độ sâu & Tính Phi tuyến (Depth & Non-Linearity):**
   Stem 1 tầng là một phép biến đổi afin cục bộ duy nhất trước LayerNorm. Micro-Stem 2 tầng thường chèn thêm 1 hàm kích hoạt (GELU/ReLU) và 1 hàm chuẩn hóa giữa 2 tầng conv, bổ sung khả năng học hàm phi tuyến bậc cao ngay tại Stem.
3. **Confounder 3: Trạng thái Khởi tạo (Pretrained vs Random Init Mismatch):**
   Stem gốc mang trọng số pretrained từ `fcmae` (ImageNet-1K). Micro-Stem mới bắt buộc phải khởi tạo ngẫu nhiên (Kaiming/He Normal). Nếu hiệu năng giảm, đó là do **mất trọng số pretrained** hay do **kiến trúc mới kém hơn**?
4. **Confounder 4: Bất đối xứng Learning Rate & Tương thích Stage 0 Đóng băng (Frozen Stage 0 Drift):**
   Nếu Stage 0 đóng băng (`freeze_encoder=True` hoặc fine-tuning với LR nhỏ), Stage 0 mong đợi tensor đầu vào nằm trên đa tạp đặc trưng đã tiền huấn luyện của ImageNet. Một Stem ngẫu nhiên đẩy các kích hoạt có trung bình/phương sai trôi dạt (distribution shift) vào Stage 0 đã đóng băng sẽ làm tê liệt toàn bộ luồng truyền tín hiệu qua `GlobalResponseNormMlp` của Stage 0.
5. **Confounder 5: Dung lượng Tham số & Receptive Field Gradient:**
   Stem 2 tầng tăng số tham số từ 2.3k lên 11k (gấp gần 5 lần). Tốc độ hội tụ của các tham số này dưới optimizer AdamW/SGD sẽ khác biệt đáng kể so với Stem 1 tầng.

---

### 2.3. Đề xuất Bộ Điều kiện Thực nghiệm Tối thiểu (Diagnostic Factorial Matrix - U0)
Để tách rời (disentangle) từng mối quan hệ nhân quả một cách độc lập, Auditor đề xuất thiết kế ma trận thực nghiệm **5 điều kiện tối thiểu (Minimal Causal Conditions)**:

| Điều kiện (Condition) | Cấu trúc Stem | Số tầng Conv | Kích hoạt giữa | Trọng số Stem | Stage 0 Backbone | Mục tiêu kiểm chứng nhân quả |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C1 (Baseline Pretrained)** | Conv $4\times4$, s4, p0 + LN | 1 | Không | Pretrained FCMAE | Giữ nguyên / Frozen | Mốc chuẩn đối sánh tối ưu hiện tại. |
| **C1b (Baseline Random Init)**| Conv $4\times4$, s4, p0 + LN | 1 | Không | Random Init (He) | Giữ nguyên / Frozen | Đo lường chính xác tổn thất do mất trọng số Pretrained (Tách Confounder 3). |
| **C2 (Single-layer Overlap)** | Conv $7\times7$, s4, p3 + LN | 1 | Không | Random Init (He) | Giữ nguyên / Frozen | **Tách biến Overlap đơn thuần:** Giữ nguyên 1 tầng, không thêm phi tuyến, chỉ đổi từ non-overlap sang overlap. |
| **C3 (Anti-Aliasing BlurPool)**| Conv $4\times4$, s1, p0 + BlurPool s4 + LN | 1 | Không | Random Init / Adapted | Giữ nguyên / Frozen | **Tách biến Anti-aliasing LPF:** Kiểm tra xem làm mịn chống răng cưa có bảo toàn được vết nứt hay làm mờ tín hiệu. |
| **C4a (Linear Micro-Stem)** | 2 tầng $3\times3$ s2 (Conv-LN-Conv-LN) | 2 | **Không có GELU** | Random Init | Giữ nguyên / Frozen | **Tách biến Progressive Downsampling:** 2 bước downsample tuyến tính thuần túy. |
| **C4b (Full Micro-Stem)** | 2 tầng $3\times3$ s2 (Conv-GELU-LN-Conv-LN) | 2 | **Có GELU** | Random Init | Giữ nguyên / Frozen | **Tách biến Phi tuyến (Non-linearity):** Đo lường đóng góp của độ sâu và hàm kích hoạt. |

---

### 2.4. Thẩm định Hợp đồng Tensor (Tensor Contract Verification)

1. **Kiểm tra Toán học Kích thước Không gian ($H, W$):**
   Công thức tổng quát:
   $$O = \left\lfloor \frac{I + 2P - K}{S} \right\rfloor + 1$$
   Với đầu vào $I = 448$:
   * **Stem gốc ($K=4, S=4, P=0$):**
     $$O = \left\lfloor \frac{448 + 0 - 4}{4} \right\rfloor + 1 = \frac{444}{4} + 1 = 111 + 1 = 112 \quad \text{(Chính xác [B, 48, 112, 112])}$$
   * **C2 Overlap $7\times7$ ($K=7, S=4, P=3$):**
     $$O = \left\lfloor \frac{448 + 6 - 7}{4} \right\rfloor + 1 = \left\lfloor \frac{447}{4} \right\rfloor + 1 = 111 + 1 = 112 \quad \text{(Khớp hoàn hảo contract)}$$
   * **C4 Micro-Stem 2 tầng ($3\times3$, $S=2$, $P=1$):**
     - Tầng 1: $O_1 = \lfloor (448 + 2 - 3)/2 \rfloor + 1 = 223 + 1 = 224$ (Feature map trung gian: $[B, 24, 224, 224]$).
     - Tầng 2: $O_2 = \lfloor (224 + 2 - 3)/2 \rfloor + 1 = 111 + 1 = 112$ (Đầu ra: $[B, 48, 112, 112]$).
     - Khớp chính xác contract yêu cầu của Stage 0.

2. **Hợp đồng Chuẩn hóa & Bias (LayerNorm2d Contract):**
   * Trong PyTorch/timm, `LayerNorm2d(48, eps=1e-6)` thực hiện chuẩn hóa trên trục Channel tại từng tọa độ không gian riêng lẻ:
     $$y = \frac{x - \mathrm{E}[x]_{c}}{\sqrt{\mathrm{Var}[x]_c + \epsilon}} \cdot \gamma + \beta$$
   * `bias=True` trong Stem Convolution là **bắt buộc** để duy trì tính tương thích phân phối nếu có sự chuyển tiếp sang Stage 0.
   * **Rủi ro kỹ thuật:** Nếu sử dụng `BatchNorm2d` trong Micro-Stem mới thay vì `LayerNorm2d`, sẽ xảy ra xung đột chuẩn hóa (normalization mismatch) với các khối ConvNeXt tiếp theo vốn sử dụng thuần túy LayerNorm2d, dẫn đến bất ổn định gradient khi batch size nhỏ (ví dụ $B=4$ hoặc $B=8$ trên Colab).

3. **Tính Tương thích với Stage 0 Đóng Băng (Frozen Stage 0 Integrity):**
   * Stage 0 của `convnextv2_femto` chứa khối `GlobalResponseNorm (GRN)`:
     $$G(X)_c = \frac{\|X_c\|_2}{\frac{1}{C}\sum_{c'} \|X_{c'}\|_2} + \beta$$
   * GRN cực kỳ nhạy cảm với sự chênh lệch độ lớn chuẩn (L2 norm) giữa các kênh. Nếu Stem mới khởi tạo ngẫu nhiên tạo ra các kênh kích hoạt không cân bằng, GRN của Stage 0 sẽ lập tức khuếch đại nhiễu hoặc triệt tiêu các kênh tín hiệu yếu.
   * **Kết luận thẩm định:** **TUYỆT ĐỐI KHÔNG ĐƯỢC đóng băng Stage 0 khi sử dụng Stem mới khởi tạo ngẫu nhiên.** Bắt buộc phải mở unfreeze Stage 0 (hoặc ít nhất là các tham số affine của LayerNorm và GRN) trong quá trình huấn luyện adaptation.

---

## TỔNG KẾT KHUYẾN NGHỊ DÀNH CHO MAIN AGENT & EXPERIMENT PIPELINE

1. **Về mặt Lý thuyết:** Không kỳ vọng BlurPool giải quyết triệt để bài toán mất thông tin vết nứt siêu mảnh. BlurPool chỉ mang lại tính ổn định shift-invariance. Để bảo toàn cấu trúc mỏng $< 4$px, hướng đi **Overlapping Convolutional Transition (C2 hoặc C4)** có nền tảng vững chắc hơn nhiều so với non-overlapping patchify.
2. **Về mặt Phương pháp luận:** Khi thực hiện giai đoạn U0, bắt buộc chạy đối sánh $C1 \to C1b \to C2$ trước tiên để xác định xem việc chuyển từ non-overlap sang overlap có mang lại lợi ích độc lập hay không trước khi phức tạp hóa mô hình thành Micro-Stem đa tầng.
3. **Về mặt Kỹ thuật:** Đảm bảo giữ nguyên chuẩn hóa `LayerNorm2d`, `bias=True`, padding đối xứng chuẩn ($P=3$ cho $K=7, S=4$; $P=1$ cho $K=3, S=2$), và luôn unfreeze Stage 0 nếu Stem là random-initialized.