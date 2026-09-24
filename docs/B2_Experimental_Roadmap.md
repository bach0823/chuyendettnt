# B2 Experimental Roadmap (SAGE-Lite)

Cập nhật kế hoạch thực nghiệm B2 theo roadmap sau. Mục tiêu là khảo sát có kiểm soát, không mở grid HPO rộng ngay từ đầu.

## Nguyên tắc chung

* Không dùng Test set để chọn hyperparameter.
* Validation là tiêu chí chọn cấu hình.
* Mỗi phase chỉ thay đổi nhóm biến cần khảo sát; các biến còn lại giữ nguyên baseline.
* Không tự ý sửa architecture/pipeline chỉ để cải thiện kết quả.
* Ghi lại đầy đủ config, checkpoint, Val Dice và các routing diagnostics cho từng run để có thể truy nguyên thí nghiệm.

## Phase 0 — Runtime preflight (HOÀN TẤT ✅)

Kết quả đo đạc thực tế trên Tesla T4 (14.56 GB usable, AMP FP16, 448×448, ViT Depth 12):
* **Batch 20 & Batch 14+**: ❌ **OOM** (Vượt ngưỡng VRAM 14.56 GB T4).
* **Batch 12 (Real Crack500 Data)**: ✅ **PASS** (12 training batches thật: Peak Alloc 13.84 GB, Peak Res 14.05 GB, **Free 0.51 GB**, Throughput 0.55 samples/s, 0 memory leak từ batch 2).
* **Batch 12 (Synthetic)**: ✅ **PASS** (12 iters: Peak Alloc 13.77 GB, Peak Res 14.10 GB, Free 0.47 GB, memory drift +0.08 MB).
* **Batch 10 (Synthetic)**: ✅ **PASS** (12 iters: Peak Alloc 13.43 GB, Peak Res 13.89 GB, Free 0.68 GB).
* **Batch 8 (Synthetic)**: ✅ **PASS** (12 iters: Peak Alloc 12.07 GB, Peak Res 12.50 GB, Free 2.07 GB).
* **Targeted Runtime Profiling (SageLayer 0 & 1 Focus, 10 Measured Steps)**: ✅ **PASS**
  - **Phát hiện Căn nguyên 43x Slowdown**: Khi ViT Layer gọi ViT Expert ($N=196$), thời gian chỉ **0.79 ms/call**. Nhưng khi Stage 0 hoặc Stage 1 gọi ViT Expert, chuỗi spatial tokens là **12,544 tokens**, khiến self-attention vọt lên **33.85 – 34.65 ms/call (chậm hơn 43 lần/call)**!
  - **Tập trung Chi phí**: Stage 0 & 1 gọi ViT experts 212 lần trong 10 steps, tiêu tốn 7,255 ms (>85% thời gian expert path của 2 tầng này).
  - **Accounting Reconciliation**: Tổng thời gian vi mô khớp **96.9% – 98.3%** thời gian Expert Path (Compute chiếm 91.8% – 94.5%, residual chỉ 1.7% – 3.1%).
* **ViT Scaling vs Local Window Attention Micro-Benchmark**: ✅ **PASS**
  - **Global ViT $\mathcal{O}(N^2)$**: Tại $N=12,544$ và $B=4$, tổng Fwd+Bwd bùng nổ lên **173.07 ms** (chậm gấp **72.2x** so với $N=196$).
  - **Local Window Attention ($W=7$) $\mathcal{O}(N)$**: Chỉ tốn **24.60 ms** tại $N=12,544$ $\implies$ **nhanh hơn 7.04 lần** so với Global ViT. Chi phí giảm từ $0.003449$ xuống $0.000490$ ms/token.
  - **Ranh giới OOM (Stress Test)**: Nhờ PyTorch SDPA, mô hình chịu tải tới 401,408 tokens ($B=32$, Peak VRAM 3.77 GB) mà không OOM, nhưng arithmetic intensity của Global ViT cao gấp 7.2x.
* **Spatial Compression before Global ViT Feasibility Micro-Benchmark**: ✅ **PASS**
  - **Giảm N qua AdaptiveAvgPool**: Từ $112 \times 112$ ($N=12544$, 177.48 ms ở $B=4$), nén xuống $56 \times 56$ ($N=3136$) tốn **15.84 ms (11.21x nhanh hơn)**; nén xuống $28 \times 28$ ($N=784$) chỉ tốn **3.36 ms (52.90x nhanh hơn)**; nén về $14 \times 14$ ($N=196$) tốn **3.16 ms (56.12x nhanh hơn)**.
  - **Tiết kiệm dự kiến**: Có thể cắt giảm tới 90% thời gian gọi ViT expert ở CNN stages mà vẫn giữ nguyên 100% pretrained weights của ViT-Tiny.
* **Kết luận Runtime Batch Size**: **`batch_size: 12`** và **`num_workers: 2`** chính thức được xác nhận khả thi và an toàn cho Full Training trên dữ liệu Crack500 thật.
* Chi tiết log xem tại: `docs/B2_Phase0_Preflight_Log.md` (Mục 9, 10, 11, 12).

| Hạng mục | Kết quả |
|---|---|
| Real Crack500 pipeline | PASS |
| B2-D12 / top-k=4 | PASS |
| Batch 12 OOM | Không |
| Forward/backward/optimizer | PASS (Đo tách bạch qua CUDA Events) |
| Numerical stability | PASS |
| 16 experts active | PASS (4.7% – 7.2%, 0 dead experts) |
| Peak VRAM (workers=2) | **13,610.5 MB (13.29 GB)** (Headroom ~0.95 GB) |
| Throughput (workers=2) | **2.26 img/s (~13.98 min / epoch)** |
| Có cần sửa architecture? | **Không** |

*Lưu ý:* Việc điều chỉnh batch size từ 20 xuống 12 là **hardware-constrained runtime setting**, không phải HPO/tuning result.

---

## Định Hướng Kiến Trúc Cốt Lõi: Tối Ưu Hóa Nút Thắt High-Resolution CNN→ViT (UPDATE 2026-09-24)

Sau khi đánh giá độc lập điểm nghẽn tính toán của các cuộc gọi CNN Stage 0/1 $\to$ ViT expert ($N=12,544$ và $N=3,136$ tokens), thứ tự ưu tiên và định hướng thực nghiệm chính thức được chốt như sau:

### 1. Thứ Tự Ưu Tiên Triển Khai (Priority Order)
1. **PRIMARY BASELINE $\to$ Proposal 3: Feature/Detail Enhancement $\to$ Spatial Compression**
2. **SECOND $\to$ Proposal 1: Spatial Reduction Attention (SRA)**
3. **THIRD $\to$ Proposal 2: Restricted High-Resolution Routing**

*Động lực định hướng*: Mục tiêu không thuần túy là tốc độ tối đa, mà là sự cân bằng tối ưu tổng thể giữa:
- Độ chính xác phân đoạn vết nứt (Crack segmentation accuracy).
- Khả năng bảo tồn chi tiết và vết nứt siêu mảnh (Thin-crack/detail preservation).
- Ngữ cảnh tầm xa (Long-range context).
- Tính tương thích với ViT-Tiny pretrained ImageNet.
- Triết lý định tuyến đa phương thức dị thể của SAGE (Heterogeneous routing).
- Runtime và VRAM thực tế trên GPU T4.
- Rủi ro kỹ thuật triển khai (Implementation risk).

---

### 2. Chi Tiết Từng Hướng Kiến Trúc

#### PROPOSAL 3 — PRIMARY BASELINE: Feature/Detail Enhancement $\to$ Spatial Compression
* **Quy trình luồng dữ liệu**:
  $$\text{CNN Stage 0/1 feature} \to \text{Lightweight learnable detail refinement} \to \text{Spatial compression} \to \text{Pretrained ViT global attention} \to \text{SA-Hub adaptation về CNN shape} \to \text{Residual fusion với Main Path}$$
* **Nguyên tắc bắt buộc**:
  - Module enhancement **CHỈ áp dụng trên nhánh CNN $\to$ ViT expert**.
  - **Main CNN path giữ nguyên 100% độ phân giải cao** ($112 \times 112$ và $56 \times 56$).
  - Các đặc trưng đưa vào Skip connections của UNet Decoder giữ nguyên vẹn 100%.
  - Khối ViT-Tiny giữ nguyên dạng black-box pretrained chuẩn từ `timm`, **tuyệt đối không sửa attention internals**.
  - Định tuyến đa phương thức (Heterogeneous routing) vẫn được kích hoạt đầy đủ.
* **Ý đồ kiến trúc**: Nén biểu diễn không gian trên expert branch để đạt hiệu năng cao, nhưng tinh chỉnh/tăng cường bằng chứng vết nứt trước khi nén để thông tin vết nứt mảnh không bị pha loãng không gian (spatial dilution).
* **Trạng thái**: Enhancement module là một giả thuyết cần được thiết kế cụ thể, nhẹ, khả học và dạng residual trên feature maps (không dùng Sobel/Laplacian cố định, không sửa RGB preprocessing). Kích thước nén đích cần được chốt trước khi code (Stage 0: $112 \to 56$ hoặc $28$; Stage 1: $56 \to 28$ hoặc $14$).

#### PROPOSAL 1 — SECOND PRIORITY: SRA (Spatial Reduction Attention)
* **Cơ chế**:
  - $Q$ giữ nguyên full spatial resolution ($112 \times 112 = 12,544$ tại Stage 0; $56 \times 56 = 3,136$ tại Stage 1).
  - $K, V$ được nén không gian (ví dụ $28 \times 28 = 784$ tại Stage 0).
  - Attention duy trì cơ chế toàn cục trên tập K/V nén.
* **Động lực chính**: Giữ lưới tọa độ Query/Output độ phân giải cao, giữ receptive field toàn cục, giảm chi phí ma trận $\mathcal{O}(N^2)$.
* **Lưu ý triển khai**: Cần can thiệp vào attention bên trong ViT block, tái sử dụng pretrained Q/K/V projections. Đặc biệt, khối ViT là module dùng chung (*shared expert*), vừa làm Bottleneck ($N=196$), vừa làm expert cho Stage 0/1 ($N=12,544$), do đó cần thiết kế API/cấu trúc an toàn để hỗ trợ cả hai chế độ mà không làm gãy vỡ đường truyền Bottleneck.
* **Vị trí**: Xếp thứ hai trong lộ trình, không phải mục tiêu triển khai ngay lập tức.

#### PROPOSAL 2 — THIRD PRIORITY: Restricted High-Resolution Routing
* **Quy tắc định tuyến**:
  - Stage 0 & Stage 1: Chỉ được chọn CNN experts (Indices 0..3).
  - Stage 2 & Stage 3: Giữ full CNN + ViT expert pool (16 experts).
  - Tất cả các tầng ViT (Layers 4..15): Giữ full 16 experts.
* **Lưu ý bắt buộc**:
  - Không sửa expert pool.
  - Không nén không gian, không sửa ViT block.
* **Điểm nghẽn cốt tử cần giải quyết trước khi code**: Cấu hình hiện tại là `top_k = 4` trong khi số CNN experts chỉ có 4. Nếu chỉ mask ViT logits, router bị buộc phải chọn tất cả 4 CNN experts cho mọi mẫu $\to$ mất hoàn toàn tính chọn lọc và đa dạng định tuyến. Phải có giải pháp cấu hình (ví dụ Dynamic `top_k = 2` cho Stage 0/1) và tài liệu hóa rõ ràng rằng điều này đưa vào một biến thực nghiệm bổ sung.

---

### 3. Các Nguyên Tắc Kiến Trúc Quan Trọng (Important Architectural Principles)
1. **Không coi 3 proposal là các mẹo vặt thay thế lẫn nhau**: Chúng trả lời 3 câu hỏi nghiên cứu độc lập:
   - *P3*: "Liệu ta có thể bảo tồn ngữ cảnh toàn cục đa phương thức trong khi nén biểu diễn của expert không?"
   - *P1*: "Liệu ta có thể giữ Query độ phân giải cao + ngữ cảnh toàn cục trong khi giảm chi phí attention của K/V không?"
   - *P2*: "Liệu ta có thực sự cần định tuyến CNN $\to$ ViT ở độ phân giải cao hay không?"
2. **Main CNN Path bắt buộc phải là nguồn gốc của chi tiết vết nứt độ phân giải cao**: Nhánh expert không thay thế main path.
3. **Không được mặc định rằng attention trên chuỗi lớn ($N=12,544$) sẽ tự động trở nên uniform**: Đây chưa phải là sự thật được xác lập trong hệ thống và không được dùng làm giả định thiết kế.
4. **Các kích thước không gian đã xác thực**:
   - Stage 0 = $112 \times 112 = 12,544$ tokens
   - Stage 1 = $56 \times 56 = 3,136$ tokens
   - Stage 2 = $28 \times 28 = 784$ tokens
   - Stage 3 / ViT bottleneck = $14 \times 14 = 196$ tokens
5. **Phân biệt rõ 4 loại tổn thất riêng biệt**:
   - Mất độ phân giải không gian (Loss of spatial resolution).
   - Mất ngữ cảnh toàn cục (Loss of global context).
   - Mất tương tác đa phương thức (Loss of cross-modal interaction).
   - Mất mát/nhòe do nội suy (Loss caused by interpolation).
6. **Các ước tính runtime từ microbenchmark chỉ là ước lượng**: Tuyệt đối không trình bày các phép ngoại suy tốc độ epoch như là hiệu năng end-to-end được đảm bảo chắc chắn.

---

### 3.1 Quy Tắc Hướng Định Tuyến Bất Biến Cốt Lõi (Core Routing Direction Invariant)
> ⚠️ **BẤT BIẾN KIẾN TRÚC BẮT BUỘC CHO CẢ 3 PROPOSALS:**
> Bất kỳ xử lý đặc biệt nào cho nhánh CNN high-resolution $\to$ ViT **CHỈ ĐƯỢC PHÉP KÍCH HOẠT KHI THỎA MÃN ĐỒNG THỜI CẢ 2 ĐIỀU KIỆN**:
> 1. **Tầng nguồn (Source SageLayer)** là CNN Stage 0 HOẶC CNN Stage 1 (`source ∈ {CNN Stage 0, CNN Stage 1}`).
> 2. **Chuyên gia đích được chọn (Selected Expert)** là một khối Transformer / ViT (`target ∈ {ViT Experts}`).
> 
> Biểu diễn toán tử logic:
> $$\mathbf{Trigger} \iff (\text{source} \in \{\text{CNN Stage 0}, \text{CNN Stage 1}\}) \land (\text{target} \in \{\text{ViT Experts}\})$$

* **TẤT CẢ CÁC HƯỚNG ĐỊNH TUYẾN CÒN LẠI BẮT BUỘC PHẢI GIỮ NGUYÊN HÀNH VI SAGE CHUẨN (Normal SAGE)**:
  - `CNN S0 → CNN expert` $\implies$ Normal SAGE.
  - `CNN S1 → CNN expert` $\implies$ Normal SAGE.
  - `CNN S2 → CNN expert` $\implies$ Normal SAGE.
  - `CNN S2 → ViT expert` $\implies$ Normal SAGE (trừ khi có proposal tương lai yêu cầu rõ ràng).
  - `CNN S3 → ViT expert` $\implies$ Normal SAGE (trừ khi có proposal tương lai yêu cầu rõ ràng).
  - `ViT → CNN S0/S1` $\implies$ **Normal SAGE 100%**.
  - `ViT → CNN S2/S3` $\implies$ Normal SAGE.
  - `ViT → ViT` $\implies$ Normal SAGE.

* **TẠI SAO ĐIỀU NÀY MANG TÍNH SINH TỬ**:
  - Điểm nghẽn bùng nổ tính toán xuất phát từ: `Feature CNN phân giải cao (112x112 / 56x56) → Chuyển sang dạng Transformer (12,544 tokens) → ViT Global Attention trên số tokens khổng lồ`.
  - Hướng ngược lại: `ViT → CNN Stage 0/1` xuất phát từ biểu diễn ViT phân giải thấp ($N=196$ tại bottleneck), nên **KHÔNG HỀ GÂY RA** điểm nghẽn attention độ phân giải cao!
  - **TUYỆT ĐỐI KHÔNG ÁP DỤNG ĐỐI XỨNG CƠ CHẾ NÉN/SRA CHO CẢ 2 CHIỀU.**
  - **Phân biệt rõ**: "Source CNN Stage 0/1" và "Target CNN Stage 0/1 Expert" là hai khái niệm hoàn toàn khác biệt. Khi một tầng ViT gọi chuyên gia Stage 0, Stage 0 là TARGET, không phải SOURCE. Không được kích hoạt cơ chế nén!

* **Áp dụng cụ thể cho từng Proposal**:
  - **P3 (Enhance + Compress)**: CHỈ nhánh `CNN S0/S1 → ViT` mới qua Detail Refinement + Spatial Compression. Main path, Router, Expert pool, ViT$\to$CNN, CNN$\to$CNN, S2/S3$\to$ViT giữ nguyên gốc. Tuyệt đối không nén toàn cục feature của Stage 0/1.
  - **P1 (SRA)**: CHỈ kích hoạt SRA khi ViT expert nhận tensor từ CNN S0/S1. Khi ViT block chạy ở bottleneck ($N=196$) hoặc được gọi bởi S2/S3, chạy standard attention chuẩn.
  - **P2 (Restricted Routing)**: CHỈ giới hạn quyền chọn của S0/S1 (`S0/S1 → CNN experts only`). Chiều ngược lại `ViT → CNN S0/S1` vẫn được phép 100% và chạy normal SAGE.

* **Yêu cầu triển khai (Implementation Requirement)**:
  - Bắt buộc kiểm tra tường minh theo cặp (Source, Target):
    ```python
    if source_is_cnn_stage_0_or_1 and target_is_transformer_expert:
        # Áp dụng cơ chế high-res CNN->ViT đặc thù của proposal
    else:
        # Chạy normal SAGE path
    ```
  - CẤM TUYỆT ĐỐI dựa vào: số chiều tensor đơn thuần, số lượng channels đơn thuần, chỉ số expert đơn thuần, hay logic generic kiểu `if transformer` hoặc `if Stage 0/1`.

---

### 4. Lộ Trình Thực Nghiệm Hiện Tại (Current Roadmap)
$$\text{B2 Current Baseline (Two-Stage Phase 1)} \implies \text{P3: Enhancement} \to \text{Spatial Compression} \implies \text{P1: SRA} \implies \text{P2: Restricted S0/S1 Routing}$$

---

## Phase 1 — Khảo sát ViT depth (Baseline Scale) (ĐÃ SẴN SÀNG CẤU HÌNH ✅)

Do thay depth ảnh hưởng đến architecture và routing scale, Phase 1 chạy đầu tiên để lock base architecture.

* **Cố định dùng chung**:
  - `batch_size = 12` (runtime setting đã xác thực qua Real-Data Preflight trên T4)
  - `img_size = 448`, `seed = 42`, `lr = 1e-4`, `epochs = 30`, `patience = 6`
  - Canonical preprocessing (Crack500 random crop 448x448, smart filter `fg_pixels >= 20`, reflect pad)
  - SAGE config: `top_k = 4`, `hidden = 64`, `gating = sigmoid`, `noise = ON`, `logit_mod = ON`, `LB = 0.01`, `dropout = 0.1`, `fusion_type = residual`, `residual_scale = 0.1`.
* **Bộ cấu hình thực nghiệm Phase 1**:
  * **Run 1: Depth 12** (`configs/b2_crack500_depth12.yaml` → `output_dir: .../B2_Crack500_Depth12`): 16 routers (4 CNN + 12 ViT), 16 experts, 13.9M params.
  * **Run 2: Depth 6** (`configs/b2_crack500_depth6.yaml` → `output_dir: .../B2_Crack500_Depth6`): 10 routers (4 CNN + 6 ViT), 10 experts, 12.0M params.
  * **Run 3: Depth 4** (`configs/b2_crack500_depth4.yaml` → `output_dir: .../B2_Crack500_Depth4`): 8 routers (4 CNN + 4 ViT), 8 experts, 10.1M params.
* **Quy tắc Quyết định**: Chọn 1 depth tốt nhất **dựa duy nhất trên Validation Dice (tập Val)** để làm base cho Phase 2 (top_k). **TUYỆT ĐỐI KHÔNG DÙNG TEST SET ĐỂ CHỌN DEPTH**.


## Phase 2 — Khảo sát Routing Capacity (top_k)

* Fix: Base depth từ Phase 1.
* Thử nghiệm (thay đổi top_k):
  * Run 4: top_k = 2
  * Run 5: top_k = 4 (đã chạy ở Phase 1)
* Decision: Chọn top_k tốt nhất. (Nếu top_k=2 gần bằng top_k=4, ưu tiên top_k=2 vì FLOPS thấp hơn).

## Phase 3 — Khảo sát Router Hidden Dim

* Fix: Base depth (P1) + Best top_k (P2).
* Thử nghiệm:
  * Run 6: router_hidden_dim = 32
  * Run 7: router_hidden_dim = 64 (từ P1)
  * Run 8: router_hidden_dim = 128
* Decision: Chọn hidden_dim.

## Phase 4 — Cân bằng tải (Load Balancing)

Chỉ mở phase này nếu log cho thấy expert bị "dead" hoặc mất cân bằng nghiêm trọng. Chú ý: LB loss scale theo số lượng routers, nên Depth 12 vs Depth 6 sẽ có total LB loss khác nhau. Phải soi trung bình LB/router.

* Fix: Base config từ P3.
* Thử nghiệm (LB weight):
  * Run 9: LB = 0.005
  * Run 10: LB = 0.01 (từ P1)
  * Run 11: LB = 0.03

## Phase 5 — Optimization Stability (SAGE LR & Warmup)

Nhóm parameters của SAGE (Routers + Adapters) thường cần LR khác backbone.
Mặc định hiện tại: Backbone=1e-5, Decoder=1e-4.

* Thử nghiệm (SAGE LR):
  * Run 12: SAGE LR = 5e-5
  * Run 13: SAGE LR = 1e-4
  * Run 14: SAGE LR = 2e-4
* Thử nghiệm Warmup epochs (nếu thấy training bị spike loss đầu epoch):
  * Run 15: Warmup = 2
  * Run 16: Warmup = 3
  * Run 17: Warmup = 5

## Phase 6 — Regularization (Nghiên cứu thêm)

* Thử nghiệm dropout cho adapter:
  * expert_dropout = {0.0, 0.1, 0.2}
* Thử nghiệm pure residual scale:
  * residual_scale = {0.05, 0.1, 0.2}

Quy trình: Đưa cho tôi script preflight và file kế hoạch này. Khi nào chuẩn bị xong Colab tôi sẽ chạy Phase 0 và gửi log.

## Phase A — Khảo sát Residual Scale
* Fix: fusion_type = "residual"
* Cố định các biến khác (dropout=0.1, top_k, depth, router, v.v.)
* Thử nghiệm residual_scale:
  * Run A1: residual_scale = 0.05
  * Run A2: residual_scale = 0.10 (Baseline từ trước)
  * Run A3: residual_scale = 0.20
* Decision: Chọn residual_scale tốt nhất trên Validation.

## Phase B — So sánh Fusion Type (Residual vs Adaptive)
* Fix: Chọn residual_scale tốt nhất từ Phase A cho Residual variant.
* Adaptive Fusion variant: Sử dụng `alpha` (learnable scalar).
* Đảm bảo cách ly (isolation): giữ nguyên cùng ViT depth, expert pool, top_k, router hidden, LB factor, learning rate, preprocessing, random seed, v.v.
* Thử nghiệm so sánh:
  * Run B1: Residual Fusion (với scale tốt nhất Phase A)
  * Run B2: Original SAGE Adaptive Fusion (với learnable alpha)
* Decision: Chọn Fusion mechanism.
