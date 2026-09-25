# Project State: SAGE-Lite: A Lite Version of Shape-Adapting Gated Experts for Crack Binary Segmentation

## Deadline
Deadline: N/A
Days Remaining: N/A
Risk Level: Safe

## Workflow: Repo-Based (Colab Driver Pattern)
- Source code nằm trong Git repo độc lập: `sage_lite/` (private GitHub repo).
- Colab chỉ đóng vai trò driver: Clone repo → `pip install -e .` → gọi entry point training.
- Mỗi run có thể truy ra từ commit + config YAML tương ứng.
- Checkpoint và log lưu vào Google Drive để không mất khi Colab session reset.

## Current Progress

Current Milestone:
Milestone 2: SAGE Core Mechanism & B2 (Full SAGE-Lite)

Current SK:
Hoàn thành giải quyết triệt để 2 launch-path inconsistencies và siết chặt cổng kiểm định nguồn gốc Locked Base:
1. `p3_mode` propagation: `scripts/train_crack.py` đã truyền chính xác `p3_mode=config.get('p3_mode', None)` vào constructor `create_b2_unet` và bổ sung `--locked-base` CLI flag.
2. PE28 Provenance & Initialization Alignment: `scripts/preflight_p3_realdata.py` đã đồng bộ `pretrained=True`, kiểm chứng derivation toán học 2D bicubic 14x14 -> 28x28 từ PE14, và bổ sung cổng kiểm định xuất xứ Locked Base (Hàng N trong bảng báo cáo). Nếu không truyền `--locked-base`, preflight sẽ cảnh báo và chặn lại với phán quyết `GATED (Locked Base Checkpoint required)`. Khi truyền `--locked-base`, preflight trích xuất và đối chiếu bitwise trực tiếp với `backbone.positional_embeddings` trong file checkpoint thật.
3. Huấn luyện 30 epoch chính thức giữ nguyên trạng thái GATED cho đến khi chạy Preflight với checkpoint cụ thể trên Tesla T4.

State:
P3 Launch-Path Inconsistencies Resolved & Strict Locked-Base Provenance Gate Enforced (Commit `31f4390` pushed to `crack500-audit`). Ready for Colab T4 Preflight with `--locked-base`.

### ⚠️ QUY TẮC BẮT BUỘC: PREPROCESSING CHÍNH THỨC ĐÃ KHÓA (FROZEN CANONICAL)
> **TUYỆT ĐỐI KHÔNG ĐƯỢC TỰ Ý THAY ĐỔI, THÊM/BỚT BẤT KỲ BƯỚC PREPROCESSING NÀO (crop, padding, resize, augmentation, mask processing) CHO ĐẾN KHI HOÀN THÀNH TOÀN BỘ SAGE-LITE.**  
> Chỉ thay đổi khi người dùng CHỦ ĐỘNG yêu cầu làm ablation hoặc sửa protocol.
> - **Crack500 Train:** Reflect Pad (nếu cần) → RandomCrop 448×448 → smart filter `fg_pixels >= 20` (retry max 20 crop attempts × 10 source resamples) → Augmentation (`HorizontalFlip`, `VerticalFlip`, `RandomRotate90`, `RandomBrightnessContrast`, `GaussianBlur`) → ImageNet Normalize → Tensor.
> - **Crack500 Eval:** Tiling Setting A (448×448 non-overlap) & Setting B (448×448 overlap 50%, stride 224, average probabilities → threshold 0.5).
> - **DeepCrack Train & Eval:** Mask `{0,255} → {0,1}` → Dynamic Pad-to-Square (`target = max(H, W, 448)`, BORDER_CONSTANT=0) → Resize 448×448 (`cv2.INTER_NEAREST` mask) → cùng bộ Augmentation trên (chỉ train) → ImageNet Normalize → Tensor. Eval chạy direct 1-pass full-image.
> - **Augmentation đã loại bỏ hoàn toàn:** `CLAHE`, `ElasticTransform`, `GridDistortion`, `ShiftScaleRotate`, `HueSaturationValue`.

Completed:
- Đã chốt kiến trúc SAGE-Lite (Implementation Plan).
- Xác định xong Bảng biến thực nghiệm (Nhóm 1 khoá, Nhóm 2 đo).
- Hoàn thiện Migration Map theo file tree.
- SK1 Hoàn thành: Shape audit pass chuẩn input 448x448, Femto channels [48, 96, 192, 384], 196 spatial tokens.
- SK2 Hoàn thành: `sage_lite/sage/networks/convnextv2_vit_hybrid.py` (Femto + ViT-Tiny dynamic blocks support).
- SK3 Hoàn thành: `decoder_block.py` + `b0_unet.py` + `train_crack.py` (hỗ trợ single-stage, loss BCE+SoftDice, optimizer AdamW, Cosine Annealing warmup).
- **B0 Baseline Training HOÀN TẤT trên cả 2 dataset:**
  + **Crack500 (Run-3):** Best Val Dice = **0.7318** (@ epoch 8). Test Setting A: Dice **0.6771**, IoU 0.5645. Test Setting B: Dice **0.6801**, IoU 0.5682.
  + **DeepCrack (Run-1):** Best Val Dice = **0.6240** (@ epoch 11). Test Direct: Dice **0.6953**, IoU 0.6041, Boundary IoU 0.1365, HD95 30.89 px.
- Infrastructure Migration (Repo-based Workflow) Hoàn thành trên branch `crack500-audit`.
- **ViT-Depth Sweep {4, 6, 8, 12} & Official Eval B1 trên Crack500 HOÀN TẤT 100% (Screening Phase):**
  + Đã huấn luyện đủ 4 cấu hình: Depth 4 (Val Dice **0.7428**, Val Loss **1.1364** - Top 1), Depth 6 (0.7420), Depth 8 (0.7379), Depth 12 (0.7419).
  + Đã chạy Official Evaluation Setting A & Setting B cho toàn bộ các checkpoint `best_model_b1.pth`.
  + **Interpretation & Chiến lược:** Val Dice của Depth 4/6/12 ngang nhau (~0.742). Test Dice của Depth 12 cao hơn nhưng chỉ phản ánh representation thuần (chưa có SAGE). Do đó **CHƯA freeze ViT depth từ B1**; B1 đóng vai trò screening. Depth tối ưu cho SAGE-Lite sẽ được đo đạc và quyết định qua **B2 Depth Ablation** (tập trung vào `{4, 6, 12}`).
- **Milestone 2: Triển khai Full B2 Architecture HOÀN TẤT 100%:**
  + [x] Khóa kỹ thuật #1 (Strict Tensor Contract): `forward()` trả pure Tensor, thu thập đủ $4 + N_{\text{vit}}$ routing info qua `_last_routing_info`.
  + [x] Khóa kỹ thuật #2 (Zero Dynamic Params): Trainable parameters bất biến tuyệt đối trước/sau forward pass.
  + [x] Pure Residual Fusion: $y = x_{\text{main}} + \text{Dropout}(0.1 \times x_{\text{expert}})$.
  + [x] Zero-cost Self-Selection Bypass `my_index`: Tự bypass khi router chọn chính layer hiện tại.
  + [x] 3 Optimizer Param Groups: Backbone ($10^{-5}$), Decoder ($10^{-4}$), SAGE ($10^{-4}$). Weight decay $0.0$ cho LN/BN/bias, $0.05$ cho rest.
  + [x] Training Loss: $\mathcal{L} = \mathcal{L}_{\text{seg}} + 1.0 \times \mathcal{L}_{\text{balance}}$.
  + [x] Configs: `configs/b2_crack500_depth12.yaml` và `configs/b2_crack500_depth6.yaml`.
  + [x] Verification: Toàn bộ 10 smoke tests trong `scripts/scratch/verify_b2.py` PASS 100% dưới AMP FP16.
  + [x] Independent Cross-Check Audit: Hoàn tất với 25/25 requirements PASS, review log lưu tại `docs/B2_Audit_Review_Log.md`.
  + [x] **Phase 0 Runtime Preflight & Throughput Profiling trên Colab Tesla T4 (HOÀN TẤT 100%):**
    - Batch 20 & Batch 14+: OOM (vượt 14.56 GB VRAM T4).
    - **Batch 12 trên Real Crack500 (12 training batches thật):** ✅ **PASS** (Peak Alloc 13.84 GB, Peak Res 14.05 GB, **Free 0.51 GB**, 0 memory leak từ batch 2, 16/16 experts chọn đều 5.0% - 6.9%).
    - **Deep Runtime Profiler (`profile_deep_b2.py` - Chạy trực tiếp trên Colab T4):**
      + **B1-D12 Baseline**: Step 206.6 ms (Fwd 60.5 ms, Bwd 137.7 ms), Throughput 58.07 img/s, Est 1 epoch 0.54 min (~32s), Peak VRAM 2.42 GB.
      + **B2-D12 SAGE-Lite**: Step 5433.0 ms (Fwd 1487.4 ms, Bwd 3934.7 ms), Throughput 2.21 img/s, Est 1 epoch 14.31 min, Peak VRAM 13.61 GB.
      + **Phát hiện Cốt lõi (Resolution Bottleneck)**: Stage 0 (490.2ms) + Stage 1 (426.1ms) chiếm **61.6% thời gian Forward** do phân giải cao ($112 \times 112$ và $56 \times 56$). Cả 12 ViT blocks chỉ chiếm ~420ms (ít hơn 1 mình Stage 0).
      + **Minh oan cho SA-Hub**: SA-Hub adapt in/out chỉ chiếm 8.7% expert loop, index_add_ chiếm 2.1%. **89.1% thời gian là tính toán FLOPs thực tế trong Expert modules**.
      + **Phân phối Routing**: 16/16 experts cân bằng 4.7% – 7.2% (uniform target 6.25%). Cross-modal chiếm 36.3%.
      + **_infer_expert_type CPU**: Chỉ tốn 14.58 ms/step (0.27% tổng step time).
    - **Targeted Runtime Profiler (`profile_targeted_stages.py` - SageLayer 0 & 1 Focus, 10 Steps Colab T4):**
      + **Nguồn gốc Slowdown 43x**: Khi ViT Layer gọi ViT Expert ($N=196$), thời gian chỉ **0.79 ms/call**. Nhưng khi Stage 0 hoặc Stage 1 gọi ViT Expert, chuỗi tokens là **12,544 tokens**, khiến self-attention vọt lên **33.85 – 34.65 ms/call (chậm hơn 43 lần/call)**!
      + **Tập trung Chi phí**: Stage 0 & 1 gọi ViT experts 212 lần trong 10 steps, tiêu tốn 7,255 ms (>85% thời gian expert path của 2 tầng này).
      + **Micro-timing Accounting**: Khớp **96.9% – 98.3%** thời gian Expert Path (Compute chiếm 91.8% – 94.5%, residual chỉ 1.7% – 3.1%).
    - **ViT Scaling vs Local Window Attention Micro-Benchmark (`benchmark_vit_scaling.py` - Colab T4):**
      + **Global ViT $\mathcal{O}(N^2)$**: Tại $N=12,544$ và $B=4$, tổng Fwd+Bwd bùng nổ lên **173.07 ms** (chậm gấp **72.2x** so với $N=196$).
      + **Local Window Attention ($W=7$) $\mathcal{O}(N)$**: Chỉ tốn **24.60 ms** tại $N=12,544$ $\implies$ **nhanh hơn 7.04 lần** so với Global ViT. Chi phí giảm từ $0.003449$ xuống $0.000490$ ms/token.
    - **Spatial Compression before Global ViT Feasibility Micro-Benchmark (`benchmark_spatial_compression.py` - Colab T4):**
      + **Hiệu quả nén qua AdaptiveAvgPool**: Từ $112 \times 112$ ($N=12544$, 177.48 ms ở $B=4$), nén xuống $56 \times 56$ ($N=3136$) tốn **15.84 ms (11.21x nhanh hơn)**; nén xuống $28 \times 28$ ($N=784$) chỉ tốn **3.36 ms (52.90x nhanh hơn)**; nén về $14 \times 14$ ($N=196$) tốn **3.16 ms (56.12x nhanh hơn)**.
      + **Bảo toàn Pretrained Weights**: Giữ nguyên 100% cấu trúc và weights của pretrained ViT-Tiny block, giải quyết trọn vẹn điểm nghẽn sequence length của các nhánh CNN $\to$ ViT.
    - Bảng nghiệm thu: Real Crack500 pipeline (PASS), B2-D12 / top-k=4 (PASS), Batch 12 OOM (Không), Forward/backward/opt (PASS), Numerical stability (PASS), 16 experts active (PASS), VRAM (An toàn với đệm ~0.95 GB), Throughput (13.98 min/epoch), Cần sửa architecture? (Không).
    - Chi tiết log lưu tại: `docs/B2_Phase0_Preflight_Log.md` (Mục 9, 10, 11, 12).


In Progress:
- Phase 1: ViT-depth ablation suite trên Crack500 (Google Colab T4, `batch_size: 12`, `num_workers: 2`, `--two-stage`):
  + Run 1: B2 Depth 12 (`configs/b2_crack500_depth12.yaml`)
  + Run 2: B2 Depth 6 (`configs/b2_crack500_depth6.yaml`)
  + Run 3: B2 Depth 4 (`configs/b2_crack500_depth4.yaml`)
  + Đánh giá và chọn depth tốt nhất dựa DUY NHẤT trên Validation Dice (không dùng test-set).
- **Cập nhật Định hướng Kiến trúc Giải quyết Nút thắt High-Resolution CNN→ViT:**
  + Đã hoàn thành đánh giá độc lập 3 proposal cho nút thắt Stage 0/1 ($N=12,544$ và $N=3,136$) gọi ViT expert.
  + **Thứ tự ưu tiên nghiên cứu & triển khai đã chốt:**
    1. **PRIMARY BASELINE $\to$ Proposal 3: Feature/Detail Enhancement $\to$ Spatial Compression**
       * Refine đặc trưng vết nứt mảnh trên feature map trước khi nén không gian $\to$ nạp vào ViT-Tiny pretrained global attention $\to$ SA-Hub adapt về CNN shape $\to$ residual fusion với main path $112 \times 112$.
       * Chỉ tác động trên nhánh CNN$\to$ViT expert. Main CNN path và UNet skips giữ nguyên 100%. ViT block giữ nguyên dạng black-box pretrained.
    2. **SECOND $\to$ Proposal 1: Spatial Reduction Attention (SRA)**
       * $Q$ giữ full resolution ($112 \times 112 = 12,544$), $K, V$ được nén không gian (ví dụ $28 \times 28 = 784$). Global attention trên tập K/V nén.
       * Cần sửa logic attention trong ViT block, tái sử dụng pretrained Q/K/V projections và hỗ trợ an toàn cho shared expert bottleneck ($N=196$).
    3. **THIRD $\to$ Proposal 2: Restricted High-Resolution Routing**
       * Giới hạn Stage 0 và 1 chỉ route tới CNN experts. Stage 2/3 và ViT layers giữ full 16 experts.
       * Cần xử lý triệt để bài toán $top\_k=4$ trên tập chỉ có 4 CNN experts (nguy cơ sụp đổ entropy và forced selection).
  + **Nguyên tắc kiến trúc cốt lõi:**
    * Main CNN path là nguồn cung cấp chi tiết vết nứt sắc nét cho UNet skip connections; expert branch không thay thế main path.
    * Phân biệt rõ các loại tổn thất: mất độ phân giải (resolution), mất ngữ cảnh (context), mất tương tác đa phương thức (cross-modal), mất do nội suy (interpolation).
    * Các số liệu microbenchmark là ước tính tham khảo, không coi là end-to-end speedup bảo đảm.
    * Kích thước không gian thực tế: Stage 0 = $112 \times 112$ ($12,544$ tokens), Stage 1 = $56 \times 56$ ($3,136$ tokens), Stage 2 = $28 \times 28$ ($784$ tokens), Stage 3 / ViT = $14 \times 14$ ($196$ tokens).

Knowledge Being Learned:
- Cơ chế Routing đa chuyên gia (MoE), tính ổn định số học trong Softmax/Sigmoid gating dưới AMP FP16, giảm thiểu overhead của self-selection qua bypass `my_index`.
- Quy trình quản lý thực nghiệm bằng Git commit + config YAML để đảm bảo tính tái lập (reproducibility).
- Đo đạc biên giới hạn phần cứng (VRAM profiling) và tách bạch giữa hardware-constrained runtime settings vs HPO.
- Kỹ thuật phân rã throughput bằng CUDA Events: nhận diện chính xác Backward computation bottleneck trong mạng MoE thay vì đoán mò về I/O hay DataLoader.
- Động lực học định tuyến và nguyên lý bảo tồn tín hiệu vết nứt mảnh trong mạng MoE lai CNN-Transformer ở độ phân giải cao.

Current Issue:
- Không có issue. Two-Stage Training & Optimizer Preflight đã PASS 100% cho cả 3 cấu hình Run A, Run B và Run C (343/343 tensors khớp tuyệt đối, 0 missing, 0 duplicates, shared experts cô lập chuẩn ở CNN main blocks, Stage 2 LR ratio 1:1 bảo toàn).

Next Step:
- Tiến hành thực thi P3-PHASE-7: Huấn luyện chính thức 3 cấu hình Run A (`b2_p3_run_a.yaml`), Run B (`b2_p3_run_b.yaml`), Run C (`b2_p3_run_c.yaml`) trên Google Colab T4 theo giao thức Two-Stage Training đã khóa.
- Thu thập metrics trên Validation set để phục vụ P3-PHASE-8 Decision Gate (tuyệt đối không truy cập tập Test).


## Milestones & SKs (Dependency-order)

### Milestone 0 (đã thêm): Infrastructure & Repo Setup
- [x] Đổi tên `SAGE_LITE` → `sage_lite`, khởi tạo Git repo độc lập, thêm `pyproject.toml`.
- [x] Phase 2: Tạo `train_crack.py` (CLI: --config, --resume, --output_dir), driver notebook Colab T4, configs cho Crack500 và DeepCrack.

### Milestone 1: Diagnostics & Core Backbone (B0 Baseline)
- [x] SK1: Viết và chạy `scripts/shape_audit.py` (channels [48,96,192,384], 196 tokens, xử lý CLS token).
- [x] SK2: Code `sage/networks/convnextv2_vit_hybrid.py` và verify forward pass.
- [x] SK3: Code `sage/networks/decoder_block.py` + hoàn thiện B0 UNet (`b0_unet.py`).
- [x] **Huấn luyện và nghiệm thu B0 Baseline:**
  - [x] Crack500 Run-3: Val Dice 0.7318, Test Setting A Dice 0.6771, Setting B Dice 0.6801.
  - [x] DeepCrack Run-1: Val Dice 0.6240, Test Direct Dice 0.6953.

### Milestone 1.5: Huấn luyện Baseline B1 & ViT-Depth Screening
- [x] Huấn luyện B1-6blocks trên Crack500 (PASS: Val Dice **0.7420**, Test Dice Setting A **0.6857**, Setting B **0.6895**, HD95 **77.43 px**)
- [x] **ViT-Depth Screening trên B1 (Crack500)**: Hoàn thành sweep tập `{4, 6, 8, 12}` trên tập Validation & Official Test:
  + [x] B0: 0 blocks — Val Dice 0.7318, Val Loss 1.3962
  + [x] B1 Depth 4: Val Dice **0.7428**, Val Loss **1.1364** (Top 1 Val), Test Dice A 0.6829, Test Dice B 0.6867
  + [x] B1 Depth 6: Val Dice 0.7420, Val Loss 1.1453 (Top 2 Val), Test Dice A 0.6857, Test Dice B 0.6895
  + [x] B1 Depth 8: Val Dice 0.7379, Val Loss 1.2006 (Top 4 Val), Test Dice A 0.6819, Test Dice B 0.6854
  + [x] B1 Depth 12: Val Dice 0.7419, Val Loss 1.2332 (Top 3 Val), Test Dice A 0.6902, Test Dice B 0.6926
  + [x] **Chiến lược:** Chưa freeze ViT depth từ B1; giữ nguyên dữ liệu screening, chuyển việc chốt depth sang **B2 Depth Ablation `{4, 6, 12}`** khi có SAGE routing.
- [x] Đánh giá Test Setting A & B cho toàn bộ sweep B1 trên Crack500: HOÀN TẤT.
- [ ] Huấn luyện B1 trên DeepCrack (Direct eval)

### Milestone 2: SAGE Core Mechanism & B2 (Full SAGE-Lite)
- [x] SK4: Port core mechanism (`sage/components/router.py`, `sage_layer.py`, `sa_hub.py`). Áp dụng fix eps 1e-5, FP32 logit modulation, bypass `my_index`, residual fusion scale 0.1.
- [x] SK5: Code `sage/networks/sage_injection.py` (Wiring full injection + isinstance guard). Hoàn thiện mô hình B2 (`b2_unet.py`) với Lock #1 (Tensor Contract) và Lock #2 (Zero Dynamic Params). Sẵn sàng huấn luyện.

### Milestone 3: Data Pipeline & Training Setup
- [x] SK6: Cập nhật `sage/utils/dataloader.py` và `losses.py` (DeepCrack pad+resize, Crack500 RandomCrop+smart filter, kết hợp BCE+SoftDice).
- [x] SK7: Viết `sage/utils/training_utils.py` và `scripts/train_crack.py` (Single-stage, AdamW, Cosine Annealing warmup, differential LR).

### Milestone 4: Diagnostic Scripts & Baseline Ladder
- [ ] SK8.6: **Generic Routing Diagnostics** (Top-K Activation Map, Expert Usage Ratio, Routing Entropy/Concentration).
- [ ] SK9: Cập nhật metrics trong evaluation (Dice primary, Boundary IoU, HD95).
- [ ] SK10: Chạy Baseline Ladder B0 → B2. **Quy tắc**: Train ĐỘC LẬP từ ImageNet gốc, KHÔNG warm-start từ bậc trước.
  - **B0**: ConvNeXtV2-Femto + U-Net (Hoàn thành trên Crack500 & DeepCrack)
  - **B1**: B0 + 6 ViT-Tiny (ConvNeXtV2-Femto + 6 ViT-Tiny + U-Net thuần, Late Fusion, không SAGE) — *[Đang tiến hành]*
  - **B2 (Full SAGE-Lite)**: B1 + full SAGE mechanism (SAGE Router + heterogeneous Expert Pool + SA-Hub + Load-Balance Loss)
- [ ] SK11: Ablation test: Sparse N_injection.
- [ ] SK12: Ablation test: Exploration Noise ON vs OFF

## Bảng chốt biến thực nghiệm (Configuration Map)

### Nhóm 1: Khóa CỨNG (đổi = phải sửa code/logic, không đổi tùy tiện)
- **ViT blocks**: 6
- **num_shared_experts**: 4 (auto = 4 CNN stages)
- **router_hidden_dim**: 64
- **top_k**: 4
- **N_injection**: Full (10 routers ở baseline 6 blocks)
- **Decoder conv**: Standard 3x3 (DWSC chỉ là flag tắt)
- **Fusion**: Residual (main + adapter), không alpha
- **SA-Hub**: O(D²) eager pairwise, giữ nguyên gốc
- **Gating**: sigmoid, khóa cứng, không ablation
- **Loss**: 1.0*BCE + 1.5*SoftDice + 1.0*L_balance, không pos_weight
- **load_balance_factor**: 0.01
- **expert_dropout**: 0.1
- **freeze_encoder/transformer**: False/False
- **Stage2 LR ratio**: 1:1 (chỉ nghiêng khi thấy collapse)
- **AMP dtype**: FP16 (không BF16 vì dùng T4)
- **Warmup**: 2-3 epoch
- **EarlyStopping patience**: 5-7
- **Weight decay**: 0.05, param groups tách LN/bias
- **Seed**: 42 cố định
- **Exploration Noise**: Giữ nguyên baseline = ON (giống code gốc). Ablation bật/tắt để riêng ra SK sau, không phải quyết định kiến trúc mặc định.

#### Nhóm 2: Cấu hình Thực nghiệm Đã Chốt
- **Batch size**: **B0 = 20** (đã benchmark T4), **B1 = 16** (config chính thức an toàn; bài học kinh nghiệm: các architecture khác nhau không được tự động dùng chung batch size nếu chưa benchmark riêng).
- **Base LR**: **1e-4** (Backbone LR: **1e-5**, Decoder LR: **1e-4**).
- **Training Budget**: **Epoch budget = 30** (ngân sách tối đa, không phải con số cố định bắt buộc chạy đủ), **EarlyStopping patience = 6**.
- **Crack500 smart filter**: **fg_pixels >= 20** (max 20 crop attempts × 10 source resamples).

## Migration Map

```text
sage-lite/
├── configs/
│   ├── datasets/
│   │   ├── colon.yaml, ebhi.yaml, glas.yaml          [DELETE]
│   │   ├── deepcrack.yaml, crack500.yaml              [NEW]
│   ├── experiments/
│   │   ├── baseline_*.yaml, sage_*.yaml (gốc)         [DELETE]
│   │   ├── b0_crack500.yaml, b0_deepcrack.yaml        [DONE] — Baseline B0
│   │   ├── b1_crack500.yaml, b1_deepcrack.yaml        [NEW] — Baseline B1
│   │   ├── b2_sage_lite.yaml                          [NEW] — B2 (Full SAGE-Lite)
├── prepare_data/
│   ├── prepare_colon*.py, prepare_ebhi.py, prepare_glas.py  [DELETE]
│   ├── prepare_deepcrack.py, prepare_crack500.py       [NEW]
├── sage/
│   ├── components/
│   │   ├── router.py                                   [MODIFY] eps fix (1e-9→1e-5), autocast(enabled=False) cho g_s, GIỮ NGUYÊN exploration noise (baseline=ON)
│   │   ├── sa_hub.py                                    [COPY nguyên]
│   │   ├── sage_layer.py                                [MODIFY] thêm `my_index` param → zero-cost self-select bypass
│   ├── networks/
│   │   ├── convnext_transformer_unet.py                 [REPLACE] → convnextv2_vit_hybrid.py (Femto backbone + ViT-Tiny 6 blocks)
│   │   ├── decoder_block.py                             [DONE] default 3x3, `use_dwsc` flag
│   │   ├── sage_injection.py                            [MODIFY] isinstance guard chặn downsampler/patch-embed/LayerNorm; thêm `num_sage_experts` property support
│   │   ├── wrappers.py                                  [COPY nguyên]
│   ├── utils/
│   │   ├── dataloader.py                                [DONE] Dataset DeepCrack (pad+resize) và Crack500 (RandomCrop + smart filter)
│   │   ├── losses.py                                     [DONE] BCE+SoftDice+L_balance combined
│   │   ├── metrics.py, advanced_metrics.py               [DONE] per-sample foreground Dice làm primary, thêm Boundary IoU + HD95
│   │   ├── evaluation.py                                 [DONE] Crack500 Setting A & B, DeepCrack direct
│   │   ├── training_utils.py                             [DONE] Single-stage, `get_param_groups` tách WD
│   │   ├── model_utils.py                                [DONE] `num_sage_experts` property
│   │   ├── gs_tracker.py, visualization.py                [COPY nguyên]
│   ├── __init__.py                                        [DONE] export mới
├── scripts/
│   ├── shape_audit.py                                     [DONE]
│   ├── train_crack.py                                     [DONE] — Single-stage training runner
│   ├── evaluate_crack_official.py                         [DONE] — Official Setting A/B evaluator
```

## Bảng Tổng hợp Hyperparameter & Baseline

| Hyperparameter | Phân loại | Giá trị Baseline | Ghi chú |
|---|---|---|---|
| **ViT blocks** | Kiến trúc | 6 | Nửa đầu của ViT-Tiny (12 blocks) |
| **num_shared_experts** | Kiến trúc | 4 | Bằng đúng 4 CNN stages |
| **router_hidden_dim** | Kiến trúc | 64 | Bộ định tuyến gọn nhẹ |
| **top_k** | Kiến trúc | 4 | Baseline |
| **N_injection** | Kiến trúc | Full (10 routers) | Ở baseline 6 ViT blocks (4 CNN + 6 ViT) |
| **Decoder conv** | Kiến trúc | Standard 3x3 | DWSC chỉ là flag tắt |
| **Fusion** | Kiến trúc | Residual | main + adapter, không alpha |
| **SA-Hub** | Kiến trúc | O(D²) eager pairwise | Giữ nguyên thuật toán gốc |
| **Gating** | Kiến trúc | Sigmoid | Khóa cứng (không ablation) |
| **Exploration Noise** | Kiến trúc | ON | Đưa vào danh sách ablation sau |
| **Patch size** | Kiến trúc | 14x14 | 196 tokens tại bottleneck |
| **Loss Weights** | Training | 1.0*BCE, 1.5*SoftDice, 1.0*L_balance | Không dùng pos_weight |
| **load_balance_factor** | Training | 0.01 | Scale nội bộ Router |
| **expert_dropout** | Training | 0.1 | Hạn chế overfitting SA-Hub |
| **freeze_encoder/transformer** | Training | False/False | Fine-tune cả mạng |
| **Stage2 LR ratio** | Training | 1:1 | Khởi đầu trung lập |
| **AMP dtype** | Training | FP16 | Tối ưu trên GPU Tesla T4 |
| **Warmup** | Training | 3 epochs | CosineAnnealingLR sau warmup |
| **EarlyStopping patience** | Training | 6 | Dừng sớm nếu Val Dice không tăng |
| **Weight decay** | Training | 0.05 | Param groups tách LN/bias (WD=0.0) |
| **Seed** | Training | 42 | Cố định toàn pipeline |
| **Batch size** | Thực nghiệm | B0 = 20, B1 = 16 | B0 đã benchmark T4; B1 dùng BS=16 an toàn trong config chính thức |
| **Base LR** | Thực nghiệm | 1e-4 | Backbone LR: 1e-5, Decoder LR: 1e-4 |
| **Training Budget** | Thực nghiệm | Epoch budget = 30 | Ngân sách tối đa; EarlyStopping patience = 6 |
| **min_pixels Crack500** | Data | fg_pixels >= 20 | Ngưỡng lọc patch có vết nứt (Canonical) |


## Progress Log
- **2026-09-20**: Hoàn thành Preflight Shape Audit (SK1) qua `scripts/shape_audit.py`. Xác nhận chính thức Shape Contract cho Milestone 1: Input 448x448, ConvNeXt-Femto channels [48, 96, 192, 384], bottleneck 14x14 = 196 tokens, ViT-Tiny slice 6 blocks, tách riêng CLS token khỏi pos_embed, N_injection = 10 (4 CNN stages + 6 ViT blocks). Đủ điều kiện bắt đầu code SAGE-Lite (SK2).
- **2026-09-20**: Khởi tạo lại `milestone_and_progress.md` dựa trên kế hoạch SAGE-Lite mới (Blueprint/Implementation Plan) theo cấu trúc Dependency-order. Rõ ràng 3 script diagnostic không chặn code.

- **2026-09-20**: Cập nhật logic đánh giá Batch Size (đo riêng B0/B2, dùng Gradient Accumulation) và Epoch (ngân sách N_total chung) để đảm bảo tính công bằng thực nghiệm (so sánh accuracy và tính complexity).
- **2026-09-20**: Fix thêm 2 lỗ hổng thực nghiệm (LR Range Test đo riêng B0/B2, nghiêm cấm warm-start ở Baseline Ladder) và đồng bộ lại file SAGE_lite_Implementation_Plan.md.
- **2026-09-22**: Hoàn thành B0 Baseline chính thức trên cả 2 dataset (Crack500 Run-3: Val Dice 0.7318, Test Setting A Dice 0.6771, Setting B Dice 0.6801; DeepCrack Run-1: Val Dice 0.6240, Test Direct Dice 0.6953). Chốt frozen canonical preprocessing cho toàn bộ dự án.
- **2026-09-22**: Chốt thang đo Baseline Ladder: B0 (ConvNeXtV2-Femto + U-Net) → B1 (B0 + 6 ViT-Tiny) → B2 (Full SAGE-Lite). Config `b1_crack500.yaml` chốt `batch_size: 16` an toàn, `Epoch budget = 30` (EarlyStopping patience = 6).
- **2026-09-22**: Hoàn thành Huấn luyện & Đánh giá chính thức B1 trên Crack500 (Run-1). Val Dice = 0.7420 (@ epoch 21). Test Setting A: Dice 0.6857 (IoU 0.5730, HD95 83.11 px). Test Setting B: Dice 0.6895 (IoU 0.5770, HD95 77.43 px). Vượt B0 toàn diện trên 100% metrics (+0.94% Dice Setting B, HD95 giảm mạnh 16.93 px). Sẵn sàng chuyển sang B1 DeepCrack.
- **2026-09-23**: Hoàn thành toàn diện ViT-Depth Sweep B1 trên Crack500 `{4, 6, 8, 12}` và đánh giá chính thức Setting A & Setting B:
  + Depth 4: Val Dice **0.7428**, Val Loss **1.1364** (Top 1) | Test Setting A Dice 0.6829 | Test Setting B Dice 0.6867 (Recall B 0.8589, HD95 90.64 px).
  + Depth 6: Val Dice 0.7420, Val Loss 1.1453 (Top 2) | Test Setting A Dice 0.6857 | Test Setting B Dice 0.6895.
  + Depth 8: Val Dice 0.7379, Val Loss 1.2006 (Top 4) | Test Setting A Dice 0.6819 | Test Setting B Dice 0.6854.
  + Depth 12: Val Dice 0.7419, Val Loss 1.2332 (Top 3) | Test Setting A Dice 0.6902 | Test Setting B Dice 0.6926.
  + **Quyết định Chiến lược:** Chưa freeze ViT depth từ B1; B1 đóng vai trò screening. Depth tối ưu của SAGE-Lite sẽ được đo đạc và chốt qua **B2 Depth Ablation `{4, 6, 12}`** khi có SAGE routing. Không cần chạy lại B1. Sẵn sàng huấn luyện B1 trên DeepCrack.
- **2026-09-23**: Khởi tạo lộ trình thực nghiệm chi tiết cho B2 (Full SAGE-Lite) từ Phase 0 đến Phase 6. Lộ trình được tài liệu hóa tại `docs/B2_Experimental_Roadmap.md`. Đã hoàn tất cài đặt kỹ thuật B2, sẵn sàng chạy Phase 0 (Runtime preflight) trên Colab.
- **2026-09-25**: Hoàn thành Chuẩn bị Giao thức Thử nghiệm Đối chứng P3 (P3-PHASE-6):
  + Đối chiếu và chuẩn hóa toàn diện thuật ngữ giữa Phase 6 và Phase 8 trong `docs/B2_Experimental_Roadmap.md`: Run A được định danh chính xác là "Compression Control under Common Fixed PE28 Treatment" (`no refinement + common fixed PE28 + AvgPool28 -> ViT`); Run B và Run C được xác định là các kiểm soát gần tương đương về dung lượng ("near-matched-capacity controls", Run B = 38,450 vs Run C = 37,874 params, lệch 576 params ở DW) cung cấp bằng chứng thực nghiệm về cấu trúc định hướng/đa tỷ lệ; Speedup được chuẩn hóa thành chỉ số chẩn đoán (diagnostic metric) so với Direct Baseline expert-path latency theo 4-Pillar Framework, loại bỏ ngưỡng bác bỏ cứng ad-hoc `< 10x` chưa đăng ký trước.
  + Khởi tạo bộ 3 file cấu hình YAML chuẩn hóa tại `sage_lite/configs/p3_ablation/`: `b2_p3_run_a.yaml` (Run A, `p3_mode: "A"`), `b2_p3_run_b.yaml` (Run B, `p3_mode: "B"`), `b2_p3_run_c.yaml` (Run C, `p3_mode: "C"`).
  + Xác minh ma trận tham số (Diff Matrix): 100% siêu tham số ngoài khối refinement (Locked Base Depth = 4, Seed = 42, Batch Size = 12, Epochs = 30, Learning Rate = 1e-4, 9 khóa của `sage_config`, data paths) hoàn toàn đẳng cấu và đồng nhất.
  + Chuẩn hóa và đóng băng giao thức cuối cùng (Final Protocol Freeze): Khẳng định `p3_mode` là biến hành vi mô hình duy nhất (`output_dir` chỉ để cô lập artifact); khóa ngữ nghĩa `epochs = 30` là ngân sách tối đa có early stopping (`patience = 6`); ghi nhận ghi chú phương pháp luận phân biệt rõ A/B/C (kiểm soát dưới PE28 chung) vs Direct Baseline (so sánh cấp độ hệ thống); chuẩn hóa kết quả nghiệm thu Phase 5 là 12/12 tests PASS. P3-PHASE-6 chính thức được đánh dấu **FROZEN**.
- **2026-09-25**: Hoàn thành Kiểm tra Runtime Run-A và Tiền kiểm tra Tối ưu hóa Hai giai đoạn (Two-Stage Optimizer Preflight) cho cả 3 Run A, Run B, Run C (P3-PHASE-6 & P3-PHASE-7 Preflight):
  + **Run-A Runtime Gate PASS**: Xác minh kiểu runtime của `stage0.p3_refinement` và `stage1.p3_refinement` là `nn.Identity()`. Nhánh nén P3 ($28 \times 28$, `backbone.pe28_fixed`, $784$ tokens) được kích hoạt chuẩn xác; đường direct high-res cũ tuyệt đối không bị gọi cho ViT experts.
  + **Run A Two-Stage Preflight PASS**: 333/333 tensors được hạch toán đầy đủ (0 duplicate, 0 missing); 132 shared tensors thuộc duy nhất về CNN main blocks; `model.set_shared_experts([0, 1, 2, 3])` thực thi chuẩn; scheduler Stage 2 khởi tạo lại với `warmup=3`.
  + **Run B & Run C Two-Stage Preflight PASS**:
    * Run B: 10 P3 tensors (38,450 params). Stage 1 phân bổ vào nhóm `backbone` (LR $10^{-5}$, WD 0.05). Stage 2 sau reload và `set_shared_experts([0,1,2,3])` thuộc strictly về `other_and_routers` (LR $10^{-4}$, WD 0.05), không lẫn vào `shared_experts`. Khớp 343/343 trainable tensors (0 missing, 0 duplicate).
    * Run C: 10 P3 tensors (37,874 params). Stage 1 phân bổ vào `backbone` (LR $10^{-5}$, WD 0.05). Stage 2 thuộc strictly về `other_and_routers` (LR $10^{-4}$, WD 0.05), không lẫn vào `shared_experts`. Khớp 343/343 trainable tensors (0 missing, 0 duplicate).
    * Tỷ lệ LR Stage 2: `stage2_shared_lr : stage2_base_lr = 1e-4 : 1e-4 = 1:1` được bảo toàn nghiêm ngặt.
  + Cả 3 cấu hình Run A, Run B, Run C đã sẵn sàng 100% để bước vào P3-PHASE-7: Huấn luyện chính thức.


