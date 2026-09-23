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
Hoàn thành triển khai và kiểm thử B2 (Full SAGE-Lite) trên codebase `sage_lite/` (Branch `crack500-audit`). Sẵn sàng chạy thực nghiệm trên Colab.

State:
Completed Implementation & Smoke Tests (Ready for Colab Training)

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
    - **Coarse Throughput Profiler (`profile_throughput_b2.py`):**
      + `num_workers = 2`: **DataWait 0.3ms (0.0% overhead)**, Forward 1.45s (27.5%), **Backward 3.81s (72.3%)**, Optimizer 9.6ms (0.2%).
      + Throughput: **2.28 samples/s**, Est 1 epoch (train thuần): **13.88 phút** (~15.5 phút cả Val).
      + Nút thắt thực sự được xác định là Backward pass qua 16 MoE routers & SA-Hub adapters, không phải CPU/DataLoader.
    - Bảng nghiệm thu: Real Crack500 pipeline (PASS), B2-D12 / top-k=4 (PASS), Batch 12 OOM (Không), Forward/backward/opt (PASS), Numerical stability (PASS), 16 experts active (PASS), VRAM (An toàn với đệm ~0.94 GB ở workers=2), Throughput (Đo chính xác ~13.88 min/epoch), Cần sửa architecture? (Không).
    - Chi tiết log lưu tại: `docs/B2_Phase0_Preflight_Log.md`.


In Progress:
- Phase 1: ViT-depth ablation suite trên Crack500 (Google Colab T4, `batch_size: 12`, `num_workers: 2`, `--two-stage`):
  + Run 1: B2 Depth 12 (`configs/b2_crack500_depth12.yaml`)
  + Run 2: B2 Depth 6 (`configs/b2_crack500_depth6.yaml`)
  + Run 3: B2 Depth 4 (`configs/b2_crack500_depth4.yaml`)
  + Đánh giá và chọn depth tốt nhất dựa DUY NHẤT trên Validation Dice (không dùng test-set).




Knowledge Being Learned:
- Cơ chế Routing đa chuyên gia (MoE), tính ổn định số học trong Softmax/Sigmoid gating dưới AMP FP16, giảm thiểu overhead của self-selection qua bypass `my_index`.
- Quy trình quản lý thực nghiệm bằng Git commit + config YAML để đảm bảo tính tái lập (reproducibility).
- Đo đạc biên giới hạn phần cứng (VRAM profiling) và tách bạch giữa hardware-constrained runtime settings vs HPO.
- Kỹ thuật phân rã throughput bằng CUDA Events: nhận diện chính xác Backward computation bottleneck trong mạng MoE thay vì đoán mò về I/O hay DataLoader.

Current Issue:
- Không có issue. Two-Stage protocol đã được unit-test 100% PASS, B2 throughput profiling đã hoàn tất trên Colab T4.

Next Step:
- Chốt budget số epoch huấn luyện cho Phase 1 (Stage 1 & Stage 2) dựa trên throughput thực tế (~14 phút/epoch).
- Khởi động Run 1 (B2 Depth 12) trên Google Colab T4 theo giao thức Two-Stage.


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

