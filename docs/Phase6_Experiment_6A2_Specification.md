# Phase 6-A.2 Specification: Pure PLU Representation Probe (Progressive Learned Upsampling)
**Pre-registered Experimental Protocol & Diagnostic Specification | Date: 2026-09-30**

> [!IMPORTANT]
> **Single-Variable Controlled Representation Probe (Pure PLU):**
> Phase 6-A.2 isolates the **architectural representation capability** of spatial upsampling without confounding objective changes.
> - **Lineage Root:** Trained from scratch from pretrained initialization through full Stage 1 (17 epochs) $\to$ Stage 2 (18 epochs), total budget 35 epochs.
> - **Zero Cross-Lineage Checkpoint Reuse:** Candidate B Stage 1 checkpoint reuse and head remapping are strictly prohibited.
> - **Architecture:** Candidate B ($D=4, K=2, H=64$, ConvNeXtV2-Femto + ViT-Tiny, P3-C ASDW) — **100% frozen**.
> - **Routing:** Sigmoid gating, load balance factor $LB = 0.010$, 8 experts, top_k = 2 — **100% frozen**.
> - **Objective:** $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Base}} = 1.0 \times \text{BCE} + 1.5 \times \text{Dice}$ across both Stage 1 and Stage 2. BoundaryIoU is **completely disabled** (`boundary_iou_weight = 0.0`).
> - **Evaluation:** Setting A official tiling ($448\times 448$ non-overlapping tiles, $\text{stride} = 448$) on all 348 validation images — **100% frozen**.
> - **Single Intervention:** Decoder Segmentation Head replaced with **Progressive Learned Upsampling Head (PLU-Head)**.

---

## 1. Experimental Design: Isolating Representation from Objective

To ensure scientific attribution, Phase 6 decomposes the investigation into two orthogonal, single-variable probes:

| Configuration | Segmentation Head | Stage 1 Loss | Stage 2 Loss | Role / Probe Target | Val Dice (Best) |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Candidate B** | Baseline (1x1 conv + 4x bilinear) | $\mathcal{L}_{\text{Base}}$ | $\mathcal{L}_{\text{Base}}$ | Phase 5 Canonical Control | 0.7641 |
| **Phase 6-A.1** | Baseline (1x1 conv + 4x bilinear) | $\mathcal{L}_{\text{Base}}$ | $\mathcal{L}_{\text{Base}} + 0.50 \times \mathcal{L}_{\text{B-IoU}}$ | **Objective Probe** (Loss limitation) | 0.7684 |
| **Phase 6-A.2** | **PLU-Head** (ConvTranspose 112 $\to$ 224 $\to$ 448) | $\mathcal{L}_{\text{Base}}$ | $\mathcal{L}_{\text{Base}}$ | **Pure Representation Probe** (Architecture) | *In Progress* |

*Note: Any combination of PLU + BoundaryIoU represents a multi-variable cocktail and is formally deferred to future exploration (Phase 6-A.3).*

---

## 2. Research Hypothesis (RQ6.2 Representation Bottleneck)

From Phase 6-A.1 diagnostics:
- Adding Soft Boundary IoU Loss ($\lambda=0.50, d=2$) recovered $+2.37\%$ Dice ($0.6404 \to 0.6641$) on thin cracks ($\text{thinness} > 0.20$), confirming that loss objective was an active limitation.
- However, thin cracks still lag behind global performance ($0.6641$ vs. $0.7684$), and thin-crack predictions remain $+64.55\%$ wider than ground truth.
- **Structural Bottleneck Observation:**
  In the baseline decoder, feature maps stop upsampling at $112\times 112$ (stride 4). The final projection from $(B, 48, 112, 112) \to (B, 1, 112, 112)$ is followed by a non-parametric $4\times$ bilinear interpolation. Any 1-pixel activation at $112\times 112$ naturally blurs across a $4\times 4 = 16$ pixel region at $448\times 448$.

### Core Hypothesis
> **Learned progressive spatial upsampling (from $112\times 112 \to 224\times 224 \to 448\times 448$) provides the spatial reconstruction capacity needed to refine fine crack boundaries and reduce remaining thin-crack area bloat without changing the loss function.**

---

## 3. Locked Architecture: Progressive Learned Upsampling (PLU-Head)

```text
Feature Map from Decoder Block 2: x_dec ∈ ℝ^(B × 48 × 112 × 112)
                     │
                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Tầng 0: Trích xuất đặc trưng cục bộ tại 112×112 (conv112 + norm112)    │
│ - Conv2d(in=48, out=24, kernel=3, stride=1, padding=1, bias=True)     │
│ - BatchNorm2d(num_features=24)                                         │
│ - ReLU(inplace=True)                                                   │
└────────────────────────────────────────────────────────────────────────┘
                     │
                     │  h_112 ∈ ℝ^(B × 24 × 112 × 112)
                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Tầng 1: Giải chập học được ×2 lên 224×224 (up224 + norm224)           │
│ - ConvTranspose2d(in=24, out=16, kernel=4, stride=2, padding=1, bias=F)│
│ - BatchNorm2d(num_features=16)                                         │
│ - ReLU(inplace=True)                                                   │
└────────────────────────────────────────────────────────────────────────┘
                     │
                     │  h_224 ∈ ℝ^(B × 16 × 224 × 224)
                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Tầng 2: Giải chập học được ×2 & Chiếu Logits ra 448×448 (up448)        │
│ - ConvTranspose2d(in=16, out=1, kernel=4, stride=2, padding=1, bias=T) │
└────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
Raw Segmentation Logits: logits ∈ ℝ^(B × 1 × 448 × 448)
```

> [!NOTE]
> **Ghi chú Phương pháp luận về Checkerboard Artifacts:**
> Cấu hình `kernel=4, stride=2` có overlap đều theo không gian và được sử dụng để giảm nguy cơ checkerboard artifact; không coi việc loại bỏ artifact là một giả định đã được chứng minh.

---

## 4. Exact Parameter Count & Delta Verification

| Component | Layer Name & Configuration | Parameters | Subtotal |
| :--- | :--- | :---: | :---: |
| **Baseline Head (Old)** | `conv3x3(48 -> 24)` ($24 \times 48 \times 9 + 24$) | 10,392 | |
| | `BatchNorm2d(24)` ($24 \times 2$) | 48 | |
| | `conv1x1(24 -> 1)` ($1 \times 24 \times 1 + 1$) | 25 | **10,465** |
| **PLU-Head (Phase 6-A.2)** | `conv112` ($24 \times 48 \times 9 + 24$) | 10,392 | |
| | `norm112` ($24 \times 2$) | 48 | |
| | `up224` ($24 \times 16 \times 16$) | 6,144 | |
| | `norm224` ($16 \times 2$) | 32 | |
| | `up448` ($16 \times 1 \times 16 + 1$) | 257 | **16,873** |

$$\Delta \text{Params} = 16,873 - 10,465 = \mathbf{+6,408} \text{ parameters}$$

- **Total Model Parameters (Candidate B / 6-A.1):** **$10,118,955$**
- **Total Model Parameters (Phase 6-A.2):** **$10,125,363$**
- **Exact Relative Parameter Delta:** **$+0.0633\%$**

---

## 5. Two-Stage Optimization Schedule & Learning Rates

Huấn luyện trọn vẹn hai giai đoạn (Full Stage 1 $\to$ Stage 2) tuân thủ quy chuẩn của Candidate B:

- **Stage 1 (Epoch 1 đến 17):**
  - Backbone: `lr_backbone = 1e-5` (huấn luyện ở LR thấp, **không đóng băng**)
  - Decoder & PLU-Head: `lr_decoder = 1e-4` (weights: WD=`0.05`, norm/bias: WD=`0.0`)
  - SAGE Routing & Adapters: `lr_sage = 2e-4`
  - P3-C ASDW: `lr_p3 = 1e-4`
  - Checkpoint Selection: Lưu `best_model_b2_stage1.pth` khi `val_dice > best_dice + 1e-4` (tie-break trên `val_loss`).

- **Stage 2 (Epoch 18 đến 35, tương đương 18 epochs):**
  - Checkpoint Loading: Nạp lại `best_model_b2_stage1.pth` do chính A2 sinh ra với cơ chế `strict=True` (0 missing, 0 unexpected).
  - All Groups (Backbone, Shared Experts, SAGE, Decoder, PLU-Head): `stage2_base_lr = 1e-4`, `stage2_shared_lr = 1e-4`.
  - Scheduler: Cosine Annealing warmup 3 epochs xuống $1\times 10^{-6}$.

---

## 6. Pre-registered Decision Matrix (vs Candidate B Baseline = 0.7641)

| Outcome | Quantitative Criteria | Scientific Interpretation | Subsequent Action |
| :--- | :--- | :--- | :--- |
| **Positive** | $\text{Global Dice} > 0.7641$ (đặc biệt $\ge 0.768$);<br>$\Delta \text{Thin Dice} \ge +0.015$;<br>$\text{Thin Area Excess} < +65\%$ | **Supports** hypothesis rằng bilinear upsampling là nút thắt cổ chai biểu diễn của Candidate B. | Xác nhận PLU là cải tiến kiến trúc hữu hiệu độc lập với loss. Mở đường cho Phase 6-A.3 (Combined PLU + B-IoU). |
| **Flat / Ineffective** | $|\text{Global Dice} - 0.7641| \le 0.003$;<br>Thin crack metrics không đổi đáng kể | PLU intervention **không mang lại ưu thế độc lập**; bilinear upsampling không phải là bottleneck chính khi thiếu boundary loss. | Giữ nguyên Candidate B baseline head. Khảo sát các hướng biểu diễn khác. |
| **Degraded** | $\text{Global Dice} < 0.7600$ | PLU gây overfitting hoặc mất ổn định tối ưu hóa khi huấn luyện từ đầu. | Bác bỏ hoàn toàn PLU head. |

---

## 7. Execution Commands (Colab / Cloud T4 — Ultra-Minimal 3-Cell Standard)

### Cell 1: Environment Setup & Dataset Preparation
```bash
!rm -rf /content/SAGE_LITE
!git clone -b crack500-audit https://github.com/bach0823/SAGE_LITE.git /content/SAGE_LITE
%cd /content/SAGE_LITE

!python prepare_data/prepare_crack500.py
```

### Cell 2: Training Execution (Full Stage 1 -> Stage 2 Run)
```python
# ==============================================================================
# Cell 2: Training Execution (Pure PLU Stage 1 -> Stage 2)
# ==============================================================================
%cd /content/SAGE_LITE

!python scripts/train_crack.py \
  --config configs/p3_ablation/b2_p3_run_c_d4_k2_h64_phase6_a2_pure_plu.yaml \
  --two-stage
```

### Cell 3: Post-Training Diagnostics, Packaging & Auto-Download
```python
# ==============================================================================
# Cell 3: Post-Training Diagnostics, Zip & Browser Download
# ==============================================================================
%cd /content/SAGE_LITE
from google.colab import files
import shutil

CONFIG_PATH = "configs/p3_ablation/b2_p3_run_c_d4_k2_h64_phase6_a2_pure_plu.yaml"
RUN_DIR = "/content/runs/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64"
CKPT_PATH = f"{RUN_DIR}/best_model_b2_global.pth"
DIAG_VAL = f"{RUN_DIR}/P3_C_Routing_Diagnostics/full_val"
DIAG_ERR = f"{RUN_DIR}/P3_C_Routing_Diagnostics/error_analysis"

# 1. Routing Diagnostics (Full Val)
!python tools/analyze_routing.py \
  --config {CONFIG_PATH} \
  --checkpoint {CKPT_PATH} \
  --output_dir {DIAG_VAL} \
  --split val \
  --data_root /content/dataset/Crack500

# 2. Error Analysis & Morphology Stratification (Setting A Tiling)
!python tools/run_p3_c_error_analysis.py \
  --config {CONFIG_PATH} \
  --checkpoint {CKPT_PATH} \
  --routing-json {DIAG_VAL}/routing_statistics.json \
  --output-dir {DIAG_ERR} \
  --data-root /content/dataset/Crack500

# 3. Zip & Download
shutil.make_archive("/content/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_Full", "zip", "/content/runs", "P3_C_Phase6_A2_Pure_PLU_D4_K2_H64")
files.download("/content/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_Full.zip")
```
