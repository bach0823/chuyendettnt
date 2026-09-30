# Phase 6-A.2 Specification: Representation Probe (Progressive Learned Upsampling Head)
**Pre-registered Experimental Protocol & Diagnostic Specification | Date: 2026-09-30**

> [!IMPORTANT]
> **Single-Variable Controlled Representation Probe:**
> Phase 6-A.2 evaluates whether learned high-resolution spatial decoding/reconstruction addresses the remaining thin-crack performance limitation, without introducing multi-scale architectural cocktails or confounding objective changes.
> - **Lineage Root (Stage 1 Ancestor):** Resumed from the exact same Stage 1 checkpoint as 6-A.1: [`best_model_b2_stage1.pth`](file:///d:/truong/SpecialSubjectTTNT/results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_stage1.pth) (SHA256: `9c1b3822011ebc9721de005dc1a2eb84ac4494ba2f25a81d2a4432e77df46fd2`, Stage 1 Val Dice 0.7333).
> - **Architecture:** Candidate B ($D=4, K=2, H=64$, ConvNeXtV2-Femto + ViT-Tiny, P3-C ASDW) — **100% frozen**.
> - **Routing:** Sigmoid gating, load balance factor $LB = 0.010$, 8 experts — **100% frozen**.
> - **Objective:** $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{base\_seg}} + 1.0 \times \mathcal{L}_{\text{LB}} + 0.50 \times \mathcal{L}_{\text{B-IoU}}$ ($d=2$) — **100% locked from Phase 6-A.1**.
> - **Evaluation:** Setting A official tiling ($448\times 448$ tile, 0.5 stride) on all 348 validation images — **100% frozen**.
> - **Single Intervention:** Decoder Segmentation Head replaced with **Progressive Learned Upsampling Head (PLU-Head)**.

---

## 1. Context & Research Hypothesis (RQ6.2 Representation Side)

From Phase 6-A.1 diagnostics:
- Adding Soft Boundary IoU Loss ($\lambda=0.50, d=2$) confirmed that the training objective was a limitation, recovering $+2.37\%$ Dice ($0.6404 \to 0.6641$) and $+4.16\%$ Precision on thin cracks ($\text{thinness} > 0.20$), while reducing area over-inflation from $+80.37\%$ to $+64.55\%$.
- However, thin cracks still lag behind global performance ($0.6641$ vs. $0.7684$), and thin-crack predictions remain $+64.55\%$ wider than ground truth.
- **Structural Bottleneck Observation:**
  In the baseline decoder, feature maps stop upsampling at $112\times 112$ (stride 4). The final projection from $(B, 48, 112, 112) \to (B, 1, 112, 112)$ is followed by a **non-parametric $4\times$ bilinear interpolation**. Any 1-pixel activation at $112\times 112$ naturally blurs across a $4\times 4 = 16$ pixel region at $448\times 448$.

### Core Hypothesis
> **Learned progressive spatial upsampling (from $112\times 112 \to 224\times 224 \to 448\times 448$) provides the spatial reconstruction capacity needed to refine fine crack boundaries and reduce remaining thin-crack area bloat.**

---

## 2. Locked Architecture Graph: Progressive Learned Upsampling (PLU-Head)

The non-parametric $4\times$ bilinear upsampling of logits is replaced by two progressive learned transposed convolution stages:

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

## 3. Exact Parameter Count & Delta Verification

Verified via PyTorch tensor inspection:

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

## 4. Optimizer Parameter Group Partitioning (Stage 2)

According to the SAGE-Lite two-stage optimization protocol:
- **Existing Parameters (Backbone, SAGE Routers, SA-Hub, P3-C ASDW, Decoder Blocks):**
  - Inherit Stage 2 schedule: base LR = $1\times 10^{-4}$, shared LR = $1\times 10^{-4}$ ($r=1.00$), Cosine Annealing warmup 3 epochs down to $1\times 10^{-6}$.
- **PLU-Head Parameters (allocated to `tier = 'others'`):**
  - **Weights (`others_decay`):** `conv112.weight`, `up224.weight`, `up448.weight` $\to$ **LR: $1.0\times 10^{-4}$, Weight Decay: $0.05$**.
  - **Norms / Biases (`others_no_decay`):** `conv112.bias`, `norm112.weight`, `norm112.bias`, `norm224.weight`, `norm224.bias`, `up448.bias` $\to$ **LR: $1.0\times 10^{-4}$, Weight Decay: $0.0$**.
  - **Initialization:** Deterministic Kaiming Normal for deconvolution weights, unit/zero for BatchNorm, zero for biases.
  - **Zero custom learning rate or scheduler modifications.**

---

## 5. Lineage & Ancestor Resumption Contract

To ensure strictly fair attribution, Phase 6-A.2 starts Stage 2 from the **exact same ancestor checkpoint** as Phase 6-A.1:

$$\text{Stage 1 Checkpoint} \longrightarrow \begin{cases} \text{Phase 6-A.1: Baseline Decoder} + \mathcal{L}_{\text{B-IoU}} \quad (\text{Peak Dice: } 0.7684) \\ \text{Phase 6-A.2: PLU-Head Decoder} + \mathcal{L}_{\text{B-IoU}} \end{cases}$$

- All 340+ backbone, router, SA-Hub, P3-C ASDW, and decoder block tensors are restored identically.
- `conv112` and `norm112` inherit the Stage 1 feature projection weights and running statistics.
- Các tensor thuộc phần upsampling mới (`up224`, `norm224`, `up448`) được khởi tạo fresh (5 parameter tensors + running stats); các tensor `conv112` và `norm112` được remap từ Stage-1 segmentation head.

---

## 6. Pre-registered Decision Matrix

| Outcome | Quantitative Criteria | Scientific Interpretation | Subsequent Action |
| :--- | :--- | :--- | :--- |
| **Positive** | $\Delta \text{Thin Dice} \ge +0.015$;<br>$\text{Thin Area Excess} < +50\%$;<br>$\text{Global Dice} \ge 0.7684$ | **Supports** hypothesis rằng learned high-resolution decoding là một limitation của Candidate B. | Khóa PLU-Head làm chuẩn biểu diễn mới; khảo sát tối ưu tiếp theo. |
| **Flat / Ineffective** | $|\Delta \text{Thin Dice}| < 0.005$;<br>$\Delta \text{Thin Area Excess} \in [\pm 2\%]$;<br>$\text{Global Dice} \approx 0.768$ | PLU intervention **không cho thấy lợi ích đáng kể**; không dùng kết quả này để kết luận encoder-resolution bottleneck đã được chứng minh. | Giữ nguyên baseline 6-A.1; dừng hướng can thiệp decoder upsampling; tìm kiếm các giả thuyết khác. |
| **Degraded** | $\text{Global Dice} < 0.764$ **hoặc**<br>$\text{Precision} \downarrow$ kèm $\text{Area Excess} \uparrow$ đáng kể | PLU intervention bị xem là không phù hợp và không tiếp tục làm baseline. | Bác bỏ hoàn toàn PLU-Head. |

---

## 7. Execution Commands (Colab / Cloud T4)

```bash
# 1. Pull latest code on crack500-audit branch
!git clone -b crack500-audit https://github.com/bach0823/SAGE_LITE.git
%cd SAGE_LITE
!git checkout crack500-audit
!git log -1 --oneline

# 2. Prepare Crack500 dataset
!python prepare_data/prepare_crack500.py

# 3. Download Stage 1 Ancestor Checkpoints
!mkdir -p /content/checkpoints
!wget -q -O /content/checkpoints/best_model_b2_stage1.pth \
  "https://raw.githubusercontent.com/bach0823/chuyendettnt/main/results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_stage1.pth"
!wget -q -O /content/checkpoints/last_model_b2_stage1_rng.pth \
  "https://raw.githubusercontent.com/bach0823/chuyendettnt/main/results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_last_model_b2_stage1_rng.pth"

# 4. Execute Stage 2 Training Run
STAGE1_CKPT = "/content/checkpoints/best_model_b2_stage1.pth"
RNG_CKPT = "/content/checkpoints/last_model_b2_stage1_rng.pth"
CONFIG_PATH = "configs/p3_ablation/b2_p3_run_c_d4_k2_h64_phase6_a2_plu.yaml"

!python scripts/train_crack.py \
  --config {CONFIG_PATH} \
  --stage2-only \
  --checkpoint {STAGE1_CKPT} \
  --rng-checkpoint {RNG_CKPT} \
  --data-root /content/dataset/Crack500
```

### Cell 2: Post-Training Diagnostics, Packaging & Auto-Download (Colab)
```python
# ==============================================================================
# Cell 2: Routing Diagnostics, Error Analysis, Artifacts Packaging & Download
# ==============================================================================
%cd /content/SAGE_LITE

import os
import shutil
from google.colab import files

CONFIG_PATH = "configs/p3_ablation/b2_p3_run_c_d4_k2_h64_phase6_a2_plu.yaml"
RUN_DIR = "/content/runs/P3_C_Phase6_A2_PLU_D4_K2_H64"
CKPT_PATH = os.path.join(RUN_DIR, "best_model_b2_global.pth")
if not os.path.exists(CKPT_PATH):
    CKPT_PATH = os.path.join(RUN_DIR, "best_model_b2_stage2.pth")

DIAG_DIR = os.path.join(RUN_DIR, "P3_C_Routing_Diagnostics")
FULL_VAL_DIR = os.path.join(DIAG_DIR, "full_val")
ERROR_ANALYSIS_DIR = os.path.join(DIAG_DIR, "error_analysis")

# 1. Routing Diagnostics (Full Val Split)
!python tools/analyze_routing.py \
  --config {CONFIG_PATH} \
  --checkpoint {CKPT_PATH} \
  --output_dir {FULL_VAL_DIR} \
  --split val \
  --data_root /content/dataset/Crack500

# 2. Comprehensive Error Analysis & Morphology Stratification
!python tools/run_p3_c_error_analysis.py \
  --config {CONFIG_PATH} \
  --checkpoint {CKPT_PATH} \
  --routing-json {FULL_VAL_DIR}/routing_statistics.json \
  --output-dir {ERROR_ANALYSIS_DIR} \
  --data-root /content/dataset/Crack500

# 3. Đóng gói Artifacts & Tải về máy cục bộ
ZIP_BASE = "/content/P3_C_Phase6_A2_PLU_D4_K2_H64_Full"
ZIP_OUTPUT = f"{ZIP_BASE}.zip"

if not os.path.exists(RUN_DIR):
    raise FileNotFoundError(f"[ERROR] Không tìm thấy thư mục: {RUN_DIR}")

print("\n" + "=" * 80)
print(f"BẮT ĐẦU ĐÓNG GÓI ARTIFACT: {RUN_DIR}")
print(f"Checkpoints: {[f for f in os.listdir(RUN_DIR) if f.endswith('.pth')]}")
print("=" * 80)

shutil.make_archive(base_name=ZIP_BASE, format="zip", root_dir=RUN_DIR)
zip_size_mb = os.path.getsize(ZIP_OUTPUT) / (1024 * 1024)
print(f"✓ ĐÃ NÉN THÀNH CÔNG: {ZIP_OUTPUT} ({zip_size_mb:.2f} MB)")
print("Đang kích hoạt tải file về máy...")
files.download(ZIP_OUTPUT)
```
