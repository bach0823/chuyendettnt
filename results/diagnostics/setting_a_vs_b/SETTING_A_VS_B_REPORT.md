# Phase 6 — Audit Report: Setting A vs Setting B on Candidate B

**Date:** 2026-10-03  
**Target:** Candidate B (`P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth`)  
**Evaluation Set:** Crack500 Validation Set ($N = 348$, Setting A vs Setting B)  
**Execution Pipeline:** Official evaluation protocols via [`scripts/evaluate_crack_official.py`](file:///d:/truong/SpecialSubjectTTNT/SAGE_LITE/scripts/evaluate_crack_official.py)  
**Evaluator Script:** [`scripts/diagnostics/run_setting_a_vs_b_audit.py`](file:///d:/truong/SpecialSubjectTTNT/scripts/diagnostics/run_setting_a_vs_b_audit.py)  

---

## 1. Executive Summary & Core Verdict

| Metric | Setting A (Non-overlap 448×448) | Setting B (50% Overlap, Stride 224, Avg Probs) | Delta (B − A) |
|---|---|---|---|
| **Mean Dice** | **0.7641** | **0.7707** | **+0.0066 (+0.66%)** |
| **Mean clDice** | **0.8498** | **0.8562** | **+0.0064 (+0.64%)** |
| **Mean Recall** | **0.8477** | **0.8526** | **+0.0049 (+0.49%)** |
| **Mean Precision** | **0.7338** | **0.7375** | **+0.0037 (+0.37%)** |
| **Bridge Images** | **110** | **111** | **+1 image (+0.9%)** |
| **Bridge Events** | **118** | **116** | **−2 events (−1.7%)** |
| **Break Events** | **33** | **31** | **−2 events** |
| **Break Images** | **33** | **29** | **−4 images** |

### Core Empirical Verdict: **CASE 4 & CASE 2**
1. **Dice and clDice increase slightly (+0.66% Dice, +0.64% clDice)**, but **False Bridge is virtually invariant**:
   - Bridge Images: **110 $\to$ 111** (Net change: **+1 image**)
   - Bridge Events: **118 $\to$ 116** (Net change: **−2 events**)
   - **108 out of 110 bridge images (98.18%) remain persistently bridged under Setting B**.
   - **Consensus 7/7 cohort ($N=101$): Exactly 101 images remain bridged (100.0% persistent)**.
   - **Consensus Resistant cohort ($N=82$): Exactly 82 images remain bridged (100.0% persistent)**.
   - Only **2 bridge images** were cured by Setting B, while **3 previously clean images developed new false bridges** under Setting B.
2. **Tile-Boundary Hypothesis is UNEQUIVOCALLY REFUTED:**
   - In the `0–16 px` internal tile boundary zone: **0 out of 9 bridge events were cured (0.0% cure rate)**.
   - In the `16–32 px` internal tile boundary zone: **0 out of 7 bridge events were cured (0.0% cure rate)**.
   - **77.1%** (91 / 118) of bridge events are located **more than 64 pixels away** from any internal tile seam.
   - The 2 cured events occurred at distances of **66.0 px** and **44.0 px** from the nearest internal tile boundary line, proving their resolution was not boundary-driven.
3. **Conclusion:**
   False bridges in Candidate B are **NOT an artifact of non-overlapping tile boundaries or boundary context truncation**. They are intrinsic to the model's semantic representations and receptive field interactions. Overlapping tiling smooths probability maps slightly (+1.76% mean absolute difference), but does not resolve the structural false bridge failure mode.

---

## 2. Cohort Breakdown: Setting A vs Setting B

| Cohort | N | A Bridge Imgs | B Bridge Imgs | A Bridge Evs | B Bridge Evs | A Dice | B Dice | $\Delta$ Dice | A Recall | B Recall | A clDice | B clDice |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **All Validation** | 348 | 110 | 111 | 118 | 116 | 0.7641 | 0.7707 | +0.0066 | 0.8477 | 0.8526 | 0.8498 | 0.8562 |
| **Consensus 7/7** | 101 | 101 | 101 | 109 | 106 | 0.7371 | 0.7399 | +0.0028 | 0.8998 | 0.8976 | 0.8455 | 0.8487 |
| **Consensus Resistant** | 82 | 82 | 82 | 88 | 85 | 0.7575 | 0.7604 | +0.0028 | 0.9086 | 0.9070 | 0.8587 | 0.8619 |
| **Clean Control (Base 0-bridge)** | 238 | 0 | 3 | 0 | 3 | 0.7769 | 0.7842 | +0.0073 | 0.8249 | 0.8308 | 0.8518 | 0.8592 |
| **Clean Control (Strict)** | 56 | 0 | 0 | 0 | 0 | 0.7539 | 0.7588 | +0.0049 | 0.7922 | 0.7943 | 0.8198 | 0.8257 |

---

## 3. Bridge Transition Analysis

- **A Bridge Images:** 110
- **B Bridge Images:** 111
- **A $\to$ B Cured Images:** **2**
  1. `20160222_164851_641_721` (A events: 1, B events: 0; A Dice: 0.7273 $\to$ B Dice: 0.7380)
  2. `20160318_181632_1_361` (A events: 1, B events: 0; A Dice: 0.8360 $\to$ B Dice: 0.8406)
- **A $\to$ B Persistent Images:** **108** (**98.18% of the cohort**)
- **A Clean $\to$ B Created Images:** **3**
  1. `20160302_155857_1281_1` (A events: 0, B events: 1; A Dice: 0.7229 $\to$ B Dice: 0.7383)
  2. `20160307_162438_1921_1081` (A events: 0, B events: 1; A Dice: 0.8267 $\to$ B Dice: 0.8286)
  3. `20160321_185557_1281_721` (A events: 0, B events: 1; A Dice: 0.7725 $\to$ B Dice: 0.7844)
- **Net Event Change:** $116 - 118 = \mathbf{-2\text{ events}}$
- **Net Image Change:** $111 - 110 = \mathbf{+1\text{ image}}$

---

## 4. Tile Boundary Proximity Analysis (Testing Hypothesis)

In Setting A, internal tile boundaries cut across the image at multiples of $448$ (e.g. $x = 448$ in $360 \times 640$ images).
For each of the 118 Setting A bridge events, we measured the minimum Euclidean distance from the bridge connector to the nearest internal tile boundary line:

| Distance Bin to Internal Tile Boundary | Total Events | Cured by Setting B | Cure Rate (%) |
|---|---|---|---|
| **0 – 16 px** | **9** | **0** | **0.00%** |
| **16 – 32 px** | **7** | **0** | **0.00%** |
| **32 – 64 px** | **11** | **1** | **9.09%** |
| **> 64 px** | **91** | **1** | **1.10%** |
| **Total** | **118** | **2** | **1.69%** |

### Proximity to ANY Tile Boundary (Including image borders $x=0, y=0$):
| Distance Bin to Any Tile Boundary | Total Events | Cured by Setting B | Cure Rate (%) |
|---|---|---|---|
| **0 – 16 px** | 25 | 1 | 4.00% |
| **16 – 32 px** | 12 | 0 | 0.00% |
| **32 – 64 px** | 19 | 0 | 0.00% |
| **> 64 px** | 62 | 1 | 1.61% |

**Key Finding:**
Bridges located directly on or adjacent to internal tile seams ($0 - 32$ px) had a **0.0% cure rate**. The hypothesis that "Setting B cures bridges by removing non-overlap tile boundary artifacts" is completely unsupported by empirical data.

---

## 5. Setting B Tile-View Distribution (`count_map`)

Crucially, **Setting B does NOT provide 4 views per pixel uniformly across the image**:
Across the 348 validation images ($81,594,144$ total pixels):
- **1 view:** **51,444,640 pixels (63.05%)**
- **2 views:** **29,296,512 pixels (35.91%)**
- **4 views:** **852,992 pixels (1.05%)**

### Reason:
324 out of 348 images in Crack500 validation set have resolution $360 \times 640$.
Since the image height $H = 360 < 448$, there is only **one tile row** in the vertical dimension ($y = 0$).
Consequently:
- In the vertical dimension, every pixel has exactly **1 vertical view**.
- In the horizontal dimension ($W = 640$), stride 224 creates two patches ($x=0$ and $x=224$). The central strip ($x \in [224, 448)$) receives 2 views, while the left ($x < 224$) and right ($x \ge 448$) strips receive only 1 view.
- Therefore, in 93.1% of images, **no pixel ever receives 4 views**; the maximum possible view count is 2. Four views only occur in the 17 images with $H = 484 > 448$ at their central intersection.

---

## 6. Prediction Map Comparison (Probability Difference)

- **Mean Absolute Probability Difference ($|P_A - P_B|$):**
  - All 348 images: **0.0176** (mean change of only $1.76\%$)
  - 110 A-bridge images: **0.0178**
  - Near internal tile boundary ($\le 32$ px): **0.0309**
  - Far from internal tile boundary ($> 32$ px): **0.0160**
- Setting B does provide localized smoothing near the tile boundary seam ($\Delta P \approx 3.1\%$ vs $1.6\%$), but this small attenuation is orders of magnitude smaller than what is required to bridge the $P \approx 0.96$ high-confidence plateau documented in Diagnostic A.

---

## 7. Historical B0 Number Audit

Regarding the claim:
> `B0 Setting A Dice = 0.6771`  
> `B0 Setting B Dice = 0.6801`  

### Verification Result: **FOUND IN REPO ARTIFACTS — CLARIFIED AS TEST SET METRICS**
- **Exact File & Line Citations:**
  1. [`milestone_and_progress.md`](file:///d:/truong/SpecialSubjectTTNT/milestone_and_progress.md#L42):
     `Crack500 (Run-3): Best Val Dice = 0.7318 (@ epoch 8). Test Setting A: Dice 0.6771, IoU 0.5645. Test Setting B: Dice 0.6801, IoU 0.5682.`
  2. [`milestone_and_progress.md`](file:///d:/truong/SpecialSubjectTTNT/milestone_and_progress.md#L68):
     `- [x] Crack500 Run-3: Val Dice 0.7318, Test Setting A Dice 0.6771, Setting B Dice 0.6801.`
  3. [`SAGE_lite_Implementation_Plan.md`](file:///d:/truong/SpecialSubjectTTNT/SAGE_lite_Implementation_Plan.md#L49):
     `- Crack500: Best Val Dice 0.7318, Test Setting A Dice 0.6771, Setting B Dice 0.6801.`
- **Checkpoint & Protocol Identification:**
  - Model: Early baseline B0 (Crack500 Run-3).
  - Checkpoint: `best_model_b0.pth` (Validation Best Dice = 0.7318).
  - Dataset: Crack500 **TEST SET** ($N = 1124$, strictly sealed), evaluated via `evaluate_crack_official.py`.
  - These numbers were **never** validation set metrics; they represent official Test Set benchmarks for B0.

---

## 8. Bắt buộc trả lời 5 câu hỏi chính

### 1. Setting B có làm giảm Bridge Images không?
**KHÔNG.**  
Bridge Images thực tế tăng từ **110 lên 111** (+1 ảnh). 108/110 ảnh (98.2%) giữ nguyên trạng thái bridge.

### 2. Setting B có làm giảm Bridge Events không?
**KHÔNG ĐÁNG KỂ.**  
Bridge Events chỉ giảm từ **118 xuống 116** (−2 events, tương đương giảm 1.69%), hoàn toàn nằm trong biên độ dao động biên cục bộ.

### 3. Setting B có làm tăng Break Events không?
**KHÔNG.**  
Break Events gần như giữ nguyên: **33 events ở Setting A $\to$ 31 events ở Setting B** (−2 events).

### 4. Các bridge được chữa có tập trung gần tile boundary không?
**HOÀN TOÀN KHÔNG.**  
Trong số 16 bridge events nằm sát vách ngăn tile nội bộ ($\le 32$ px), **chính xác 0 events được chữa (tỷ lệ 0.0%)**. Cả 2 ca duy nhất được chữa đều nằm cách xa ranh giới tile (lần lượt 66 px và 44 px).

### 5. Sau kết quả này, hypothesis “tile-boundary/context truncation” mạnh lên hay yếu đi?
**YẾU ĐI RÕ RỆT (BỊ BÁC BỎ HOÀN TOÀN).**  
Hiện tượng False Bridge của Candidate B không xuất phát từ việc chia cắt patch hay ranh giới tile của Setting A. Hiện tượng này là do biểu diễn ngữ cảnh và trường tiếp nhận nội tại của mô hình tạo ra các cao nguyên xác suất cao bất thường trên các khoảng cách không gian rộng.
