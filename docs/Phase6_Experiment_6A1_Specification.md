# Phase 6-A.1 Specification: Objective Probe (Soft Boundary IoU Loss)
**Pre-registered Experimental Protocol & Diagnostic Specification | Date: 2026-09-30**

> [!IMPORTANT]
> **Zero-Parameter & Frozen Architecture Constraint:**
> Experiment 6-A.1 evaluates purely whether the ~0.76 Dice ceiling and associated crack morphology errors are caused by training objective signal allocation, without introducing any architectural modifications or additional parameters.
> - **Architecture:** Candidate B ($D=4, K=2, H=64$, ConvNeXtV2-Femto + ViT-Tiny, P3-C ASDW) — **100% frozen (0 params added)**.
> - **Routing:** Sigmoid gating, load balance factor $LB = 0.010$ — **100% frozen**.
> - **Decoder:** Standard UNet decoder up to $112\times 112$ + bilinear upsample to $448\times 448$ — **100% frozen**.
> - **Optimization:** Stage 2 Resumption from Candidate B Stage 1 checkpoint (`best_model_b2_stage1.pth`, Val Dice 0.7333), 18 epochs, $r = 1.00$ ($1\times 10^{-4}$ shared, $1\times 10^{-4}$ base, $1\times 10^{-4}$ SAGE).
> - **Evaluation:** Setting A official tiling ($448\times 448$ non-overlapping tiles, $\text{stride} = 448$) on all 348 validation images.

---

## 1. Context & Research Question (RQ6.2)

From the failure taxonomy of Candidate B on the 348 validation samples:
1. **`boundary_margin_error` (n=127, 36.5%):** Model is recall-biased (Precision $0.7337$ vs. Recall $0.8477$, Pred area $+8.16\%$ above GT). Predictions are systematically slightly wider than GT at crack borders.
2. **`thin_crack` failure chain (n=66 with thinness > 0.20, n=5 low-area failure):** Strong negative correlation between thinness and Dice (Spearman $\rho = -0.618$).

### Central Probe Question
> **Does reshaping the loss landscape by adding explicit boundary supervision ($\mathcal{L}_{\text{B-IoU}}$) provide sufficient gradient signal to penalize boundary over-prediction and recover fine crack boundaries without architectural capacity expansion?**

---

## 2. Mathematical Formulation & PyTorch Implementation

### 2.1 Formulation
Let predicted logits be $Z \in \mathbb{R}^{B \times 1 \times H \times W}$ and binary ground truth be $Y \in \{0, 1\}^{B \times 1 \times H \times W}$.
Continuous foreground probability:
$$P = \sigma(Z) \in [0, 1]^{B \times 1 \times H \times W}$$

Boundary extraction is performed via differentiable morphological erosion:
$$\text{Erode}_d(X) = 1.0 - \text{MaxPool2D}(1.0 - X, \text{kernel\_size}=2d+1, \text{stride}=1, \text{padding}=d)$$
$$B(X) = \text{ReLU}(X - \text{Erode}_d(X))$$

Soft Boundary Intersection-over-Union:
$$\text{Boundary-IoU}(P, Y) = \frac{\sum_{h, w} (B(P) \odot B(Y)) + \epsilon}{\sum_{h, w} (B(P) + B(Y) - B(P) \odot B(Y)) + \epsilon}$$

$$\mathcal{L}_{\text{B-IoU}} = 1.0 - \frac{1}{B} \sum_{b=1}^B \text{Boundary-IoU}(P_b, Y_b)$$

### 2.2 Composite Total Loss
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{base\_seg}} + 1.0 \times \mathcal{L}_{\text{LB}} + \lambda_{\text{boundary}} \times \mathcal{L}_{\text{B-IoU}}$$
where $\mathcal{L}_{\text{base\_seg}} = 1.0 \times \text{BCEWithLogits} + 1.5 \times \text{SoftDiceLoss}$.

---

## 3. Pre-locked Hyperparameters & Empirical Scale Validation

All hyperparameters are locked prior to training (*ex-ante*):

| Parameter | Symbol | Locked Value | Epistemic Status & Rationale |
|:---|:---:|:---:|:---|
| **Boundary Loss Weight** | $\lambda_{\text{boundary}}$ | **0.50** | **LOCKED EXPERIMENTAL VALUE** (pre-registered). Empirically measured gradient norm ratio $\|\nabla \mathcal{L}_{\text{B-IoU}}\| / \|\nabla \mathcal{L}_{\text{base}}\| \approx 0.0454$ on Candidate B model weights, yielding an effective gradient contribution of $\approx 2.27\%$ (a gentle, stable boundary regularizer). *Not an unmeasured claim of "35% gradient pressure".* |
| **Erosion Dilation Radius** | $d$ | **2** | **LOCKED EXPERIMENTAL VALUE** (Kernel size $5\times 5$). Targets fine structural boundaries. |
| **Numerical Epsilon** | $\epsilon$ | **1e-5** | Standard epsilon for zero-division avoidance in background-only patches. |
| **Epochs (Stage 2)** | $E_2$ | **18** | Identical budget to Candidate B Stage 2 (total 35 epochs equivalent). |
| **Patience** | — | **6** | Identical early stopping criterion. |

### 3.1 Diagnostic Morphology Sanity Audit ($d=2$, Kernel $5\times 5$)
*Empirical measurement across synthetic crack widths on $32\times 32$ grid:*
- **Width $w = 1\text{ px}$:** Total $24\text{ px} \to$ Boundary $B(Y) = 24\text{ px}$ (**$100.0\%$ support**, Core $0\text{ px}$).
- **Width $w = 2\text{ px}$:** Total $48\text{ px} \to$ Boundary $B(Y) = 48\text{ px}$ (**$100.0\%$ support**, Core $0\text{ px}$).
- **Width $w = 3\text{ px}$:** Total $72\text{ px} \to$ Boundary $B(Y) = 72\text{ px}$ (**$100.0\%$ support**, Core $0\text{ px}$).
- **Width $w = 4\text{ px}$:** Total $96\text{ px} \to$ Boundary $B(Y) = 96\text{ px}$ (**$100.0\%$ support**, Core $0\text{ px}$).
- **Width $w = 5\text{ px}$:** Total $120\text{ px} \to$ Boundary $B(Y) = 100\text{ px}$ ($83.33\%$, Core $20\text{ px}$).
- **Width $w = 8\text{ px}$:** Total $192\text{ px} \to$ Boundary $B(Y) = 112\text{ px}$ ($58.33\%$, Core $80\text{ px}$).
- **Width $w = 16\text{ px}$:** Total $384\text{ px} \to$ Boundary $B(Y) = 144\text{ px}$ ($37.50\%$, Core $240\text{ px}$).
- **Cracks touching border:** For widths $w \in \{1, 2, 3, 4\}\text{ px}$ running into the tile border, lateral boundary support remains **$100.0\%$**. Tile boundary is not artificially penalized as a false interior border.
- **Background-only patch:** Produces exactly **$0$ boundary pixels** (no false boundaries).

---

## 4. Stage-1 Checkpoint Lineage & Integrity Contract

Before Stage 2 resumption starts, the preflight script executes hard assertions on checkpoint identity:

| Verification Field | Expected Contract | Actual Measurement | Status |
|:---|:---:|:---:|:---:|
| **File** | `P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_stage1.pth` | Same | **MATCH** |
| **SHA256 Checksum** | `9c1b3822011ebc9721de005dc1a2eb84ac4494ba2f25a81d2a4432e77df46fd2` | `9c1b3822...` | **MATCH** |
| **Stage Index** | `1` | `1` | **MATCH** |
| **Epoch** | `13` | `13` | **MATCH** |
| **Stage 1 Best Dice** | `0.7332886585387659` (~0.7333) | `0.7333` | **MATCH** |
| **Model Type / D / P3** | `B2`, $D=4$, `p3_mode='C'` | `B2`, $D=4$, `p3_mode='C'` | **MATCH** |
| **SAGE Config** | $\text{top\_k}=2, H=64, LB=0.010$ | $\text{top\_k}=2, H=64, LB=0.010$ | **MATCH** |
| **Parent Lineage** | Direct ancestor of Candidate B Global Best (`147f7840...`, Dice 0.7641) | Verified in Phase 5.1/5.3 log | **MATCH** |

*Rule:* If SHA256 or any metadata field mismatches $\to$ **STOP immediately**, do not resume.

---

## 5. Pre-registered Baseline Comparison Anchors (Candidate B)

*Source: `results/P3_C_Routing_Diagnostics_D4_K2_H64_Phase5_SAGELR2e-4/diagnostics/error_analysis/per_sample_metrics.csv`*

| Stratum / Diagnostic Slice | Sample Count ($n$) | Candidate B Baseline Val Dice | Baseline Precision | Baseline Recall | Baseline Pred/GT Area Bias | Key Diagnostic Signal |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Global Validation Set** | 348 | **0.7641** ($\pm 0.165$) | 0.7337 | 0.8477 | **+8.16%** | Primary global benchmark |
| **Boundary Margin Error** | 127 | **0.7812** | 0.7452 | 0.8387 | **+5.85%** | Tests boundary tightening / over-prediction reduction |
| **Thin Cracks ($\text{thinness} > 0.20$)** | 66 | **0.6404** | 0.5232 | 0.9028 | **+80.37%** | High-frequency detail recovery vs boundary bloat |
| **Thin-Low-Area Failures** | 5 | **0.5370** | 0.3799 | 0.9205 | **+141.66%** | Extreme failure severity recovery |
| **Complex Topology** | 27 | **0.7513** | 0.7431 | 0.8329 | +3.54% | Secondary connectivity sanity check |

---

## 6. Pre-registered Decision Matrix & Absolute Delta Reporting

All outcomes will be reported using **absolute deltas** ($\Delta$) relative to Candidate B anchors rather than rigid binary cutoff assertions:

$$\Delta \text{Global Dice}, \quad \Delta \text{Boundary Dice}, \quad \Delta \text{Thin Dice}, \quad \Delta \text{Precision}, \quad \Delta \text{Recall}, \quad \Delta \text{Area Ratio}$$

| Pattern | Descriptive Direction of Deltas | Diagnostic Interpretation | Subsequent Action |
|:---|:---|:---|:---|
| **Pattern 1: Boundary Specific Fix** | $\Delta \text{Boundary Dice} > 0$, $\Delta \text{Precision} > 0$, $\Delta \text{Area Ratio} < 0$, $\Delta \text{Thin Dice} \approx 0$ | **Objective / boundary supervision bottleneck** confirmed for medium cracks. Thin cracks remain limited by spatial representation. | Keep objective; Proceed to **6-A.2 (Representation Probe)** specifically targeting thin cracks. |
| **Pattern 2: Dual Chain Recovery** | $\Delta \text{Boundary Dice} > 0$, $\Delta \text{Thin Dice} > 0$, $\Delta \text{Area Ratio} < 0$ across all strata | Objective signal alone was sufficient to guide both boundary localization and thin-crack delineation. | SAGE-Lite representation has adequate capacity; explore hybrid objective tuning. |
| **Pattern 3: Ineffective / Flat** | $|\Delta \text{Boundary Dice}| < 0.005$, $|\Delta \text{Thin Dice}| < 0.005$, $|\Delta \text{Global Dice}| < 0.005$ | **Objective probe did NOT resolve failure modes.** Does NOT prove representation bottleneck, but proves gradient reallocation alone is insufficient under Candidate B architecture. | Formally motivates **6-A.2 (Representation Probe)** (e.g. high-res skip / explicit edge feature branch). |
| **Pattern 4: Recall Penalty / Precision Degrade** | $\Delta \text{Precision} < 0$, $\Delta \text{Area Ratio} > 0$, or $\Delta \text{Dice} < 0$ | Loss destabilized boundary gradients, exacerbating false positives or triggering gradient distortion. | Reject Boundary IoU objective; re-examine loss landscape. |

---

## 7. Strict Epistemic Protocol

> [!CAUTION]
> **Non-negotiable Reporting Rule:**
> If Experiment 6-A.1 yields negative or flat results (Pattern 3), it is **STRICTLY FORBIDDEN** to conclude:
> *"Boundary loss does not help, therefore representation bottleneck is proven."*
>
> The ONLY mathematically and methodologically valid conclusion is:
> *"Under the fixed Candidate B architecture and the tested Boundary IoU objective, no meaningful improvement was observed in the targeted failure strata; this motivates testing a representation-side intervention (Phase 6-A.2)."*
