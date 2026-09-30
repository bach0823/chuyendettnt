# Phase 6-A.1 Specification: Objective Probe (Soft Boundary IoU Loss)
**Pre-registered Experimental Protocol & Diagnostic Specification | Date: 2026-09-30**

> [!IMPORTANT]
> **Zero-Parameter & Frozen Architecture Constraint:**
> Experiment 6-A.1 evaluates purely whether the ~0.76 Dice ceiling and associated crack morphology errors are caused by training objective signal allocation, without introducing any architectural modifications or additional parameters.
> - **Architecture:** Candidate B ($D=4, K=2, H=64$, ConvNeXtV2-Femto + ViT-Tiny, P3-C ASDW) — **100% frozen**.
> - **Routing:** Sigmoid gating, load balance factor $LB = 0.010$ — **100% frozen**.
> - **Decoder:** Standard UNet decoder up to $112\times 112$ + bilinear upsample to $448\times 448$ — **100% frozen**.
> - **Optimization:** Stage 2 Resumption from Candidate B Stage 1 checkpoint (`best_model_b2_stage1.pth`, Val Dice 0.7333), 18 epochs, $r = 1.00$ ($1\times 10^{-4}$ shared, $1\times 10^{-4}$ base, $1\times 10^{-4}$ SAGE).
> - **Evaluation:** Setting A official tiling ($448\times 448$ tile, 0.5 stride) on all 348 validation images.

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

- For 1-2 pixel cracks: $\text{Erode}_d(Y) = 0$, so $B(Y) = Y$ (100% of the crack is preserved as boundary).
- For thick cracks / margins: $B(Y)$ isolates exactly the outer $d$-pixel contour band.
- For over-predicted boundaries: If $P$ bleeds into the background, $B(P)$ shifts outward, creating a disjoint boundary with zero overlap against $B(Y)$.

Soft Boundary Intersection-over-Union:
$$\text{Boundary-IoU}(P, Y) = \frac{\sum_{h, w} (B(P) \odot B(Y)) + \epsilon}{\sum_{h, w} (B(P) + B(Y) - B(P) \odot B(Y)) + \epsilon}$$

$$\mathcal{L}_{\text{B-IoU}} = 1.0 - \frac{1}{B} \sum_{b=1}^B \text{Boundary-IoU}(P_b, Y_b)$$

### 2.2 Composite Total Loss
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{base\_seg}} + 1.0 \times \mathcal{L}_{\text{LB}} + \lambda_{\text{boundary}} \times \mathcal{L}_{\text{B-IoU}}$$
where $\mathcal{L}_{\text{base\_seg}} = 1.0 \times \text{BCEWithLogits} + 1.5 \times \text{SoftDiceLoss}$.

---

## 3. Pre-locked Hyperparameters (Locked Ex-Ante)

To avoid post-hoc selection bias, all hyperparameters are locked prior to training:

| Parameter | Symbol | Locked Value | Rationale |
|:---|:---:|:---:|:---|
| Boundary Loss Weight | $\lambda_{\text{boundary}}$ | **0.50** | Balances with $\mathcal{L}_{\text{base\_seg}} \approx 0.30-0.45$; provides ~30-40% gradient pressure without destabilizing regional convergence. |
| Erosion Dilation Radius | $d$ | **2** | Kernel size $5\times 5$. Perfectly covers crack widths $\le 4$px as 100% boundary, while targeting the 2px margin error in medium cracks. |
| Numerical Epsilon | $\epsilon$ | **1e-5** | Standard epsilon for zero-division avoidance in background-only patches. |
| Epochs (Stage 2) | $E_2$ | **18** | Identical budget to Candidate B Stage 2 (total 35 epochs equivalent). |
| Patience | — | **6** | Identical early stopping criterion. |

---

## 4. Pre-registered Baseline Comparison Table (Candidate B Anchors)

*Source: `results/P3_C_Routing_Diagnostics_D4_K2_H64_Phase5_SAGELR2e-4/diagnostics/error_analysis/per_sample_metrics.csv`*

| Stratum / Diagnostic Slice | Sample Count ($n$) | Candidate B Baseline Val Dice | Baseline Precision | Baseline Recall | Baseline Pred/GT Area Bias | Key Diagnostic Signal |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Global Validation Set** | 348 | **0.7641** ($\pm 0.165$) | 0.7337 | 0.8477 | **+8.16%** | Primary global benchmark |
| **Boundary Margin Error** | 127 | **0.7812** | 0.7452 | 0.8387 | **+5.85%** | Tests boundary tightening / over-prediction reduction |
| **Thin Cracks ($\text{thinness} > 0.20$)** | 66 | **0.6404** | 0.5232 | 0.9028 | **+80.37%** | High-frequency detail recovery vs boundary bloat |
| **Thin-Low-Area Failures** | 5 | **0.5370** | 0.3799 | 0.9205 | **+141.66%** | Extreme failure severity recovery |
| **Complex Topology** | 27 | **0.7513** | 0.7431 | 0.8329 | +3.54% | Secondary connectivity sanity check |

---

## 5. Pre-registered Decision Matrix

Results will be evaluated strictly across this multi-metric matrix rather than relying solely on global Dice:

| Pattern | Metric Response | Diagnostic Interpretation | Subsequent Action |
|:---|:---|:---|:---|
| **Pattern 1: Boundary Specific Fix** | Boundary Dice $\uparrow$, Precision $\uparrow$ ($> 0.75$), Area delta $\to 0\%$, Thin Dice $\approx 0.64$ | **Objective / boundary supervision bottleneck** confirmed for medium cracks. Thin cracks remain limited by spatial resolution. | Proceed to 6-A.2 (Representation Probe) specifically targeting thin cracks. |
| **Pattern 2: Dual Chain Recovery** | Boundary Dice $\uparrow$, Thin Dice $\uparrow$ ($> 0.67$), Area delta reduced across all strata | Objective signal alone was sufficient to guide both boundary localization and thin-crack delineation. | SAGE-Lite representation has adequate capacity; explore hybrid objective optimization. |
| **Pattern 3: Ineffective / Flat** | Boundary Dice $\approx 0.78$, Thin Dice $\approx 0.64$, Global Dice $\approx 0.764$ ($\pm 0.005$) | **Objective probe did NOT resolve failure modes.** Does NOT prove representation bottleneck, but proves gradient reallocation alone is insufficient under Candidate B architecture. | Formally motivates **6-A.2 (Representation Probe)** (e.g. multi-scale high-res skip / explicit edge feature branch). |
| **Pattern 4: Recall Penalty / Precision Degrade** | Precision $\downarrow$, Pred area $\uparrow$ ($> +12\%$), or Dice $\downarrow$ | Loss destabilized boundary gradients, exacerbating false positives or triggering gradient distortion. | Reject Boundary IoU objective; re-examine loss landscape. |

---

## 6. Strict Epistemic Protocol

> [!CAUTION]
> **Non-negotiable Reporting Rule:**
> If Experiment 6-A.1 yields negative or flat results (Pattern 3), it is **STRICTLY FORBIDDEN** to conclude:
> *"Boundary loss does not help, therefore representation bottleneck is proven."*
>
> The ONLY mathematically and methodologically valid conclusion is:
> *"Under the fixed Candidate B architecture and the tested Boundary IoU objective, no meaningful improvement was observed in the targeted failure strata; this motivates testing a representation-side intervention (Phase 6-A.2)."*
