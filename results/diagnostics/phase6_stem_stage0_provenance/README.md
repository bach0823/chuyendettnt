# Phase 6: Stem to Stage-0 Block-Level Provenance Diagnostic Report

**Target Model:** Candidate B (`P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth`, 10,118,955 parameters)  
**Evaluation Protocol:** Official Setting A ($448 \times 448$, stride 448, non-overlapping, reflect padding, exact cropping, $\tau = 0.5$)  
**Cohorts Evaluated ($N=164$ validation samples total):**
- **Consensus 7/7 ($N=101$):** False bridges persistent across all 7 Phase 6 models (Base, A1, A2, B1, C1, D1, D2).
- **Cured by D2 ($N=7$):** False bridges dissolved under D.2 InterComponentSeparationLoss.
- **Clean / No-Bridge Control ($N=56$):** Ground truth with $\ge 2$ connected components and strictly 0 bridge events across all 7 models.  
**Diagnostic Status:** Completed, frozen, and verified bitwise reproducible.

---

## 1. Protocol & Integrity Verification

To guarantee rigorous adherence to the **DIAGNOSTIC-ONLY** protocol:
- **Zero Training:** Zero backward passes, zero optimizer steps, zero gradient updates.
- **Zero Checkpoint Mutation:** Checkpoint file on disk strictly bitwise verified before and after execution:
  - SHA256: `147f784021414efd...` (Bitwise Identical)
- **Zero Parameter Drift:** SHA256 across all `named_parameters`:
  - SHA256: `4aeda58ce6d2fb32...` (Bitwise Identical)
- **Baseline Bridge Invariance:** Official Setting A validation bridge count strictly verified at **$110/348$**.
- **Dataset Boundaries:** Sealed test set ($N=1124$) strictly unaccessed. Validation set ($N=348$) used exclusively.

---

## 2. Exact Architectural Hook Locations

The physical architecture of the ConvNeXt-V2-Femto backbone was verified directly from the module graph:

```text
Input (448×448 RGB)
  │
  ▼
[Stem]
  Conv2d(3 -> 48, kernel=4, stride=4) + LayerNorm2d((48,), eps=1e-6)
  │ Output: (B, 48, 112, 112) ──► Hook: 'stem'
  ▼
[Stage 0 Block 0]
  ConvNeXtBlock(
    conv_dw: Conv2d(48 -> 48, kernel=7, stride=1, padding=3, groups=48)
    norm: LayerNorm2d((48,))
    mlp: GlobalResponseNormMlp(48 -> 192 -> 48)
    shortcut: Identity (Residual Add)
  )
  │ Output: (B, 48, 112, 112) ──► Hook: 'stage0_block0'
  ▼
[Stage 0 Block 1] (Stage 0 Final Pre-SAGE)
  ConvNeXtBlock(
    conv_dw: Conv2d(48 -> 48, kernel=7, stride=1, padding=3, groups=48)
    norm: LayerNorm2d((48,))
    mlp: GlobalResponseNormMlp(48 -> 192 -> 48)
    shortcut: Identity (Residual Add)
  )
  │ Output: (B, 48, 112, 112) ──► Hook: 'stage0_block1'
  ▼
[Stage 0 SAGE Layer]
  SageLayer(
    main_block: Stage 0 (TupleSafeWrapper)
    router: SageRouter(H64, K2)
    expert_pool: Shared Multi-Scale Experts
    residual_scale: 0.1
  )
  │ Output: (B, 48, 112, 112) ──► Hook: 'stage0_post_sage'
```

*Architectural Invariance:* All 4 hook points operate at the exact same spatial grid ($112 \times 112$) and channel dimension ($48$). No spatial interpolation or channel adaptation is needed to compare them.

---

## 3. Metric 1: Normalized Emergence Index ($R_{\text{norm}}$) Across Blocks

$$R_{\text{norm}} = \frac{E_{\text{neck}} - E_{\text{bg}}}{E_{\text{crack}} - E_{\text{bg}} + \epsilon}$$

### Table 1: Block-Level Emergence Statistics

| Cohort | Hook Point | Median $R_{\text{norm}}$ | Mean $R_{\text{norm}}$ | P10 | P25 | P75 | P90 | $\% \ge 0.50$ | $\% \ge 0.70$ | Median $\frac{E_{\text{neck}}}{E_{\text{crack}}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Consensus 7/7** ($N=101$) | **`stem`** ($112 \times 112$) | **0.7872** | 1.1180 | 0.0385 | 0.3878 | 1.3592 | 2.1456 | **69.31%** | **55.45%** | 0.9900 |
| | **`stage0_block0`** | **0.8470** | 1.1632 | 0.0077 | 0.3692 | 1.2697 | 1.7669 | **67.33%** | **55.45%** | 1.0119 |
| | **`stage0_block1`** (Pre-SAGE) | **0.7459** | 0.9522 | -0.1512 | 0.2653 | 1.1683 | 1.8721 | **62.38%** | **51.49%** | 1.0210 |
| | **`stage0_post_sage`** | **0.7437** | 0.9616 | -0.1873 | 0.2624 | 1.1701 | 1.8523 | **62.38%** | **51.49%** | 1.0207 |
| **Cured by D2** ($N=7$) | **`stem`** ($112 \times 112$) | 0.2601 | 0.5316 | -0.0501 | 0.1947 | 0.7665 | 1.3237 | 42.86% | 28.57% | 0.9472 |
| | **`stage0_block0`** | 0.3785 | 0.0398 | -1.7221 | -0.1047 | 0.9221 | 1.4285 | 42.86% | 42.86% | 1.0112 |
| | **`stage0_block1`** (Pre-SAGE) | 0.6666 | 2.0880 | 0.2215 | 0.5846 | 1.1530 | 5.1277 | 85.71% | 42.86% | 1.0235 |
| | **`stage0_post_sage`** | 0.6707 | 2.0469 | 0.2197 | 0.5775 | 1.1551 | 5.0184 | 85.71% | 42.86% | 1.0246 |
| **Clean Control** ($N=56$) | **`stem`** ($112 \times 112$) | -0.1751 | -0.8103 | -3.3608 | -0.8429 | 1.1649 | 3.1500 | 32.14% | 30.36% | 0.9579 |
| | **`stage0_block0`** | 0.0140 | -0.1101 | -1.5335 | -0.5590 | 0.8057 | 2.5662 | 32.14% | 26.79% | 1.0339 |
| | **`stage0_block1`** (Pre-SAGE) | 0.0615 | 0.1777 | -1.0061 | -0.3448 | 1.0958 | 1.7680 | 35.71% | 32.14% | 1.0406 |
| | **`stage0_post_sage`** | 0.0650 | 0.1822 | -1.0099 | -0.3524 | 1.0871 | 1.7770 | 35.71% | 32.14% | 1.0399 |

---

## 4. Metric 2: Pairwise Spatial Separation on the $112 \times 112$ Grid

For every bridged ground-truth component pair $(g_i, g_j)$:
1. **Grid-Cell Collision:** Whether components $g_i$ and $g_j$ share any common $4 \times 4$ downsampling cell on the $112 \times 112$ grid.
2. **Feature Separation Margin:** $\text{SepMargin} = 1.0 - \frac{E_{\text{neck}}}{E_{\text{crack}} + \epsilon}$.
   - $\text{SepMargin} \ge 0.20$: Substantial energy valley between the components ($\ge 20\%$ lower activation at the neck than on crack).
   - $\text{SepMargin} < 0.20$ or $\le 0.00$: Neck energy matches or exceeds crack energy (feature representation collapse).

### Table 2: Pairwise Spatial Separation Statistics

| Cohort | Hook Point | Pairs Evaluated | Median Gap (px) | Grid-Cell Collision | % Spatially Separable | % Representation Collapsed | Median SepMargin |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Consensus 7/7** | **`stem`** | 513 | 20.6 px | 9.36% | **8.77%** | **91.23%** | **+0.0101** |
| | **`stage0_block0`** | 513 | 20.6 px | 9.36% | **8.77%** | **91.23%** | **-0.0051** |
| | **`stage0_block1`** | 513 | 20.6 px | 9.36% | **9.16%** | **90.84%** | **-0.0126** |
| | **`stage0_post_sage`**| 513 | 20.6 px | 9.36% | **9.16%** | **90.84%** | **-0.0133** |
| **Cured by D2** | **`stem`** | 7 | 11.0 px | 0.00% | 0.00% | 100.00% | +0.0528 |
| | **`stage0_block0`** | 7 | 11.0 px | 0.00% | 0.00% | 100.00% | -0.0112 |
| | **`stage0_block1`** | 7 | 11.0 px | 0.00% | 14.29% | 85.71% | -0.0235 |
| | **`stage0_post_sage`**| 7 | 11.0 px | 0.00% | 14.29% | 85.71% | -0.0246 |
| **Clean Control** | **`stem`** | 56 | 43.3 px | 10.71% | 0.00% | 100.00% | +0.0421 |
| | **`stage0_block0`** | 56 | 43.3 px | 10.71% | 0.00% | 100.00% | -0.0339 |
| | **`stage0_block1`** | 56 | 43.3 px | 10.71% | 0.00% | 100.00% | -0.0406 |
| | **`stage0_post_sage`**| 56 | 43.3 px | 10.71% | 0.00% | 100.00% | -0.0399 |

---

## 5. First-Emergence Distribution

### Table 3: Earliest Stage Reaching Emergence Thresholds

| Earliest Stage Reaching Threshold | Consensus 7/7 ($\ge 50\%$) | Consensus 7/7 ($\ge 70\%$) | Cured by D2 ($\ge 50\%$) | Clean Control ($\ge 50\%$) |
| :--- | :---: | :---: | :---: | :---: |
| **`stem`** ($112 \times 112$) | **70 (69.31%)** | **56 (55.45%)** | 3 (42.86%) | 18 (32.14%) |
| **`stage0_block0`** | 11 (10.89%) | 10 (9.90%) | 1 (14.29%) | 7 (12.50%) |
| **`stage0_block1`** (Pre-SAGE) | 2 (1.98%) | 3 (2.97%) | 2 (28.57%) | 5 (8.93%) |
| **`stage0_post_sage`** | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| Later Stages / Did Not Emerge in Stage 0 | 18 (17.82%) | 32 (31.68%) | 1 (14.29%) | 26 (46.43%) |
| **Total Within Stage 0 Pipeline** | **83 / 101 (82.18%)** | **69 / 101 (68.32%)** | **6 / 7 (85.71%)** | **30 / 56 (53.57%)** |

---

## 6. Scientific Decision Gate Evaluation

### Criterion Application:
- **Case A — Stem / Early Downsampling Representation Collapse:**
  *Condition:* Spatial separation is already degraded at the stem before Stage-0 learned blocks.
- **Case B — Stage-0 Learned Processing Transition:**
  *Condition:* Stem preserves a clear separation valley, but Block 0 or Block 1 creates a prominent transition.
- **Case C — Unresolved:**
  *Condition:* Transitions are noisy or small across all points.

### **Diagnostic Verdict: Supported Hypothesis is CASE A (Stem / Early Downsampling Representation Collapse)**

1. **Stem Emergence Dominance:**
   - In **$69.31\%$** (70/101) of Consensus 7/7 cases, the bridge representation crosses the $R_{\text{norm}} \ge 0.50$ threshold **immediately after the stem projection** (`Conv2d(3 -> 48, 4, 4, s=4)` + `LayerNorm2d`).
   - The median $R_{\text{norm}}$ at the stem is already **$0.7872$**, well above the $0.50$ emergence cutoff.

2. **Feature Valley Absence at Stem:**
   - Across 513 bridged component pairs, the median separation margin at the stem is only **$+0.0101$** (less than $1\%$ valley depth between neck and crack activation).
   - In **$91.23\%$** of bridged pairs, representation collapse has already occurred at the stem output, even though in $90.64\%$ of these pairs the components occupied disjoint $4 \times 4$ spatial grid cells.

3. **Marginal Impact of Learned ConvNeXt Blocks:**
   - Stage 0 Block 0 and Block 1 do not initiate a sharp representational collapse:
     $$\Delta R_{\text{block0}} = R_{\text{block0}} - R_{\text{stem}} = 0.8470 - 0.7872 = \mathbf{+0.0598}$$
     $$\Delta R_{\text{block1}} = R_{\text{block1}} - R_{\text{block0}} = 0.7459 - 0.8470 = \mathbf{-0.1011}$$
     $$\Delta R_{\text{SAGE0}} = R_{\text{post\_SAGE}} - R_{\text{block1}} = 0.7437 - 0.7459 = \mathbf{-0.0022}$$
   - The learned $7 \times 7$ depthwise blocks slightly fluctuate the representation, but the high-contrast activation on the bridge neck was already established at the stem output.

---

## 7. Limitations & Epistemic Boundaries

1. **Representation Correlation vs Causal Necessity:**
   While the earliest observed emergence of the bridge signal is localized to the stem output, this diagnostic measures **representation ambiguity** and does not constitute causal proof that modifying the stem alone is sufficient to eliminate false bridges.
2. **Receptive Field Interaction:**
   The stem performs non-overlapping $4 \times 4$ projection followed by `LayerNorm2d((48,))`. In asphalt images where cracks are narrow ($1$–$4$ px) and local contrast between asphalt matrix and crack edges is subtle, the stem projection creates representation ambiguity that downstream $7 \times 7$ convolutions and decoder spatial mixing propagate and amplify.
3. **No Training Authorization:**
   In accordance with the frozen Phase 6 diagnostic protocol, no model training, loss tuning, or architectural modifications have been or will be performed based solely on this diagnostic.
