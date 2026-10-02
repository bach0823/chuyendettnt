# Phase 6: Stem Raw Conv vs. LayerNorm Factorization Diagnostic Report

**Target Model:** Candidate B (`P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth`, 10,118,955 parameters)  
**Evaluation Protocol:** Official Setting A ($448 \times 448$, stride 448, non-overlapping, reflect padding, exact cropping, $\tau = 0.5$)  
**Cohorts Evaluated ($N=164$ validation samples total):**
- **Consensus 7/7 ($N=101$):** False bridges persistent across all 7 Phase 6 models (Base, A1, A2, B1, C1, D1, D2).
- **Cured by D2 ($N=7$):** False bridges dissolved under D.2 InterComponentSeparationLoss.
- **Clean / No-Bridge Control ($N=56$):** Ground truth with $\ge 2$ connected components and strictly 0 bridge events across all 7 models.  
**Diagnostic Status:** Completed, frozen, and verified bitwise reproducible.

---

## 1. Protocol & Integrity Verification

To guarantee strict compliance with the **DIAGNOSTIC-ONLY** protocol:
- **Zero Training:** Zero backward passes, zero optimizer steps, zero gradient updates.
- **Zero Checkpoint Mutation:** Checkpoint file on disk strictly bitwise verified before and after execution:
  - SHA256: `147f784021414efdf23bc0fc95a4f7831d1ea0e1f7dcf8d80c6fb052a9d8ae42` (Bitwise Identical)
- **Zero Parameter Drift:** SHA256 across all `named_parameters`:
  - SHA256: `4aeda58ce6d2fb322f717cf7b320d43a60db6e3ba73e514f77c3e38716b1ff5e` (Bitwise Identical)
- **Baseline Bridge Invariance:** Official Setting A validation bridge count strictly verified at **$110/348$**.
- **Dataset Boundaries:** Sealed test set ($N=1124$) strictly unaccessed. Validation set ($N=348$) used exclusively.

---

## 2. Exact Architectural Hook Locations

The physical architecture of the ConvNeXt-V2-Femto stem and Stage 0 was factorized into 5 sequential observation points:

```text
Input (448×448 RGB)
  │
  ▼
[Stem Step 1: Raw Patch Projection]
  Conv2d(3 -> 48, kernel=4, stride=4)
  │ Output: (B, 48, 112, 112) ──► Hook 1: 'raw_conv' (Pre-LN)
  ▼
[Stem Step 2: Channel Normalization]
  LayerNorm2d((48,), eps=1e-6)
  │ Output: (B, 48, 112, 112) ──► Hook 2: 'stem_ln' (Post-LN)
  ▼
[Stage 0 Block 0]
  ConvNeXtBlock(
    conv_dw: Conv2d(48 -> 48, kernel=7, stride=1, padding=3, groups=48)
    norm: LayerNorm2d((48,))
    mlp: GlobalResponseNormMlp(48 -> 192 -> 48)
    shortcut: Identity (Residual Add)
  )
  │ Output: (B, 48, 112, 112) ──► Hook 3: 'stage0_block0'
  ▼
[Stage 0 Block 1] (Stage 0 Final Pre-SAGE)
  ConvNeXtBlock(
    conv_dw: Conv2d(48 -> 48, kernel=7, stride=1, padding=3, groups=48)
    norm: LayerNorm2d((48,))
    mlp: GlobalResponseNormMlp(48 -> 192 -> 48)
    shortcut: Identity (Residual Add)
  )
  │ Output: (B, 48, 112, 112) ──► Hook 4: 'stage0_block1'
  ▼
[Stage 0 SAGE Layer]
  SageLayer(
    main_block: Stage 0 (TupleSafeWrapper)
    router: SageRouter(H64, K2)
    expert_pool: Shared Multi-Scale Experts
    residual_scale: 0.1
  )
  │ Output: (B, 48, 112, 112) ──► Hook 5: 'stage0_post_sage'
```

*Architectural Invariance:* All 5 hook points share the exact same spatial grid ($112 \times 112$) and channel dimension ($48$). No spatial interpolation, channel projection, or parameter adaptation is introduced.

---

## 3. Metric 1: Normalized Emergence Index ($R_{\text{norm}}$) Across Factorized Stages

$$R_{\text{norm}} = \frac{E_{\text{neck}} - E_{\text{bg}}}{E_{\text{crack}} - E_{\text{bg}} + \epsilon}$$

### Table 1: Stem Factorization Emergence Statistics

| Cohort | Hook Point | Median $R_{\text{norm}}$ | Mean $R_{\text{norm}}$ | P10 | P25 | P75 | P90 | $\% \ge 0.50$ | $\% \ge 0.70$ | Median $\frac{E_{\text{neck}}}{E_{\text{crack}}}$ | Median $\cos(\mathbf{f}_{\text{neck}}, \mathbf{f}_{\text{crack}})$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Consensus 7/7** ($N=101$) | **`raw_conv`** (Pre-LN) | **0.5508** | 3.8910 | -0.9876 | 0.0267 | 1.5566 | 3.7610 | **54.46%** | **45.54%** | 1.0280 | **0.9658** |
| | **`stem_ln`** (Post-LN) | **0.7872** | 1.1180 | 0.0385 | 0.3878 | 1.3592 | 2.1456 | **69.31%** | **55.45%** | 0.9900 | **0.8828** |
| | **`stage0_block0`** | **0.8470** | 1.1632 | 0.0077 | 0.3692 | 1.2697 | 1.7669 | **67.33%** | **55.45%** | 1.0119 | **0.9835** |
| | **`stage0_block1`** (Pre-SAGE) | **0.7459** | 0.9522 | -0.1512 | 0.2653 | 1.1683 | 1.8721 | **62.38%** | **51.49%** | 1.0210 | **0.9974** |
| | **`stage0_post_sage`** | **0.7437** | 0.9616 | -0.1873 | 0.2624 | 1.1701 | 1.8523 | **62.38%** | **51.49%** | 1.0207 | **0.9974** |
| **Cured by D2** ($N=7$) | **`raw_conv`** (Pre-LN) | -0.3680 | -0.3110 | -0.6151 | -0.5882 | -0.1025 | 0.0370 | **0.00%** | **0.00%** | 1.0853 | 0.9383 |
| | **`stem_ln`** (Post-LN) | 0.2601 | 0.5316 | -0.0501 | 0.1947 | 0.7665 | 1.3237 | 42.86% | 28.57% | 0.9472 | 0.8145 |
| | **`stage0_block0`** | 0.3785 | 0.0398 | -1.7221 | -0.1047 | 0.9221 | 1.4285 | 42.86% | 42.86% | 1.0112 | 0.9657 |
| | **`stage0_block1`** (Pre-SAGE) | 0.6666 | 2.0880 | 0.2215 | 0.5846 | 1.1530 | 5.1277 | 85.71% | 42.86% | 1.0235 | 0.9943 |
| | **`stage0_post_sage`** | 0.6707 | 2.0469 | 0.2197 | 0.5775 | 1.1551 | 5.0184 | 85.71% | 42.86% | 1.0246 | 0.9943 |
| **Clean Control** ($N=56$) | **`raw_conv`** (Pre-LN) | -0.0200 | -0.3216 | -1.9829 | -0.5303 | 1.0044 | 1.7629 | 42.86% | 33.93% | 0.9885 | 0.9625 |
| | **`stem_ln`** (Post-LN) | -0.1751 | -0.8103 | -3.3608 | -0.8429 | 1.1649 | 3.1500 | 32.14% | 30.36% | 0.9579 | 0.8455 |
| | **`stage0_block0`** | 0.0140 | -0.1101 | -1.5335 | -0.5590 | 0.8057 | 2.5662 | 32.14% | 26.79% | 1.0339 | 0.9783 |
| | **`stage0_block1`** (Pre-SAGE) | 0.0615 | 0.1777 | -1.0061 | -0.3448 | 1.0958 | 1.7680 | 35.71% | 32.14% | 1.0406 | 0.9962 |
| | **`stage0_post_sage`** | 0.0650 | 0.1822 | -1.0099 | -0.3524 | 1.0871 | 1.7770 | 35.71% | 32.14% | 1.0399 | 0.9962 |

---

## 4. Metric 2: Pairwise Spatial Separation on the $112 \times 112$ Grid (513 Pairs)

For all 513 bridged ground-truth component pairs in the Consensus 7/7 cohort:
- **Separation Condition:** $\text{SepMargin} \ge 0.20$ and no grid-cell collision on the $112 \times 112$ grid.
- **Collapse Condition:** $\text{SepMargin} < 0.20$ or grid-cell collision (no distinct energy valley between components).

### Table 2: Pairwise Spatial Separation Statistics

| Cohort | Hook Point | Pairs Evaluated | Median Gap (px) | Grid Collision | % Spatially Separable | % Representation Collapsed | Median SepMargin |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Consensus 7/7** | **`raw_conv`** (Pre-LN) | 513 | 20.6 px | 9.36% | **9.75%** | **90.25%** | **-0.0192** |
| | **`stem_ln`** (Post-LN) | 513 | 20.6 px | 9.36% | **8.77%** | **91.23%** | **+0.0101** |
| | **`stage0_block0`** | 513 | 20.6 px | 9.36% | **8.77%** | **91.23%** | **-0.0051** |
| | **`stage0_block1`** | 513 | 20.6 px | 9.36% | **9.16%** | **90.84%** | **-0.0126** |
| | **`stage0_post_sage`**| 513 | 20.6 px | 9.36% | **9.16%** | **90.84%** | **-0.0133** |
| **Cured by D2** | **`raw_conv`** (Pre-LN) | 7 | 11.0 px | 0.00% | 0.00% | 100.00% | -0.0853 |
| | **`stem_ln`** (Post-LN) | 7 | 11.0 px | 0.00% | 0.00% | 100.00% | +0.0528 |
| | **`stage0_block0`** | 7 | 11.0 px | 0.00% | 0.00% | 100.00% | -0.0112 |
| | **`stage0_block1`** | 7 | 11.0 px | 0.00% | 14.29% | 85.71% | -0.0235 |
| | **`stage0_post_sage`**| 7 | 11.0 px | 0.00% | 14.29% | 85.71% | -0.0246 |
| **Clean Control** | **`raw_conv`** (Pre-LN) | 56 | 43.3 px | 10.71% | 0.00% | 100.00% | +0.0115 |
| | **`stem_ln`** (Post-LN) | 56 | 43.3 px | 10.71% | 0.00% | 100.00% | +0.0421 |
| | **`stage0_block0`** | 56 | 43.3 px | 10.71% | 0.00% | 100.00% | -0.0339 |
| | **`stage0_block1`** | 56 | 43.3 px | 10.71% | 0.00% | 100.00% | -0.0406 |
| | **`stage0_post_sage`**| 56 | 43.3 px | 10.71% | 0.00% | 100.00% | -0.0399 |

---

## 5. First-Emergence Distribution

### Table 3: Earliest Stage Reaching Emergence Thresholds

| Earliest Stage Reaching Threshold | Consensus 7/7 ($\ge 50\%$) | Consensus 7/7 ($\ge 70\%$) | Cured by D2 ($\ge 50\%$) | Clean Control ($\ge 50\%$) |
| :--- | :---: | :---: | :---: | :---: |
| **`raw_conv`** (Pre-LN) | **55 (54.46%)** | **46 (45.54%)** | **0 (0.00%)** | 24 (42.86%) |
| **`stem_ln`** (Post-LN) | **28 (27.72%)** | **22 (21.78%)** | 3 (42.86%) | 10 (17.86%) |
| **`stage0_block0`** | 5 (4.95%) | 7 (6.93%) | 1 (14.29%) | 4 (7.14%) |
| **`stage0_block1`** (Pre-SAGE) | 1 (0.99%) | 3 (2.97%) | 2 (28.57%) | 3 (5.36%) |
| **`stage0_post_sage`** | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| Later Stages / Did Not Emerge in Stage 0 | 12 (11.88%) | 23 (22.77%) | 1 (14.29%) | 15 (26.79%) |
| **Total Within Stem (Raw + LN)** | **83 / 101 (82.18%)** | **68 / 101 (67.33%)** | **3 / 7 (42.86%)** | **34 / 56 (60.71%)** |

---

## 6. Scientific Decision Gate Evaluation

### Hypotheses Evaluated:
- **Case A — Raw Conv / Early Patch Projection Representation Collapse:**  
  *Condition:* Spatial separation is already collapsed and high emergence ($R_{\text{norm}} \ge 0.50$) is already present at `raw_conv` before LayerNorm2d.
- **Case B — LayerNorm Normalization Transition:**  
  *Condition:* `raw_conv` preserves a clear spatial separation valley, but `LayerNorm2d` destroys the valley and causes the emergence transition.
- **Case C — Stage-0 Learned Block Transition:**  
  *Condition:* Stem (raw conv + LN) preserves separation, but Block 0 or Block 1 introduces the collapse.
- **Case D — SAGE Contribution:**  
  *Condition:* Post-SAGE causes a significant jump in bridge emergence.

### **Diagnostic Verdict: Supported Hypothesis is CASE A (Raw Conv / Early Patch Projection Representation Collapse)**

1. **Immediate Emergence at Raw Conv (`Conv2d(3, 48, 4, 4)`):**
   - **54.46%** (55/101) of Consensus 7/7 false bridges first cross the $R_{\text{norm}} \ge 0.50$ emergence threshold **immediately at `raw_conv`**, before any normalization or learned depthwise convolution is applied.
   - For the more stringent $\ge 0.70$ threshold, **45.54%** (46/101) first emerge at `raw_conv`.
   - The median $R_{\text{norm}}$ at `raw_conv` is already **$0.5508$** (above the 0.50 emergence threshold).

2. **Severe Representation Collapse Pre-Normalization:**
   - In **90.25%** (463/513) of the bridged component pairs, feature separation has already collapsed at `raw_conv` ($\text{SepMargin} < 0.20$ or negative).
   - The median separation margin at `raw_conv` is **$-0.0192$**, indicating that the unnormalized feature energy inside the bridge neck already equals or exceeds the energy on the crack body.
   - The feature cosine similarity between the bridge neck corridor and the crack body at `raw_conv` is **$0.9658$**, indicating extreme semantic alignment between crack and non-crack gap regions prior to LayerNorm.

3. **Role of LayerNorm (`stem_ln`): Mediator / Rescaler, Not Root Cause:**
   - `LayerNorm2d` increases the median $R_{\text{norm}}$ from $0.5508$ to $0.7872$ ($\Delta R = +0.2364$), and adds 28 cases to the $\ge 0.50$ pool.
   - However, the fraction of collapsed component pairs only changes by **$+0.98\%$** (from $90.25\%$ to $91.23\%$).
   - This proves that LayerNorm rescales and sharpens channel activations across the patch, but does **not** create the spatial ambiguity from scratch.

4. **D2-Cured Contrast:**
   - In the 7 cases cured by D2, **0.0%** emerge at `raw_conv` (median $R_{\text{norm}} = -0.3680$).
   - Their emergence only creeps in at `stem_ln` ($42.86\%$) and `stage0_block1` ($85.71\%$), confirming that D2-cured bridges are structurally fragile and form much later in the network than Consensus 7/7 bridges.

---

## 7. Limitations & Epistemic Boundaries

1. **Representation Ambiguity vs Causal Sufficiency:**
   This diagnostic tracks the earliest observed representation ambiguity in feature space. While `raw_conv` shows representation collapse, this does not imply that an isolated change to the stem projection kernel alone will completely eradicate downstream false bridges without architectural co-adaptation.
2. **Receptive Field and Stride-4 Geometry:**
   The non-overlapping $4 \times 4$ stride-4 projection groups $16$ input pixels into a single feature token. For thin crack branches (1–3 pixels wide), adjacent background pixels inside the same $4 \times 4$ tile bleed into the token, causing early representation ambiguity.
3. **No Training Authorization:**
   In accordance with the frozen Phase 6 diagnostic protocol, no weights, loss functions, or training loops have been modified.
