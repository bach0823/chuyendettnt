# Phase 6: AC Attenuation Counterfactual Diagnostic Report

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
- **Baseline Invariance at $\alpha=1.00$:** Exactly reproduces the official baseline bridge count:
  - Consensus 7/7: **$101/101$** ($100.0\%$).
  - Clean Control: **$0/56$** ($0.0\%$).
- **Dataset Boundaries:** Sealed test set ($N=1124$) strictly unaccessed. Validation set ($N=348$) used exclusively.

---

## 2. Hypothesis & Mathematical Intervention

### Scientific Hypothesis:
The preceding diagnostic demonstrated that the AC component (spatially contrast-sensitive / edge-like, zero-mean component) of the stem `Conv2d(3, 48, 4, 4, s=4)` accounts for $96.26\%$ of the filter weight energy and isolates the earliest observed representation collapse.
**The Question:** Is this AC component **causally necessary** for downstream false-bridge commitment, or does the downstream network maintain the false bridge even when AC spatial contrast is attenuated?

### Mathematical Intervention:
By linearity of convolution, we scale the zero-mean spatial component $\widetilde{W}$ while preserving the DC component $\bar{W}$ and bias $\mathbf{b}$:
$$W_\alpha = \bar{W} + \alpha \widetilde{W}$$
$$\mathbf{y}_\alpha = \text{Conv2d}(X; W_\alpha, \mathbf{b}) = \mathbf{b} + \mathbf{y}_{\text{DC}} + \alpha \mathbf{y}_{\text{AC}}$$
where:
- $\alpha = 1.00$: Exact baseline full Conv2d ($100\%$ AC).
- $\alpha = 0.75$: $25\%$ attenuation of the spatially contrast-sensitive component.
- $\alpha = 0.50$: $50\%$ attenuation.
- $\alpha = 0.25$: $75\%$ attenuation.
- $\alpha = 0.00$: Pure DC-only (complete elimination of intra-patch spatial contrast, retaining only spatial average pooling + $1 \times 1$ channel projection).

---

## 3. Results: Downstream Topologic & Segmentation Readout

### Table 1: Downstream Bridge Events, Transitions, and Crack Continuity

| Cohort | $\alpha$ | False Bridge Cases | Pct Bridge Cases | Bridge Events | Cured | Persistent | Created | Broken Cases (% Fragmented) | Median Dice | Median Recall | Median clDice |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Consensus 7/7** ($N=101$) | **$1.00$** | **101** | **100.0%** | 109 | 0 | 101 | 0 | 10 (9.9%) | **0.7607** | **0.9279** | **0.8873** |
| | **$0.75$** | **101** | **100.0%** | 108 | 0 | 101 | 0 | 10 (9.9%) | 0.7583 | 0.9287 | 0.8885 |
| | **$0.50$** | **101** | **100.0%** | 108 | 0 | 101 | 0 | 9 (8.9%) | 0.7621 | 0.9203 | 0.8897 |
| | **$0.25$** | **101** | **100.0%** | 108 | 0 | 101 | 0 | 8 (7.9%) | 0.7584 | 0.9218 | 0.8846 |
| | **$0.00$** | **82** | **81.2%** | 89 | **19** | **82** | 0 | **32 (31.7%)** | 0.7437 | **0.8748** | 0.8607 |
| **Cured by D2** ($N=7$) | **$1.00$** | 7 | 100.0% | 7 | 0 | 7 | 0 | 0 (0.0%) | 0.7636 | 0.9391 | 0.9012 |
| | **$0.75$** | 7 | 100.0% | 7 | 0 | 7 | 0 | 0 (0.0%) | 0.7629 | 0.9409 | 0.8870 |
| | **$0.50$** | 6 | 85.7% | 6 | 1 | 6 | 0 | 0 (0.0%) | 0.7659 | 0.9430 | 0.8862 |
| | **$0.25$** | 6 | 85.7% | 6 | 1 | 6 | 0 | 0 (0.0%) | 0.7834 | 0.9591 | 0.8973 |
| | **$0.00$** | **1** | **14.3%** | 1 | **6** | **1** | 0 | 1 (14.3%) | 0.7792 | 0.8747 | 0.8417 |
| **Clean Control** ($N=56$) | **$1.00$** | 0 | 0.0% | 0 | 0 | 0 | 0 | 8 (14.3%) | 0.7829 | 0.8207 | 0.8587 |
| | **$0.75$** | 0 | 0.0% | 0 | 0 | 0 | 0 | 10 (17.9%) | 0.7794 | 0.8216 | 0.8599 |
| | **$0.50$** | 0 | 0.0% | 0 | 0 | 0 | 0 | 10 (17.9%) | 0.7802 | 0.8222 | 0.8577 |
| | **$0.25$** | 1 | 1.8% | 1 | 0 | 0 | **1** | 11 (19.6%) | 0.7741 | 0.8221 | 0.8562 |
| | **$0.00$** | 0 | 0.0% | 0 | 0 | 0 | 0 | **23 (41.1%)** | **0.7175** | **0.6879** | **0.7950** |

---

## 4. Results: Raw-Stem Feature Evolution

### Table 2: Raw-Stem Emergence & Separation Profile Across $\alpha$

| Cohort | $\alpha$ | Median $R_{\text{norm}}$ | P25 | P75 | $\% \ge 0.50$ | $\% \ge 0.70$ | Median $\text{SepMargin}$ | Median $\frac{E_{\text{neck}}}{E_{\text{crack}}}$ | Median $\cos(\mathbf{f}_{\text{neck}}, \mathbf{f}_{\text{crack}})$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Consensus 7/7** | **$1.00$** | **0.5508** | 0.0267 | 1.5566 | 54.46% | 45.54% | **-0.0280** | 1.0280 | **0.9658** |
| | **$0.75$** | 0.6253 | 0.0487 | 1.4554 | 55.45% | 47.52% | -0.0169 | 1.0169 | 0.9785 |
| | **$0.50$** | 0.5006 | 0.0034 | 1.3411 | 50.50% | 45.54% | -0.0053 | 1.0053 | 0.9876 |
| | **$0.25$** | 0.8952 | -0.0631 | 1.5222 | 60.40% | 56.44% | +0.0018 | 0.9982 | 0.9933 |
| | **$0.00$** | 0.9440 | -0.2294 | 1.1541 | 70.30% | 65.35% | +0.0045 | 0.9955 | 0.9968 |
| **Cured by D2** | **$1.00$** | -0.3680 | -0.5882 | -0.1025 | 0.00% | 0.00% | -0.0853 | 1.0853 | 0.9383 |
| | **$0.75$** | -0.1605 | -0.4673 | -0.0832 | 0.00% | 0.00% | -0.0536 | 1.0536 | 0.9582 |
| | **$0.50$** | 0.0579 | -0.2541 | -0.0433 | 14.29% | 0.00% | -0.0223 | 1.0223 | 0.9737 |
| | **$0.25$** | 0.2415 | -0.0841 | 0.2514 | 28.57% | 28.57% | -0.0247 | 1.0247 | 0.9835 |
| | **$0.00$** | 0.2648 | -0.0777 | 0.8957 | 14.29% | 14.29% | -0.0163 | 1.0163 | 0.9870 |
| **Clean Control** | **$1.00$** | -0.0200 | -0.5303 | 1.0044 | 42.86% | 33.93% | +0.0115 | 0.9885 | 0.9625 |
| | **$0.75$** | 0.1209 | -0.4851 | 0.8845 | 42.86% | 39.29% | -0.0003 | 1.0003 | 0.9722 |
| | **$0.50$** | -0.1341 | -0.5401 | 0.7712 | 37.50% | 33.93% | -0.0046 | 1.0046 | 0.9820 |
| | **$0.25$** | -0.1169 | -0.4994 | 0.6552 | 35.71% | 30.36% | -0.0098 | 1.0098 | 0.9893 |
| | **$0.00$** | 0.0917 | -0.5844 | 0.2647 | 35.71% | 26.79% | -0.0133 | 1.0133 | 0.9921 |

---

## 5. Decision Gate Evaluation: Sufficiency vs. Necessity

### Criterion 1: Continuous Attenuation Regime ($\alpha \in [0.25, 0.75]$)
* When the AC component is systematically attenuated by $25\%$, $50\%$, and $75\%$, the downstream false bridge count in Consensus 7/7 remains **strictly invariant at $101 / 101$ ($100.0\%$)**:
  - $\alpha = 1.00$: $101 / 101$ bridges
  - $\alpha = 0.75$: $101 / 101$ bridges ($0$ cured)
  - $\alpha = 0.50$: $101 / 101$ bridges ($0$ cured)
  - $\alpha = 0.25$: $101 / 101$ bridges ($0$ cured)
* Across this entire range, downstream Dice ($0.7607 \to 0.7584$) and Recall ($0.9279 \to 0.9218$) remain completely stable.
* **Interpretation:** Reducing the magnitude of the spatially contrast-sensitive component at the stem by three-quarters ($75\%$) does **not** cure a single Consensus false bridge. The downstream network (Stage 0 blocks, ViT bottleneck, and decoder spatial mixing) possesses sufficient learned capacity to amplify the remaining precursor and commit the bridge.

### Criterion 2: Extreme Zero-AC Regime ($\alpha = 0.00$)
* At $\alpha = 0.00$ (completely eliminating the AC filter bank and passing only average color pooling + $1 \times 1$ channel projection):
  - In Consensus 7/7, false bridges decrease from $101 \to 82$ ($19$ cured = $18.8\%$).
  - In D2-cured, false bridges decrease from $7 \to 1$ ($6$ cured = $85.7\%$).
* **The Catastrophic Trade-off:**
  - In Clean Controls, crack breakage (fragmented components) **nearly triples from $8 \to 23$ cases ($14.3\% \to 41.1\%$)**. Median Recall collapses by **$13.3$ percentage points** (from $0.8207 \to 0.6879$), and median Dice drops from $0.7829 \to 0.7175$.
  - In Consensus 7/7, broken cases **more than triple from $10 \to 32$ cases ($9.9\% \to 31.7\%$)**, with Recall dropping from $0.9279 \to 0.8748$.
* **Interpretation:** Total elimination of the AC component does dissolve a fraction of false bridges, but it destroys the network's fundamental ability to trace thin cracks, leading to massive crack fragmentation and continuity failure.

### **Scientific Decision Gate Verdict:**
$$\boxed{\text{Sufficiency: SUPPORTED (as early precursor)} \quad \Big| \quad \text{Necessity: REFUTED / NOT DEMONSTRATED}}$$

1. **Sufficiency of AC for expressing the earliest precursor:**
   - Isolated AC filters reproduce the raw-stem representation collapse ($90.64\%$ collapse rate).
2. **Necessity of AC for downstream false-bridge commitment:**
   - **Refuted in the non-destructive regime ($\alpha \in [0.25, 0.75]$):** Attenuating AC by up to $75\%$ does not cure persistent false bridges ($0/101$ cured). Downstream architectural components are fully capable of committing the bridge without relying on full AC amplitude.
   - **Severe Trade-off at $\alpha = 0.00$:** Zeroing AC reduces bridges by only $18.8\%$ while causing catastrophic crack fragmentation ($41.1\%$ breakage on clean controls).
3. **Architectural Implication:**
   The persistent false-bridge phenotype is **not** a localized defect caused solely by the stem projection kernel. It is a distributed network property where downstream stages (Stage 0 blocks, ViT bottleneck, and Decoder Block 1 spatial mixing) actively amplify and commit the ambiguity.

---

## 6. Epistemic Limits & Wording Boundaries

1. **Higher-Rank Channel-Spatial Structure:**
   The $15.94\%$ non-separable SVD energy is properly designated as **higher-rank channel-spatial structure** rather than cross-channel interaction.
2. **Spatially Contrast-Sensitive Characterization:**
   The zero-mean AC component is characterized as **spatially contrast-sensitive / edge-like** rather than definite edge/gradient filters.
3. **Distribution-Based Conclusions:**
   All scientific conclusions rely on medians, interquartile ranges, and binary threshold fractions rather than heavy-tailed mean emergence values.
4. **No Causal Root Overclaim:**
   This counterfactual test definitively rules out the hypothesis that the stem AC component is an isolated causal root whose simple attenuation could fix false bridges.
