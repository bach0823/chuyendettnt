# Phase 6: Upstream Genesis / Stem Information Preservation Diagnostic ($U_0\text{-C0}$ vs $U_0\text{-C1}$)

## 1. Executive Summary & Epistemic Boundaries

This diagnostic investigation evaluates the causal hypothesis of **Upstream Genesis ($H_1$)**:
> *Does early false-bridge emergence originate from non-overlapping patch downsampling ($4\times4$ stride 4) in the Stem losing boundary separation for thin crack structures ($<4$ px)?*

### Causal Isolation Protocol
To isolate spatial overlap from confounding factors (such as depth, normalization drift, parameter capacity, and pretrained core fine-tuning), the experiment executed a strictly controlled pair:
* **$U_0\text{-C0}$ (Identity Control):** $7\times7$ Conv2d (stride 4, padding 3) with center $4\times4$ containing pretrained Candidate B weights, and all 33 halo positions permanently frozen at $0.0$.
  * **Verified bitwise invariant:** $\max |\text{Stem}_{\text{base}} - \text{Stem}_{C0}| = 0.00\text{e}+00$.
  * Exactly reproduces Candidate B baseline on Validation ($N=348$): **110 Bridge Images, 118 Bridge Events, 88 Resistant Events**.
* **$U_0\text{-C1}$ (Learnable Spatial Overlap):** Identical architecture, identical center $4\times4$ frozen pretrained weights, identical LayerNorm2d.
  * **Trainable degrees of freedom:** Exactly **4,752 halo parameters** (33 spatial positions $\times 48 \text{ channels} \times 3 \text{ RGB}$).
  * Initialized to $0.0$ at $t=0$. Center $4\times4$ strictly frozen via backward slice zeroing and bitwise post-step assertion.
  * Trained for 8 epochs on Crack500 train set ($N=1896$, batch 14, lr $10^{-4}$, AdamW, loss = BCE + Dice).
  * Downstream network (Stage 0 through Decoder B1) **100% FROZEN**.

---

## 2. Experimental Readout & Subgroup Breakdown

### Primary Validation Results ($N=348$, Setting A Tiling, $\tau=0.5$)

| Cohort | $N$ | $C0$ Bridge Img | $C1$ Bridge Img | $\Delta$ Bridge Img | $C0$ Bridge Ev | $C1$ Bridge Ev | $\Delta$ Bridge Ev | Cured Img | Created Img | $C0$ Break Ev | $C1$ Break Ev | $\Delta$ Break Ev | $C0$ Dice | $C1$ Dice | $\Delta$ Dice | $C0$ clDice | $C1$ clDice | $\Delta$ clDice |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Full Validation** | 348 | 110 | 110 | **0** | 118 | 116 | **-2 (-1.7%)** | 4 | 4 | 35 | 38 | **+3** | 0.7641 | 0.7478 | **-0.0163** | 0.8499 | 0.8342 | **-0.0157** |
| **Consensus 7/7** | 101 | 101 | 101 | **0** | 109 | 107 | **-2** | 0 | 0 | 11 | 8 | **-3** | 0.7371 | 0.7288 | -0.0083 | 0.8455 | 0.8399 | -0.0056 |
| **Resistant 82** | 82 | 82 | 82 | **0** | 88 | 86 | **-2** | 0 | 0 | 8 | 7 | **-1** | 0.7575 | 0.7497 | -0.0078 | 0.8586 | 0.8516 | -0.0070 |
| **Sensitive 19** | 19 | 19 | 19 | **0** | 21 | 21 | **0** | 0 | 0 | 3 | 1 | **-2** | 0.6491 | 0.6385 | -0.0106 | 0.7892 | 0.7896 | +0.0004 |
| **Clean Control** | 56 | 0 | 0 | **0** | 0 | 0 | **0** | 0 | 0 | 8 | 7 | **-1** | 0.7540 | 0.7372 | -0.0168 | 0.8200 | 0.8007 | -0.0193 |

---

## 3. Layer 1 Stem Representation Audit ($N=157$ Cohort Samples)

Activation energy after Stem LayerNorm2d ($112\times112$, 48 channels):

| Cohort | Metric | $U_0\text{-C0}$ | $U_0\text{-C1}$ | $\Delta (C1 - C0)$ | Interpretation |
|:---|:---|:---:|:---:|:---:|:---|
| **Resistant ($N=82$)** | **SepMargin** | 0.0125 | 0.0319 | **+0.0194** | Marginal increase, but neck corridor remains at $96.8\%$ crack energy. |
| | **$\cos(\text{neck}, \text{crack})$** | 0.8397 | 0.8256 | **-0.0141** | Minimal angular change; representations remain collinear ($>0.82$). |
| | **$E_{\text{neck}}$** | 3.3122 | 4.2015 | +0.8893 | Energy increased uniformly across all spatial loci. |
| | **$E_{\text{crack}}$** | 3.3535 | 4.3457 | +0.9922 | Energy increased uniformly. |
| | **$E_{\text{bg}}$** | 3.0966 | 4.0850 | +0.9884 | Energy increased uniformly. |
| **Sensitive ($N=19$)** | **SepMargin** | 0.0170 | 0.0280 | **+0.0110** | Slight increase, representation still unseparated. |
| | **$\cos(\text{neck}, \text{crack})$** | 0.8104 | 0.8051 | -0.0052 | Unchanged collinearity. |
| **Clean Control ($N=56$)** | **SepMargin** | 0.0270 | 0.0037 | **-0.0233** | Margin degraded on clean cracks. |
| | **$\cos(\text{neck}, \text{crack})$** | 0.8108 | 0.7917 | -0.0191 | Unchanged collinearity. |

---

## 4. Key Scientific Findings & Decision Tree Conclusion

1. **Spatial Overlap Halo Alone Fails to Resolve False Bridges:**
   * Across the entire Validation set ($N=348$), bridge images remained **110/348 $\to$ 110/348** (0% reduction).
   * 4 bridge images were cured, but exactly 4 new bridge images were created (**net 0 cured**).
   * On the Consensus Resistant Cohort ($N=82$), **82/82 images remained bridged** ($88 \to 86$ events, trivial $-2.3\%$).

2. **Downstream Coupling Mismatch:**
   * Downstream frozen layers (ConvNeXt Stages 0–3, ViT, SAGE Routers, Decoder) were pretrained and optimized for non-overlapping $4\times4$ patch features.
   * Allowing the Stem halo to adapt independently caused feature distribution shift, dropping global Dice by **$-1.63\%$** ($0.7641 \to 0.7478$) and clDice by **$-1.57\%$** ($0.8499 \to 0.8342$).

3. **Causal Conclusion:**
   * **Hypothesis $H_1$ (Single-layer stride-4 non-overlap is the sole causal driver) is REFUTED.**
   * Adding spatial overlap at the Stem level ($7\times7$ with learnable halo) without retraining the downstream architecture does NOT eliminate false bridges.
   * The false-bridge phenomenon is not merely an aliasing artifact of the single-layer $4\times4$ patchification; it is deeply coupled with multi-stage spatial receptive field integration and downstream decoder reconvergence.
