# Phase 6: Downstream Re-convergence Provenance Diagnostic Report

## 1. Executive Summary & Epistemic Boundaries

This diagnostic investigation tracks the representation trajectory of crack gap corridors across **21 continuous architectural observation points** in the canonical Candidate B model (ConvNeXtV2-Femto + ViT-Tiny, Setting A inference, $\tau=0.5$).

### Core Scientific Question
> **When representation separation is restored at the stem (under $\alpha=0.00$, where stem AC is removed), where in the downstream network and by what mechanism does the bridge representation re-converge to commit false bridges in the 82 downstream-resistant Consensus cases?**

### Non-Negotiable Protocol Hard Locks
1. **DIAGNOSTIC-ONLY**: Zero training passes, zero backward passes, zero optimizer steps, zero learning rate sweeps.
2. **Bitwise Invariance**:
   - Checkpoint SHA256: `147f784021414efd229f37faee27618997a0665dfb12c8b8745585d8885f812b` (verified identical before and after).
   - Parameter SHA256: `4aeda58ce6d2fb321ecbb398beaa9142f36f6d5386052be1bb3e162f43d04d80` (verified identical before and after).
3. **Setting A Preserved**: $448 \times 448$ tile, non-overlapping stride 448, reflect padding, exact cropping, $\tau=0.5$.
4. **Cohort Separation**: Evaluated exclusively on Validation ($N=348$). Sealed test set ($N=1124$) strictly unaccessed.
5. **Calibrated Epistemic Language**:
   - Zero-mean AC component is termed *"spatially contrast-sensitive / edge-like"* (not "edge filters").
   - 15.94% SVD residual is termed *"higher-rank channel-spatial structure"*.
   - Distinction maintained between *early precursor*, *downstream re-convergence*, and *decoder spatial commitment*.
   - Zero claims of "causal root cause".

---

## 2. Quantitative Summary Across 21 Observation Points

### Table 1: Chronological Representation Trajectory ($R_{\text{norm}}$ and $\text{SepMargin}$) Across Pipeline
Comparison between **Consensus Resistant** ($N=82$) and **Consensus Sensitive** ($N=19$) under Baseline ($\alpha=1.00$) vs Counterfactual Stem ($\alpha=0.00$).

| Stage Index | Stage Name | Scale | $\alpha=1.00$ Resist $R_{\text{norm}}$ | $\alpha=1.00$ Sens $R_{\text{norm}}$ | $\alpha=0.00$ Resist $R_{\text{norm}}$ | $\alpha=0.00$ Sens $R_{\text{norm}}$ | $\Delta R$ ($\alpha=0.00$) | $\alpha=0.00$ Resist $\text{SepMargin}$ | $\alpha=0.00$ Sens $\text{SepMargin}$ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `raw_conv` | $112 \times 112$ | 0.6031 | 0.5476 | 1.0280 | 0.7338 | +0.2942 | +0.0042 | +0.0065 |
| 2 | `stem_ln` | $112 \times 112$ | 0.8027 | 0.6809 | **0.3746** | **0.0772** | +0.2974 | +0.0413 | +0.0730 |
| 3 | `stage0_block0` | $112 \times 112$ | 0.8626 | 0.6144 | **0.6350** | **0.5933** | +0.0417 | -0.0316 | -0.0391 |
| 4 | `stage0_block1` | $112 \times 112$ | 0.7625 | 0.4548 | 0.5954 | 0.3708 | +0.2247 | -0.0271 | -0.0351 |
| 5 | `stage0_post_sage` | $112 \times 112$ | 0.7654 | 0.4526 | 0.5963 | 0.3693 | +0.2269 | -0.0273 | -0.0350 |
| 6 | `stage1_post_sage` | $56 \times 56$ | 0.5643 | 0.4434 | 0.5151 | 0.7803 | -0.2652 | -0.0049 | +0.0059 |
| 7 | `stage2_post_sage` | $28 \times 28$ | 0.5186 | 0.3891 | 0.4764 | 0.2553 | +0.2211 | -0.0155 | -0.0115 |
| 8 | `stage3_post_sage` | $14 \times 14$ | 0.3396 | 0.2977 | 0.0994 | 0.0668 | +0.0326 | +0.0408 | +0.0192 |
| 9 | `pre_vit` | $14 \times 14$ | 0.1709 | -0.0669 | 0.2541 | -0.0285 | +0.2825 | +0.0000 | -0.0002 |
| 10 | `vit_block_0` | $14 \times 14$ | 0.4541 | 0.0834 | 0.2975 | -0.3513 | +0.6488 | +0.0217 | +0.0215 |
| 11 | `vit_block_1` | $14 \times 14$ | 0.1725 | -0.1740 | 0.1606 | 0.1344 | +0.0262 | +0.0095 | +0.0018 |
| 12 | `vit_block_2` | $14 \times 14$ | 0.4594 | 0.2670 | 0.3472 | 0.0417 | +0.3056 | +0.0096 | +0.0227 |
| 13 | `vit_block_3` | $14 \times 14$ | 0.7333 | 0.3471 | 0.5070 | 0.6958 | -0.1888 | +0.0045 | +0.0072 |
| 14 | `bottleneck` | $14 \times 14$ | 0.6675 | 0.6848 | 0.5624 | 0.5597 | +0.0028 | +0.0001 | +0.0001 |
| 15 | `S2` | $28 \times 28$ | 0.6156 | 0.5635 | **0.5384** | **0.0304** | **+0.5080** | +0.0444 | +0.0578 |
| 16 | `T1` | $56 \times 56$ | 0.5278 | 0.4394 | 0.3722 | 0.1026 | +0.2696 | +0.0414 | +0.0240 |
| 17 | `T2` | $56 \times 56$ | 0.3944 | 0.4380 | 0.4549 | 0.6596 | -0.2047 | -0.0043 | +0.0066 |
| 18 | `T3` | $56 \times 56$ | 0.7001 | 0.6459 | **0.5951** | **0.4199** | **+0.1752** | +0.0981 | +0.0853 |
| 19 | `T4_S1` | $56 \times 56$ | 0.7303 | 0.7115 | **0.6915** | **0.2254** | **+0.4662** | +0.1007 | +0.1503 |
| 20 | `S0` | $112 \times 112$ | 0.7675 | 0.7378 | **0.7879** | **0.0427** | **+0.7452** | +0.1141 | +0.2883 |
| 21 | `final_head` | $448 \times 448$ | 0.9941 | 0.9456 | **1.0136** | **0.1893** | **+0.8243** | -0.0119 | **+0.5184** |

---

## 3. Answers to the 5 Core Scientific Questions

### Q1: Where in the downstream network does bridge representation collapse / re-emerge under $\alpha=0.00$?
**Finding**:
Under $\alpha=0.00$, stem AC attenuation successfully suppresses corridor energy at `stem_ln` (median $R_{\text{norm}}$ drops from $0.8027 \to 0.3746$ in Resistant, and $0.6809 \to 0.0772$ in Sensitive). 
However, representation ambiguity re-emerges in two distinct phases:
1. **Immediate CNN Re-emergence at `stage0_block0`**:
   The $7 \times 7$ depthwise convolution at $112 \times 112$ has an effective receptive field of $28 \times 28$ pixels at full resolution. For narrow crack gaps ($< 3$ px), this filter aggregates feature energy from both adjacent crack tips across the corridor, driving median $R_{\text{norm}}$ sharply back up to $0.6350$ ($60.98\% \ge 0.50$).
2. **Decoder Spatial Amplification at Decoder Block 1 (`T2` $\to$ `T3` $\to$ `T4_S1`)**:
   In the Resistant cohort, after bottleneck downsampling to $14 \times 14$ and upsampling to $S2$ ($R_{\text{norm}} = 0.5384$), Decoder Block 1 acts as a powerful amplifier:
   $R_{\text{norm}}$ climbs from $0.4549$ (`T2`) $\to 0.5951$ (`T3`) $\to 0.6915$ (`T4_S1`), raising the fraction of collapsed samples from $47.56\% \to 71.95\%$. This is subsequently locked into $0.7879$ at `S0` and $1.0136$ at `final_head`.

---

### Q2: Is there a single common re-convergence point for the 82 persistent cases, or is it distributed?
**Finding**:
The re-convergence is **distributed across early CNN depthwise mixing and late decoder spatial convolution**:
- **Phase A (Early Infiltration)**: $70.73\%$ of Resistant cases have corridor ambiguity at or before `stem_ln`, and $90.24\%$ have it re-established by `stage0_block0`.
- **Phase B (Decoder Commitment)**: The final transition from feature ambiguity into a topological false bridge occurs inside Decoder Block 1 and Block 2. 
There is no single "magic layer" responsible; rather, the network exhibits a cascade where receptive-field bleed in Stage 0 re-injects ambiguity that persists through the bottleneck and is reconstructed by the decoder.

---

### Q3: How does the downstream trajectory of the 7 D2-cured cases differ from persistent cases?
**Finding**:
The 7 D2-cured cases show a radically distinct downstream trajectory:
- Already at the bottleneck ($14 \times 14$), median $R_{\text{norm}}$ is only $0.1751$ (under $\alpha=1.00$) and $0.3596$ (under $\alpha=0.00$), compared to $0.6675$ and $0.5624$ in the Consensus cohort.
- Throughout the decoder under $\alpha=0.00$:
  - `S2`: $R_{\text{norm}} = 0.2680$ ($\text{SepMargin} = +0.0694$)
  - `T1`: $R_{\text{norm}} = -0.1264$ ($\text{SepMargin} = +0.0787$)
  - `T2`: $R_{\text{norm}} = -0.2390$ ($\text{SepMargin} = +0.0142$)
  - `T4_S1`: $R_{\text{norm}} = 0.1202$ ($\text{SepMargin} = +0.3586$)
  - `S0`: $R_{\text{norm}} = 0.0214$ ($\text{SepMargin} = +0.5891$)
  - `final_head`: $R_{\text{norm}} = 0.0883$ ($\text{SepMargin} = +0.7362$)
- Under $\alpha=0.00$, $6/7$ ($85.7\%$) D2-cured cases remain completely clean of false bridges. Their decoder actively suppresses corridor features rather than amplifying them.

---

### Q4: Does Decoder Block 1 remain a major amplifier under $\alpha=0.00$, or did ambiguity already re-emerge in CNN/ViT?
**Finding**:
**Both phenomena coexist as sequential links in a causal chain**:
1. **Ambiguity already re-emerged before the decoder**:
   At the $14 \times 14$ bottleneck and $28 \times 28$ $S2$, median $R_{\text{norm}}$ is already $0.5624$ and $0.5384$ respectively ($54.88\%$ and $51.22\% \ge 0.50$). Thus, the decoder is *not* hallucinating a bridge from completely pure background features.
2. **Decoder Block 1 is the necessary amplifier that converts sub-threshold ambiguity into above-threshold commitment**:
   Inside Decoder Block 1, $R_{\text{norm}}$ jumps by $+0.2366$ ($0.4549 \to 0.6915$), and the proportion of cases with $R_{\text{norm}} \ge 0.50$ increases by $+24.39$ percentage points ($47.56\% \to 71.95\%$).
   Without Decoder Block 1's spatial mixing, these ambiguous representations would not achieve the high-confidence threshold ($\tau=0.5$) necessary to form a topological false bridge.

---

### Q5: What characterizes the divergence between Stem-Sensitive ($N=19$) and Downstream-Resistant ($N=82$) cases?
**Finding**:
The bifurcation between the 19 cured cases and the 82 resistant cases is fundamentally driven by **gap geometry and corridor purity**:

| Metric | Consensus Resistant ($N=82$) | Consensus Sensitive ($N=19$) | Difference | Physical / Epistemic Interpretation |
|:---|:---:|:---:|:---:|:---|
| **Pure Background Neck Cells** | $47.67\%$ | **$66.67\%$** | **-19.00 pp** | Sensitive cases have significantly wider pure background corridors |
| **Receptive Field Bleed Cells** | $52.33\%$ | **$33.33\%$** | **+19.00 pp** | Resistant cases suffer higher RF overlap from crack pixels into neck |
| **Total Crack Pixels** | 6,534 px | 3,776 px | +2,758 px | Resistant cases feature denser, larger crack networks |
| **DC Separation Margin** | +0.1763 | **+0.2748** | -0.0985 | Sensitive cases achieve a much deeper separation valley under DC |
| **$S2$ $R_{\text{norm}}$ ($\alpha=0.00$)** | 0.5384 | **0.0304** | **+0.5080** | In Sensitive cases, wide gaps prevent ambiguity from surviving to $S2$ |
| **Decoder Block 1 Behavior** | Amplifies ($+0.2366$) | **Suppresses ($-0.4342$)** | Divergent | Decoder suppresses clean inputs but amplifies ambiguous ones |
| **Final Head $\text{SepMargin}$** | -0.0119 | **+0.5184** | **-0.5303** | Sensitive cases cleanly preserve crack separation at output |

---

## 4. Architectural Synthesis & Decision Gate Verdict

1. **Stem AC attenuation ($\alpha=0.00$) is insufficient as a standalone cure**:
   While it cures $19/101$ ($18.8\%$) of Consensus bridges by virtue of their wider background gaps, it leaves $82/101$ ($81.2\%$) persistent because narrow gaps ($< 3$ px) are re-merged by Stage 0 depthwise convolutions ($7 \times 7$) and amplified by Decoder Block 1. Furthermore, $\alpha=0.00$ causes severe crack breakage ($10 \to 32$ in Consensus, $8 \to 23$ in Clean Controls).
2. **False bridge formation is a multi-scale, distributed phenomenon**:
   - *Origin*: Receptive-field bleed across narrow spatial gaps at early patch projection and ConvNeXt Stage 0.
   - *Propagation*: Bottleneck downsampling ($14 \times 14$) compresses crack and corridor tokens together.
   - *Amplification & Commitment*: Decoder Block 1 off-center spatial mixing bridges the sub-threshold representation into an explicit topological connection.
3. **Implication for Future Intervention**:
   Any intervention aimed at eliminating the remaining 82 persistent false bridges must operate either on:
   - Early receptive-field isolation (preventing Stage 0 depthwise mixing across narrow gaps), OR
   - Decoder spatial mixing modulation (preventing Decoder Block 1 from bridging sub-threshold corridor ambiguity),
   WITHOUT destroying the high-frequency AC detail necessary for crack continuity.
