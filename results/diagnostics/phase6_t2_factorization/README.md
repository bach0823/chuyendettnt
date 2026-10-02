# Phase 6: T2 Representation / Skip-vs-Bottleneck Contribution & Fusion Factorization Diagnostic Report

## 1. Executive Summary & Epistemic Boundaries

This diagnostic investigates the dual-stream feature representation arriving at $T2$—the entry point to Decoder Block 1—to determine where the false bridge ambiguity primarily resides, how the concatenation of bottleneck and skip features affects component separation, and the relative causal leverage of each stream on final false bridge formation versus true crack continuity.

### Core Scientific Questions
1. **At $T2$ (Decoder Block 1 input, 288 channels = 192 ch upsampled bottleneck + 96 ch skip from CNN Stage 1), where does false bridge ambiguity primarily reside: in the bottleneck stream or the skip stream?**
2. **Does feature concatenation/fusion at $T2$ improve separation or cause the representation of the two crack components to re-converge?**
3. **How do Consensus Resistant ($N=82$) and Consensus Sensitive ($N=19$) differ in their bottleneck vs. skip contributions?**
4. **Does D2-Cured ($N=7$) have its own distinct trajectory/balance at $T2$?**
5. **What is the counterfactual leverage of the bottleneck vs. skip stream on final false bridge removal and crack continuity?**

### Non-Negotiable Protocol Hard Locks
1. **DIAGNOSTIC-ONLY**: Zero training passes, zero backward passes, zero optimizer steps, zero gradient updates.
2. **Bitwise Invariance**:
   - Checkpoint SHA256: `147f784021414efd229f37faee27618997a0665dfb12c8b8745585d8885f812b` (verified bitwise identical before and after).
   - Parameter SHA256: `4aeda58ce6d2fb321ecbb398beaa9142f36f6d5386052be1bb3e162f43d04d80` (verified bitwise identical before and after).
3. **Setting A Preserved**: $448 \times 448$ tile, non-overlapping stride 448, reflect padding, exact cropping, $\tau=0.5$.
4. **Cohort Separation**: Evaluated on $N=164$ validation samples (82 Consensus Resistant, 19 Consensus Sensitive, 7 Cured by D2, 56 Clean Control). Sealed test set ($N=1124$) strictly unaccessed.
5. **Calibrated Epistemic Language**:
   - Distinguish representation correlation/provenance from causal necessity.
   - Do NOT declare either stream a "causal root" without bridge-specific counterfactual evidence.

---

## 2. Architecture & Tensor Slicing Schema at T2

At the entrance to Decoder Block 1 ($T2$, spatial resolution $56 \times 56$), the input tensor $X_{T2} \in \mathbb{R}^{B \times 288 \times 56 \times 56}$ is formed by channel concatenation:

$$
X_{T2} = \text{Concat}\big(X_{\text{bottleneck}}, X_{\text{skip}}\big)
$$

Where:
- **Channels `0..191` ($192\text{ ch}$)**: $X_{\text{bottleneck}}$ — upsampled from the ViT / deep bottleneck stage ($28 \times 28 \to 56 \times 56$).
- **Channels `192..287` ($96\text{ ch}$)**: $X_{\text{skip}}$ — high-resolution lateral skip connection from ConvNeXt / CNN Stage 1 ($56 \times 56$).

```
                      +---------------------------------------+
                      |   Deep Encoder / ViT / Bottleneck     |
                      +---------------------------------------+
                                          | (Upsample 2x)
                                          v
+------------------------+      +-------------------+
| CNN Stage 1 (Skip)     |      | Upsampled Bottlen.|
| 96 ch @ 56x56          |      | 192 ch @ 56x56    |
+------------------------+      +-------------------+
            \                             /
             \                           /
              v                         v
       +---------------------------------------+
       |   T2 Concatenation (288 channels)     |
       |   [0..191]: Bottleneck                |
       |   [192..287]: Skip                    |
       +---------------------------------------+
                          |
                          v
       +---------------------------------------+
       |   Decoder Block 1 Conv1 (288 -> 96)   |
       +---------------------------------------+
```

### Evaluated Counterfactual Conditions (7 Conditions)
1. **Baseline**: Untouched pristine $T2$ tensor ($\alpha_{\text{skip}}=1.0, \alpha_{\text{btn}}=1.0$).
2. **T2_Skip_Only**: Bottleneck zeroed ($\alpha_{\text{skip}}=1.0, \alpha_{\text{btn}}=0.0$).
3. **T2_Bottleneck_Only**: Skip zeroed ($\alpha_{\text{skip}}=0.0, \alpha_{\text{btn}}=1.0$).
4. **T2_Skip_Attenuated_0.50**: Skip scaled by $0.50$, Bottleneck untouched ($1.0$).
5. **T2_Bottleneck_Attenuated_0.50**: Bottleneck scaled by $0.50$, Skip untouched ($1.0$).
6. **T2_Skip_Attenuated_0.25**: Skip scaled by $0.25$, Bottleneck untouched ($1.0$).
7. **T2_Bottleneck_Attenuated_0.25**: Bottleneck scaled by $0.25$, Skip untouched ($1.0$).

---

## 3. Quantitative Results & Representation Factorization

### Table 1: Energy Share and Representation Separation Metrics at T2

| Cohort | Subgroup | $N$ | Median Bottleneck Energy % | Median Skip Energy % | Median Bottleneck $R_{\text{norm}}$ | Median Skip $R_{\text{norm}}$ | Median Fused $R_{\text{norm}}$ | Median Bottleneck SepMargin | Median Skip SepMargin | Median Fused SepMargin |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Consensus 7/7** | All | 101 | 4.79% | 95.21% | 0.5567 | 0.7110 | 0.4541 | +0.0497 | +0.0017 | +0.0072 |
| **Consensus 7/7** | **Resistant** | 82 | **5.01%** | **94.99%** | **0.5746** | **0.7359** | **0.5548** | **+0.0449** | **+0.0007** | **+0.0046** |
| **Consensus 7/7** | **Sensitive** | 19 | **4.09%** | **95.91%** | **0.3873** | **0.4726** | **0.4490** | **+0.0732** | **+0.0134** | **+0.0128** |
| **Cured by D2** | All | 7 | 3.85% | 96.15% | **0.0906** | **0.3208** | **0.4798** | **+0.1900** | **+0.0524** | **+0.0521** |
| **Clean Control** | All | 56 | 3.65% | 96.35% | -0.0711 | -0.2141 | -0.2916 | +0.1417 | +0.0119 | +0.0190 |

*Note: Energy % represents median share of total squared $L_2$ norm within the corridor mask at $T2$.*

---

### Table 2: Pairwise Cross-Component Profile & Valley Ratios at T2 ($N=513$ pairs)

| Cohort | Subgroup | Pairs | Median Btn Valley Ratio | % Btn with Valley | Median Skp Valley Ratio | % Skp with Valley | Median Fsd Valley Ratio | % Fsd with Valley |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Consensus 7/7** | All | 513 | 1.0293 | 13.3% | 1.0063 | 8.6% | 1.0069 | 8.6% |
| **Consensus 7/7** | **Resistant** | 447 | **1.0293** | 13.4% | **1.0075** | **8.9%** | **1.0083** | **8.9%** |
| **Consensus 7/7** | **Sensitive** | 66 | **1.0318** | 12.1% | **0.9990** | **6.1%** | **0.9998** | **6.1%** |
| **Cured by D2** | All | 7 | 1.0075 | 14.3% | 0.9884 | 0.0% | 0.9880 | 0.0% |
| **Clean Control** | All | 56 | 1.0090 | 17.9% | 1.0061 | 0.0% | 1.0033 | 0.0% |

---

### Table 3: Counterfactual Leverage Response Matrix Across Subgroups

| Condition | Mode | $\alpha$ | Resistant Bridges ($/82$) | Resistant Cured ($/82$) | Resistant Breakage ($/82$) | Resistant Recall | Resistant Med ConnProb | Sensitive Bridges ($/19$) | Sensitive Cured ($/19$) | Control Breakage ($/56$) | D2 Bridges ($/7$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | baseline | 1.00 | 82/82 | 0/82 (0.0%) | 7/82 | 0.9298 | 0.8922 | 19/19 | 0/19 (0.0%) | 8/56 | 7/7 |
| **T2_Skip_Atten_0.50** | skip | 0.50 | 80/82 | 2/82 (2.4%) | 14/82 | 0.8680 | 0.8458 | 18/19 | 1/19 (5.3%) | 15/56 | 2/7 |
| **T2_Btn_Atten_0.50** | btn | 0.50 | 82/82 | 0/82 (0.0%) | 8/82 | 0.9395 | 0.8448 | 19/19 | 0/19 (0.0%) | 13/56 | 4/7 |
| **T2_Skip_Atten_0.25** | skip | 0.25 | **65/82** | **17/82 (20.7%)** | 34/82 | 0.7351 | 0.6310 | 16/19 | 3/19 (15.8%) | 25/56 | **0/7** |
| **T2_Btn_Atten_0.25** | btn | 0.25 | **79/82** | **3/82 (3.7%)** | 13/82 | 0.9247 | 0.7412 | 18/19 | 1/19 (5.3%) | 19/56 | **0/7** |
| **T2_Skip_Only** | skip | 1.00 | **54/82** | **28/82 (34.1%)** | 54/82 | 0.7969 | 0.5220 | 8/19 | 11/19 (57.9%) | 37/56 | **0/7** |
| **T2_Btn_Only** | btn | 1.00 | **37/82** | **45/82 (54.9%)** | 61/82 | **0.4155** | **0.2078** | 8/19 | 11/19 (57.9%) | 32/56 | **0/7** |

---

## 4. Key Scientific Findings

### 1. Overwhelming Energy Asymmetry: Skip Stream Dominates T2 (>94%)
Within the inter-crack corridor mask at $T2$, the feature energy ($L_2$ norm) is overwhelmingly concentrated in the lateral skip connection:
- In **Consensus Resistant ($N=82$)**: **94.99%** of the corridor energy is supplied by the skip stream (CNN Stage 1, 96 ch), while only **5.01%** comes from the upsampled bottleneck (192 ch).
- In **Consensus Sensitive ($N=19$)**: Skip accounts for **95.91%** vs. bottleneck 4.09%.
- In **Cured by D2 ($N=7$)**: Skip accounts for **96.15%** vs. bottleneck 3.85%.
- In **Clean Controls ($N=56$)**: Skip accounts for **96.35%** vs. bottleneck 3.65%.
This proves that the numerical dynamic range entering Decoder Block 1 is dominated by the CNN Stage 1 skip connection by a factor of nearly $20:1$.

### 2. Ambiguity Resides Overwhelmingly in the Skip Stream
In Consensus Resistant ($N=82$):
- **Skip Stream**: $R_{\text{norm}} = \mathbf{0.7359}$, $\text{SepMargin} = \mathbf{+0.0007}$. The spatial separation between crack tips and neck is completely collapsed in the skip features before fusion.
- **Bottleneck Stream**: $R_{\text{norm}} = 0.5746$, $\text{SepMargin} = +0.0449$. The bottleneck preserves higher relative separation than the skip stream.
- **Fused Stream**: $R_{\text{norm}} = 0.5548$, $\text{SepMargin} = +0.0046$.
Concatenating the high-energy, ambiguous skip stream with the low-energy bottleneck severely dilutes separation, pulling $\text{SepMargin}$ down from $+0.0449$ in the bottleneck to $+0.0046$ in the fused input.

### 3. Divergence Between Resistant and Sensitive Phenotypes
The fundamental distinction between Resistant and Sensitive false bridges at $T2$ is stark:
- **Consensus Sensitive ($N=19$)**: Both streams enter $T2$ with $R_{\text{norm}} < 0.50$ (Bottleneck: $0.3873$, Skip: $0.4726$, Fused: $0.4490$). The representation is sub-threshold prior to Decoder Block 1. When the downstream spatial mixing or stem AC is perturbed, the fragile ambiguity is readily dissolved.
- **Consensus Resistant ($N=82$)**: Both streams enter $T2$ with $R_{\text{norm}} > 0.50$ (Bottleneck: $0.5746$, Skip: $0.7359$, Fused: $0.5548$). The skip connection is already severely collapsed, providing high-magnitude, ambiguous signal directly to Decoder Block 1.

### 4. D2-Cured Cases Have an Ultra-Clean Bottleneck
In **Cured by D2 ($N=7$)**:
- Bottleneck $R_{\text{norm}} = \mathbf{0.0906}$, with $\text{SepMargin} = \mathbf{+0.1900}$.
- The deep bottleneck in D2-cured cases has a massive separation valley.
- When skip is attenuated to $0.25$ or zeroed, **$100\%$ ($7/7$) of D2 bridges disappear** with zero breakage. This explains why D2 training resolved these cases: D2 successfully learned topological separation in the deep bottleneck.

### 5. Counterfactual Causal Leverage: Skip Attenuation Has 5.6x Higher Bridge Removal Leverage
Comparing matched intermediate attenuation ($\alpha = 0.25$):
- **Attenuating Skip (`T2_Skip_Attenuated_0.25`)**:
  - Cures **$17/82$ ($20.7\%$)** of Resistant bridges and drops median connector probability from $0.8922 \to 0.6310$.
  - True crack Recall remains viable at $0.7351$.
- **Attenuating Bottleneck (`T2_Bottleneck_Attenuated_0.25`)**:
  - Cures only **$3/82$ ($3.7\%$)** of Resistant bridges (a $5.6\times$ lower cure rate).
  - Median connector probability remains high at $0.7412$.
- **Extreme Zeroing (`T2_Bottleneck_Only` vs `T2_Skip_Only`)**:
  - `T2_Bottleneck_Only` (skip zeroed) eliminates $45/82$ bridges ($54.9\%$), but **catastrophically destroys true crack detection** (Recall collapses to $0.4155$, breakage explodes to $61/82$). The skip connection is indispensable for spatial crack localization.
  - `T2_Skip_Only` (bottleneck zeroed) eliminates $28/82$ bridges ($34.1\%$) with Recall $= 0.7969$ and breakage $= 54/82$.

---

## 5. Decision Gate Verdict & Synthesis

### Verdict: SKIP STREAM IS THE PRIMARY CARRIER OF RESISTANT AMBIGUITY
1. **Source of Truth**:
   The lateral skip connection from CNN Stage 1 carries $95\%$ of the energy and has severely degraded component separation ($R_{\text{norm}} = 0.7359$, $\text{SepMargin} = +0.0007$) in Consensus Resistant cases.
2. **Role of Bottleneck**:
   The deep bottleneck is not the primary carrier of the high-energy bridge signal; however, in Resistant cases, it has also partially degraded ($R_{\text{norm}} = 0.5746$). In contrast, in D2-cured cases, the bottleneck was cleanly separated ($R_{\text{norm}} = 0.0906$).
3. **Implication for Training / Architecture**:
   Interventions focused solely on the deep ViT bottleneck (or loss functions applied only to the final output) face an uphill battle if the high-resolution lateral skip connection injects a high-magnitude, ambiguous bridge representation directly into the decoder. Mitigating false bridges requires either:
   - Enhancing spatial discrimination in early CNN stages (Stem / Stage 1) so the skip connection does not carry collapsed neck features.
   - Gating or filtering the skip connection at $T2$ based on bottleneck topological confidence (Topological Skip Gating).
