# Phase 6: Downstream Spatial-Mixing Causality Diagnostic Report

## 1. Executive Summary & Epistemic Boundaries

This diagnostic performs a direct, counterfactual causal ablation on the two primary downstream spatial-mixing loci identified in prior provenance tracking:
1. **Locus A — Stage 0 Block 0 ($7 \times 7$ Depthwise Conv)**: The earliest observed CNN locus where corridor ambiguity re-emerged after stem AC suppression.
2. **Locus B — Decoder Block 1 ($\text{Conv1}$ and $\text{Conv2}$ $3 \times 3$ Convs)**: The late decoder locus where sub-threshold ambiguity was amplified into high-confidence bridge commitment.

### Core Scientific Question
> **Is off-center spatial mixing at Stage 0 Block 0 or Decoder Block 1 causally necessary for false bridge formation in the 82 Downstream-Resistant Consensus cases, or do they merely correlate with activation trajectories?**

### Non-Negotiable Protocol Hard Locks
1. **DIAGNOSTIC-ONLY**: Zero training passes, zero backward passes, zero optimizer steps, zero gradient updates.
2. **Bitwise Invariance**:
   - Checkpoint SHA256: `147f784021414efd229f37faee27618997a0665dfb12c8b8745585d8885f812b` (verified identical before and after).
   - Parameter SHA256: `4aeda58ce6d2fb321ecbb398beaa9142f36f6d5386052be1bb3e162f43d04d80` (verified identical before and after).
3. **Setting A Preserved**: $448 \times 448$ tile, non-overlapping stride 448, reflect padding, exact cropping, $\tau=0.5$.
4. **Cohort Separation**: Evaluated on $N=164$ validation samples (82 Consensus Resistant, 19 Consensus Sensitive, 7 Cured by D2, 56 Clean Control). Sealed test set ($N=1124$) strictly unaccessed.
5. **Epistemic Distinction**:
   - *Re-emergence locus*: A layer where bridge-like signal re-appears in forward activation maps.
   - *Amplification locus*: A layer where bridge signal magnitude increases.
   - *Causal necessity*: Claimed ONLY if counterfactual attenuation of that layer's specific operation eliminates the failure phenotype without destroying baseline functionality.

---

## 2. Quantitative Summary Across Interventions

### Table 1: Subgroup Response Matrix (Bridge Removal vs Continuity Cost)

| Intervention Condition | Locus | $\alpha_{\text{spatial}}$ | Resistant Bridges ($/82$) | Resistant Cured ($/82$) | Resistant Breakage ($/82$) | Resistant Recall | Sensitive Cured ($/19$) | Sensitive Breakage ($/19$) | Control Breakage ($/56$) | D2-Cured Bridges ($/7$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | Baseline | 1.00 | 82/82 | 0 (0.0%) | 7/82 | 0.9298 | 0 (0.0%) | 3/19 | 8/56 | 7/7 |
| **Stage0_DW_0.75** | Stage0 DW | 0.75 | 82/82 | 0 (0.0%) | 7/82 | 0.9292 | 0 (0.0%) | 3/19 | 5/56 | 7/7 |
| **Stage0_DW_0.50** | Stage0 DW | 0.50 | 82/82 | 0 (0.0%) | 7/82 | 0.9267 | 0 (0.0%) | 2/19 | 7/56 | 7/7 |
| **Stage0_DW_0.25** | Stage0 DW | 0.25 | 82/82 | 0 (0.0%) | 4/82 | 0.9235 | 0 (0.0%) | 3/19 | 9/56 | 6/7 |
| **Stage0_DW_0.00** | Stage0 DW | 0.00 | **77/82** | **5 (6.1%)** | **12/82** | **0.8841** | **2 (10.5%)** | **5/19** | **14/56** | **2/7** |
| **Decoder_B1_0.75** | Decoder B1 | 0.75 | 82/82 | 0 (0.0%) | 8/82 | 0.9439 | 0 (0.0%) | 4/19 | 13/56 | 5/7 |
| **Decoder_B1_0.50** | Decoder B1 | 0.50 | 81/82 | 1 (1.2%) | 10/82 | 0.9148 | 0 (0.0%) | 5/19 | 18/56 | **1/7** |
| **Decoder_B1_0.25** | Decoder B1 | 0.25 | **37/82** | **45 (54.9%)** | **70/82** | **0.5563** | **13 (68.4%)** | **19/19** | **45/56** | **0/7** |
| **Decoder_B1_0.00** | Decoder B1 | 0.00 | **0/82** | **82 (100%)** | 1/82 | 0.0000 | 19 (100%) | 0/19 | 0/56 | 0/7 |

---

### Table 2: Representation & Probability Dynamics Across Conditions (Consensus 7/7, $N=101$)

| Condition | Median Dice | Median Recall | Median Precision | Median clDice | Mean Connector Prob | Median Connector Prob | Stage0 Block0 $R_{\text{norm}}$ | Decoder B1 $T4\_S1$ $R_{\text{norm}}$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | 0.7607 | 0.9279 | 0.6830 | 0.8873 | 0.8894 | 0.8921 | 0.8264 | 0.7250 |
| **Stage0_DW_0.75** | 0.7611 | 0.9248 | 0.6843 | 0.8886 | 0.8840 | 0.8885 | 0.7956 | 0.7297 |
| **Stage0_DW_0.50** | 0.7592 | 0.9220 | 0.6833 | 0.8769 | 0.8816 | 0.8947 | 0.8313 | 0.7140 |
| **Stage0_DW_0.25** | 0.7540 | 0.9200 | 0.6787 | 0.8699 | 0.8839 | 0.8963 | 0.8720 | 0.7097 |
| **Stage0_DW_0.00** | 0.7511 | 0.8738 | 0.7032 | 0.8632 | 0.7629 | 0.7958 | 0.9173 | 0.6212 |
| **Decoder_B1_0.75** | 0.7619 | 0.9343 | 0.6789 | 0.8878 | 0.8889 | 0.9006 | 0.8264 | 0.7116 |
| **Decoder_B1_0.50** | 0.7742 | 0.9111 | 0.7245 | 0.8886 | 0.7896 | 0.7991 | 0.8264 | 0.6702 |
| **Decoder_B1_0.25** | 0.6760 | 0.5372 | 0.9064 | 0.7679 | **0.3548** | **0.3131** | 0.8264 | 0.6282 |
| **Decoder_B1_0.00** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0612 | 0.0566 | 0.8264 | 0.6278 |

---

## 3. Detailed Findings by Scientific Question

### A. Stage 0 Block 0 ($7 \times 7$ Depthwise Conv) Counterfactual
1. **Hypothesis**: The spatial mixing of the $7 \times 7$ depthwise kernel at $112 \times 112$ was the causal root that regenerated corridor ambiguity when stem AC was removed.
2. **Empirical Result**:
   - At $\alpha_{\text{dw}} \in [0.25, 0.75]$, **$0/82$** Resistant bridges and **$0/19$** Sensitive bridges were cured ($100\%$ persistence).
   - Even at $\alpha_{\text{dw}} = 0.00$ (complete elimination of off-center spatial mixing, reducing the depthwise conv to purely pointwise channel scaling), **$77/82$ ($93.9\%$)** of Resistant bridges and **$17/19$ ($89.5\%$)** of Sensitive bridges persisted.
   - Connector probability remained strongly above threshold (mean $0.7629$, median $0.7958$).
   - Crack breakage increased in Clean Controls from $8/56 \to 14/56$ ($25.0\%$).
3. **Verdict**: **NOT CAUSALLY NECESSARY / NOT A CAUSAL BOTTLENECK**.
   Stage 0 Block 0 is an *early activation re-emergence locus*, but its off-center spatial mixing is NOT the causal mechanism driving false bridge formation. The network easily bypasses or compensates for the loss of Stage 0 depthwise mixing.

---

### B. Decoder Block 1 (Conv1 & Conv2 $3 \times 3$) Counterfactual
1. **Hypothesis**: Off-center spatial mixing in Decoder Block 1 is the necessary engine that amplifies sub-threshold corridor representations into topological binary bridges.
2. **Empirical Result**:
   - At $\alpha_{\text{dec}} = 0.50$, bridge removal in Consensus is negligible ($1/82$, $1.2\%$), but in the D2-cured cohort, $6/7$ ($85.7\%$) bridges disappear cleanly.
   - At $\alpha_{\text{dec}} = 0.25$, **$45/82$ ($54.9\%$)** of Resistant bridges and **$13/19$ ($68.4\%$)** of Sensitive bridges are cured! Total Consensus bridge count drops from $101 \to 43$ ($57.4\%$ cured). Mean connector probability collapses from $0.889 \to 0.355$ (median $0.313$), successfully breaking the false bridge corridor.
   - **Severe Continuity Cost**: At $\alpha_{\text{dec}} = 0.25$, crack breakage surges from $7/82 \to 70/82$ ($85.4\%$) in Resistant, and from $8/56 \to 45/56$ ($80.4\%$) in Clean Controls. Recall collapses from $0.9298 \to 0.5563$ ($-37.35$ percentage points).
   - At $\alpha_{\text{dec}} = 0.00$, prediction collapses completely to near-empty masks (Recall $= 0.000$).
3. **Verdict**: **AMPLIFICATION & COMMITMENT ENGINE (Structurally Coupled to Continuity)**.
   Decoder Block 1 off-center spatial mixing is causally necessary for bridge commitment, but it is *symmetrically necessary* for thin crack continuity. Uniformly attenuating its off-center taps acts as a blunt spatial erosion operator.

---

## 4. Synthesis Matrix: Locus $\times$ Subgroup $\times$ Causal Role

| Locus | Subgroup | Bridge Response | Continuity Cost | Causal Classification | Epistemic Verdict |
|:---|:---|:---:|:---:|:---:|:---|
| **Stage 0 Block 0** ($7 \times 7$ DW) | Consensus Resistant ($N=82$) | Insensitive ($93.9\%$ persist at $\alpha=0.0$) | Mild breakage ($8.5\% \to 14.6\%$) | **NOT CAUSAL** | Activation re-emergence is a correlational symptom, not causal root. |
| **Stage 0 Block 0** ($7 \times 7$ DW) | Consensus Sensitive ($N=19$) | Insensitive ($89.5\%$ persist at $\alpha=0.0$) | Moderate breakage ($15.8\% \to 26.3\%$) | **NOT CAUSAL** | No meaningful effect on bridge phenotype. |
| **Stage 0 Block 0** ($7 \times 7$ DW) | Cured by D2 ($N=7$) | Moderately sensitive ($71.4\%$ cured at $\alpha=0.0$) | Low breakage | **CONTRIBUTORY (Weak)** | Only affects weakly-bridged cases. |
| **Decoder Block 1** ($3 \times 3$ Convs) | Consensus Resistant ($N=82$) | Highly sensitive at $\alpha \le 0.25$ ($54.9\%$ cured) | Severe breakage ($8.5\% \to 85.4\%$) | **AMPLIFICATION ONLY** | Necessary for bridge commitment, but strictly coupled to crack continuity. |
| **Decoder Block 1** ($3 \times 3$ Convs) | Consensus Sensitive ($N=19$) | Highly sensitive at $\alpha \le 0.25$ ($68.4\%$ cured) | Catastrophic breakage ($15.8\% \to 100\%$) | **AMPLIFICATION ONLY** | Strong coupling to thin structure survival. |
| **Decoder Block 1** ($3 \times 3$ Convs) | Cured by D2 ($N=7$) | Exceptionally sensitive at $\alpha=0.50$ ($85.7\%$ cured) | Minimal breakage ($0 \to 2$) | **AMPLIFICATION ONLY** | D2 cases sit at boundary; small spatial damping cures them cleanly. |

---

## 5. Decision Gate Verdict & Epistemic Boundaries

1. **Elimination of Stage 0 Block 0 as an Intervention Target**:
   Counterfactual attenuation proves that Stage 0 Block 0 depthwise spatial mixing is NOT the causal bottleneck for Resistant bridges ($93.9\%$ persist even when depthwise spatial mixing is completely zeroed). We formally close Stage 0 depthwise mixing as a primary candidate for architectural intervention.
2. **Identification of Decoder Block 1 as a Spatial Amplifier, NOT an Isolated Solution**:
   While Decoder Block 1 spatial mixing is causally required to thicken and bridge the corridor across $\tau=0.5$, it cannot be disabled or uniformly attenuated without simultaneously destroying thin crack continuity (Recall drops by $-37.4$ pp; breakage jumps to $85.4\%$).
3. **The Core Dilemma Established**:
   - The false bridge in Consensus Resistant cases is NOT caused by a single rogue convolution kernel.
   - It is a **multi-scale topological failure**: receptive-field bleed across narrow gaps ($< 3$ px) at the stem/stage0 enters the bottleneck and is reconstructed as crack mass by the decoder.
   - Uniform weight interventions (stem AC attenuation, depthwise attenuation, decoder attenuation) all hit the same fundamental trade-off: **bridge reduction is mechanically coupled to crack breakage**.
4. **Recommendation for Subsequent Research**:
   Any successful resolution must be **topologically selective** (e.g., boundary-aware separation loss, anisotropic connectivity constraints, or direction-sensitive lateral inhibition), rather than isotropic spatial kernel attenuation.
