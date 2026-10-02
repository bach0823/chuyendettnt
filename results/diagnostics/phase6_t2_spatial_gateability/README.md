# Phase 6: T2 Spatial Discriminability / Gateability Diagnostic Report

## 1. Executive Summary & Epistemic Boundaries

This diagnostic investigates whether there exists an intrinsic spatial statistic or feature relationship between Skip ($S$, $96\text{ ch}$) and Bottleneck ($B$, $192\text{ ch}$) at the $T2$ junction that can discriminate:
1. **True Crack** (`CRACK`),
2. **False-Bridge Corridor** (`NECK`),
3. **Ordinary Background** (`BACKGROUND`),
without using ground-truth masks at inference time.

### Epistemic Wording Boundaries
- **Diagnostic Gateability, NOT a Solution**: High or low discriminability of a feature statistic does not prove that a trained gating module would succeed or fail. It evaluates whether the raw feature representations arriving at $T2$ in the baseline checkpoint possess sufficient separability to serve as an input signal for selective gating.
- **Strict Distinction between Association and Causality**: A spatial difference between $B$ and $S$ does not prove "Skip contamination" or "Bottleneck topological reliability".
- **Zero Ground Truth at Inference**: All candidate statistics were computed strictly from the channel tensors $B(x, y)$ and $S(x, y)$. Ground-truth masks were used exclusively post-hoc as evaluation targets.

### Non-Negotiable Protocol Hard Locks
1. **DIAGNOSTIC-ONLY**: Zero training passes, zero backward passes, zero optimizer steps, zero gradient updates.
2. **Bitwise Invariance**:
   - Checkpoint SHA256: `147f784021414efd229f37faee27618997a0665dfb12c8b8745585d8885f812b` (verified bitwise identical before and after).
   - Parameter SHA256: `4aeda58ce6d2fb321ecbb398beaa9142f36f6d5386052be1bb3e162f43d04d80` (verified bitwise identical before and after).
3. **Setting A Preserved**: $448 \times 448$ tile, non-overlapping stride 448, reflect padding, exact cropping, $\tau=0.5$.
4. **Cohort Separation**: Evaluated on $N=164$ validation samples (82 Consensus Resistant, 19 Consensus Sensitive, 7 Cured by D2, 56 Clean Control). Sealed test set ($N=1124$) strictly unaccessed.

---

## 2. Mathematical Definition of Evaluated Spatial Statistics

At the entry to Decoder Block 1 ($T2$, spatial resolution $56 \times 56$), we evaluate candidate spatial statistics $M(x, y)$ computed purely from $B \in \mathbb{R}^{192 \times 56 \times 56}$ and $S \in \mathbb{R}^{96 \times 56 \times 56}$:

1. **Channel $L_2$ Energies**:
   - $E_B(x, y) = \|B(:, x, y)\|_2$
   - $E_S(x, y) = \|S(:, x, y)\|_2$
   - $\text{log\_ratio}(x, y) = \log\big((E_S + \epsilon) / (E_B + \epsilon)\big)$
2. **Normalized Energy & Contrast**:
   - $z_B, z_S$: within-sample z-score normalization of $E_B$ and $E_S$.
   - $\text{norm\_E\_B}, \text{norm\_E\_S}$: min-max normalization into $[0, 1]$.
   - $\text{norm\_ratio} = (\text{norm\_E\_S} + 0.05) / (\text{norm\_E\_B} + 0.05)$
3. **Cross-Stream Disagreement**:
   - $\text{disagree\_diff} = z_S - z_B$ (measures relative elevation of Skip over Bottleneck).
   - $\text{disagree\_abs} = |z_S - z_B|$
4. **Stream Agreement**:
   - $\text{cosine\_BS}(x, y)$: cosine similarity between 96-channel average-pooled $B$ and $S$.
5. **Local Spatial Gradients & Variance**:
   - $\text{grad\_E\_B}, \text{grad\_E\_S}$: Sobel gradient magnitude $\|\nabla E\|_2$.
   - $\text{var\_E\_B}, \text{var\_E\_S}$: local spatial variance in a $3 \times 3$ window.
6. **Composite Candidate Gating Signals**:
   - $\text{candidate\_gate\_B} = \text{norm\_E\_S} \cdot (1.0 - \text{norm\_E\_B})$ (Hypothesis: High Skip Energy + Low Bottleneck Energy indicates false bridge corridor).

---

## 3. Quantitative Discriminability Results

### Table 1: Regional Values (Median & Inter-Regional Delta) in Consensus Resistant ($N=82$)

| Spatial Statistic | NECK Median | CRACK Median | BG Median | NECK $-$ CRACK | NECK $-$ BG | Interpretation |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **$E_S$ (Skip Energy)** | **15.75** | **15.55** | **15.62** | **+0.20 (+1.3%)** | +0.13 | **Uniform across entire image; zero spatial selectivity** |
| **$E_B$ (Bottleneck Energy)** | **3.65** | **3.86** | **3.23** | **-0.20 (-5.3%)** | +0.42 | Only 5.3% contrast between neck and crack |
| **$\text{log\_ratio}$** | 1.48 | 1.40 | 1.57 | +0.08 | -0.09 | Slight elevation in neck, but BG is even higher |
| **$z_S$** | 0.37 | 0.28 | 0.16 | +0.09 | +0.21 | Skip is slightly above mean in both crack and neck |
| **$z_B$** | 0.26 | 0.65 | -0.62 | -0.38 | +0.88 | Strong separation of Foreground vs. BG, but NOT neck vs. crack |
| **$\text{disagree\_diff}$ ($z_S - z_B$)** | **+0.16** | **-0.18** | **+0.85** | **+0.34** | -0.69 | Neck has slightly higher Skip elevation, but BG has highest |
| **$\text{cosine\_BS}$** | -0.028 | -0.018 | -0.030 | -0.010 | +0.002 | Near-zero stream correlation everywhere |
| **$\text{grad\_E\_B}$** | 2.48 | 2.22 | 1.82 | +0.26 | +0.66 | Border effect; higher gradient at transitions |
| **$\text{candidate\_gate\_B}$** | **0.238** | **0.199** | **0.283** | **+0.039** | -0.045 | Weak elevation in neck (+0.039), but BG is higher (0.283) |

---

### Table 2: Discriminability Matrix (ROC-AUC & Cliff's Delta across Cohorts)

| Cohort | Statistic | Directional AUC (NECK > CRACK) | 2-Sided Discriminability (NECK vs CRACK) | Cliff's $\delta$ (NECK vs CRACK) | Directional AUC (NECK > BG) | 2-Sided Discriminability (NECK vs BG) | Cliff's $\delta$ (NECK vs BG) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Consensus Resistant** ($N=82$) | **$E_S$** | 0.5067 | 0.6606 | +0.2337 | 0.5000 | 0.6540 | +0.2007 |
| | **$E_B$** | 0.4079 | 0.6267 | -0.1554 | **0.7000** | **0.7170** | **+0.3551** |
| | **$\text{log\_ratio}$** | 0.5890 | 0.6206 | +0.2059 | 0.3173 | 0.7500 | -0.2852 |
| | **$\text{disagree\_diff}$** | **0.5902** | **0.6633** | **+0.0902** | 0.3794 | 0.6948 | -0.1877 |
| | **$\text{cosine\_BS}$** | 0.4561 | 0.6701 | -0.0754 | 0.5191 | 0.6250 | +0.0252 |
| | **$\text{candidate\_gate\_B}$**| 0.5766 | 0.6443 | +0.0588 | 0.3539 | 0.6948 | -0.2397 |
| **Consensus Sensitive** ($N=19$) | $E_S$ | 0.3327 | 0.6774 | -0.0999 | 0.4000 | 0.6724 | -0.0101 |
| | $E_B$ | 0.2857 | 0.7425 | -0.2232 | **0.8611** | **0.8611** | **+0.3336** |
| | $\text{log\_ratio}$ | 0.5556 | 0.7703 | +0.2072 | 0.1389 | 0.8611 | -0.3365 |
| | $\text{disagree\_diff}$ | 0.5000 | 0.7284 | -0.0054 | 0.2274 | 0.7726 | -0.3861 |
| | $\text{candidate\_gate\_B}$| 0.5556 | 0.7193 | +0.0500 | 0.2362 | 0.7638 | -0.3888 |
| **Cured by D2** ($N=7$) | $E_S$ | **0.2673** | **0.7327** | **-0.4897** | 0.1375 | 0.8625 | -0.5417 |
| | $E_B$ | 0.3019 | 0.7823 | -0.2740 | **1.0000** | **1.0000** | **+0.3565** |
| | $\text{log\_ratio}$ | 0.5532 | 0.7231 | +0.1210 | 0.0500 | 0.9583 | -0.4630 |
| | $\text{disagree\_diff}$ | 0.3820 | 0.7823 | -0.2055 | 0.1000 | 0.9000 | -0.5370 |
| | $\text{candidate\_gate\_B}$| 0.4868 | 0.7651 | -0.0183 | 0.0500 | 0.9500 | -0.4630 |

---

## 4. Answers to the 5 Core Scientific Questions

### Q1. Does any statistic distinguish NECK from CRACK?
**NO. NO STATISTIC PROVIDES ACTIONABLE SEPARATION BETWEEN NECK AND CRACK.**
- In Consensus Resistant ($N=82$), the directional ROC-AUC for all candidate statistics falls within a narrow, uninformative band of **$0.41 - 0.59$** (equivalent to random guessing around $0.50$).
- $E_S$ has identical median energy across NECK ($15.75$) and CRACK ($15.55$) ($\text{AUC} = 0.5067$).
- $E_B$ drops by merely $5.3\%$ in the neck ($3.65$ vs $3.86$), yielding a weak directional $\text{AUC} = 0.4079$.
- Cross-stream disagreement ($z_S - z_B$) achieves the highest directional AUC, but peaks at only **$0.5902$**, with massive distributional overlap ($>80\%$).

### Q2. Does any statistic distinguish NECK from BACKGROUND?
**YES, BUT IT ONLY DETECTS GENERAL FOREGROUND PRESENCE (NOT BRIDGE SPECIFICITY).**
- Bottleneck Energy $E_B$ reliably separates NECK from BACKGROUND ($\text{AUC} = 0.7000$ in Resistant, $0.8611$ in Sensitive, $1.0000$ in D2-Cured).
- However, $E_B$ is equally high (or higher) on true cracks ($3.86$). Thus, $E_B$ functions as a general crack detector, incapable of discerning where the true crack terminates and the false bridge begins.
- Candidate Gate B ($\text{norm\_E\_S} \cdot (1 - \text{norm\_E\_B})$) is actually **higher in ordinary background ($0.283$) than in the neck ($0.238$)**, making background rejection problematic without an external foreground mask.

### Q3. How do Resistant vs. Sensitive differ in spatial statistics?
- **Consensus Resistant**:
  - Exhibits an elevated, ambiguous Bottleneck energy in the neck corridor ($E_B = 3.65$), closely tracking the true crack ($3.86$).
  - As a result, cross-stream disagreement is slightly positive ($z_S - z_B = +0.16$), indicating that the neck features closely mimic the crack features.
- **Consensus Sensitive**:
  - $E_B$ in the neck experiences a sharper drop ($3.26$ vs $3.43$).
  - Cross-stream disagreement is negative ($z_S - z_B = -0.014$).
  - In Sensitive cases, the gap corridor is representationally closer to background than in Resistant cases.

### Q4. Does D2-Cured resemble Sensitive?
**NO — D2-CURED EXHIBITS A UNIQUE, INVERTED SIGNATURE.**
- In Cured by D2 ($N=7$), Skip Energy in the neck is **sharply depressed** compared to both crack and background ($E_S = 15.39$ in neck vs. $16.21$ in crack and $16.40$ in background).
- Directional $\text{AUC}(E_S)$ drops to **$0.2673$** (Cliff's $\delta = -0.4897$), completely inverting the Resistant pattern.
- In D2-Cured, the neck corridor already possesses a distinct topological energy valley at $T2$, explaining why mild downstream regularization easily dissolves D2 bridges.

### Q5. Is there a scalar spatial score good enough to be a candidate gate signal?
**NO. NO POINTWISE SCALAR COMBINATION OF B AND S QUALIFIES AS A STANDALONE GATE SIGNAL.**
- A functional selective gate would require an AUC $\ge 0.85$ separating NECK from CRACK to avoid suppressing true crack features while removing the bridge.
- The highest achieved directional AUC in Resistant is **$0.5902$** (`disagree_diff`).
- Applying a threshold on any of these pointwise statistics would indiscriminately suppress true cracks, causing severe crack breakage and catastrophic Recall collapse.

---

## 5. Decision Gate Verdict

### Formal Classification: NOT SUPPORTED (for Pointwise Spatial Skip Gating)

1. **Failure of the Pointwise Separability Assumption**:
   - The hypothesis that "a simple algebraic function of $B(x, y)$ and $S(x, y)$ at $T2$ can cleanly mask false bridge corridors" is **refuted by the data**.
   - Raw $L_2$ energy $E_S$ is globally flat across the image ($\sim 15.6$), carrying zero localized boundary contrast.
   - Raw $E_B$ in the neck corridor of Resistant cases is only $5\%$ lower than on the true crack.
2. **Pointwise Gate Formulation is a Dead End**:
   - Designing an algebraic or $1 \times 1$ pointwise gating mechanism at $T2$ without topological context will fail the trade-off test, behaving identically to the channel attenuation attempts by destroying true cracks.
3. **Plausibility Boundary for Future Exploration**:
   - While *pointwise* statistics fail ($\text{AUC} \approx 0.59$), *spatial context / topological operators* (e.g. non-local attention, geodesic distance, or boundary-conditioned routing) remain mathematically open because they do not rely on local pointwise energy contrast.

---

## 6. What a Viable Selective Skip Gate Would Need to Detect

Based on the empirical measurements, a successful gating mechanism cannot rely on local feature magnitude ($E_S, E_B$). To be viable, a mechanism must detect:

1. **Non-Local Topological Connectivity**:
   - Pointwise $E_B$ at the neck ($3.65$) is indistinguishable from crack ($3.86$).
   - A viable detector must compute whether the point $(x, y)$ lies on a path that connects two disjoint ground-truth components or represents a continuous single component. This requires non-local receptive fields or graph/topological representations.
2. **Phase / Structural Alignment Rather Than Norm**:
   - $\text{cosine\_BS}$ is near zero everywhere ($-0.028$). The channel spaces of $B$ and $S$ are largely orthogonal.
   - Gating cannot rely on scalar product alignment between raw channels; it requires learned semantic alignment.
3. **Boundary Enclosure Constraints**:
   - Rather than gating the internal neck corridor (which shares crack-like energy), the intervention should enforce **boundary margin penalties** at the crack tips to prevent features from propagating into the corridor in the first place.

---

## 7. System Integrity Report

- **Checkpoint SHA256**:
  `147f784021414efd229f37faee27618997a0665dfb12c8b8745585d8885f812b` (bitwise invariant before and after).
- **Named Parameters SHA256**:
  `4aeda58ce6d2fb321ecbb398beaa9142f36f6d5386052be1bb3e162f43d04d80` (bitwise invariant before and after).
- **Baseline Bridge Invariance**: $82/82$ in Resistant, $19/19$ in Sensitive, $7/7$ in D2-Cured, $0/56$ in Clean Control.
- **Sealed Test Set**: Strictly untouched ($N=1124$).
- **Working Tree**: Completely clean.
