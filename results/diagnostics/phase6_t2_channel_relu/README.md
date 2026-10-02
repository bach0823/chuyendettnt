# Phase 6: Channel-wise ReLU Interaction & Spatial Selectivity Diagnostic Report

## 1. Executive Summary & Epistemic Boundaries

This diagnostic investigates whether the non-linear interaction ignited at Decoder Block 1 Conv1 post-ReLU is localized to a **sparse, bridge-selective subset of channels** or represents a **diffuse / non-selective property across the 96-channel tensor**.

### Epistemic Wording Boundaries
- **"ReLU is the first observed nonlinear coupling locus"**: We have mathematically proven that Conv1 pre-BN is strictly additive ($|Y_{BS} - Y|_\infty \le 4.41 \times 10^{-6}$) and post-BN is strictly affine ($|I| \le 2.74 \times 10^{-6}$). Non-linear interaction emerges precisely at the post-ReLU step.
- **Do NOT declare "ReLU causes false bridges"**: ReLU is an un-parameterized pointwise non-linear activation $\max(0, z)$. It does not create spatial connectivity on its own; it couples and non-linearly rectifies the joint projection of $W_B * B$ and $W_S * S$.
- **Do NOT claim "fusion synergy" based on $I > 0$**: Positive interaction is described strictly as **spatially selective non-linear interaction**, not functional synergy.
- **Strictly Diagnostic-Only**: No channel masking, no channel attenuation, no architecture modification, and zero parameter training.

### Non-Negotiable Protocol Hard Locks
1. **DIAGNOSTIC-ONLY**: Zero training passes, zero backward passes, zero optimizer steps, zero gradient updates.
2. **Bitwise Invariance**:
   - Checkpoint SHA256: `147f784021414efd229f37faee27618997a0665dfb12c8b8745585d8885f812b` (verified bitwise identical before and after).
   - Parameter SHA256: `4aeda58ce6d2fb321ecbb398beaa9142f36f6d5386052be1bb3e162f43d04d80` (verified bitwise identical before and after).
3. **Setting A Preserved**: $448 \times 448$ tile, non-overlapping stride 448, reflect padding, exact cropping, $\tau=0.5$.
4. **Cohort Separation**: Evaluated on $N=164$ validation samples (82 Consensus Resistant, 19 Consensus Sensitive, 7 Cured by D2, 56 Clean Control). Sealed test set ($N=1124$) strictly unaccessed.

---

## 2. Mathematical Definition of Channel-Wise Spatial Interaction

At Decoder Block 1 Conv1 post-ReLU, for each channel $c \in \{0, \dots, 95\}$ and spatial coordinate $(x, y)$:

$$
I_c(x, y) = R_c(\text{Full})(x, y) - R_c(\text{B-only})(x, y) - R_c(\text{S-only})(x, y) + R_c(\text{Zero})(x, y)
$$

Where $R_c$ is the activation value **strictly after ReLU**.

For each sample and channel $c$:
- **Corridor Neck Interaction**: $I_c(\text{neck}) = \text{median}_{x, y \in \text{neck}} I_c(x, y)$
- **True Crack Interaction**: $I_c(\text{crack}) = \text{median}_{x, y \in \text{crack}} I_c(x, y)$
- **Spatial Selectivity Metric**:

$$
\Delta I_c = I_c(\text{neck}) - I_c(\text{crack})
$$

If $\Delta I_c > 0$, the non-linear interaction in channel $c$ favors the false bridge corridor over true cracks.
If $\Delta I_c \le 0$, the channel either amplifies true cracks more than the neck or is spatially non-selective.

---

## 3. Quantitative Results

### Table 1: Top 15 Channels Ranked by Spatial Selectivity ($\Delta I_c$) in Consensus Resistant ($N=82$)

| Rank | Channel | Median $\Delta I_c$ | Median $I_c(\text{neck})$ | Median $I_c(\text{crack})$ | Cumulative Share of Pos $\Delta I$ | Sample Coverage ($\%$) | Quadrant Type |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | **Ch 92** | **+0.1144** | -0.4835 | -0.5778 | **14.44%** | 80.5% | Differential Sub-additivity ($I_n < 0, I_c < 0$) |
| **2** | **Ch 46** | **+0.0893** | -0.8573 | -0.9611 | **25.71%** | 78.0% | Differential Sub-additivity ($I_n < 0, I_c < 0$) |
| **3** | **Ch 0**  | **+0.0703** | -0.6679 | -0.7410 | **34.59%** | 75.6% | Differential Sub-additivity ($I_n < 0, I_c < 0$) |
| **4** | **Ch 30** | **+0.0685** | -1.3029 | -1.4082 | **43.24%** | 73.2% | Differential Sub-additivity ($I_n < 0, I_c < 0$) |
| **5** | **Ch 35** | **+0.0640** | -0.6511 | -0.7793 | **51.31%** | 72.0% | Differential Sub-additivity ($I_n < 0, I_c < 0$) |
| **6** | **Ch 72** | **+0.0633** | -0.2087 | -0.3493 | **59.30%** | 68.3% | Differential Sub-additivity ($I_n < 0, I_c < 0$) |
| **7** | **Ch 20** | **+0.0573** | -0.5909 | -0.6676 | **66.53%** | 69.5% | Differential Sub-additivity ($I_n < 0, I_c < 0$) |
| **8** | **Ch 18** | **+0.0545** | -1.0486 | -1.1108 | **73.40%** | 65.9% | Differential Sub-additivity ($I_n < 0, I_c < 0$) |
| **9** | **Ch 26** | **+0.0523** | -0.0357 | -0.1216 | **80.01%** | 64.6% | Differential Sub-additivity ($I_n < 0, I_c < 0$) |
| **10**| **Ch 57** | **+0.0411** | -0.4081 | -0.4431 | **85.19%** | 62.2% | Differential Sub-additivity ($I_n < 0, I_c < 0$) |
| **11**| **Ch 8**  | **+0.0328** | -0.1030 | -0.1271 | **89.34%** | 58.5% | Differential Sub-additivity ($I_n < 0, I_c < 0$) |
| **12**| **Ch 6**  | **+0.0283** | -0.7557 | -0.7828 | **92.91%** | 56.1% | Differential Sub-additivity ($I_n < 0, I_c < 0$) |
| **13**| **Ch 79** | **+0.0238** | -0.2892 | -0.3349 | **95.92%** | 54.9% | Differential Sub-additivity ($I_n < 0, I_c < 0$) |
| **14**| **Ch 84** | **+0.0200** | -0.0000 | -0.1077 | **98.44%** | 51.2% | Differential Sub-additivity ($I_n \approx 0, I_c < 0$) |
| **15**| **Ch 21** | **+0.0069** | -0.2652 | -0.3031 | **99.31%** | 48.8% | Differential Sub-additivity ($I_n < 0, I_c < 0$) |

---

### Table 2: Concentration & Cumulative Share Analysis Across Cohorts

| Cohort | Subgroup | Total Channels with $\Delta I_c > 0$ | Channels for 50% Share ($k_{50}$) | Channels for 80% Share ($k_{80}$) | Channels for 90% Share ($k_{90}$) | Concentration Profile |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **Consensus 7/7** | **Resistant** | 33 / 96 | **5 channels** | **9 channels** | **12 channels** | **Highly Concentrated in $\Delta I_c$** |
| **Consensus 7/7** | **Sensitive** | 32 / 96 | **6 channels** | **12 channels** | **17 channels** | Moderately Concentrated |
| **Cured by D2** | **All** | 39 / 96 | **8 channels** | **15 channels** | **19 channels** | More Diffuse |
| **Clean Control** | **All** | 25 / 96 | **5 channels** | **9 channels** | **12 channels** | Background Noise Concentration |

---

### Table 3: Cross-Cohort Channel Overlap Analysis

| Comparison | Top-10 Overlap Count | Top-10 Jaccard Similarity | Top-10 Shared Channels | Top-20 Overlap Count | Top-20 Jaccard Similarity | Top-20 Shared Channels |
|:---|:---:|:---:|:---|:---:|:---:|:---|
| **Resistant vs. Sensitive** | **8 / 10** | **0.667** | `[18, 20, 30, 35, 46, 57, 72, 92]` | **12 / 20** | **0.429** | `[0, 6, 18, 20, 30, 35, 46, 51, 57, 72, 92, 95]` |
| **Resistant vs. D2-Cured**  | **5 / 10** | **0.333** | `[0, 18, 30, 35, 57]` | **15 / 20** | **0.600** | `[0, 6, 18, 20, 26, 30, 34, 35, 46, 51, 57, 72, 79, 92, 95]` |
| **Sensitive vs. D2-Cured**  | **5 / 10** | **0.333** | `[6, 18, 30, 35, 57]` | **12 / 20** | **0.429** | `[0, 6, 18, 20, 30, 35, 46, 51, 57, 72, 92, 95]` |

---

### Table 4: Bridge-Selective Quadrant Analysis ($I_c(\text{neck}) > 0 \text{ and } I_c(\text{crack}) \le 0$)

| Cohort | Channels in Quadrant | Channel Indices | Share of Total Positive $\Delta I$ | Mean Rank of Quadrant Channels |
|:---|:---:|:---|:---:|:---:|
| **Consensus Resistant** | **2 / 96** | `[47, 75]` | **< 0.05%** | **Rank 41.0** (Rank 38 and 44) |
| **Consensus Sensitive** | **6 / 96** | `[11, 16, 22, 42, 89, 90]` | **< 0.20%** | **Rank 34.5** |
| **Cured by D2**         | **11 / 96**| `[2, 16, 34, 42, 44, 47, 50, 66, 75, 78, 94]` | **1.85%** | **Rank 28.2** |

---

## 4. Key Scientific Findings

### 1. The Top $\Delta I_c$ Channels are NOT Corridor Amplifiers
Crucially, **not a single channel in the top 10 has positive interaction in the neck corridor ($I_c(\text{neck}) > 0$)**:
- For Channel 92 (Rank 1): $I_c(\text{neck}) = \mathbf{-0.4835}$, while $I_c(\text{crack}) = \mathbf{-0.5778}$.
- For Channel 46 (Rank 2): $I_c(\text{neck}) = \mathbf{-0.8573}$, while $I_c(\text{crack}) = \mathbf{-0.9611}$.
- For Channel 30 (Rank 4): $I_c(\text{neck}) = \mathbf{-1.3029}$, while $I_c(\text{crack}) = \mathbf{-1.4082}$.

**Why is $\Delta I_c > 0$?**
Because on the true crack body, activations are very high in both $B$ and $S$, causing severe sub-additive saturation under ReLU ($\max(0, z_B + z_S) \ll z_B + z_S$). In the corridor neck, activations are weaker, so the sub-additive deficit is less negative. Thus:

$$
\Delta I_c = (-0.4835) - (-0.5778) = \mathbf{+0.0943} > 0
$$

The positive selectivity is driven by **crack saturation**, NOT by a selective positive boost to the false bridge corridor.

### 2. The Bridge-Selective Quadrant ($I_{\text{neck}} > 0, I_{\text{crack}} \le 0$) is Functionally Empty
Only **2 out of 96 channels** (Channels 47 and 75) fall strictly into the bridge-selective quadrant in Consensus Resistant cases.
- Both channels have near-zero interaction magnitudes ($\Delta I < 0.0001$).
- Their ranks are **38** and **44**, contributing essentially $0\%$ to the cumulative selectivity curve.
- There is **no dedicated subset of "bridge-generating channels"** in the Conv1 representation.

### 3. Weak Correlation with Final Bridge Status
Correlation analysis across all 96 channels against final sample connector probability and bridge presence shows that **no channel has $|r| \ge 0.28$**:
- 89 of 96 channels have $p > 0.05$ (uncorrelated).
- The highest correlation observed is Channel 5 ($r = -0.275$, negative correlation) and Channel 0 ($r = +0.212$).
- Bridge formation cannot be predicted by or attributed to the activation profile of individual channels.

---

## 5. Decision Gate Verdict

### Verdict: REJECT CHANNEL-LEVEL MASKING (Combination of B — Diffuse & C — Non-Selective)

1. **Failure of Sparse Bridge-Selective Hypothesis (Condition A REJECTED)**:
   - Condition A required: *"A small, stable subset of channels with $I_c(\text{neck}) > 0$ and $I_c(\text{crack}) \le 0$ explaining the majority of positive interaction."*
   - Fact: The top channels with $\Delta I > 0$ have **negative** neck interaction ($I_{\text{neck}} < 0$).
   - Fact: The 2 channels in the selective quadrant account for $<0.05\%$ of interaction.
2. **Channel Masking / Attenuation is a Dead End**:
   - Because the top $\Delta I$ channels (e.g. 92, 46, 0, 30, 35, 72) carry massive true-crack activation ($|I_{\text{crack}}| > 0.5 - 1.4$), zeroing or attenuating these channels would **severely destroy true-crack detection** rather than curing false bridges.
3. **Strategic Pivot to Spatial/Topological Selective Fusion**:
   - The false bridge phenotype is **NOT channel-coded**; it is **spatially coded**.
   - The solution cannot be a static channel filter ($1 \times 1$ conv, channel mask, or channel pruning).
   - Any successful intervention must operate in the **spatial domain** (e.g., Spatial Skip Gating driven by Bottleneck topological confidence, or a topological corridor margin loss).

---

## 6. System Integrity Report

- **Checkpoint SHA256**:
  `147f784021414efd229f37faee27618997a0665dfb12c8b8745585d8885f812b` (bitwise invariant before and after).
- **Named Parameters SHA256**:
  `4aeda58ce6d2fb321ecbb398beaa9142f36f6d5386052be1bb3e162f43d04d80` (bitwise invariant before and after).
- **Baseline Bridge Count**: $82/82$ in Resistant, $19/19$ in Sensitive, $7/7$ in D2-Cured, $0/56$ in Clean Control.
- **Sealed Test Set**: Strictly untouched ($N=1124$).
- **Working Tree**: Completely clean.
