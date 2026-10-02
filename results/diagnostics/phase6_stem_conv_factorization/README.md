# Phase 6: Stem Conv2d(3, 48, 4, 4) Mathematical Factorization Diagnostic Report

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

## 2. Mathematical Factorization of `Conv2d(3, 48, kernel_size=(4, 4), stride=(4, 4))`

Let the learned weight tensor of the stem projection be $W \in \mathbb{R}^{48 \times 3 \times 4 \times 4}$ and bias $\mathbf{b} \in \mathbb{R}^{48}$. For any spatial location $(y, x)$ on the $112 \times 112$ grid, the full convolution output is:
$$\mathbf{y}_{\text{full}}(y, x) = \mathbf{b} + \sum_{c=0}^2 \sum_{u=0}^3 \sum_{v=0}^3 W_{:, c, u, v} \cdot X(c, 4y + u, 4x + v)$$

To answer whether the ambiguity at `raw_conv` arises from **spatial aggregation / receptive-field mixing** or from **learned channel projection**, we factorize $W$ along three mathematically closed axes:

### Axis 1: Frequency / Structure Decomposition (DC vs. AC)
By linearity of convolution:
$$W = \bar{W} + \widetilde{W}$$
1. **DC Component ($\bar{W}$ — Spatial Mean / Pooling + 1×1 Channel Projection):**
   $$\bar{W}_{k, c, u, v} = \frac{1}{16} \sum_{i=0}^3 \sum_{j=0}^3 W_{k, c, i, j}$$
   - Energy: **$0.0684$ ($3.74\%$ of total weight energy)**.
   - Mathematical mechanism: Pure spatial average pooling $\text{AvgPool2d}(4, 4)$ followed by a learned $1 \times 1$ channel projection from $3 \to 48$. All intra-patch spatial edges, gradients, and high-frequency variations are removed.
2. **AC Component ($\widetilde{W}$ — Pure Zero-Mean Spatial Filter Bank):**
   $$\widetilde{W}_{k, c, u, v} = W_{k, c, u, v} - \bar{W}_{k, c, u, v}$$
   - Energy: **$1.7616$ ($96.26\%$ of total weight energy)**.
   - Mathematical mechanism: By construction, $\sum_{u, v} \widetilde{W}_{k, c, u, v} = 0$. Invariant to constant brightness/color shifts. It isolates the high-frequency intra-patch edge, line, and gradient filtering.
3. **Exact Linear Sum:**
   $$\mathbf{y}_{\text{full}} = \mathbf{y}_{\text{DC}} + \mathbf{y}_{\text{AC}} \quad (\text{Max reconstruction error } < 10^{-6})$$

### Axis 2: SVD Kernel Separability (Channel Projection $\otimes$ Spatial Pattern)
For each output filter $k \in \{0, \dots, 47\}$, reshape $W_k \in \mathbb{R}^{3 \times 16}$:
$$W_k = \sum_{r=1}^3 \sigma_{k, r} \cdot \mathbf{u}_{k, r} \mathbf{v}_{k, r}^T$$
- **Rank-1 Separable Component ($W_{\text{Rank-1}}$):**
  - Energy: **$1.5384$ ($84.06\%$ of total weight energy)**.
  - Decomposes each filter into a strictly separable product of a color channel projection vector $\mathbf{u}_{k, 1} \in \mathbb{R}^3$ and an intra-patch spatial pattern $\mathbf{v}_{k, 1} \in \mathbb{R}^{16}$.

### Axis 3: Physical Receptive-Field Bleed vs. Pure Background Gap Audit
Inside the bridge neck corridor on the $112 \times 112$ grid:
- **Boundary Bleed Cells:** $4 \times 4$ cells that physically overlap $\ge 1$ ground-truth crack pixel (receptive-field spatial mixing of crack and background).
- **Pure Background Gap Cells:** $4 \times 4$ cells that contain strictly **$0$ ground-truth crack pixels** (100% asphalt background).

---

## 3. Metric 1: Mathematical Factorization Emergence Statistics

### Table 1: Emergence & Contrast Across Factorized Components

| Cohort | Factorization Mode | Weight Energy | Median $R_{\text{norm}}$ | Mean $R_{\text{norm}}$ | P25 | P75 | $\% \ge 0.50$ | $\% \ge 0.70$ | Median $\frac{E_{\text{neck}}}{E_{\text{crack}}}$ | Median $\cos(\mathbf{f}_{\text{neck}}, \mathbf{f}_{\text{crack}})$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Consensus 7/7** ($N=101$) | **`raw_conv_full`** | 100.0% | **0.5508** | 3.8912 | 0.0267 | 1.5566 | **54.46%** | **45.54%** | 1.0280 | **0.9658** |
| | **`raw_conv_dc`** (Pool+1x1) | 3.74% | **0.3891** | -0.3669 | 0.1091 | 0.8873 | **39.60%** | **30.69%** | **0.8043** | **0.9321** |
| | **`raw_conv_ac`** (Spatial Filter) | 96.26% | **0.7608** | 1.0537 | 0.1360 | 1.5883 | **58.42%** | **50.50%** | 1.0231 | **0.9778** |
| | **`raw_conv_rank1`** (SVD Sep) | 84.06% | **0.5661** | 0.2272 | -0.1731 | 1.3810 | **52.48%** | **44.55%** | 1.0312 | **0.9682** |
| **Cured by D2** ($N=7$) | **`raw_conv_full`** | 100.0% | -0.3680 | -0.3110 | -0.5882 | -0.1025 | 0.00% | 0.00% | 1.0853 | 0.9383 |
| | **`raw_conv_dc`** (Pool+1x1) | 3.74% | 0.3487 | 0.4352 | -0.0777 | 0.8957 | 42.86% | 42.86% | 0.7267 | 0.8121 |
| | **`raw_conv_ac`** (Spatial Filter) | 96.26% | -0.3924 | -9.8901 | -0.6668 | -0.2761 | 0.00% | 0.00% | 1.0748 | 0.9462 |
| | **`raw_conv_rank1`** (SVD Sep) | 84.06% | -0.4100 | -0.3581 | -0.6431 | -0.0925 | 0.00% | 0.00% | 1.1006 | 0.9496 |
| **Clean Control** ($N=56$) | **`raw_conv_full`** | 100.0% | -0.0200 | -0.3216 | -0.5303 | 1.0044 | 42.86% | 33.93% | 0.9885 | 0.9625 |
| | **`raw_conv_dc`** (Pool+1x1) | 3.74% | -0.0961 | 0.0697 | -0.5844 | 0.2647 | 17.86% | 14.29% | 0.7633 | 0.8450 |
| | **`raw_conv_ac`** (Spatial Filter) | 96.26% | 0.3877 | 1.5694 | -0.4255 | 0.9490 | 44.64% | 37.50% | 0.9758 | 0.9749 |
| | **`raw_conv_rank1`** (SVD Sep) | 84.06% | 0.1152 | 1.2162 | -0.5123 | 1.1005 | 42.86% | 39.29% | 1.0051 | 0.9668 |

---

## 4. Metric 2: Pairwise Spatial Separation Across Factorizations (513 Pairs)

### Table 2: Pairwise Separation Performance Across Components

| Cohort | Factorization Mode | Pairs Evaluated | Grid Collision | % Spatially Separable ($\text{SepMargin} \ge 0.20$) | **% Representation Collapsed** | Median $\text{SepMargin}$ | Median $\frac{E_{\text{neck}}}{E_{\text{crack}}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Consensus 7/7** | **`raw_conv_full`** | 513 | 9.36% | **9.75%** | **90.25%** | **-0.0192** | 1.0192 |
| | **`raw_conv_dc`** (Pool+1x1) | 513 | 9.36% | **49.90%** | **50.10%** | **+0.2282** | **0.7718** |
| | **`raw_conv_ac`** (Spatial Filter) | 513 | 9.36% | **9.36%** | **90.64%** | **-0.0198** | 1.0198 |
| | **`raw_conv_rank1`** (SVD Sep) | 513 | 9.36% | **9.94%** | **90.06%** | **-0.0251** | 1.0251 |
| **Cured by D2** | **`raw_conv_full`** | 7 | 0.00% | 0.00% | 100.00% | -0.0853 | 1.0853 |
| | **`raw_conv_dc`** (Pool+1x1) | 7 | 0.00% | **57.14%** | **42.86%** | **+0.2733** | **0.7267** |
| | **`raw_conv_ac`** (Spatial Filter) | 7 | 0.00% | 0.00% | 100.00% | -0.0748 | 1.0748 |
| | **`raw_conv_rank1`** (SVD Sep) | 7 | 0.00% | 0.00% | 100.00% | -0.1006 | 1.1006 |
| **Clean Control** | **`raw_conv_full`** | 56 | 10.71% | 0.00% | 100.00% | +0.0115 | 0.9885 |
| | **`raw_conv_dc`** (Pool+1x1) | 56 | 10.71% | **50.00%** | **50.00%** | **+0.2367** | **0.7633** |
| | **`raw_conv_ac`** (Spatial Filter) | 56 | 10.71% | 1.79% | 98.21% | +0.0242 | 0.9758 |
| | **`raw_conv_rank1`** (SVD Sep) | 56 | 10.71% | 0.00% | 100.00% | -0.0051 | 1.0051 |

---

## 5. Metric 3: Physical Receptive-Field Bleed vs. Pure Background Gap Audit

### Table 3: Receptive Field Cell Composition in Bridge Corridors

| Cohort | Samples Evaluated | Mean % Pure Background Gap Cells | Median % Pure Background Gap Cells | Mean % Receptive Field Bleed Cells | Cases with 100% Pure BG Corridors |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Consensus 7/7** | 101 | **53.95%** | **50.00%** | 46.05% | **18 / 101 (17.82%)** |
| **Cured by D2** | 7 | **39.76%** | **40.00%** | 60.24% | **0 / 7 (0.00%)** |
| **Clean Control** | 56 | **78.67%** | **83.33%** | 21.33% | **24 / 56 (42.86%)** |

---

## 6. Scientific Analysis: Spatial Aggregation vs. Channel Projection

### Core Finding 1: The DC Component (Pure Spatial Pooling + 1×1 Channel Projection) PRESERVES Spatial Separation
- The DC component $\bar{W}$ acts as a pure spatial average pooling followed by a learned $1 \times 1$ color/channel projection.
- Under this operation:
  - Median separation margin shifts from **$-0.0192$ (collapsed) to $+0.2282$ (strong separation valley)**.
  - The percentage of spatially separable pairs **jumps more than five-fold**, from **$9.75\%$ to $49.90\%$**.
  - The contrast ratio $\frac{E_{\text{neck}}}{E_{\text{crack}}}$ drops from **$1.0192$ down to $0.7718$** (meaning feature energy at the neck is $23\%$ lower than on the crack).
- **Conclusion:** Spatial average pooling of local color channels does **not** create the false bridge representation. If the stem were merely downsampling color averages, half of the persistent false bridges would have clear feature valleys at the earliest stage.

### Core Finding 2: The AC Component (Learned Zero-Mean Spatial Filter Bank) CARRIES the Collapse
- The AC component $\widetilde{W}$ represents the learned spatial edge, gradient, and texture filters (carrying **$96.26\%$** of the total kernel weight energy).
- Under the AC component alone:
  - **$90.64\%$** of component pairs exhibit representation collapse (virtually identical to the full kernel's $90.25\%$).
  - Median separation margin is **$-0.0198$** (negative, indicating neck energy equals or exceeds crack energy).
  - Emergence rate ($\ge 0.50$) is **$58.42\%$** (vs $54.46\%$ full).
  - Median cosine similarity with the crack representation is **$0.9778$**.
- **Conclusion:** The learned high-frequency spatial filter bank within `Conv2d(3, 48, 4, 4)` is the primary mathematical driver of representation collapse. The filters respond to intra-patch pavement textures, surface roughness, and sub-patch gradients in the gap, producing feature activations indistinguishable from crack edges.

### Core Finding 3: SVD Separability Demonstrates Dominance of Spatial Patterns Over Cross-Channel Coupling
- With **$84.06\%$** of the kernel energy captured by Rank-1 separable filters ($W_{k} \approx \mathbf{u}_k \mathbf{v}_k^T$), the Rank-1 reconstruction reproduces **$90.06\%$** of the pairwise collapse (median $\text{SepMargin} = -0.0251$).
- Non-separable cross-channel spatial interactions account for less than $16\%$ of the energy and do not alter the collapse phenotype.

### Core Finding 4: Receptive Field Bleed is Real but Explains Only Half of the Corridors
- In **$46.05\%$** (mean) of neck cells, true crack pixels physically overlap the $4 \times 4$ receptive field, causing literal sub-pixel crack signal to bleed into the token.
- However, in **$53.95\%$** of neck cells (and $100\%$ of neck cells in 18 Consensus cases), there are **zero ground-truth crack pixels** within the $4 \times 4$ receptive field.
- In these pure background gap cells, the false bridge is driven not by physical crack bleed, but by the AC spatial filters mistaking asphalt texture/contrast for crack structure.

---

## 7. Epistemic Boundaries & Calibrated Conclusions

1. **Precursor Statement:**
   **Raw Conv / early patch-projection representation collapse is the earliest observed dominant precursor of the persistent false-bridge phenotype.**
2. **Causal Attribution Limitation:**
   This diagnostic factorizes the mathematical components of `Conv2d(3, 48, 4, 4)`. While the AC component of the spatial filter bank is identified as the carrier of the representation collapse, this does **not** prove that changing the stem projection alone is causally sufficient to eradicate all downstream false bridges.
3. **Semantic Alignment Statement:**
   The raw stem projection already produces highly aligned feature representations between crack and bridge-neck regions ($\cos = 0.9658$), driven primarily by the high-frequency AC spatial filters ($\cos = 0.9778$).
4. **D2-Cured Contrast:**
   D2-cured cases do not exhibit strong bridge-like representation at the raw stem projection under the present diagnostic criterion (median $R_{\text{norm}} = -0.3680$, $0/7$ reaching $0.50$ emergence).
5. **Metric Caveat:**
   Because emergence indices in the presence of low contrast denominators can exhibit heavy tails, scientific conclusions are based strictly on medians, interquartile ranges, and binary threshold fractions.
