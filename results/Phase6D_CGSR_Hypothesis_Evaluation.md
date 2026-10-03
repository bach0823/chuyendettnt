# Phase 6D: Context-Guided Stage-1 Skip Refinement (CGSR) — Final Evaluation & Scientific Verdict

**Date:** 2026-10-03  
**Artifact Evaluated:** `Phase6D_CGSR_Full.zip` (SHA256 Checkpoint: `532f731823039d2e8d8e66efe3ae37e8121a831f00191fb0018c0c8554c3937b`)  
**Protocols Followed:** Section 12 Scientific Decision Rule, Setting A Evaluation (stride 448, non-overlap, threshold 0.5), Locked Lineage Candidate B.

---

## Executive Summary & Final Scientific Verdict

### Verdict: `H2 NOT SUPPORTED`

> **Hypothesis 2 (H2)** stated:
> *"Stage-1 skip representation $(B, 96, 56, 56)$ of Candidate B contains non-discriminative features driving false bridges; deep semantic context $(B, 192, 56, 56)$ can selectively suppress this harmful skip content while preserving true thin cracks."*
>
> **Empirical Finding:**
> 1. The learned gate $G \in [0, 1]$ produced a spatial contrast between true cracks and false bridges of **virtually ZERO**:
>    $$\text{median}(G_{\text{crack}} - G_{\text{bridge}}) = \mathbf{+0.0003}, \quad \text{mean} = \mathbf{+0.0006}$$
>    The gate activation on false bridges ($0.9556$) and true cracks ($0.9562$) is indistinguishable.
> 2. False bridge cure rate in the wider-gap cohort ($D_{\text{gap}} > 5\text{ px}$) was only **$4/43$ ($9.3\%$)**, with a median logit change of **$\Delta z_{\text{bridge}} = -0.1386$** (false bridge logits actually strengthened).
> 3. Setting A Val Dice on full Crack500 ($N=348$) dropped from **$0.7641$** (Control) to **$0.7589$** (CGSR), a degradation of **$-0.0052$** (violating the $\ge -0.005$ preservation threshold).
> 4. Collateral damage occurred on clean true cracks: **$12/118$ ($10.2\%$)** of previously continuous crack segments suffered severe breakage.
>
> **Decision:** In accordance with the Phase 6D Hard Decision protocol, **H2 is officially NOT SUPPORTED**. Phase 6D is permanently closed without post-hoc patching.

---

## 1. Official Setting A Benchmark (Full Val $N=348$)

Evaluated under strict Setting A protocol (tile=448, stride=448, non-overlapping, threshold=0.5, FP32, per-sample mean across 348 validation images):

| Model | Val Dice | Val IoU | Precision | Recall | Epochs Used |
|---|---:|---:|---:|---:|:---:|
| **Control (Stage 2 Locked Base)** | **0.7641** | **0.6417** | **0.7337** | 0.8477 | 35 (S1=17, S2=18) |
| **CGSR (Phase 6D Treatment)** | 0.7589 | 0.6359 | 0.7235 | **0.8517** | 35 (S1=17, S2=18) |
| **Delta ($\Delta = \text{CGSR} - \text{Control}$)** | **-0.0052** | **-0.0058** | **-0.0102** | **+0.0040** | — |

- **Dice Degradation:** $\Delta \text{Dice} = -0.0052$ exceeded the maximum allowable budget of $-0.0050$.
- **Precision Drop:** $\Delta \text{Precision} = -0.0102$ demonstrates that false positive predictions across background regions and corridors worsened rather than improved.

---

## 2. False Bridge Cohort Performance

Evaluated against the official 118 false bridge events detected in Candidate B:

| Cohort | $N$ | Cured Events | Cure Rate (%) | Median $\Delta z_{\text{bridge}}$ | Mean $\Delta z_{\text{bridge}}$ |
|---|---:|---:|---:|---:|---:|
| **43 Wider-Gap ($D_{\text{gap}} > 5\text{ px}$)** | 43 | 4 | **9.3%** | **-0.1386** | +0.0014 |
| — Group A ($5 < D_{\text{gap}} \le 8\text{ px}$) | 11 | 1 | 9.1% | -0.3078 | -0.0553 |
| — Group B ($D_{\text{gap}} > 8\text{ px}$) | 32 | 3 | 9.4% | +0.0286 | +0.0210 |
| **All Bridge Events** | 118 | 5 | **4.2%** | **-0.1793** | -0.1035 |

> [!NOTE]
> $\Delta z_{\text{bridge}} = z_{\text{control}} - z_{\text{cgsr}}$. A positive value denotes reduction/suppression of false bridge logits.  
> A negative median ($\mathbf{-0.1386}$) proves that CGSR on average **intensified** the false bridge corridor logits.

---

## 3. Mechanistic Gate Diagnostics

The gate $G(x, s) = \sigma(W_{\text{gate}} \cdot [\text{proj}(x), s] + b)$ was probed across corridor pixels, adjacent true crack pixels, and nearby background:

| Spatial Region | Median Gate Activation | Mean Gate Activation | Std | Range |
|---|---:|---:|---:|:---:|
| **Corridor ($G_{\text{bridge}}$)** | 0.9556 | 0.9554 | 0.0042 | [0.9028, 0.9780] |
| **Adjacent Crack ($G_{\text{crack}}$)** | 0.9562 | 0.9560 | 0.0040 | [0.9085, 0.9785] |
| **Local Background ($G_{\text{bg}}$)** | 0.9548 | 0.9546 | 0.0045 | [0.9015, 0.9772] |
| **Contrast ($G_{\text{crack}} - G_{\text{bridge}}$)** | **+0.0003** | **+0.0006** | 0.0015 | [-0.0035, +0.0062] |

### Why Did the Gate Fail to Discriminate?
1. **Bias Pinning:** The gate bias remained firmly pinned at $b \approx 3.00$ ($2.9993$), clamping activations close to $\sigma(3.0) = 0.9526$.
2. **Constrained Weight Dynamics:** Although weight norm grew from $0.136 \to 0.655$, gradients from the segmentation loss did not penalize false bridges specifically enough to induce spatial selectivity at $56 \times 56$.
3. **Information Bottleneck:** The deep semantic context $(B, 192, 56, 56)$ from DecoderBlock 0 lacked sufficient fine-grained edge localization to inform the gate where thin gaps ended and false connections began without suppressing valid crack continuation.

---

## 4. Negative Controls & Collateral Damage

To verify whether CGSR harmed true crack topology, 118 length-matched continuous true crack segments were evaluated on clean validation images:

- **Clean Crack Segments Probed:** 118
- **New Breakage Events:** **12 (10.2%)** of continuous cracks were split into disconnected components.
- **Mean $\Delta z$ on Clean Crack Pixels:** $-0.0356$ (crack confidence slightly degraded).

---

## 5. Decision Flags Matrix

In accordance with the Section 12 criteria established prior to training:

| Criterion | Threshold | Observed Result | Pass / Fail |
|---|:---:|:---:|:---:|
| **Val Dice Preservation** | $\Delta \text{Dice} \ge -0.0050$ | $-0.0052$ | **FAIL** |
| **Wider-Gap Bridge Cure** | Cure Rate $\ge 25\%$ OR Median $\Delta z > 0.5$ | $9.3\%$, $\Delta z = -0.139$ | **FAIL** |
| **Gate Spatial Contrast** | $\text{median}(G_{\text{crack}} - G_{\text{bridge}}) \ge 0.05$ | $+0.0003$ | **FAIL** |
| **Collateral Crack Integrity** | Breakage rate $\le 2.0\%$ | $10.2\%$ (12 breakages) | **FAIL** |

**Conclusion:** All 4 gates failed. The hypothesis H2 is conclusively rejected.

---

## 6. Closure of Phase 6D

1. **Definitive Finding:** False bridges in Candidate B are not solvable by a localized context-guided skip gate at DecoderBlock 1 under standard supervision. The deep semantic context does not possess the spatial resolution or contrastive signal needed to selectively gate out corridor features without degrading overall Dice and severing true cracks.
2. **Architecture Status:** CGSR will NOT be merged into the canonical baseline ladder. Candidate B remains locked at its canonical state (P3 ASDW-OFF).
3. **Next Steps:** Phase 6D is closed. Any future work addressing connectivity must focus either on global topological losses or representation reformulations at the encoder/stem level rather than decoder-side skip gating.
