# Phase 6D: Context-Guided Stage-1 Skip Refinement (CGSR) — Final Evaluation & Scientific Verdict

**Date:** 2026-10-04  
**Artifacts Evaluated:**
- **Run 1:** `Phase6D_CGSR_Full.zip` (SHA256 Checkpoint: `532f731823039d2e8d8e66efe3ae37e8121a831f00191fb0018c0c8554c3937b`)
- **Run 2 (Rerun):** `Phase6D_CGSR_Full.zip` (SHA256 Checkpoint: `30716b69c05c491d5f37ae579f8c94fc3414960596db49948061cc005a222361`)  
**Protocols Followed:** Section 12 Scientific Decision Rule, Setting A Evaluation (stride 448, non-overlap, threshold 0.5), Locked Lineage Candidate B (Interim ASDW-OFF).

---

## Executive Summary & Final Scientific Verdict

### Verdict: `H2 NOT SUPPORTED` (Re-confirmed across both Run 1 & Run 2)

> **Hypothesis 2 (H2)** stated:
> *"Stage-1 skip representation $(B, 96, 56, 56)$ of Candidate B contains non-discriminative features driving false bridges; deep semantic context $(B, 192, 56, 56)$ can selectively suppress this harmful skip content while preserving true thin cracks."*
>
> **Empirical Findings (Run 1 vs Run 2 Comparison):**
> 1. **Global Performance:** In Run 2, overall Setting A Val Dice increased from **$0.7641$** (Control) to **$0.7667$** ($\Delta = \mathbf{+0.0027}$, Val IoU $+0.0012$), contrasting with Run 1 where Val Dice dropped to $0.7589$.
> 2. **Gate Selectivity (The Critical Causal Mechanism):** Despite the global Dice increase in Run 2, the learned gate $G \in [0, 1]$ produced a spatial contrast between true cracks and false bridges of **virtually ZERO** in both runs:
>    - Run 1: $\text{median}(G_{\text{crack}} - G_{\text{bridge}}) = \mathbf{+0.0003}$
>    - Run 2: $\text{median}(G_{\text{crack}} - G_{\text{bridge}}) = \mathbf{+0.0004}$
>    Gate activation on false bridges ($0.9549$) and true cracks ($0.9555$) is virtually indistinguishable, proving that the gate is globally pinned open ($\approx 95.5\%$ skip passthrough) and fails to discriminate between true cracks and false bridge corridors.
> 3. **False Bridge Cure Rate:** In the wider-gap cohort ($D_{\text{gap}} > 5\text{ px}$, $N=43$), cure rate remained dismal:
>    - Run 1: **$4/43$ ($9.3\%$)** (Median $\Delta z_{\text{bridge}} = -0.1386$)
>    - Run 2: **$5/43$ ($11.6\%$)** (Median $\Delta z_{\text{bridge}} = +0.1595$)
>    Both runs fall overwhelmingly short of the registered threshold ($\ge 25\%-35\%$), leaving $\approx 88\%$ of false bridges persistent.
> 4. **Collateral Damage:** Both runs caused substantial collateral breakage on clean, continuous true cracks ($N=118$):
>    - Run 1: $12/118$ ($10.2\%$) breakages
>    - Run 2: $13/118$ ($11.0\%$) breakages
>    (Significantly exceeding the safe guidepost of $\le 2.0\%$).
>
> **Decision:** In accordance with the Phase 6D Hard Decision Protocol, **H2 is officially NOT SUPPORTED**. The global metric bump in Run 2 is attributable to general feature regularization rather than the intended false-bridge corridor suppression mechanism. Phase 6D is permanently closed.

---

## 1. Official Setting A Benchmark (Full Val $N=348$)

Evaluated under strict Setting A protocol (tile=448, stride=448, non-overlapping, threshold=0.5, FP32, per-sample mean across 348 validation images):

| Model | Val Dice | Val IoU | Precision | Recall | Epochs Used | Best Epoch |
|---|---:|---:|---:|---:|:---:|:---:|
| **Control (Stage 2 Locked Base)** | **0.7641** | **0.6417** | **0.7337** | **0.8477** | 35 (S1=17, S2=18) | 18 |
| **CGSR Run 1** | 0.7589 | 0.6359 | 0.7235 | 0.8517 | 35 (S1=17, S2=18) | 14 |
| **CGSR Run 2 (Rerun)** | **0.7667** | **0.6429** | **0.7337** | 0.8458 | 35 (S1=17, S2=18) | 14 |
| **Delta Run 1 vs Control** | **-0.0052** | **-0.0058** | **-0.0102** | **+0.0040** | — | — |
| **Delta Run 2 vs Control** | **+0.0027** | **+0.0012** | **+0.0000** | **-0.0019** | — | — |

---

## 2. False Bridge Cohort Performance (Run 1 vs Run 2)

Evaluated against the official 118 false bridge events detected in Candidate B:

| Cohort | $N$ | Run 1 Cured (%) | Run 1 Med $\Delta z$ | Run 2 Cured (%) | Run 2 Med $\Delta z$ | Run 2 Mean $\Delta z$ |
|---|---:|---:|---:|---:|---:|---:|
| **43 Wider-Gap ($D_{\text{gap}} > 5\text{ px}$)** | 43 | 4 (**9.3%**) | -0.1386 | 5 (**11.6%**) | **+0.1595** | +0.2170 |
| — Group A ($5 < D_{\text{gap}} \le 8\text{ px}$) | 11 | 1 (9.1%) | -0.3078 | 1 (9.1%) | +0.1595 | +0.2322 |
| — Group B ($D_{\text{gap}} > 8\text{ px}$) | 32 | 3 (9.4%) | +0.0286 | 4 (12.5%) | +0.1467 | +0.2117 |
| **All Bridge Events** | 118 | 5 (**4.2%**) | -0.1793 | 6 (**5.1%**) | **+0.0165** | +0.0284 |

> [!NOTE]
> $\Delta z_{\text{bridge}} = z_{\text{control}} - z_{\text{cgsr}}$. Positive value denotes suppression of false bridge logits.  
> In Run 2, median $\Delta z_{\text{bridge}}$ became slightly positive ($+0.1595$ vs $-0.1386$), indicating modest logit attenuation, but this was insufficient to cross the decision threshold ($z < 0$), leaving $38/43$ ($88.4\%$) wider bridges un-cured.

---

## 3. Mechanistic Gate Diagnostics

Probed across corridor pixels, adjacent true crack pixels, and nearby background:

| Metric | Run 1 Median | Run 1 Mean | Run 2 Median | Run 2 Mean |
|---|---:|---:|---:|---:|
| **Corridor Gate ($G_{\text{bridge}}$)** | 0.9556 | 0.9554 | 0.9549 | 0.9547 |
| **Adjacent Crack Gate ($G_{\text{crack}}$)** | 0.9562 | 0.9560 | 0.9555 | 0.9552 |
| **Gate Contrast ($G_{\text{crack}} - G_{\text{bridge}}$)** | **+0.0003** | **+0.0006** | **+0.0004** | **+0.0006** |

### Mechanistic Root Cause of Gate Failure
1. **Bias Pinning & Saturation:** Both runs remained pinned around $b \approx 3.00$ ($\sigma(3.0) \approx 0.9526$). The network found it optimal to simply keep the skip connection fully open everywhere to preserve general boundary detail.
2. **Absence of Contrastive Spatial Signal:** The standard segmentation loss (BCE + Dice) does not provide a specialized localized gradient to DecoderBlock 1's skip connection to penalize bridging without simultaneously harming nearby crack continuation.
3. **Resolution Mismatch:** At $56 \times 56$, the semantic context from DecoderBlock 0 lacks the sub-pixel geometric precision required to identify gaps narrower than 8 pixels.

---

## 4. Negative Controls & Collateral Damage (118 Clean Crack Segments)

- **Clean Crack Segments Probed:** 118
- **Run 1 Breakage Events:** **12 (10.2%)**
- **Run 2 Breakage Events:** **13 (11.0%)**
- **Safety Guidepost:** Breakage rate $\le 2.0\%$ $\implies$ **VIOLATED in both runs**.

---

## 5. Pre-registered Decision Matrix

| Criterion | Pre-registered Threshold | Observed Run 1 | Observed Run 2 | Verdict |
|---|:---:|:---:|:---:|:---:|
| **Val Dice Preservation** | $\Delta \text{Dice} \ge -0.0050$ | $-0.0052$ (FAIL) | $\mathbf{+0.0027}$ (PASS) | Mixed (PASS in Run 2) |
| **Wider-Gap Bridge Cure** | Cure Rate $\ge 25\%$ OR Median $\Delta z > 0.5$ | $9.3\%$, $-0.139$ | $11.6\%$, $+0.160$ | **FAIL (Both Runs)** |
| **Gate Spatial Contrast** | $\text{median}(G_{\text{crack}} - G_{\text{bridge}}) \ge 0.05$ | $+0.0003$ | $+0.0004$ | **FAIL (Both Runs)** |
| **Collateral Crack Integrity** | Breakage rate $\le 2.0\%$ | $10.2\%$ | $11.0\%$ | **FAIL (Both Runs)** |

---

## 6. Scientific Conclusion & Protocol Action

1. **Rejection of Hypothesis 2:** While Run 2 yielded higher global Dice ($0.7667$), the mechanistic probe conclusively demonstrates that this gain **did NOT originate from false bridge suppression**. The gate is inert (contrast $+0.0004$), wider-gap bridges remain $88.4\%$ un-cured, and collateral crack breakage remains high ($11.0\%$).
2. **Architectural Disposition:** CGSR is officially **rejected** from the SAGE-Lite canonical architecture. Candidate B remains locked in its clean state (interim ASDW-OFF).
3. **Phase 6D Status:** **PERMANENTLY CLOSED**. No further iterations, sweeps, or patches will be conducted on skip-connection gating.
