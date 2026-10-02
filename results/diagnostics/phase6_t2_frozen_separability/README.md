# Phase 6: Frozen T2 Representational Separability Probe

**Script:** `scripts/diagnostics/phase6_t2_frozen_representational_separability_probe.py`
**Run date:** 2026-10-02  
**Duration:** ~6 minutes total (T2 extraction: 123s + 10 bootstrap × 10 probes: ~3.5min)  
**Status:** COMPLETED — bitwise integrity verified before and after

---

## Integrity Verification

| Hash Target | SHA256 | Status |
|---|---|---|
| Checkpoint file (init) | `147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66` | PASS |
| Checkpoint file (final) | `147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66` | PASS |
| Named Parameters (init) | `4aeda58ce6d2fb32f5b772cdbf92dec7fb913f9510b392910efe6674b0adf5d6` | PASS |
| Named Parameters (final) | `4aeda58ce6d2fb32f5b772cdbf92dec7fb913f9510b392910efe6674b0adf5d6` | PASS |

No Candidate B weights modified. No backward pass. No optimizer. Test N=1124 sealed.

---

## Scientific Objective

> **Is T2 representation separable (CRACK vs NECK information exists but requires learned integration), or is component-separation information fundamentally absent at T2?**

---

## Setup

| Parameter | Value |
|---|---|
| Setting A | 448×448 tiles, stride 448, reflect padding |
| T2 feature resolution | 56×56 per tile (image / 8) |
| Representations | B (192ch), S (96ch), T2-fused (288ch) |
| PCA_K | 32 (fitted per bootstrap on train pointwise pixels) |
| N_BOOTSTRAP | 10 (image-level splits) |
| Train/Held-out split | 70% / 30% image-level (42 train / 18 test images) |
| MAX_PIX_PER_CLASS | 30 per image per class |
| Probe classifiers | `LogisticRegression(lbfgs, C=0.1)` and `MLPClassifier(64 units)` |
| RNG base seed | 42 |

### Pixel sample statistics (Consensus Resistant, N=60 with NECK)

| Class | Total pixels | Avg per image |
|---|---|---|
| NECK | 197 | 3.3 |
| CRACK | 10,873 | 181.2 |

> [!IMPORTANT]
> NECK pixels are extremely sparse (avg 3.3 per image in 56×56 feature space). Per bootstrap: ~143 NECK train / ~54 NECK test pixels. This data scarcity limits conclusions about spatial context probes — see interpretation section.

---

## Probe Summary — Mean ± Std AUC (NECK vs CRACK) over 10 Bootstraps

| Probe | Rep | Window | Shuffle | Clf | AUC NC (mean±std) | AUC NB | AUC CB | BACC |
|---|---|---|---|---|---|---|---|---|
| `pointwise_B` | B | 1 | No | LogReg(PCA32) | 0.537 ± 0.034 | 0.894 | 0.957 | 0.592 |
| `pointwise_S` | S | 1 | No | LogReg(PCA32) | 0.638 ± 0.023 | 0.763 | 0.940 | 0.580 |
| `pointwise_T2` | T2 | 1 | No | LogReg(PCA32) | 0.641 ± 0.036 | 0.891 | 0.968 | 0.613 |
| **`pointwise_T2_raw`** | T2 | 1 | No | **LogReg(raw)** | **0.696 ± 0.036** | 0.874 | 0.975 | 0.630 |
| `local5_T2_lin` | T2 | 5 | No | LogReg(PCA32 flat) | 0.618 ± 0.027 | 0.798 | 0.945 | 0.620 |
| `local11_T2_lin` | T2 | 11 | No | LogReg(PCA32 flat) | 0.611 ± 0.024 | 0.761 | 0.943 | 0.607 |
| `local5_T2_mlp` | T2 | 5 | No | MLP(PCA32 flat) | 0.611 ± 0.038 | 0.808 | 0.955 | 0.606 |
| `local11_T2_mlp` | T2 | 11 | No | MLP(PCA32 flat) | 0.592 ± 0.052 | 0.778 | 0.949 | 0.585 |
| `shuffle5_T2_lin` | T2 | 5 | **Yes** | LogReg(PCA32 flat) | 0.523 ± 0.038 | 0.638 | 0.741 | 0.468 |
| `shuffle11_T2_lin` | T2 | 11 | **Yes** | LogReg(PCA32 flat) | 0.528 ± 0.043 | 0.559 | 0.611 | 0.385 |

---

## Key Deltas

| Comparison | Delta AUC NC | Interpretation |
|---|---|---|
| `pointwise_T2_raw` − `pointwise_T2` (PCA32) | **+0.054** | PCA(32) discards discriminative information |
| `local5_T2_lin` − `pointwise_T2` | −0.024 | Local 5×5 context does NOT improve over pointwise |
| `local11_T2_lin` − `pointwise_T2` | −0.031 | Local 11×11 context does NOT improve over pointwise |
| `local5_T2_mlp` − `local5_T2_lin` | −0.006 | MLP adds no gain over linear at 5×5 |
| `local11_T2_mlp` − `local11_T2_lin` | −0.019 | MLP slightly worse at 11×11 |
| `shuffle5_T2_lin` − `local5_T2_lin` | **−0.095** | Spatial arrangement carries real information |
| `shuffle11_T2_lin` − `local11_T2_lin` | **−0.082** | Spatial arrangement important for 11×11 too |

---

## Decision Gate

### Formal case detection

**Primary case: CASE C — Representation Basis Mismatch**

Evidence:
- `pointwise_T2_raw` (288-dim, no PCA) = **0.696** > `pointwise_T2` (PCA32) = 0.641 → Δ = +0.054
- PCA(32) compression discards ~5.4pp of discriminative information
- Raising PCA basis to the full linear rank (288-dim LogReg) improves separability
- This indicates the discriminative directions in T2 are NOT aligned with the top-32 variance components

**Secondary evidence: Spatial arrangement carries information (consistent with Case A precursor)**

Evidence:
- `shuffle5_T2_lin` = 0.523 vs `local5_T2_lin` = 0.618 → drop of **9.5pp** from shuffling
- `shuffle11_T2_lin` = 0.528 vs `local11_T2_lin` = 0.611 → drop of **8.2pp** from shuffling
- Shuffling spatial positions (while preserving channel marginals) degrades performance significantly
- This means the **spatial arrangement within 5×5 and 11×11 patches carries real information**

**Why local context probes underperform pointwise despite spatial information existing:**

> [!NOTE]
> This is NOT a contradiction. With only ~143 NECK train pixels and ~54 NECK test pixels per bootstrap, flattening a 5×5 patch (25 positions × 32 PCA components = 800 features) creates a severely underdetermined problem. The local context probe has 800 features and only ~143 NECK training samples — a 5.6× underdetermination ratio for the NECK class. The probe classifier cannot effectively leverage spatial structure with such few samples, even though the information exists. This is a **data scarcity problem in the probe**, not evidence that spatial information is absent.

### Case summary

| Case | Evidence | Status |
|---|---|---|
| **CASE A** — Local context >> pointwise AND shuffle drops | Shuffle drops 9pp; BUT local context probes *worse* than pointwise (data scarcity confound) | **PARTIALLY SUPPORTED by shuffle; inconclusive from context probe due to data scarcity** |
| **CASE B** — All probes near chance | Best AUC = 0.696, all probes well above 0.5 | **NOT SUPPORTED** |
| **CASE C** — Raw linear >> PCA-compressed; context adds little | raw_lin − PCA_lin = +0.054; confirmed | **PRIMARY CASE — SUPPORTED** |
| **CASE D** — Large context >> small context >> pointwise | local11 ≈ local5 < pointwise | **NOT SUPPORTED** |

---

## Nuanced Interpretation

### 1. T2 is separable above chance — information IS present

Best AUC NECK vs CRACK = **0.696 ± 0.036** (`pointwise_T2_raw`). This is significantly above chance (0.5) and above the best handcrafted feature from the prior probe (0.599). T2 representation contains discriminative information for NECK vs CRACK. The previous probe's failure was a hand-crafted feature limitation, not information absence.

> Correct wording: "T2 representation contains discriminative information for NECK vs CRACK separation (best measured AUC = 0.696 ± 0.036 over 10 bootstrap splits). This is evidence against the hypothesis that information is fundamentally absent at T2."

### 2. PCA(32) is a lossy basis — the discriminative axes are low-variance

`pointwise_T2` (PCA32) = 0.641 vs `pointwise_T2_raw` (288-dim) = 0.696. The +5.4pp gap means PCA discards discriminative information. The top-32 principal components (by variance) do not align with the directions that best separate NECK from CRACK. This is consistent with the prior finding that E_S is globally flat (high variance, low discriminativity for NECK vs CRACK).

> Correct wording: "The discriminative directions for NECK vs CRACK separation are not aligned with the top-32 principal components of T2. PCA(32) projects out ~5.4pp of discriminative information. The full 288-dim feature space is needed for the best pointwise linear separation."

### 3. Spatial arrangement carries information — but probe data is insufficient to quantify

The 9.5pp shuffle drop (5×5) and 8.2pp shuffle drop (11×11) are statistically robust (10 bootstrap mean). Shuffling spatial positions within a patch — while preserving per-channel statistics — destroys these gains. This means the **spatial arrangement of T2 features around a pixel** contributes meaningful signal.

However, local context probes (PCA32 × 5×5 = 800-dim) are NOT better than pointwise (288-dim raw), which can be explained by: only ~197 NECK pixels total in the Resistant cohort means context probes face severe class imbalance in high-dimensional space. This is a probe data scarcity confound, not definitive evidence that spatial context cannot help.

> Correct wording: "Shuffled spatial controls drop 8–10pp AUC relative to un-shuffled context probes, confirming that spatial arrangement of T2 features carries real information. However, the current probe cannot quantify whether a learned spatial integrator would outperform pointwise probes, due to the scarcity of NECK pixels (197 total, ~54 in test per bootstrap)."

### 4. B-stream vs S-stream separability

`pointwise_B` = 0.537 (barely above chance) vs `pointwise_S` = 0.638. The Skip stream carries far more NECK vs CRACK discriminative information than the Bottleneck stream. The T2-fused probe (0.641–0.696) only marginally improves over S-only, suggesting the Bottleneck contributes limited additional discriminative signal beyond the Skip.

> Consistent with prior T2 stream factorization finding: Skip is the dominant high-resolution carrier.

### 5. AUC NB and AUC CB confirm CRACK is well-separated from background

| Probe | AUC NC (NECK vs CRACK) | AUC NB (NECK vs BG) | AUC CB (CRACK vs BG) |
|---|---|---|---|
| `pointwise_T2_raw` | 0.696 | 0.874 | 0.975 |

CRACK vs BG AUC = 0.975 confirms the model encodes crack-like features strongly. The difficulty is specifically NECK vs CRACK — the bridge corridor has a representation similar enough to true crack that T2 features alone cannot cleanly separate them at the 0.70+ level.

---

## Answer to the Final Question

**Is T2 representation separable (but not pointwise-separable), or is information fundamentally absent?**

> **Measured answer:** T2 representation IS separable above chance. The best measured AUC is 0.696 ± 0.036 using a learned linear projection over all 288 channels (no PCA compression). This is significantly above both chance (0.50) and the best hand-crafted feature from the prior probe (0.599). Information is therefore present in T2.
>
> **Remaining open question:** Whether a learned spatial integrator could push separability substantially beyond 0.696. The shuffle drop evidence (−9pp) suggests spatial arrangement carries additional signal, but probe data scarcity (197 NECK pixels) prevents us from drawing a firm conclusion about spatial integration gain. This is unresolved.
>
> **What can be concluded:** The representation basis is misaligned (PCA-top-32 discards discriminative variance). The primary bottleneck is **representation basis**, not information absence. A lightweight learned projection over the full T2 feature space is the most evidence-supported intervention at this point.

---

## Architecture Implication (Observed Evidence → Hypothesis Only)

Based on CASE C evidence, the most supported hypothesis for intervention — if proceeding to architecture:

1. **Lightweight linear adapter / learned projection** on T2 (288-dim → K-dim) trained to align discriminative directions. This does NOT require a new topology module or architectural change to Candidate B.
2. **Not recommended by this evidence**: pure spatial gating without learned projection (spatial info exists but PCA basis already fails at extracting it).
3. **Underdetermined by this evidence**: whether spatial integration beyond pointwise would help significantly (requires more NECK samples to probe reliably).

> [!WARNING]
> These are architecture hypotheses derived from probe evidence. No architecture modification should be made until the user explicitly authorizes it. This probe is diagnostic-only.

---

## Data Scarcity Caveat

| Metric | Value |
|---|---|
| Total NECK pixels (60 images) | 197 |
| Avg NECK per image | 3.3 pixels |
| Train NECK per bootstrap | ~143 pixels |
| Test NECK per bootstrap | ~54 pixels |

With only 197 NECK pixels in the Resistant cohort feature space, all probe results carry substantial sampling variance (std ≈ 0.024–0.052 across bootstraps). The bootstrap CIs should be interpreted as rough estimates, not tight bounds.

**Specific limitation**: local context probes (local5: 800-dim, local11: 3872-dim) with only ~143 NECK train pixels are severely underdetermined. The observed degradation of context probes vs. pointwise may be entirely explained by data scarcity rather than information absence in the spatial context.

---

## Output Files

```
results/diagnostics/phase6_t2_frozen_separability/
  t2_frozen_probe_per_bootstrap.csv    100 rows (10 boot x 10 probes) with per-bootstrap metrics
  t2_frozen_probe_summary.csv          10 rows (one per probe) with mean/std across bootstraps
  decision_gate.txt                    Text output of decision gate with case analysis
  README.md                            This file
  figures/
    t2_frozen_probe_auc_comparison.png      Bar chart: mean AUC ± std per probe
    t2_frozen_probe_bootstrap_distribution.png  Box plots: AUC distribution over bootstraps
    t2_frozen_probe_context_vs_shuffle.png  Local context vs shuffled spatial control comparison
```

---

## Phase 6 Probe Series — Updated Closed Conclusions

| Probe | Key Finding | Closed? |
|---|---|---|
| Stem AC attenuation | Early precursor, not isolated bottleneck | YES |
| Stage0 DW 7×7 | Re-emergence locus, not causally necessary | YES |
| Decoder B1 isotropic/directional attenuation | High spatial reconstruction leverage, not bridge-specific | YES |
| T2 stream factorization | Skip dominant carrier; Bottleneck not "clean" | YES |
| Conv1 linearity | Y = Y_B + Y_S − b proved exactly | YES |
| ReLU nonlinear coupling | First nonlinear locus; 22% Resistant bridges are pure interaction artifacts | YES |
| Channel-wise ReLU | NOT sparse channel mechanism; spatially coded | YES |
| Pointwise spatial gating | NOT SUPPORTED; best AUC 0.599 | YES |
| Patch spatial context (handcrafted) | PARTIAL; no handcrafted feature achieves > 0.60 directional AUC | YES |
| **Frozen T2 representational separability (this probe)** | **CASE C: raw linear 0.696 > PCA32 0.641; spatial arrangement carries info (shuffle −9pp); information present in T2** | **YES** |

### Remaining open questions (not resolved by any probe so far)

1. Whether a learned spatial integrator (e.g., learned 5×5 conv over T2) can push AUC beyond 0.696 — requires more NECK samples or a different probe design.
2. Whether the discriminative directions in T2 (missed by PCA) are learnable during training without architectural change (fine-tuning of decoder head with bridge-specific supervision).
3. Whether the causal mechanism for false bridge commitment is the representation basis mismatch at T2, or a downstream mechanism that uses the T2 representation in a bridge-amplifying way regardless of basis.

---

## Epistemic Constraints (respected throughout)

- No "information absent" conclusion drawn from prior probe failure alone — this probe shows information IS present (AUC = 0.696).
- No "spatial integration will definitely work" — shuffle evidence supports spatial information, but data scarcity prevents quantifying the gain.
- No "CASE A confirmed" — context probes worse than pointwise; confounded by data scarcity; not definitive.
- No "intervention recommended" — architecture implications listed as hypotheses, not prescriptions.
- AUC computed per-sample first (within each bootstrap), aggregated as mean across bootstraps.
- GT used ONLY as evaluation target; never as input feature to any probe.
