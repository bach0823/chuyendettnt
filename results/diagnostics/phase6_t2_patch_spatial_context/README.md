# Phase 6: T2 Patch-Level / Spatial-Context Separability Diagnostic

**Script:** `scripts/diagnostics/phase6_t2_patch_spatial_context_diagnostic.py`
**Run date:** 2026-10-02  
**Duration:** 205.2s (164 samples processed)  
**Status:** COMPLETED — bitwise integrity verified before and after

---

## Integrity Verification

| Hash Target | SHA256 | Status |
|---|---|---|
| Checkpoint file (init) | `147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66` | PASS |
| Checkpoint file (final) | `147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66` | PASS |
| Named Parameters (init) | `4aeda58ce6d2fb32f5b772cdbf92dec7fb913f9510b392910efe6674b0adf5d6` | PASS |
| Named Parameters (final) | `4aeda58ce6d2fb32f5b772cdbf92dec7fb913f9510b392910efe6674b0adf5d6` | PASS |

No weights modified. No backward pass. No optimizer. Test set (N=1124) sealed.

---

## Scientific Objective

Answer the single question:

> **If pointwise statistics at T2 cannot separate NECK from CRACK (established in previous probe, best AUC ~0.59), does spatial context (local neighborhood geometry) around each point contain enough information to discriminate between false-bridge neck corridors and true crack regions?**

---

## Setup

| Parameter | Value |
|---|---|
| Setting A | 448×448 tiles, stride 448, reflect padding, τ=0.5 |
| T2 feature resolution | 56×56 per tile (image / 8) |
| Window sizes | 1×1, 3×3, 5×5, 7×7, 11×11 |
| Feature families | 10 (see below) |
| Total (window, feature) pairs | ~155 per sample |
| AUC computation | Per-sample first, then aggregate medians — never from group medians |

### Cohorts

| Cohort | N | Samples with NECK > 0 |
|---|---|---|
| Consensus Resistant | 82 | 60 |
| Consensus Sensitive | 19 | (subset) |
| Cured by D2 | 7 | (subset) |
| Clean Control | 56 | N/A (used as negative control) |

---

## Feature Families

| Family | Features |
|---|---|
| 1. Local mean/std/variance | `mean_EB`, `mean_ES`, `std_EB`, `std_ES` |
| 2. Local max/min | `max_EB`, `max_ES`, `min_EB`, `min_ES` |
| 3. Local gradient magnitude | `grad_EB`, `grad_ES` (Sobel, windowed mean) |
| 4. Structure tensor anisotropy | `anisotropy_EB`, `anisotropy_ES` (λ_max / λ_min) |
| 5. Local covariance / correlation | `cov_EBES`, `corr_EBES` |
| 6. Center-vs-ring contrast | `center_ring_contrast_EB`, `center_ring_contrast_ES` |
| 7. Radial energy profile | `radial_ratio_01_EB`, `radial_ratio_12_EB` |
| 8. Directional energy ratio | `dir_ratio_HV_EB`, `dir_ratio_HV_ES`, `dir_ratio_diag_EB` |
| 9. Local entropy | `entropy_ES` |
| 10. Relational features | `center_nbhd_ratio_EB/ES`, `BS_energy_ratio`, `BS_agreement` |

Special geometry: **Two-Lobe Score** (`two_lobe_score_EB/ES`) — `max(E_left_arm, E_right_arm) - E_center` along 4 directions; positive = two peaks flanking a valley (bridge geometry).

---

## Results — Consensus Resistant (N=82, 60 with NECK pixels)

### Top 20 (window, feature) pairs by two-sided discriminability

| Rank | Window | Feature | median_auc_nc | discriminability_nc | neck_median | crack_median |
|---|---|---|---|---|---|---|
| 1 | 3 | `dir_ratio_HV_EB` | **0.462** | **0.732** | 0.990 | 1.006 |
| 2 | 7 | `max_ES` | 0.526 | 0.728 | 17.33 | 17.36 |
| 3 | 11 | `anisotropy_EB` | 0.523 | 0.726 | 2.78 | 2.82 |
| 4 | 11 | `min_EB` | 0.499 | 0.724 | 2.39 | 2.54 |
| 5 | 11 | `BS_energy_ratio` | 0.401 | 0.723 | 0.217 | 0.230 |
| 6 | 11 | `BS_agreement` | **0.599** | 0.723 | 0.643 | 0.626 |
| 7 | 11 | `grad_EB` | 0.414 | 0.721 | 2.02 | 2.11 |
| 8 | 5 | `mean_ES` | 0.428 | 0.719 | 15.65 | 15.60 |
| 9 | 5 | `mean_EB` | 0.370 | 0.713 | 3.46 | 3.75 |
| 10 | 5 | `max_ES` | 0.512 | 0.712 | 16.95 | 16.98 |
| 11 | 3 | `grad_ES` | 0.502 | 0.711 | 2.13 | 2.31 |
| 12 | 11 | `std_EB` | 0.458 | 0.711 | 0.438 | 0.474 |
| 13 | 5 | `dir_ratio_HV_EB` | 0.513 | 0.710 | 1.000 | 1.014 |
| 14 | 7 | `dir_ratio_HV_EB` | 0.529 | 0.710 | 1.011 | 1.017 |
| 15 | 5 | `min_ES` | 0.527 | 0.709 | 14.86 | 14.84 |
| 16 | 3 | `mean_ES` | **0.559** | 0.707 | 15.71 | 15.57 |
| 17 | 5 | `anisotropy_EB` | 0.464 | 0.707 | 3.88 | 4.02 |
| 18 | 11 | `mean_EB` | 0.394 | 0.707 | 3.43 | 3.58 |
| 19 | 11 | `anisotropy_ES` | 0.522 | 0.704 | 2.44 | 2.02 |
| 20 | 7 | `entropy_ES` | 0.528 | 0.704 | 1.38 | 1.43 |

### AUC Improvement from w=1 to best window (top 5)

| Feature | Best window | AUC at best w | AUC at w=1 | Delta |
|---|---|---|---|---|
| `dir_ratio_HV_EB` | 3 | 0.462 | 0.500 (degenerate) | N/A (new at w>1) |
| `max_ES` | 7 | 0.526 | 0.507 | +0.019 |
| `anisotropy_EB` | 11 | 0.523 | 0.497 | +0.026 |
| `min_EB` | 11 | 0.499 | 0.408 | +0.091 |
| `BS_energy_ratio` | 11 | 0.401 | 0.411 | -0.010 |

---

## Critical Interpretation: Two-Sided Discriminability vs. Directional AUC

> [!IMPORTANT]
> The discriminability metric is **two-sided**: `discriminability = median_sample(max(auc_i, 1 - auc_i))`. A value of 0.73 does NOT automatically mean the feature achieves AUC = 0.73 as a classifier — it means per-sample separability is strong, but the **direction may be inconsistent across samples**.

This distinction is critical for interpreting the top-ranked features:

### Type A — High discriminability + directional AUC near 0.5: **Bidirectionally-active, inconsistent direction**

Features like `dir_ratio_HV_EB` (disc=0.732, AUC=0.462), `min_EB` (disc=0.724, AUC=0.499), `BS_energy_ratio` (disc=0.723, AUC=0.401):
- Within each sample, NECK and CRACK are well-separated by this feature.
- But the direction (which is higher — NECK or CRACK) **varies across samples**.
- This means the feature captures **sample-specific local structural variation**, not a universal amplitude difference.
- **Not actionable as a global scalar gate.**

### Type B — High discriminability + directional AUC consistently > 0.5: **Weakly consistent direction**

| Feature | Window | median_auc_nc | Direction |
|---|---|---|---|
| `BS_agreement` | 11 | 0.599 | NECK > CRACK (neck 0.643 > crack 0.626) |
| `mean_ES` | 3 | 0.559 | NECK > CRACK (neck 15.71 > crack 15.57) |
| `disagree_diff` (w=1) | 1 | 0.590 | NECK > CRACK (established in prior probe) |

These show **weak but consistent directional signal**: NECK tends to have slightly higher Skip energy (`E_S`) and `BS_agreement` relative to CRACK. However, directional AUC max = **0.599**, which is below any actionable gate threshold.

---

## Decision Gate Verdict

### Consensus Resistant (N=82, primary cohort)

| Criterion | Value | Status |
|---|---|---|
| Best two-sided discriminability | 0.732 (`dir_ratio_HV_EB`, w=3) | discriminability >= 0.70 → mechanistic trigger met |
| Best directional AUC (NECK vs CRACK) | 0.599 (`BS_agreement`, w=11) | < 0.60 → NOT actionable gate |
| Improvement from w=1 → best window | max Δ = +0.091 (`min_EB`, w=11) | Marginal |

**Formal verdict: PARTIAL**

The decision gate code triggered "SUPPORTED" because discriminability >= 0.70. However, the **correct scientific interpretation is PARTIAL**, for the following reasons:

1. **Discriminability is two-sided.** High discriminability with directional AUC near 0.5 means per-sample separability exists but direction is inconsistent — this is mechanistically interesting but not a basis for a universal pointwise gate.
2. **Best directional AUC = 0.599** (`BS_agreement` at w=11). This is below the 0.60–0.70 "partial" threshold for actionable signal.
3. **Spatial context provides minimal additional information over pointwise baseline:** The best directional AUC at w=1 was 0.590 (`disagree_diff`); at best window it is 0.599 (`BS_agreement` w=11). Delta = +0.009. This is negligible gain from spatial context.

> Correct wording: "Spatial context at T2 shows weak local structural variation between NECK and CRACK (per-sample discriminability ~0.70–0.73 for top features), but direction is inconsistent across samples, and no handcrafted spatial feature achieves directional AUC > 0.60 for Consensus Resistant. Spatial context provides marginal additional information over the pointwise baseline."

---

## Two-Lobe Score Analysis

| Cohort | NECK median | CRACK median | Direction |
|---|---|---|---|
| Consensus Resistant | 0.164 | 0.124 | NECK > CRACK (+0.040) |
| Consensus Sensitive | (similar) | (similar) | — |
| Cured by D2 | — | — | — |

Two-lobe score at w=1 (half=2): directional AUC = 0.584 (NECK > CRACK), discriminability = 0.626.  
This is **not in the top 20** by discriminability. Observation: NECK does show slightly higher two-lobe score than CRACK on average, consistent with a "two-peak" geometry interpretation, but the effect is weak and within the range of other pointwise features.

> Correct wording: "The two-lobe score shows weak directional advantage for NECK over CRACK (AUC ~0.58), but does not demonstrate that bridge geometry is distinctly captured at this scale. This does not constitute evidence for a topological two-lobe mechanism."

---

## What Changed from Pointwise to Spatial Context?

| Level | Best feature | Best AUC | Note |
|---|---|---|---|
| A (w=1, pointwise) | `disagree_diff` | 0.590 | Established in prior gateability probe |
| B (w=3 to 11) | `BS_agreement` w=11 | 0.599 | +0.009 over pointwise baseline |
| C (relational) | `BS_agreement` | 0.599 | Relational features are also pointwise-like |

**Spatial context adds essentially no separative information.** The best feature at any window barely edges out the pointwise baseline.

---

## Findings by Cohort

| Cohort | Best feature | Best disc | Best directional AUC | Interpretation |
|---|---|---|---|---|
| Consensus Resistant | `dir_ratio_HV_EB` w=3 | 0.732 | 0.599 (`BS_agreement`) | PARTIAL — local variation exists, not actionable |
| Consensus Sensitive | `max_EB` w=11 | 0.833 | ~0.28 (inverted) | Bidirectional, inconsistent — too few samples |
| Cured by D2 | `dir_ratio_HV_ES` w=7 | 0.885 | ~0.53 | N=7, not statistically robust |
| Clean Control | `mean_EB` w=11 | 0.917 | ~0.22 (inverted) | Control: high disc expected for different reasons |

---

## Synthesis: Phase 6 T2 Diagnostic Series — Updated State

This probe closes the question of **spatial context separability at T2**.

| Probe | Finding | Closed? |
|---|---|---|
| Pointwise spatial statistics (E_B, E_S, z-scores) | No pointwise scalar separates NECK from CRACK (best AUC ~0.59) | YES |
| **Patch spatial context (3×3–11×11, 10 families)** | **Spatial context adds no actionable separation (best AUC 0.599, Δ+0.009)** | **YES** |

### Current Phase 6 closed conclusions

1. **Stem AC** — early ambiguity precursor, not isolated causal bottleneck.
2. **Stage0 DW 7×7** — re-emergence locus, not causally necessary (77/82 Resistant survive α=0).
3. **Decoder B1 directional/isotropic attenuation** — high spatial reconstruction leverage, but not bridge-specific; attenuation breaks true continuity.
4. **T2 stream factorization** — Skip is dominant high-resolution carrier; Bottleneck alone cannot create majority bridges; neither stream is "clean".
5. **Conv1 linearity** — pre-BN is exactly linear: Y = Y_B + Y_S − b.
6. **ReLU** — first nonlinear coupling locus; 22% Resistant bridges are pure interaction artifacts (neither stream alone creates bridge).
7. **Channel-wise ReLU** — NOT sparse channel mechanism; top ΔI channels all have negative I_neck.
8. **Pointwise spatial gating** — NOT SUPPORTED; best AUC 0.59, E_S globally flat.
9. **Patch-level spatial context (this probe)** — PARTIAL; spatial context shows weak per-sample local variation (disc ~0.73) but direction inconsistent, directional AUC max 0.599. **No spatial neighborhood descriptor at T2 can reliably gate bridge vs. crack.**

### Open conclusion

> The false bridge commitment is a **distributed, multi-scale phenomenon**. It cannot be localized to a single locus, a single direction, a single channel subset, or a single spatial scale at T2. The information needed to distinguish bridge from crack is either:
> - Absent or degraded in the representation at T2 (information is too entangled by the time it reaches T2), or
> - Present in a form that requires **learning-based integration** (not accessible via hand-crafted spatial statistics at any single scale).
>
> This does NOT constitute a proof that the information is absent in intermediate representations. It constitutes evidence that pointwise and local spatial hand-crafted features at T2 are insufficient, and that any intervention must target **upstream representation learning** or a **learned spatial integrator** trained on labeled false-bridge examples.

---

## Output Files

```
results/diagnostics/phase6_t2_patch_spatial_context/
  t2_patch_context_per_sample.csv       164 rows × ~800 columns (per sample × window × feature)
  t2_patch_context_by_region.csv        Long-format: (sample, window, feature, neck/crack/bg medians, AUC)
  t2_patch_context_summary.csv          Aggregated: (subgroup, window, feature) → median AUC, disc, effect sizes
  README.md                             This file
  figures/
    t2_auc_vs_window_size.png           AUC vs window plot for top 8 features, Resistant cohort
    t2_top_features_distribution.png    Boxplots NECK/CRACK/BG for top 6 (window, feature) by discriminability
    t2_two_lobe_score.png               Two-lobe score distributions across cohorts and window sizes
```

---

## Epistemic Constraints (respected throughout)

- "SUPPORTED" triggered by discriminability >= 0.70 — but correctly re-interpreted as PARTIAL because directional AUC max = 0.599 < 0.60.
- No claim of "spatial gate will work" — evidence does NOT support this.
- No claim of "information is definitively absent" — only that hand-crafted context features at T2 cannot access it.
- No claim of "topological two-lobe mechanism proven" — two-lobe score shows weak trend, not mechanism.
- No claim of causal mechanism for any single locus or feature.
- AUC computed per-sample first; medians reported across samples, never from group-level pooled pixels.
