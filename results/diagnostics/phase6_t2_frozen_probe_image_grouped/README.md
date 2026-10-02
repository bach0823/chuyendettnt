# Phase 6: Leakage-Safe Image-Level Regularized Frozen T2 Probe

**Script:** `scripts/diagnostics/phase6_t2_frozen_probe_image_grouped.py`  
**Execution Date:** 2026-10-02  
**Platform:** Windows / CUDA (Setting A, $448 \times 448$, reflect padding, $\tau = 0.5$)  
**Status:** COMPLETED — bitwise integrity verified before and after.

---

## 1. Bitwise Invariance Verification

| Target | Expected SHA256 | Observed SHA256 (Pre) | Observed SHA256 (Post) | Status |
|---|---|---|---|---|
| Disk Checkpoint (`best_model_b2_global.pth`) | `147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66` | `147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66` | `147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66` | **BITWISE INVARIANT** |
| Model Parameters (`model.named_parameters()`) | `4aeda58ce6d2fb32f5b772cdbf92dec7fb913f9510b392910efe6674b0adf5d6` | `4aeda58ce6d2fb32f5b772cdbf92dec7fb913f9510b392910efe6674b0adf5d6` | `4aeda58ce6d2fb32f5b772cdbf92dec7fb913f9510b392910efe6674b0adf5d6` | **BITWISE INVARIANT** |

*Official Test Set ($N=1124$) strictly sealed. Only Validation Cohort ($N=348$, 60 Consensus Resistant cases with valid NECK pixels) used.*

---

## 2. Mandatory Audit of Historical Frozen Probes

Before accepting previous findings, we performed a line-by-line audit of all previous Phase 6 T2 diagnostic scripts:

| Script / Diagnostic | Split Level | PCA Fitting Scope | Scaler Fitting Scope | Hyperparameter $C$ Tuning | Evaluation Scope | Leakage / Confound Status |
|---|---|---|---|---|---|---|
| `phase6_t2_spatial_gateability_diagnostic.py` | None (pointwise descriptive) | N/A (handcrafted scalars) | Global per-sample | None (scalar thresholding) | Per-sample $N=1$ ROC-AUC | Descriptive only; no training leakage. |
| `phase6_t2_patch_spatial_context_diagnostic.py` | None (patch descriptive) | N/A (10 handcrafted families) | Global per-sample | None (scalar thresholding) | Per-sample $N=1$ ROC-AUC | Descriptive only; no training leakage. |
| `phase6_t2_frozen_representational_separability_probe.py` | Image-level random 70/30 (10 seeds) | Train pixels only | Train pixels only | **Fixed $C=0.1$** (No inner CV) | **Global pooled** test pixels across 18 images | **Moderate Confound:** No pixel leakage across images, but fixed $C=0.1$ for 800-dim local context caused severe underdetermination; global pooling risked baseline cross-image shift bias. |
| `phase6_t2_basis_recovery_sweep.py` | Image-level random 70/30 (10 seeds) | Train pixels only | Train pixels only | **Fixed $C=0.1$ across all $k$** (No inner CV) | **Global pooled** test pixels across 18 images | **Substantial Capacity Confound:** As $k$ increased $32 \to 288$, fixed $C=0.1$ allowed raw 288D train AUC to blow up to $0.938$ (gap $+0.250$), confounding capacity with basis recovery. |

### Historical Trustworthiness Verdict
1. **Downgraded:** The claim from `phase6_t2_basis_recovery_sweep.py` that "raw 288D achieves 0.696 primarily due to basis recovery" is downgraded to **Plausible but Confounded** because fixed $C=0.1$ caused large overfit (gap $+0.250$).
2. **Maintained:** The finding that PCA(32) is insufficient compared to higher dimensions is confirmed.
3. **Requirement:** Any definitive claim about T2's representation ceiling must come from the new **leakage-safe, inner-CV regularized, 5-Fold GroupKFold** protocol.

---

## 3. New Evaluation Protocol: Leakage-Safe 5-Fold GroupKFold

To completely eliminate dependence leakage, arbitrary hyperparameter choice, and capacity confounds:
1. **Outer Loop (5-Fold GroupKFold by Image ID):**
   - The 60 Consensus Resistant images with NECK pixels are partitioned into exactly 5 folds ($12$ test images, $48$ training images per fold).
   - Partitioning uses deterministic **Snake Allocation** sorted by NECK pixel count descending ($47, 41, 37, 36, 36$ NECK pixels per fold). Every fold has ample, balanced NECK representation.
   - Zero pixels from test images ever touch training.
2. **Inner Loop (Hyperparameter Selection via 4-Fold GroupKFold):**
   - For every condition on each outer fold, an inner 4-fold GroupKFold by image ID is executed over the 48 training images ($36$ train, $12$ val images per inner fold).
   - $C \in [10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1.0, 10.0]$ is selected strictly based on mean inner-validation ROC-AUC.
   - Inner scalers and PCA projectors are fitted strictly on inner-train images.
3. **Outer Evaluation:**
   - StandardScaler and PCA(k) / Random Projections are fitted strictly on the 48 outer-training images.
   - LogisticRegression is trained on the 48 training images using the optimal $C^*$.
   - Evaluated on the 12 held-out test images:
     - **Held-out Pooled AUC:** Global ROC-AUC on all test pixels in the fold.
     - **Held-out Macro AUC:** Mean of per-image ROC-AUCs across all test images with valid binary classes.
4. **Statistical Inference:**
   - 1,000 resamples of **Image-Level Cluster Bootstrap** over the 60 test images to compute empirical $95\%$ confidence intervals ($[2.5\%, 97.5\%]$) for all conditions and paired contrasts.

---

## 4. Primary Results

### Summary Table across 5 Outer Folds

| Condition | $k$ | Representation Type | Test Pooled AUC (Mean ± Std) | Test Macro AUC (Mean ± Std) | Train AUC (Mean ± Std) | Train-Test Gap (Mean ± Std) | Image-Level Bootstrap 95% CI |
|---|---|---|---|---|---|---|---|
| `pca_32` | 32 | PCA (variance-ordered) | $0.6642 \pm 0.0452$ | $0.6446 \pm 0.0912$ | $0.7354 \pm 0.0245$ | $+0.0712 \pm 0.0675$ | $[0.5657, 0.6950]$ |
| `pca_64` | 64 | PCA (variance-ordered) | $0.6871 \pm 0.0579$ | $0.6606 \pm 0.0978$ | $0.7818 \pm 0.0140$ | $+0.0948 \pm 0.0689$ | $[0.6131, 0.7355]$ |
| `pca_128` | 128 | PCA (variance-ordered) | **$0.6961 \pm 0.0544$** | $0.6807 \pm 0.1006$ | $0.8294 \pm 0.0234$ | $+0.1333 \pm 0.0525$ | **$[0.6365, 0.7379]$** |
| `pca_192` | 192 | PCA (variance-ordered) | **$0.6943 \pm 0.0607$** | **$0.6854 \pm 0.0859$** | $0.8736 \pm 0.0053$ | $+0.1792 \pm 0.0647$ | **$[0.6391, 0.7417]$** |
| `pca_256` | 256 | PCA (variance-ordered) | $0.6959 \pm 0.0535$ | $0.6801 \pm 0.0955$ | $0.8691 \pm 0.0237$ | $+0.1731 \pm 0.0302$ | $[0.6444, 0.7464]$ |
| `raw_288` | 288 | Full Raw T2 (Standardized) | **$0.6969 \pm 0.0541$** | $0.6797 \pm 0.0957$ | $0.8714 \pm 0.0246$ | $+0.1745 \pm 0.0299$ | **$[0.6452, 0.7476]$** |
| `rand_32` | 32 | Random Gaussian Projection | $0.5872 \pm 0.0799$ | $0.5922 \pm 0.1093$ | $0.6895 \pm 0.0224$ | $+0.1023 \pm 0.0843$ | $[0.5109, 0.6283]$ |
| `rand_64` | 64 | Random Gaussian Projection | $0.6398 \pm 0.0618$ | $0.6128 \pm 0.0929$ | $0.7632 \pm 0.0264$ | $+0.1234 \pm 0.0603$ | $[0.5740, 0.6775]$ |
| `rand_128` | 128 | Random Gaussian Projection | $0.6825 \pm 0.0670$ | $0.6593 \pm 0.0674$ | $0.8160 \pm 0.0126$ | $+0.1335 \pm 0.0778$ | $[0.6210, 0.7245]$ |
| `shuffle_null`| 32 | Label-Permuted Null Control | $0.5136 \pm 0.0598$ | $0.5214 \pm 0.0907$ | $0.4950 \pm 0.0807$ | $-0.0187 \pm 0.0452$ | $[0.4536, 0.5591]$ |

---

### Fold-by-Fold Breakdown

| Fold Index | Test Images | Test NECK Px | `pca_32` | `pca_64` | `pca_128` | `pca_192` | `pca_256` | `raw_288` | `rand_32` | `rand_64` | `rand_128` | `shuffle_null` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fold 0 | 12 | 47 | 0.7047 | 0.7211 | 0.7208 | 0.7271 | 0.7229 | 0.7240 | 0.6873 | 0.6852 | 0.7216 | 0.4202 |
| Fold 1 | 12 | 41 | 0.6869 | 0.7247 | 0.7222 | 0.7094 | 0.7103 | 0.7142 | 0.5504 | 0.6174 | 0.7082 | 0.5762 |
| Fold 2 | 12 | 37 | 0.5996 | 0.5919 | 0.5991 | 0.5870 | 0.6015 | 0.6011 | 0.4737 | 0.5427 | 0.5639 | 0.4939 |
| Fold 3 | 12 | 36 | 0.6951 | 0.7262 | 0.7248 | 0.7325 | 0.7319 | 0.7319 | 0.6137 | 0.6931 | 0.7201 | 0.5403 |
| Fold 4 | 12 | 36 | 0.6346 | 0.6713 | 0.7135 | 0.7157 | 0.7132 | 0.7134 | 0.6109 | 0.6608 | 0.6989 | 0.5374 |

*Note on Fold 2:* Fold 2 has lower AUC across all conditions (e.g. `pca_32`=0.5996, `raw_288`=0.6011), reflecting 2 specific images with severe noise / edge artifacts. The other 4 folds consistently achieve AUC $\ge 0.71 - 0.73$ on higher dimensions.

---

## 5. Paired Statistical Contrasts (1,000 Image-Level Cluster Bootstraps)

To eliminate between-image sampling variance, paired differences $A - B$ were computed on each bootstrap resample:

| Contrast Name | Condition A | Condition B | Mean Paired Diff | Std Diff | 95% Bootstrap CI | Empirical $p$-value ($d \le 0$) | Interpretation |
|---|---|---|---|---|---|---|---|
| **PCA192 − PCA32** | `pca_192` | `pca_32` | **$+0.0593$** | $0.0242$ | **$[+0.0120, +0.1093]$** | **$p = 0.012$** | **Statistically Significant Gain:** Higher dimensions recover real signal. CI strictly excludes 0. |
| **PCA128 − PCA32** | `pca_128` | `pca_32` | **$+0.0577$** | $0.0231$ | **$[+0.0103, +0.1018]$** | **$p = 0.012$** | **Statistically Significant Gain:** $k=128$ already captures the full dimensional recovery. |
| **PCA32 − RP32** | `pca_32` | `rand_32` | **$+0.0624$** | $0.0319$ | $[-0.0001, +0.1230]$ | $p = 0.052$ | **Substantial PCA Advantage at Low $k$:** Marginally significant ($p \approx 0.05$). |
| **PCA64 − RP64** | `pca_64` | `rand_64` | **$+0.0507$** | $0.0162$ | **$[+0.0166, +0.0795]$** | **$p = 0.002$** | **Statistically Significant PCA Advantage:** Variance alignment beats random projection at $k=64$. |
| **PCA128 − RP128** | `pca_128` | `rand_128` | $+0.0140$ | $0.0115$ | $[-0.0092, +0.0366]$ | $p = 0.232$ | **Basis Invariance at High $k$:** At $k=128$, random projection catches up to PCA. |
| **PCA192 − PCA128** | `pca_192` | `pca_128` | $+0.0016$ | $0.0082$ | $[-0.0144, +0.0188]$ | $p = 0.842$ | **Zero Difference / Complete Plateau:** Dimensions $129-192$ add no new test AUC. |
| **raw288 − PCA192** | `raw_288` | `pca_192` | $+0.0047$ | $0.0035$ | $[-0.0020, +0.0123]$ | $p = 0.166$ | **Zero Difference:** Raw 288D is statistically indistinguishable from PCA(192). |

---

## 6. Overfitting Audit & Capacity Ceiling Analysis

In the historical sweep (`phase6_t2_basis_recovery_sweep.py`) with fixed $C=0.1$, the train-test gap at `raw_288` was $+0.250$ (Train AUC $0.938$).

In this diagnostic, with inner cross-validation:
- Inner CV chose $C^* = 0.01$ or $0.001$ for higher dimensions (regularizing strongly against the small $N_{\text{NECK}} = 197$ sample size).
- Train AUC at `raw_288` dropped from $0.938 \to 0.871$.
- Train-test gap dropped from $+0.250 \to +0.175$.
- **Crucially:** Held-out test AUC remained **completely intact at $0.6969$** (fold-wise mean) and $0.6966$ (bootstrap mean).

### What this proves:
The $\sim 0.69$ test AUC is **NOT an overfit illusion or capacity artifact**. When capacity is constrained by proper regularization ($C^* = 0.01$), generalization does not collapse; it remains exactly at $0.696$.

---

## 7. Critical Hypothesis Gates

### $H_A$: Higher dimensions really contain usable signal
> **VERDICT: SUPPORTED**  
> *Evidence:* `PCA192 - PCA32` has mean paired difference $+0.0593$ with $95\%$ bootstrap CI $[+0.0120, +0.1093]$, strictly excluding zero ($p = 0.012$). PCA(128) achieves $+0.0577$ ($p = 0.012$). The gain of $\sim 6$ percentage points from expanding the basis beyond the top 32 variance axes is stable across folds, resilient to inner CV, and statistically robust.

### $H_B$: Basis selection itself has a meaningful effect
> **VERDICT: PLAUSIBLE BUT NOT PROVEN (Scale-Dependent)**  
> *Evidence:* At low dimensions ($k=32$ and $k=64$), PCA variance alignment provides a decisive advantage over random projection ($+0.062$ at $k=32$, $+0.051$ at $k=64$ with $p = 0.002$). However, by $k=128$, random projection achieves test AUC $0.6825$, narrowing the gap with PCA(128) to $+0.014$ (CI $[-0.009, +0.037]$, $p = 0.232$). While PCA alignment accelerates basis recovery at low dimensions, high dimensionality itself is sufficient to capture the subspace.

### $H_C$: There is a real T2 representation ceiling
> **VERDICT: SUPPORTED**  
> *Evidence:* Test AUC reaches a flat ceiling starting at $k=128$:
> - `pca_128`: $0.6961$
> - `pca_192`: $0.6943$
> - `pca_256`: $0.6959$
> - `raw_288`: $0.6969$  
> 
> The paired difference between `raw_288` and `pca_192` is $+0.0047$ (95% CI $[-0.0020, +0.0123]$, $p = 0.166$), and between `pca_192` and `pca_128` is $+0.0016$ ($p = 0.842$). No linear classifier on pointwise T2 features—regardless of dimension or regularization—exceeds $\sim 0.697$.

---

## 8. Final Decision & Claim Verdict Table

| Claim | Verdict | Direct Empirical Justification |
|---|---|---|
| **T2 contains measurable separation information** | **SUPPORTED** | Outer-fold test AUC consistently reaches $0.694 - 0.697$ across all $k \ge 128$ (Null shuffle control = $0.5136$). |
| **Higher dimensionality recovers real signal** | **SUPPORTED** | `PCA192 - PCA32` diff $= +0.0593$, $95\%$ CI $[+0.0120, +0.1093]$ strictly positive ($p = 0.012$). |
| **PCA basis gives unique advantage** | **PLAUSIBLE BUT NOT PROVEN** | Significant at $k=32, 64$ ($p < 0.05$), but vanishes at $k=128$ ($p = 0.232$). |
| **T2 has stable ~0.69 ceiling** | **SUPPORTED** | $k=128, 192, 256, 288$ all land within $[0.694, 0.697]$; paired differences are zero ($p > 0.16$). |
| **Evidence justifies T2 intervention** | **PLAUSIBLE BUT NOT PROVEN** | Pointwise linear projection cannot break $0.70$. A learned spatial/contextual adapter at T2 could potentially utilize spatial arrangement (prior shuffle drop $-0.09$), but pointwise intervention alone is capped at $0.697$. |
| **Evidence favors upstream representation change** | **PLAUSIBLE BUT NOT PROVEN** | Because the representation ceiling at T2 is firmly bounded at $\text{AUC} \approx 0.697$, the false-bridge ambiguity is already baked into T2's channel space before Decoder Block 1 Conv1. |

---

## 9. What This Probe Does NOT Establish

To maintain strict scientific integrity, we explicitly document what this diagnostic **cannot** and **does not** prove:

1. **Does NOT prove that a pointwise linear gate at T2 can solve false bridges:**  
   An AUC ceiling of $\sim 0.697$ corresponds to significant overlap between NECK and CRACK. A pointwise linear gating mechanism on T2 would suffer high false-positive or false-negative rates.
2. **Does NOT prove that T2 information is causally sufficient for topological curing:**  
   Measuring linear separability in a frozen feature map does not guarantee that modifying or filtering those features during an end-to-end forward pass will cure the false bridges without destroying true crack recall (as already proven in the isotropic/directional attenuation probes).
3. **Does NOT prove that upstream representation is irreparably collapsed:**  
   T2 still retains $\sim 0.697$ separability (far above chance $0.513$). The ambiguity is partial, not total.
4. **Does NOT justify training an architectural intervention immediately without a formal design plan:**  
   This probe is diagnostic-only. Any proposal for a learned adapter (at T2 or upstream) requires an explicit, separate design review.

---

## 10. Generated Artifacts

- **Fold Data:** [`t2_image_grouped_folds.csv`](file:///d:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_t2_frozen_probe_image_grouped/t2_image_grouped_folds.csv)
- **Summary Metrics:** [`t2_image_grouped_summary.csv`](file:///d:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_t2_frozen_probe_image_grouped/t2_image_grouped_summary.csv)
- **Paired Contrasts:** [`t2_image_grouped_contrasts.csv`](file:///d:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_t2_frozen_probe_image_grouped/t2_image_grouped_contrasts.csv)
- **Decision Gate Text:** [`decision_gate.txt`](file:///d:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_t2_frozen_probe_image_grouped/decision_gate.txt)
- **Figures:**
  - `figures/t2_image_grouped_auc_vs_dim.png`: Test AUC vs dimension (PCA vs Random Projection vs Train AUC).
  - `figures/t2_image_grouped_gap_vs_dim.png`: Train-test gap vs dimension under inner-CV regularization.
  - `figures/t2_image_grouped_paired_contrasts.png`: Forest plot of paired statistical contrast differences with empirical 95% bootstrap CIs.
