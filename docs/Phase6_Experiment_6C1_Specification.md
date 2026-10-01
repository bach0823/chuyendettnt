# Phase 6-C.1 Specification: Topology-Preserving Soft-clDice Probe
**Pre-registered Experimental Protocol & Diagnostic Specification | Date: 2026-10-01**

> [!IMPORTANT]
> **Single-Variable Controlled Topology Probe (Soft-clDice):**
> Phase 6-C.1 isolates the effect of **differentiable skeleton-level topological regularization** on crack continuity and centerline alignment without confounding architectural or boundary-margin alterations.
> - **Lineage Root:** Resumed strictly from Candidate B Stage 1 checkpoint (`best_model_b2_stage1.pth`).
> - **Stage 2 Only:** 18 epochs budget, `patience = 8`.
> - **Zero Added Parameters:** Architecture identical to Candidate B (10,118,955 params, **+0 params**).
> - **Routing:** Sigmoid gating, load balance factor $LB = 0.010$, 8 experts, top_k = 2 — **100% frozen**.
> - **Objective:** $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Base}} + 0.030 \times \mathcal{L}_{\text{SoftclDice}}(\text{iters}=5) + 1.0 \times \mathcal{L}_{\text{LB}}$.
> - **Isolation:** Soft Boundary IoU (`boundary_iou_weight = 0.0`) and AB-BPL (`ab_bpl_weight = 0.0`) are **strictly disabled**.
> - **Evaluation Set:** Official Setting A ($448\times 448$ non-overlapping tiles, $\text{stride} = 448$) on all $N=348$ Validation images. Test set ($N=1124$) strictly sealed.

---

## 1. Background & Empirical Motivation (Phase 6-C Pre-Intervention Diagnostics)

In the Phase 6-C paired diagnostic on all 348 validation images across 4 models:
1. **Breakage / Fragmentation:**
   - Candidate B baseline exhibited 33 fragmented GT components across 33 samples (9.5%).
   - In Thin Cracks ($n=66$), breakage affected 12.1% of samples.
   - When PLU (A.2) narrowed predictions, breakage surged to **30.3%** in thin cracks, showing that area compression without topological preservation destroys crack continuity.
2. **False Bridges:**
   - False bridges affect **31.6% - 33.0%** of all samples across all existing models (~110-115 samples).
   - In complex topologies ($n=27$), false bridge rate is **88.9% - 96.3%**; in thin cracks ($n=66$), it is **62.1% - 66.7%**.
3. **Core Research Question (RQ6.3):**
   Does soft-clDice supervision successfully heal crack breakages and improve centerline precision without worsening false bridges or causing catastrophic boundary dilation?

---

## 2. Mathematical Formulation & Gradient Calibration

### 2.1 Soft Skeletonization Operator
Following Shit et al. (CVPR 2021), soft skeletonization uses cross-shaped structuring elements:
- $\text{Erosion}(X) = \min\left(-\text{MaxPool}_{3\times 1}(-X), -\text{MaxPool}_{1\times 3}(-X)\right)$
- $\text{Dilation}(X) = \text{MaxPool}_{3\times 3}(X)$
- $\text{Opening}(X) = \text{Dilation}(\text{Erosion}(X))$
- Iterative skeleton extraction ($K=5$ iterations):
  $$\Delta S_k = \text{ReLU}(S_{k-1} - \text{Opening}(S_{k-1}))$$
  $$S_k = \text{Erosion}(S_{k-1})$$
  $$S_{\text{soft}}(V) = \sum_{k=1}^K \text{ReLU}\left(\Delta S_k - S_{\text{soft}} \cdot \Delta S_k\right)$$

### 2.2 Objective Definition
For predicted probabilities $P = \sigma(z)$ and binary target $Y$:
$$Tprec = \frac{\sum (S_{\text{soft}}(P) \odot Y) + \epsilon}{\sum S_{\text{soft}}(P) + \epsilon}$$
$$Tsens = \frac{\sum (S_{\text{soft}}(Y) \odot P) + \epsilon}{\sum S_{\text{soft}}(Y) + \epsilon}$$
$$\mathcal{L}_{\text{SoftclDice}} = 1.0 - \frac{2 \cdot Tprec \cdot Tsens}{Tprec + Tsens}$$
*Note: $S_{\text{soft}}(Y)$ is computed under `torch.no_grad()` to conserve memory and execution time.*

### 2.3 Empirical Gradient Calibration on Real Crack500 Data
Direct measurement on real validation batches revealed:
- $\text{Base Loss Grad Norm} = 0.000520$
- $\text{clDice Loss Grad Norm (iters=5)} = 0.004871$
- **Gradient Norm Ratio ($\text{clDice} / \text{Base}$) = $9.3622$**

To supply a balanced, non-destructive auxiliary regularizing force ($\sim 30\%$ of Base loss gradient), the primary probe weight is locked ex-ante:
$$\lambda_{\text{clDice}} = 0.030$$

---

## 3. Preflight Gate Verification (1–5 PASS)

Verified on local runtime before deployment:
```text
================================================================================
RUNNING PREFLIGHT TEST SUITE FOR PHASE 6-C.1 (SOFT-CLDICE PROBE)
================================================================================
[PASS] Preflight 1: Loss & Gradient exact equivalence when cldice_weight=0.0 (torch.equal verified)
[PASS] Preflight 2: Topology sensitivity verified (continuous=0.0034 vs broken=0.2491, diff=+0.2457)
[PASS] Preflight 3: AMP FP16 numerical stability verified on device: cuda
[PASS] Preflight 4: Exact parameter invariance (10,118,955 params matched, +0 dynamic)
[PASS] Preflight 5: Strict checkpoint loading from Candidate B Stage-1 (0 missing, 0 unexpected)
================================================================================
PREFLIGHT GATE: 5/5 PASSED
================================================================================
```

---

## 4. Pre-registered Guideposts & Hypotheses

| Indicator | Candidate B Baseline | Primary Target / Direction | Rationale |
| :--- | :---: | :---: | :--- |
| **Global Val Dice** | 0.7641 | $\ge 0.7641$ | No regression in global segmentation |
| **Global clDice (Mean)** | 0.8498 | $\ge 0.8550$ | Improved centerline overlap |
| **Thin Cracks Breakage Rate ($n=66$)** | 12.1% (8 samples) | $\le 9.0\%$ | Direct healing of thin crack fractures |
| **Global False Bridge Rate** | 31.6% (110 samples) | Monitor | Check whether clDice worsens or leaves bridges unchanged |
| **Thin-low-area Dice ($n=5$)** | 0.5370 | Monitor | Secondary observation |

---

## 5. Execution Command (Colab T4)

```bash
# 1. Update repo and run runtime preflight gate
cd /content/SAGE_LITE
git pull origin crack500-audit
python scripts/tests/test_phase6_c1_cldice.py

# 2. Execute Stage 2 Training
python scripts/train_crack.py \
  --config configs/p3_ablation/b2_p3_run_c_d4_k2_h64_phase6_c1_cldice.yaml \
  --stage2-only \
  --checkpoint /content/drive/MyDrive/crack_seg/P3_C_Canonical_Base_D4_K2/best_model_b2_stage1.pth \
  --rng-checkpoint /content/drive/MyDrive/crack_seg/P3_C_Canonical_Base_D4_K2/last_model_b2_stage1_rng.pth \
  --stage2-epochs 18
```

---

## 6. Empirical Outcomes & Guidepost Evaluation

### 6.1 Guidepost Scorecard

| Indicator | Candidate B Baseline | Pre-registered Target | Phase 6-C.1 Result | Guidepost Status |
| :--- | :---: | :---: | :---: | :---: |
| **Global Val Dice** | 0.7641 | $\ge 0.7641$ | **0.7613** | ❌ **Missed** ($\Delta = -0.0028$) |
| **Global clDice (Mean)** | 0.8498 | $\ge 0.8550$ | **0.8525** | ⚠️ **Improved but below target** (+0.0027) |
| **Thin Cracks Breakage Rate ($n=66$)** | 12.1% (8 samples) | $\le 9.0\%$ | **12.1% (8 samples)** | ❌ **Unimproved** (Invariant) |
| **Global False Bridge Rate** | 31.6% (110 samples) | Monitor | **32.5% (113 samples)** | Invariant ($\approx +0.9\%$) |
| **Spurious Islands Count** | 108 | Monitor | **102** | ✅ **Cleanest across all models** |
| **Global Recall** | 0.8477 | Monitor | **0.8591** | Highest among all models (+0.0114) |
| **Global Precision** | 0.7338 | Monitor | **0.7210** | Regressed (-0.0128) |
| **Global Area Excess (Aggregated)** | +8.16% | Monitor | **+11.82%** | Expanded over-dilation (+3.66%) |

---

## 7. Final Scientific Verdict & Mechanistic Finding

$$\boxed{\textbf{C.1 = Negative global result + confirmed topology trade-off}}$$

> **Core Mechanistic Finding:**  
> Soft-clDice ($\lambda_{\text{clDice}} = 0.030$) improves centerline/topological coverage ($T_{\text{sens}}: 0.8872 \to 0.8899$) and modestly suppresses spurious artifacts/islands ($108 \to 102$), but it completely fails to resolve false bridges ($31.6\% \to 32.5\%$). The observed result is consistent with a mechanism where optimizing skeleton sensitivity rewards maintaining prediction coverage around the ground-truth centerline, which in this crack segmentation task comes with an empirical tendency to widen foreground regions. This induces an unfavorable trade-off: increased predicted area ($+8.16\% \to +11.82\%$), degraded precision ($0.7338 \to 0.7210$), and a slight regression in Global Dice ($0.7641 \to 0.7613$).

### Key Hypothesis Disproven:

$$\boxed{\text{Breakage preservation} \neq \text{False-bridge correction}}$$

A continuity-preserving topology objective addresses network fragmentation, but cannot disentangle or separate merged structures. Because Candidate B's dominant topological bottleneck is False Bridges ($31.6\%$) rather than Breakage ($9.5\%$), soft-clDice regularized the minor failure mode while aggravating the major failure mode.

---

## 8. Final Status & Artifact Manifest

- **Status:** **FROZEN at $\lambda = 0.030$**. No hyperparameter sweep.
- **Checkpoints Saved:**
  * `results/checkpoints/P3_C_Phase6_C1_clDice_D4_K2_H64_best_model_b2_global.pth`
  * `results/checkpoints/P3_C_Phase6_C1_clDice_D4_K2_H64_best_model_b2_stage2.pth`
  * `results/checkpoints/P3_C_Phase6_C1_clDice_D4_K2_H64_last_model_b2_stage2.pth`
- **Full Diagnostics & Logs:**
  * Directory: `results/P3_C_Routing_Diagnostics_Phase6_C1_clDice_D4_K2_H64/`
  * Archive: `results/P3_C_Phase6_C1_clDice_D4_K2_H64_Full.zip`
  * Topology Metrics: `results/diagnostics/phase6_c_topology/topology_c1_metrics.csv`
