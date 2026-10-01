# Phase 6-D.1 Specification: Representation × Boundary Synergy Probe (PLU Head + AB-BPL)

**Status:** Ready for Deployment (Preflight Gate 5/5 PASSED)  
**Date:** 2026-10-02  
**Config:** [`configs/p3_ablation/b2_p3_run_c_d4_k2_h64_phase6_d1_plu_abbpl.yaml`](file:///d:/truong/SpecialSubjectTTNT/SAGE_LITE/configs/p3_ablation/b2_p3_run_c_d4_k2_h64_phase6_d1_plu_abbpl.yaml)  
**Parent Lineage Checkpoint:** [`results/checkpoints/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_best_model_b2_stage1.pth`](file:///d:/truong/SpecialSubjectTTNT/results/checkpoints/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_best_model_b2_stage1.pth)  
**Parent Lineage RNG:** [`results/checkpoints/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_last_model_b2_stage1.pth`](file:///d:/truong/SpecialSubjectTTNT/results/checkpoints/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_last_model_b2_stage1.pth)  

---

## 1. Architectural & Objective Invariants

Phase 6-D.1 evaluates the interaction between high-resolution spatial representation recovery (PLU Head) and asymmetric boundary-band regularization (AB-BPL).

| Component | Pure A2 Control Lineage | Phase 6-D.1 Intervention | Status / Invariant |
| :--- | :---: | :---: | :--- |
| **Model Architecture** | B2 UNet ($D=4, K=2, H=64$, ASDW) | B2 UNet ($D=4, K=2, H=64$, ASDW) | Strictly identical |
| **Decoder Head** | `PLUHead` (`use_plu_head=True`) | `PLUHead` (`use_plu_head=True`) | Strictly identical ($10,125,363$ params, $+0$ dynamic) |
| **Stage 1 Checkpoint** | Pure A2 Stage-1 ($17$ epochs) | Pure A2 Stage-1 ($17$ epochs) | **100% Shared Lineage** (`strict=True`, $0$ missing, $0$ unexpected) |
| **Stage 1 RNG / Scaler** | Pure A2 Stage-1 last checkpoint | Pure A2 Stage-1 last checkpoint | **100% Identical Random State** |
| **Stage 2 Optimizer & Scheduler**| AdamW fresh ($LR=10^{-4}$), Cosine Anneal | AdamW fresh ($LR=10^{-4}$), Cosine Anneal | Identical fresh initialization |
| **Stage 2 Objective** | $\mathcal{L}_{\text{Base}} + \mathcal{L}_{\text{balance}}$ | $\mathcal{L}_{\text{Base}} + \mathcal{L}_{\text{balance}} + \mathbf{0.040} \times \mathcal{L}_{\text{AB-BPL}}$ | **Single Isolated Objective Difference** ($\lambda=0.040, r=2$) |
| **BoundaryIoU & clDice** | OFF ($\lambda=0.0$) | OFF ($\lambda=0.0$) | Completely disabled to avoid confounding |

---

## 2. Scientific Motivation & Core Hypotheses

From the empirical findings of Phase 6-D.0, False Bridge failures decompose into two distinct components:
$$\boxed{\text{False Bridge} = \text{Dilation-mediated Component} + \text{Residual Separation Component}}$$

- **Over-dilation as a Strong Risk Multiplier ($OR = 5.67$, $p < 0.0001$):**
  $90.0\%$ of Candidate B bridges occur when $\text{AreaExcess} > 0$, and the bridge rate escalates monotonically from $10.7\% \to 52.5\%$ as dilation severity increases.
- **The PLU Trade-off in Phase 6-A.2:**
  Pure PLU achieved the largest thin-crack recovery across the study ($+0.0689$ Dice on thin-low-area, cutting AreaExcess from $+80.37\% \to +57.00\%$), but concurrently produced topological fragmentation ($614$ spurious islands vs $108$ in Base, and $6$ newly created bridges).

### Hypotheses:

> **Primary Synergy Hypothesis:**  
> Boundary penalty ($\mathcal{L}_{\text{AB-BPL}}$, $\lambda=0.040, r=2$) may constrain the excess foreground expansion introduced by PLU, potentially reducing its topological side-effects while preserving its thin-low-area spatial representation gain.

> **Counter-Hypothesis (Interference):**  
> Simultaneous application of high-frequency convolutional upsampling and outer boundary penalty may suppress thin crack signals prematurely, causing loss of the PLU representation gain on ultra-thin filaments ($n=5$).

---

## 3. Preflight Gate Verification (1–5 PASS)

Verified on local runtime before deployment:
```text
================================================================================
RUNNING PREFLIGHT TEST SUITE FOR PHASE 6-D.1 (PLU + AB-BPL SYNERGY PROBE)
================================================================================
[PASS] Preflight 1: Base and D.1 (lambda=0) loss are identical (diff=0.00e+00)
[PASS] Preflight 2: AB-BPL penalizes over-dilation (clean=0.0067, dilated=5.0067, diff=+5.0000)
[PASS] Preflight 3: AMP FP16 forward/backward/optimizer step stable on device: cuda (loss=1.4814)
[PASS] Preflight 4: Parameter invariance verified (exactly 10,125,363 params, +0 dynamic)
[PASS] Preflight 5: Strict checkpoint loading verified (0 missing, 0 unexpected, complete RNG/scaler state present)
================================================================================
PREFLIGHT GATE: 5/5 PASSED — READY FOR PHASE 6-D.1 STAGE 2 DEPLOYMENT
================================================================================
```

---

## 4. Pre-registered Guideposts & Readout Matrix

### 4.1 Four Core Guideposts

| Indicator | Candidate B Baseline | Pure A2 Control | Phase 6-B.1 | Phase 6-D.1 Target | Status & Scientific Rationale |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Thin-low-area Dice ($n=5$)** | 0.5370 | **0.6059** | 0.5362 | $\ge \mathbf{0.5900}$ | **Pre-registered practical retention guidepost**: Retain majority of PLU representation gain |
| **Boundary Margin Dice ($n=127$)** | 0.7812 | 0.7779 | **0.7802** | $\ge \mathbf{0.7800}$ | Retain B1 boundary precision and recall preservation |
| **Global Area Excess (Aggregated)** | +8.16% | **+4.93%** | +9.46% | $< \mathbf{+5.0\%}$ | Enforce strict mask tightness; resist over-dilation |
| **Global False Bridge Count** | 110 (31.6%) | 111 (31.9%) | 115 (33.0%) | $\le \mathbf{107}$ ($< 31.0\%$) | **Integer threshold**: Net change $\text{NetChange} = N_{\text{created}} - N_{\text{cured}} \le -5$ |

### 4.2 Full Mandatory Readout Matrix across 5 Cohorts
Evaluated on post-hoc Setting A tiling ($448 \times 448$, threshold $0.5$):
$$\boxed{\text{Dice},\ \text{Precision},\ \text{Recall},\ \text{AreaExcess},\ \text{FalseBridge Count/Rate},\ \text{Breakage Count/Rate},\ \text{Spurious Islands},\ \text{clDice}}$$
Across cohorts:
1. Global ($N=348$)
2. Thin Cracks ($n=66$, thinness $> 0.20$)
3. Thin-low-area ($n=5$)
4. Boundary Margin ($n=127$)
5. Complex Topology ($n=27 - 35$)

---

## 5. Execution Command (Colab T4)

```bash
# 1. Update repository and run local runtime preflight gate
cd /content/SAGE_LITE
git pull origin main
python scripts/tests/test_phase6_d1_plu_abbpl.py

# 2. Download Phase 6-A.2 Stage 1 checkpoints (if not present)
mkdir -p /content/checkpoints
wget -q -O /content/checkpoints/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_best_model_b2_stage1.pth \
  "https://raw.githubusercontent.com/bach0823/chuyendettnt/main/results/checkpoints/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_best_model_b2_stage1.pth"
wget -q -O /content/checkpoints/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_last_model_b2_stage1.pth \
  "https://raw.githubusercontent.com/bach0823/chuyendettnt/main/results/checkpoints/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_last_model_b2_stage1.pth"

# 3. Execute Stage 2 Training
python scripts/train_crack.py \
  --config configs/p3_ablation/b2_p3_run_c_d4_k2_h64_phase6_d1_plu_abbpl.yaml \
  --stage2-only \
  --checkpoint /content/checkpoints/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_best_model_b2_stage1.pth \
  --rng-checkpoint /content/checkpoints/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_last_model_b2_stage1.pth \
  --stage2-epochs 18
```

---

## 6. Branching Decision Logic after Phase 6-D.1

- **Branch 1 (Synergy Supported):**  
  If $\text{AreaExcess} < +5.0\%$ **AND** False Bridge count $\le 107$ ($< 31.0\%$) while Thin-low-area Dice $\ge 0.5900$:  
  $\implies$ Phase 6-D.1 supports the representation-boundary synergy hypothesis, demonstrating that the dilation-mediated component of False Bridges can be resolved through joint spatial representation and boundary control.
- **Branch 2 (Separation Bottleneck Confirmed):**  
  If $\text{AreaExcess}$ shrinks substantially ($< +5.0\%$) **BUT** False Bridge count remains invariant at $\sim 110 - 115$ ($\approx 32\%$):  
  $\implies$ Provides definitive empirical evidence that False Bridge in crack topologies is overwhelmingly governed by the residual separation component rather than dilation. This establishes the necessary empirical justification to proceed directly to **Phase 6-D.2 (Explicit Distance-weighted Repulsive / Separation Objective)**.
