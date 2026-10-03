# Candidate B Routing Ablation Study: Static vs Random vs Adaptive vs SAGE-Off

**Date:** 2026-10-03 23:38:08

**Model Checkpoint:** `P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth`

**Dataset:** Crack500 Val ($N=348$), Setting A (tile=448, stride=448, threshold=0.5)

---

## 1. Overall Setting A Validation Performance

| Mode | Val Dice | Val IoU | Precision | Recall | Delta Dice vs Adaptive | Delta IoU vs Adaptive |
|---|---:|---:|---:|---:|---:|---:|
| **Adaptive** | 0.7641 ± 0.1649 | 0.6417 | 0.7338 | 0.8477 | +0.00000 | +0.00000 |
| **Static_Val** | 0.7644 ± 0.1648 | 0.6421 | 0.7359 | 0.8460 | +0.00031 | +0.00040 |
| **Static_Train** | 0.7638 ± 0.1653 | 0.6414 | 0.7310 | 0.8504 | -0.00030 | -0.00030 |
| **Random_Avg** | 0.7633 ± 0.1666 | 0.6412 | 0.7320 | 0.8483 | -0.00074 | -0.00055 |
| **SAGE_Off** | 0.7643 ± 0.1650 | 0.6420 | 0.7328 | 0.8487 | +0.00019 | +0.00028 |

## 2. Paired Statistical Tests against Adaptive Baseline ($N=348$)

| Comparison | Mean Delta | Median Delta | 95% CI | Paired t-stat (p-val) | Wilcoxon W (p-val) | Win Rate (Win/Tie/Loss) |
|---|---:|---:|:---:|:---:|:---:|:---:|
| **Adaptive vs Static_Val** | -0.00031 | -0.00013 | [-0.00102, +0.00039] | t=-0.88 (p=3.80e-01) | W=24849.0 (p=7.07e-03) | **44.0%** (153/4/191) |
| **Adaptive vs Static_Train** | +0.00030 | -0.00001 | [-0.00108, +0.00169] | t=+0.43 (p=6.66e-01) | W=27703.0 (p=2.14e-01) | **48.6%** (169/2/177) |
| **Adaptive vs Random_Avg** | +0.00074 | +0.00013 | [+0.00007, +0.00141] | t=+2.18 (p=2.96e-02) | W=26182.0 (p=3.95e-02) | **53.7%** (187/2/159) |
| **Adaptive vs SAGE_Off** | -0.00019 | -0.00001 | [-0.00114, +0.00076] | t=-0.39 (p=6.99e-01) | W=28752.0 (p=5.56e-01) | **48.9%** (170/3/175) |

## 3. Thinness Quartile Breakdown (Q1: Thicked -> Q4: Thinnest Cracks)

| Thinness Quartile | Adaptive Dice | Static (Val) Dice | Static (Train) Dice | Random Dice | SAGE-Off Dice |
|---|---:|---:|---:|---:|---:|
| **Q1** | 0.8562 | 0.8556 | 0.8561 | 0.8559 | 0.8566 |
| **Q2** | 0.8117 | 0.8114 | 0.8120 | 0.8118 | 0.8125 |
| **Q3** | 0.7546 | 0.7551 | 0.7562 | 0.7540 | 0.7559 |
| **Q4** | 0.6338 | 0.6355 | 0.6307 | 0.6316 | 0.6321 |

## 4. False Bridge Cohort Impact

| Routing Mode | 43 Wider-Gap Cured (%) | 43 Wider Mean Corridor Logit | 118 All Cured (%) | 118 All Mean Corridor Logit |
|---|---:|---:|---:|---:|
| **Adaptive** | 0/43 (**0.0%**) | +2.1276 | 0/118 (**0.0%**) | +2.7637 |
| **Static_Val** | 1/43 (**2.3%**) | +2.1199 | 2/118 (**1.7%**) | +2.7617 |
| **Static_Train** | 0/43 (**0.0%**) | +2.1780 | 1/118 (**0.8%**) | +2.8347 |
| **Random_Seed42** | 0/43 (**0.0%**) | +2.1319 | 1/118 (**0.8%**) | +2.7816 |
| **SAGE_Off** | 0/43 (**0.0%**) | +2.1563 | 1/118 (**0.8%**) | +2.8085 |

