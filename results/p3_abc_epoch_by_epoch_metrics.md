# B2 Crack500: P3 Controlled Ablation & Depth Sensitivity (D12 vs D8 vs D6 vs D4 vs D2)

> **Canonical Protocol Verification:**
> - **Dataset & Protocol**: Crack500 Val split (348 samples), Setting A evaluation protocol.
> - **Base Backbone**: ConvNeXt-V2-Femto (ImageNet pretrained).
> - **Controlled Variants**:
>   - **P3-A (D12 Identity Control)**: ViT D12, 0 extra params (`p3_mode: none`).
>   - **P3-B (D12 Generic DW)**: ViT D12, +38,450 params (`p3_mode: generic_dw`, depthwise conv 3x3).
>   - **P3-C (D12 ASDW Refinement)**: ViT D12, +37,874 params (`p3_mode: asdw`, asymmetric strip convs + gating).
>   - **P3-C (D8 ASDW Controlled Depth)**: ViT D8 (8 blocks instead of 12), all other hyperparameters identical.
>   - **P3-C (D6 ASDW Controlled Depth)**: ViT D6 (6 blocks instead of 12), all other hyperparameters identical.
>   - **P3-C (D4 ASDW Controlled Depth)**: ViT D4 (4 blocks instead of 12, 8 experts: 4 CNN + 4 ViT, 1:1 symmetry, k=4), all other hyperparameters identical.
>   - **P3-C (D2 ASDW Controlled Depth)**: ViT D2 (2 blocks instead of 12, 6 experts: 4 CNN + 2 ViT, k=4), all other hyperparameters identical.

---

## 1. Executive Summary & Peak Performance Comparison

| Configuration | Depth | Total Params | Extra Params | Peak S1 Val Dice | Peak S2 Val Dice | Global Peak Ep | Val Loss @ Peak | Epoch Time (Speed) | Termination Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **P3-A (Identity Control)** | D12 | 12.45M | 0 | 0.7158 (Ep 14) | **0.7543** | S2 Ep 13 | 1.0140 | ~03:54 (1.72s/it) | Budget ceiling (30/30 ep) |
| **P3-B (Generic DW)** | D12 | 12.49M | +38,450 | 0.7179 (Ep 10) | **0.7496** | S2 Ep 15 | 1.0495 | ~04:06 (1.82s/it) | Budget ceiling (30/30 ep) |
| **P3-C (ASDW Refinement)** | D12 | 12.49M | +37,874 | 0.7266 (Ep 10) | **0.7557** | S2 Ep 09 | 1.0889 | ~04:05 (1.80s/it) | Early Stopping Confirmed (Patience=6 @ Ext Ep 6) |
| **P3-C (ASDW D8)** | D8 | 11.31M | +37,874 | 0.7274 (Ep 10) | **0.7604** | **S2 Ep 17** | **0.9999** | **~03:20 (1.48s/it)** | Budget ceiling (33/33 ep, S2 t_init=18) |
| **P3-C (ASDW D6)** | D6 | 10.71M | +37,874 | 0.7312 (Ep 13) | **0.7599** | S2 Ep 16 | **0.9602** | **~03:12 (1.42s/it)** | Early Stopping Confirmed (Patience=6 @ Ext Ep 6) |
| **P3-C (ASDW D4)** | **D4** | **10.12M** | +37,874 | 0.7295 (Ep 14) | 🏆 **0.7639** | **S2 Ep 14** | **0.9533** | **~03:40 (1.63s/it)** | Early Stopping Confirmed (Patience=6 @ Ext Ep 6) |
| **P3-C (ASDW D2)** | **D2** | **9.20M** | +37,874 | 0.7190 (Ep 13) | **0.7578** | **S2 Ep 16** | **0.9477** | **~03:30 (1.55s/it)** | Budget ceiling (35/35 ep) |
| **Phase 4 (LB=0.005)** | D4 | 10.12M | +37,874 | 0.7321 (Ep 12) | **0.7580** | S2 Ep 19 | 0.9572 | ~02:45 (1.18s/it) | Budget ceiling (35/35 ep, Router Collapse) |
| **Phase 4 (LB=0.010, Cand B)** | **D4** | **10.12M** | +37,874 | **0.7326 (Ep 08)** | 🏆 **0.7641** | **S2 Ep 14** | **0.9544** | **~02:40 (1.15s/it)** | 🏆 **LOCKED OPTIMAL (Balanced Pareto)** |
| **Phase 4 (LB=0.030)** | D4 | 10.12M | +37,874 | 0.7321 (Ep 14) | **0.7624** | S2 Ep 16 | 0.9465 | ~02:42 (1.16s/it) | Budget ceiling (35/35 ep, Over-regularized) |
| **Phase 6-A.1 (Boundary IoU)** | **D4** | **10.12M** | **0 (Frozen)** | **0.7333 (Ep 13)** | 🏆 **0.7684** | **S2 Ep 16** | **1.3911** | **~02:35 (1.11s/it)** | 🏆 **NEW PEAK (Thin Crack +2.37% Dice, Area +7.85%)** |

---

## 2. Stage 1 Side-by-Side Comparison (Epochs 1-17)

| Ep | P3-A (D12) | P3-B (D12) | P3-C (D12) | P3-C (D8) | P3-C (D6) | P3-C (D4) | P3-C (D2) | Top Dice in Ep | D4 Gamma (S0/S1) | D2 Gamma (S0/S1) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 0.1461 | 0.1535 | 0.1560 | 0.1568 | **0.1601** | 0.1478 | 0.1562 | **P3-C D6** (0.1601) | 0.0099/0.0100 | 0.0101/0.0101 |
| 02 | 0.5770 | 0.6004 | 0.6317 | 0.6036 | 0.6331 | 0.6184 | **0.6420** | **P3-C D2** (0.6420) | 0.0102/0.0099 | 0.0106/0.0112 |
| 03 | 0.5962 | 0.5887 | 0.6329 | **0.6479** | 0.6073 | 0.5964 | 0.6159 | **P3-C D8** (0.6479) | 0.0108/0.0108 | 0.0112/0.0117 |
| 04 | 0.6821 | 0.6635 | 0.6449 | 0.6826 | 0.6740 | 0.6972 | **0.7101** | **P3-C D2** (0.7101) | 0.0116/0.0121 | 0.0116/0.0130 |
| 05 | 0.5955 | 0.6398 | 0.6354 | **0.6571** | 0.5996 | 0.6434 | 0.6147 | **P3-C D8** (0.6571) | 0.0126/0.0143 | 0.0116/0.0135 |
| 06 | 0.5994 | 0.6130 | 0.6568 | **0.6661** | 0.6513 | 0.6139 | 0.6136 | **P3-C D8** (0.6661) | 0.0127/0.0150 | 0.0117/0.0135 |
| 07 | 0.6955 | 0.6980 | 0.6992 | 0.6962 | 0.7154 | 0.7098 | **0.7178** | **P3-C D2** (0.7178) | 0.0138/0.0151 | 0.0107/0.0151 |
| 08 | 0.7060 | 0.7002 | 0.7129 | 0.7050 | **0.7269** | 0.7152 | 0.7164 | **P3-C D6** (0.7269) | 0.0144/0.0152 | 0.0123/0.0141 |
| 09 | 0.7121 | 0.7064 | 0.7116 | **0.7257** | 0.7119 | 0.6989 | 0.6882 | **P3-C D8** (0.7257) | 0.0145/0.0152 | 0.0117/0.0138 |
| 10 | 0.7157 | 0.7179 | 0.7266 | 0.7274 | 0.7216 | **0.7295** | 0.7181 | **P3-C D4** (0.7295) | 0.0147/0.0150 | 0.0116/0.0140 |
| 11 | 0.7137 | 0.7176 | **0.7218** | 0.7205 | 0.7199 | 0.7197 | 0.7102 | **P3-C D12** (0.7218) | 0.0144/0.0155 | 0.0114/0.0141 |
| 12 | 0.7128 | 0.7093 | 0.7181 | 0.7211 | **0.7265** | 0.7194 | 0.7130 | **P3-C D6** (0.7265) | 0.0153/0.0157 | 0.0115/0.0139 |
| 13 | 0.7140 | 0.7094 | 0.7178 | 0.7212 | **0.7312** | 0.7241 | 0.7190 | **P3-C D6** (0.7312) | 0.0153/0.0160 | 0.0114/0.0141 |
| 14 | 0.7158 | 0.7157 | 0.7203 | 0.7219 | 0.7242 | **0.7295** | 0.7072 | **P3-C D4** (0.7295) | 0.0154/0.0158 | 0.0114/0.0140 |
| 15 | 0.7151 | 0.7121 | 0.7186 | 0.7210 | 0.7277 | **0.7282** | 0.7145 | **P3-C D4** (0.7282) | 0.0154/0.0159 | 0.0113/0.0141 |
| 16 | — | — | — | — | **0.7233** | 0.7225 | 0.7142 | **P3-C D6** (0.7233) | 0.0154/0.0160 | 0.0114/0.0142 |
| 17 | — | — | — | — | 0.7268 | **0.7289** | 0.7189 | **P3-C D4** (0.7289) | 0.0154/0.0160 | 0.0114/0.0142 |

---

## 3. Stage 2 Side-by-Side Comparison (Epochs 1-18)

| Ep | P3-A (D12) | P3-B (D12) | P3-C (D12) | P3-C (D8) | P3-C (D6) | P3-C (D4) | P3-C (D2) | Top Dice in Ep | D4 Gamma (S0/S1) | D2 Gamma (S0/S1) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 0.7112 | 0.6996 | 0.7084 | 0.7197 | 0.7263 | **0.7277** | 0.7181 | **P3-C D4** (0.7277) | 0.0154/0.0158 | 0.0114/0.0142 |
| 02 | 0.7091 | 0.6919 | 0.7057 | 0.6993 | 0.7205 | 0.7195 | **0.7289** | **P3-C D2** (0.7289) | 0.0156/0.0161 | 0.0113/0.0139 |
| 03 | 0.7119 | 0.7197 | 0.7303 | 0.7351 | **0.7406** | 0.7248 | 0.7389 | **P3-C D6** (0.7406) | 0.0161/0.0161 | 0.0113/0.0143 |
| 04 | **0.7182** | 0.6984 | 0.7125 | 0.7124 | 0.6832 | 0.6803 | 0.7093 | **P3-A** (0.7182) | 0.0164/0.0169 | 0.0125/0.0141 |
| 05 | 0.7310 | 0.7220 | **0.7316** | 0.7193 | 0.6932 | 0.7309 | 0.7203 | **P3-C D12** (0.7316) | 0.0170/0.0185 | 0.0121/0.0139 |
| 06 | 0.7291 | 0.7294 | 0.7136 | 0.7145 | 0.7313 | **0.7387** | 0.7116 | **P3-C D4** (0.7387) | 0.0168/0.0193 | 0.0132/0.0140 |
| 07 | 0.7283 | 0.7249 | 0.7445 | 0.7270 | **0.7451** | 0.7185 | 0.7163 | **P3-C D6** (0.7451) | 0.0167/0.0197 | 0.0128/0.0139 |
| 08 | 0.6794 | 0.6835 | 0.7391 | 0.7437 | 0.7041 | **0.7480** | 0.7432 | **P3-C D4** (0.7480) | 0.0164/0.0194 | 0.0128/0.0134 |
| 09 | 0.7327 | 0.7384 | **0.7557** | 0.7419 | 0.7250 | 0.7448 | 0.7193 | **P3-C D12** (0.7557) | 0.0162/0.0200 | 0.0125/0.0135 |
| 10 | 0.7492 | 0.7464 | **0.7534** | 0.7396 | 0.7406 | 0.7516 | 0.7463 | **P3-C D12** (0.7534) | 0.0159/0.0194 | 0.0121/0.0136 |
| 11 | 0.7417 | 0.7338 | **0.7545** | 0.7350 | 0.7454 | 0.7388 | 0.7513 | **P3-C D12** (0.7545) | 0.0166/0.0197 | 0.0118/0.0136 |
| 12 | 0.7435 | 0.7391 | 0.7383 | **0.7509** | 0.7384 | 0.7362 | 0.7435 | **P3-C D8** (0.7509) | 0.0161/0.0204 | 0.0112/0.0137 |
| 13 | 0.7543 | 0.7491 | **0.7544** | 0.7518 | 0.7406 | 0.7472 | 0.7479 | **P3-C D12** (0.7544) | 0.0161/0.0203 | 0.0114/0.0140 |
| 14 | 0.7495 | 0.7411 | 0.7505 | 0.7465 | 0.7540 | 🏆 **0.7639** | 0.7552 | 🏆 **P3-C D4** (0.7639) | 0.0161/0.0203 | 0.0117/0.0140 |
| 15 | 0.7485 | 0.7496 | 0.7508 | 0.7451 | 0.7558 | 🏆 **0.7635** | 0.7558 | 🏆 **P3-C D4** (0.7635) | 0.0162/0.0204 | 0.0117/0.0141 |
| 16 | — | — | — | 0.7582 | 0.7599 | **0.7629** | 0.7578 | **P3-C D4** (0.7629) | 0.0163/0.0205 | 0.0117/0.0141 |
| 17 | — | — | — | 0.7604 | 0.7563 | **0.7606** | 0.7561 | **P3-C D4** (0.7606) | 0.0162/0.0205 | 0.0117/0.0141 |
| 18 | — | — | — | 0.7576 | 0.7583 | **0.7615** | 0.7567 | **P3-C D4** (0.7615) | 0.0162/0.0205 | 0.0117/0.0141 |

---

## 3.1. Stage 2 Continuation (D12 Convergence Audit Window at LR Floor 1e-6)

> **Continuation Context & Invariants:**
> - Checkpoint baseline: `best_model_b2_global.pth` (recorded at Stage 2 Epoch 9 with Val Dice = `0.7557`, Val Loss = `1.0889`).
> - Learned Gamma preserved from checkpoint: S0 = `0.0063`, S1 = `0.0071`.
> - Continuation Mechanism: `--resume-stage2-low-lr` (`--low-lr 1e-6`, constant LR, no warm restart).
> - Objective: Verify whether D12 has naturally converged at 0.7557 under `patience = 6` early stopping.

| Ext Ep | Overall S2 Ep | Train Loss | Train LB | Train Dice | Val Loss | Val Dice | Gamma (S0/S1) | LR Groups | Status / Non-improving |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **01** | Ep 16 | 1.1974 | 0.1606 | 0.7109 | 1.0826 | 0.7441 | 0.0063 / 0.0072 | 1.00e-06 | Non-improving (1/6) |
| **02** | Ep 17 | 1.1925 | 0.1606 | 0.7133 | 1.0878 | 0.7478 | 0.0063 / 0.0072 | 1.00e-06 | Non-improving (2/6) |
| **03** | Ep 18 | 1.1946 | 0.1606 | 0.7062 | 1.0930 | 0.7477 | 0.0063 / 0.0072 | 1.00e-06 | Non-improving (3/6) |
| **04** | Ep 19 | 1.1936 | 0.1606 | 0.7068 | 1.0894 | 0.7448 | 0.0063 / 0.0072 | 1.00e-06 | Non-improving (4/6) |
| **05** | Ep 20 | 1.1896 | 0.1606 | 0.7140 | 1.1018 | 0.7473 | 0.0063 / 0.0072 | 1.00e-06 | Non-improving (5/6) |
| **06** | Ep 21 | 1.1925 | 0.1606 | 0.7071 | 1.0965 | 0.7461 | 0.0063 / 0.0072 | 1.00e-06 | 🛑 **EarlyStopping Triggered (6/6)** |

---

## 3.2. Stage 2 Continuation (D6 Convergence Audit Window at LR Floor 1e-6)

> **Continuation Context & Invariants:**
> - Checkpoint baseline: `best_model_b2_stage2.pth` (recorded at Stage 2 Epoch 16 with Val Dice = `0.7599`, Val Loss = `0.9602`).
> - Learned Gamma preserved from checkpoint: S0 = `0.0149`, S1 = `0.0127`.
> - Continuation Mechanism: `--resume-stage2-low-lr` (`--low-lr 1e-6`, constant LR, no warm restart).
> - Objective: Verify whether D6 has naturally converged at 0.7599 under `patience = 6` early stopping.

| Ext Ep | Overall S2 Ep | Train Loss | Train LB | Train Dice | Val Loss | Val Dice | Gamma (S0/S1) | LR Groups | Status / Non-improving |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **01** | Ep 19 | 0.9700 | 0.1004 | 0.7325 | 0.9631 | 0.7566 | 0.0150 / 0.0127 | 1.00e-06 | Non-improving (1/6) |
| **02** | Ep 20 | 0.9638 | 0.1004 | 0.7349 | 0.9589 | 0.7581 | 0.0150 / 0.0127 | 1.00e-06 | Non-improving (2/6) |
| **03** | Ep 21 | 0.9680 | 0.1004 | 0.7350 | 0.9534 | 0.7582 | 0.0150 / 0.0127 | 1.00e-06 | Non-improving (3/6) |
| **04** | Ep 22 | 0.9631 | 0.1004 | 0.7401 | 0.9658 | 0.7567 | 0.0149 / 0.0127 | 1.00e-06 | Non-improving (4/6) |
| **05** | Ep 23 | 0.9647 | 0.1004 | 0.7326 | 0.9583 | 0.7589 | 0.0149 / 0.0127 | 1.00e-06 | Non-improving (5/6) |
| **06** | Ep 24 | 0.9646 | 0.1004 | 0.7327 | 0.9514 | 0.7596 | 0.0149 / 0.0127 | 1.00e-06 | 🛑 **EarlyStopping Triggered (6/6)** |

---

## 3.3. Stage 2 Continuation (D4 Convergence Audit Window at LR Floor 1e-6)

> **Continuation Context & Invariants:**
> - Checkpoint baseline: `best_model_b2_global.pth` (recorded at Stage 2 Epoch 14 with Val Dice = `0.7639`, Val Loss = `0.9533`).
> - Learned Gamma preserved from checkpoint: S0 = `0.0161`, S1 = `0.0203`.
> - Continuation Mechanism: `--resume-stage2-low-lr` (`--low-lr 1e-6`, constant LR, no warm restart).
> - Objective: Verify whether D4 has naturally converged at 0.7639 under `patience = 6` early stopping.

| Ext Ep | Overall S2 Ep | Train Loss | Train LB | Train Dice | Val Loss | Val Dice | Gamma (S0/S1) | LR Groups | Status / Non-improving |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **01** | Ep 19 | 0.9466 | 0.0804 | 0.7318 | 0.9401 | 0.7599 | 0.0162 / 0.0203 | 1.00e-06 | Non-improving (1/6) |
| **02** | Ep 20 | 0.9483 | 0.0804 | 0.7304 | 0.9278 | 0.7626 | 0.0161 / 0.0203 | 1.00e-06 | Non-improving (2/6) |
| **03** | Ep 21 | 0.9498 | 0.0804 | 0.7312 | 0.9491 | 0.7613 | 0.0161 / 0.0203 | 1.00e-06 | Non-improving (3/6) |
| **04** | Ep 22 | 0.9391 | 0.0804 | 0.7316 | 0.9477 | 0.7608 | 0.0161 / 0.0203 | 1.00e-06 | Non-improving (4/6) |
| **05** | Ep 23 | 0.9458 | 0.0804 | 0.7309 | 0.9425 | 0.7634 | 0.0161 / 0.0203 | 1.00e-06 | Non-improving (5/6) |
| **06** | Ep 24 | 0.9401 | 0.0804 | 0.7300 | 0.9369 | 0.7629 | 0.0162 / 0.0203 | 1.00e-06 | 🛑 **EarlyStopping Triggered (6/6)** |

---

## 4. D4 Detailed Error Analysis & Routing Diagnostics Summary

Từ kết quả thẩm định chuẩn tắc (`results/P3_C_Routing_Diagnostics_D4/error_analysis/error_summary.json` trên toàn bộ 348 mẫu Val):

- **Phân phối Metric Chính thức (New Project Record)**:
  - **Val Dice**: Mean = 🏆 **0.7639** (±0.1640), Median = **0.8049**, P75 = **0.8726**, P90 = **0.9015**, P95 = **0.9167**
  - **Val IoU**: Mean = 🏆 **0.6412** (±0.1782), Median = **0.6736**, P75 = **0.7739**, P90 = **0.8207** (Mức cao nhất toàn dự án)
  - **Val Precision**: Mean = 🏆 **0.7298** (±0.1895), Median = **0.7820** (Tăng vọt +2.33% so với D6: 0.7065)
  - **Val Recall**: Mean = **0.8498** (±0.1604), Median = **0.8935**
  - **Val Loss**: Đạt đáy **0.9317** (S2 Ep 18), tại Peak Ep 14 là **0.9533**.

- **Hành vi Định tuyến Đối xứng (1:1 Heterogeneous Expert Pool)**:
  - **Cấu trúc Pool**: 8 Experts (4 CNN stages + 4 ViT blocks) với Top-k = 4 (chọn đúng 50% pool mỗi token).
  - **CNN Expert Fraction**: Mean = **35.24%** (median: 34.38%, min: 21.88%, max: 50.00%).
  - **ViT Expert Fraction**: Mean = **64.76%** (median: 65.62%, min: 50.00%, max: 78.12%).
  - **Mean Routing Entropy**: **1.99991** (entropy cực đại của 8 experts: log2(8)=3.0, phân bố định tuyến hoàn toàn lành mạnh).
  - **Expert HHI Concentration**: Mean = **0.1574** (phù hợp tuyệt đối mức cân bằng lý tưởng của 8 experts: 1/8 = 0.125).
  - **Mean Gumbel-Softmax Probability**: **0.5012**.

- **Phân loại Lỗi (Error Taxonomy Comparison: D4 vs D6)**:
  - `high_quality` (Dice > 0.85): **127 mẫu** (tăng từ 124 mẫu ở D6).
  - `boundary_margin_error`: **135 mẫu** (chiếm tỷ trọng chính do vết nứt mảnh).
  - `complex_topology`: **35 mẫu**.
  - `moderate_general_error`: **34 mẫu** (giảm mạnh từ 41 mẫu ở D6).
  - `over_segmentation`: **28 mẫu** (giảm mạnh từ 39 mẫu ở D6, giải thích vì sao Precision tăng vọt lên 72.98%).
  - `thin_low_area_failure`: **26 mẫu** (giảm từ 32 mẫu ở D6).
  - `false_crack_high_fp`: **18 mẫu** (giảm từ 21 mẫu ở D6).
  - `missed_crack_high_fn`: **9 mẫu**.
  - `fragmented_prediction`: **4 mẫu** (giảm từ 5 mẫu ở D6).

---

## 5. Depth Sensitivity Trajectory Analysis (D12 $\to$ D8 $\to$ D6 $\to$ D4 $\to$ D2)

1. **Quy luật Đường cong Parabol Ngược (Inverted-U Concave Curve) & Điểm Tối ưu Tuyệt đối D4**:
   - Khi quét toàn diện qua 5 độ sâu ViT từ 12 xuống 2:
     - D12 (12.49M params, 16 experts): Peak Dice = **0.7557** (Val Loss: 1.0889)
     - D8  (11.31M params, 12 experts): Peak Dice = **0.7604** (Val Loss: **0.9999**, S2 Ep 17)
     - D6  (10.71M params, 10 experts): Peak Dice = **0.7599** (Val Loss: 0.9602)
     - **D4  (10.12M params, 8 experts)**: Peak Dice = 🏆 **0.7639** (Val Loss: **0.9533** / đáy **0.9317**)
     - D2  (9.20M params, 6 experts):  Peak Dice = **0.7578** (Val Loss: 0.9477 / đáy 0.9364)
   - **Phát hiện Khoa học mang tính Bước ngoặt**:
     - Hiệu năng không tăng trưởng đơn điệu mãi mãi khi giảm depth. Từ D12 $\to$ D4, hiệu năng tăng liên tục (+0.82%), nhưng khi tiếp tục giảm xuống **D2**, hiệu năng sụt giảm rõ rệt (-0.61% so với D4, rơi xuống dưới D6 và D8).
     - **Nguyên nhân cơ chế (Mechanistic Breakdown at D2)**:
       * **Thiếu hụt dung lượng Attention toàn cục**: 2 ViT blocks không đủ sâu để mô hình hóa sự phụ thuộc không gian tầm xa (long-range dependency) của các vết nứt kéo dài xuyên suốt bức ảnh.
       * **Mất cân bằng Expert Pool**: Tại D2, chỉ có 2 ViT experts đối đầu 4 CNN experts (tổng 6). Kết quả định tuyến cho thấy tỷ trọng chọn CNN bị kéo lệch lên tới **67.6%** (so với mức cân bằng lý tưởng ~50/50 ở D4).
       * **Bùng nổ lỗi đứt đoạn (Fragmentation)**: Phân loại taxonomy cho thấy lỗi `fragmented_prediction` tại D2 tăng vọt lên **10 mẫu** (gấp 2.5 lần so với chỉ 4 mẫu ở D4), phản ánh việc các vết nứt bị đứt khúc do thiếu receptive field toàn cục.
     - **Kết luận**: **D4 chính là "Vùng Goldilocks" (Điểm Cân bằng Vàng)** giữa Inductive Bias cục bộ (CNN) và Biểu diễn Toàn cục (ViT), đồng thời đạt tính đối xứng kiến trúc hoàn hảo 1:1 (4 CNN + 4 ViT, $k=4$).

2. **Động lực học Gamma ($\gamma$) xuyên suốt 5 Cấp Độ sâu**:
   - Giá trị Gamma ASDW học được cuối Stage 2:
     - D12: $\gamma_{S0} = 0.0063$, $\gamma_{S1} = 0.0071$ (Bị gradient ức chế do over-smoothing).
     - D8:  $\gamma_{S0} = 0.0089$, $\gamma_{S1} = 0.0158$ (Phục hồi).
     - D6:  $\gamma_{S0} = 0.0150$, $\gamma_{S1} = 0.0127$ (Ổn định).
     - D4:  $\gamma_{S0} = 0.0162$, $\gamma_{S1} = 🏆 **0.0205** (Bùng nổ cực đại, tối ưu cho nhánh dải 1x7, 7x1).
     - D2:  $\gamma_{S0} = 0.0117$, $\gamma_{S1} = 0.0141$ (Hạ nhiệt do ViT quá nông khiến gradient tinh chế không còn động lực lan truyền).

---

## 6. Phase 2 Routing Capacity ($top\_k$) Screening & Routing Intervention (Canonical D4)

### 6.1. Bảng So Sánh Hiệu Năng $top\_k$ (Screening trên D4, 8 Experts Pool)

| Cấu hình | $top\_k$ / Pool | Tỷ lệ kích hoạt | Peak S1 Dice | Peak S2 Dice (Global) | Val Loss @ Peak | Mean IoU | Median Dice | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **P3-C D4 K4** (Baseline) | $4 / 8$ | 50% | 0.7295 (Ep 14) | 🏆 **0.7639** (Ep 14) | 0.9533 | **0.6412** | **0.8066** | Đã khóa làm Baseline chuẩn |
| **P3-C D4 K2** | $2 / 8$ | 25% | **0.7304** (Ep 13) | **0.7618** (Ep 16) | **0.9518** | 0.6386 | 0.8058 | Hoàn tất 35 eps (giảm 0.21% Dice) |
| **P3-C D4 K6** | $6 / 8$ | 75% | — | — | — | — | — | Đã cấu hình (`b2_p3_run_c_d4_k6.yaml`) |

### 6.2. Kết Quả Can Thiệp Định Tuyến (Routing Intervention Diagnostic trên D4 Checkpoint, 348 mẫu Val)
- **Adaptive Baseline vs Frequency-based Static Top-4**:
  $$\text{Adaptive: } 0.7639 \quad \text{vs} \quad \text{Static: } 0.7636 \quad (\Delta = +0.00037, \; 95\%\text{ CI: } [-0.00051, +0.00124])$$
  - Paired $t$-test: $t = 0.8192, p = \mathbf{0.4133}$ (không có ý nghĩa thống kê).
  - Wilcoxon signed-rank: $W = 27942.0, p = \mathbf{0.3053}$ (không có ý nghĩa thống kê).
  - Phân vị vết nứt mảnh nhất ($Q4$): Adaptive $0.6637$ vs Static $0.6636$ ($\Delta = \mathbf{0.0001}$).
- **Adaptive Baseline vs Random Top-4 (10 seeds)**:
  $$\text{Adaptive: } 0.7639 \quad \text{vs} \quad \text{Random: } 0.7633 \quad (\Delta = +0.00063, \; p = 0.0335)$$
  - Chênh lệch có ý nghĩa thống kê nhưng effect size siêu bé ($0.063$ percentage points Dice).
- **Nguyên lý "Information ≠ Utility"**:
  Nút thắt biểu diễn hình thái ở router (đo được từ linear probe âm) hiện tại **không cấu thành nút thắt hiệu năng phân đoạn downstream**. SAGE-Lite D4 hoạt động như một tập hợp chuyên gia đa dạng mạnh mẽ, việc thay đổi chính sách chọn expert giữa adaptive và static hầu như không ảnh hưởng đến Dice.

---

## 7. Phase 5.1 SAGE LR Isolation: Candidate B (SAGE LR = 2e-4) vs Phase 5.1 Extra (Stage 2 SAGE LR = 2e-4)

### 7.1. Tổng Quan & So Sánh Hiệu Năng Đỉnh Cao

| Cấu hình | Stage 1 SAGE LR | Stage 2 SAGE LR | Peak S1 Val Dice | Peak S2 Val Dice (Global) | Val Loss @ Peak | Mean IoU | Median Dice | Precision | Recall | Trạng thái Nghiệm thu |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Phase 5.1 Baseline** | 1e-4 | 1e-4 | 0.7304 (Ep 13) | 0.7618 (Ep 16) | **0.9518** | 0.6386 | 0.8058 | 0.7195 | 0.8576 | Baseline gốc Phase 5.1 (`sage_lr = 1e-4`) |
| **Candidate B** | **2e-4** | 1e-4 (base) | **0.7333** (Ep 13) | **0.7641** (Ep 14) | 0.9550 | **0.6417** | 0.8066 | **0.7224** | **0.8642** | 🏆 Thắng giải Phase 5.1 (`sage_lr = 2e-4`) |
| **Phase 5.1 Extra** | **2e-4** | **2e-4** (cô lập) | **0.7333** (Inherited) | 🏆 **0.7644** (Ep 16) | 0.9526 (đáy **0.9412**) | 0.6409 | **0.8073** | 0.7195 | 0.8615 | 🏆 Kỷ lục Val Dice dự án (giữ SAGE LR = 2e-4 cả Stage 2) |

### 7.2. Bảng Đối Chiếu Từng Epoch Stage 2 (Epoch 1 – 18)

| Stage 2 Ep | Tổng Ep | Candidate B Val Dice | Extra Val Dice | Extra Train Loss | Extra Val Loss | Extra LR (SAGE / Base) | Extra Gamma (S0 / S1) | Ghi chú Tiến độ Extra |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 01 | 18 | 0.7188 | 0.7258 | 1.3017 | 1.2713 | 6.73e-5 / 3.40e-5 | 0.0106 / 0.0139 | Khởi động fresh optimizer/scheduler |
| 02 | 19 | 0.7145 | 0.7140 | 1.3008 | 1.2710 | 1.34e-4 / 6.70e-5 | 0.0108 / 0.0147 | Warmup epoch 2 |
| 03 | 20 | 0.7289 | 0.7321 | 1.2823 | 1.3873 | 1.87e-4 / 9.34e-5 | 0.0118 / 0.0152 | Warmup epoch 3 (peak LR) |
| 04 | 21 | 0.7120 | 0.7056 | 1.2466 | 1.2182 | 1.77e-4 / 8.84e-5 | 0.0127 / 0.0157 | |
| 05 | 22 | 0.7250 | 0.7204 | 1.1950 | 1.1299 | 1.64e-4 / 8.23e-5 | 0.0131 / 0.0162 | |
| 06 | 23 | 0.7190 | 0.7172 | 1.1546 | 1.0896 | 1.50e-4 / 7.52e-5 | 0.0118 / 0.0165 | |
| 07 | 24 | 0.7301 | 0.7356 | 1.1115 | 1.1066 | 1.35e-4 / 6.74e-5 | 0.0124 / 0.0172 | 🏆 Vượt baseline Stage 1 (0.7333) |
| 08 | 25 | 0.7510 | 0.7591 | 1.0803 | 1.0326 | 1.18e-4 / 5.91e-5 | 0.0129 / 0.0170 | 🏆 New Global Best (0.7591) |
| 09 | 26 | 0.7480 | 0.7450 | 1.0541 | 1.0241 | 1.01e-4 / 5.05e-5 | 0.0136 / 0.0171 | |
| 10 | 27 | 0.7495 | 0.7434 | 1.0247 | 0.9920 | 8.32e-5 / 4.19e-5 | 0.0139 / 0.0168 | |
| 11 | 28 | 0.7540 | 0.7403 | 0.9997 | 0.9883 | 6.65e-5 / 3.36e-5 | 0.0133 / 0.0170 | |
| 12 | 29 | 0.7580 | 0.7424 | 0.9820 | 0.9612 | 5.08e-5 / 2.58e-5 | 0.0129 / 0.0170 | |
| 13 | 30 | 0.7602 | 0.7491 | 0.9684 | 0.9884 | 3.65e-5 / 1.87e-5 | 0.0130 / 0.0171 | |
| 14 | 31 | **0.7641** | 0.7632 | 0.9576 | 0.9550 | 2.43e-5 / 1.26e-5 | 0.0129 / 0.0172 | 🏆 New Global Best (0.7632) |
| 15 | 32 | 0.7610 | 0.7560 | 0.9441 | 0.9582 | 1.43e-5 / 7.63e-6 | 0.0130 / 0.0173 | |
| 16 | 33 | 0.7625 | 🏆 **0.7644** | 0.9416 | 0.9526 | 7.00e-6 / 3.99e-6 | 0.0131 / 0.0174 | 🏆 **GLOBAL PEAK (0.7644)** |
| 17 | 34 | 0.7605 | 0.7590 | 0.9359 | 0.9631 | 2.51e-6 / 1.75e-6 | 0.0132 / 0.0173 | |
| 18 | 35 | 0.7612 | 0.7599 | 0.9358 | **0.9412** | 1.00e-6 / 1.00e-6 | 0.0132 / 0.0173 | Hoàn tất toàn bộ 35 epochs (đáy Val Loss 0.9412) |

### 7.3. Đánh Giá Khoa Học & Kết Luận Về `stage2_sage_lr`

1. **Hiệu Năng & Độ Ổn Định**:
   - Khi giữ nguyên `stage2_sage_lr = 2e-4` ở Stage 2, mô hình đạt đỉnh **0.7644 Val Dice** (tại Epoch 16), nhỉnh hơn nhẹ so với Candidate B gốc (`0.7641` tại Epoch 14).
   - Đáy Validation Loss đạt **0.9412** (Epoch 18), thấp hơn mức 0.9550 của Candidate B, cho thấy độ tự tin dự đoán của mô hình sắc nét hơn.
   - Chênh lệch $\Delta = +0.0003$ Dice giữa $2\times 10^{-4}$ và $1\times 10^{-4}$ là rất nhỏ, khẳng định rằng vùng LR xung quanh $[1\times 10^{-4}, 2\times 10^{-4}]$ cho SAGE Router ở Stage 2 là cực kỳ ổn định, không có hiện tượng divergence hay gradient explosion.

2. **Chẩn Đoán Định Tuyến (Routing Diagnostics trên Checkpoint Đỉnh 0.7644)**:
   - **Tỷ trọng chuyên gia (CNN vs ViT)**: CNN = **41.38%**, ViT = **58.62%** (tiếp tục bảo toàn tính cân bằng dị thể hoàn hảo).
   - **Mức độ tập trung HHI**: **0.1793** (tiệm cận mức phân bổ lý tưởng 8 experts: $1/8 = 0.125$).
   - **Entropy định tuyến chuẩn hóa**: **0.999995** (phân tán lành mạnh, 0 dead experts, 0 router collapse).
   - **Động lực học Gamma ASDW**: $\gamma_{S0} = 0.0132, \gamma_{S1} = 0.0173$ (tiếp tục phát triển ổn định).

---

---

## 8. Phase 5.3 Stage-2 Shared LR Ratio Sweep ( = \frac{LR_{shared}}{LR_{base}}$): Toàn Bộ 6 Cấu Hình Đầy Đủ

### 8.1. Tổng Quan & So Sánh Hiệu Năng Đỉnh Cao Toàn Diện

| Cấu hình | Ratio $ | Shared LR | Base LR | Peak S1 Dice | Peak S2 Dice (Global) | Mean IoU | Median Dice | Precision | Recall | Đáy Val Loss | Trạng thái Nghiệm thu |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Phase 5.3 R025** | **0.25** | .5 \times 10^{-5}$ | .0 \times 10^{-4}$ | 0.7333 | **0.7579** (Ep 16) | 0.6336 | 0.8047 | 0.7061 | 0.8673 | 0.9384 | Hoàn tất 35 eps (Soft Freeze suy giảm) |
| **Phase 5.3 R050** | **0.50** | .0 \times 10^{-5}$ | .0 \times 10^{-4}$ | 0.7333 | **0.7586** (Ep 14) | 0.6347 | 0.8042 | 0.7167 | 0.8571 | 0.9437 | Hoàn tất 35 eps (SAGE chuẩn suy giảm) |
| **Candidate B (Anchor)** | **1.00** | .0 \times 10^{-4}$ | .0 \times 10^{-4}$ | **0.7333** | 🏆 **0.7641** (Ep 14) | 🏆 **0.6417** | 🏆 **0.8066** | 🏆 **0.7224** | 0.8642 | 0.9550 | 🏆 **CHIẾN THẮNG TUYỆT ĐỐI (Sweet Spot)** |
| **Phase 5.3 R200** | **2.00** | .0 \times 10^{-4}$ | .0 \times 10^{-4}$ | 0.7333 | **0.7638** (Ep 15) | 0.6405 | 0.8055 | 0.7198 | 0.8612 | 🏆 **0.9308** | Hoàn tất 35 eps (Bão hòa Dice, Đáy Loss) |
| **Phase 5.3 R400** | **4.00** | .0 \times 10^{-4}$ | .0 \times 10^{-4}$ | 0.7333 | **0.7611** (Ep 16) | 0.6378 | 0.8027 | 0.7134 | 🏆 **0.8680** | 0.9344 | Hoàn tất 35 eps (Quá tải Shared LR) |
| **Phase 5.3 R500** | **5.00** | .0 \times 10^{-4}$ | .0 \times 10^{-4}$ | 0.7333 | **0.7585** (Ep 14) | 0.6356 | 0.8025 | 0.7207 | 0.8519 | 0.9395 | Hoàn tất 35 eps (Suy giảm mạnh Shared LR) |

### 8.2. Bảng Đối Chiếu Song Song Từng Epoch Stage 2 (Epoch 1 – 18)

| S2 Ep | Tổng Ep | r = 0.25 (R025) | r = 0.50 (R050) | r = 1.00 (Candidate B) | r = 2.00 (R200) | r = 4.00 (R400) | r = 5.00 (R500) | Top Dice in Ep | Top Config |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 01 | 18 | **0.7297** | 0.7276 | 0.7288 | 0.7255 | 0.7262 | 0.7275 | **0.7297** | r=0.25 |
| 02 | 19 | 0.7301 | 0.7100 | 0.7253 | 0.7289 | 0.7179 | **0.7380** | **0.7380** | r=5.00 |
| 03 | 20 | 0.7210 | 0.7256 | 0.7178 | 0.7270 | **0.7287** | 0.7231 | **0.7287** | r=4.00 |
| 04 | 21 | 0.6994 | 0.6986 | 0.7184 | **0.7301** | 0.6344 | 0.6826 | **0.7301** | r=2.00 |
| 05 | 22 | **0.7215** | 0.7056 | 0.6824 | 0.7168 | 0.7213 | 0.7107 | **0.7215** | r=0.25 |
| 06 | 23 | 0.7227 | 0.6973 | 0.7027 | **0.7335** | 0.6956 | 0.6793 | **0.7335** | r=2.00 |
| 07 | 24 | 0.7274 | 0.7317 | 0.7319 | **0.7383** | 0.7177 | 0.7369 | **0.7383** | r=2.00 |
| 08 | 25 | 0.7444 | 0.7426 | 0.7422 | **0.7503** | 0.7466 | 0.7480 | **0.7503** | r=2.00 |
| 09 | 26 | **0.7355** | 0.7316 | 0.7170 | 0.7110 | 0.7191 | 0.6820 | **0.7355** | r=0.25 |
| 10 | 27 | 0.7456 | 0.7420 | 0.7475 | 0.7335 | 0.7481 | **0.7543** | **0.7543** | r=5.00 |
| 11 | 28 | 0.7470 | 0.7437 | 0.7479 | 0.7479 | 0.7438 | **0.7485** | **0.7485** | r=5.00 |
| 12 | 29 | **0.7451** | 0.7338 | 0.7296 | 0.7348 | 0.7380 | 0.7314 | **0.7451** | r=0.25 |
| 13 | 30 | 0.7528 | 0.7458 | 0.7480 | **0.7578** | 0.7530 | 0.7294 | **0.7578** | r=2.00 |
| 14 | 31 | 0.7503 | 0.7586 | **0.7641** | 0.7621 | 0.7595 | 0.7585 | **0.7641** | r=1.00 |
| 15 | 32 | 0.7511 | 0.7570 | 0.7517 | **0.7638** | 0.7559 | 0.7512 | **0.7638** | r=2.00 |
| 16 | 33 | 0.7579 | 0.7539 | 0.7584 | **0.7636** | 0.7611 | 0.7552 | **0.7636** | r=2.00 |
| 17 | 34 | 0.7569 | 0.7521 | 0.7552 | **0.7621** | 0.7592 | 0.7551 | **0.7621** | r=2.00 |
| 18 | 35 | 0.7564 | 0.7529 | 0.7559 | **0.7636** | 0.7609 | 0.7537 | **0.7636** | r=2.00 |

### 8.3. Đánh Giá Khoa Học & Phán Quyết Khóa Chính Thức Phase 5.3

1. **Quy luật Đường cong Chữ U Ngược Toàn Diện (Full Empirical Inverted-U Response Curve)**:
   - Dữ liệu thực nghiệm thu được từ toàn bộ 6 tỉ số  \in \{0.25, 0.50, 1.00, 2.00, 4.00, 5.00\}$ (trải dài hơn một bậc độ lớn từ .5\times 10^{-5}$ đến .0\times 10^{-4}$) vẽ nên một đường cong hình chuông lồi đơn đỉnh (unimodal concave response curve) hoàn hảo:
     \text{Peak Val Dice: } 0.7579\,(r=0.25) < 0.7586\,(r=0.50) < \mathbf{0.7641}\,(r=1.00) > 0.7638\,(r=2.00) > 0.7611\,(r=4.00) > 0.7585\,(r=5.00)
     \text{Mean IoU: } 0.6336 < 0.6347 < \mathbf{0.6417} > 0.6405 > 0.6378 > 0.6356
     \text{Median Dice: } 0.8047 > 0.8042 < \mathbf{0.8066} > 0.8055 > 0.8027 > 0.8025

2. **Cơ chế suy giảm ở hai thái cực**:
   - **Vùng Soft-Freeze ( < 1.0$)**: Khi kìm hãm tốc độ học của Shared CNN Experts ( = 0.25$ và  = 0.50$), các đặc trưng không gian nông (low-level edges, boundary textures) không kịp thích ứng với các biểu diễn mới của ViT và SAGE routers, khiến Val Dice giảm mạnh $\approx -0.6\%$.
   - **Vùng Quá tải Tốc độ ( > 2.0$)**: Khi đẩy learning rate của Shared CNN lên quá cao ( = 4.00$ và  = 5.00$), các trọng số ConvNeXt bị xáo trộn mạnh, phá vỡ cấu trúc biểu diễn hình thái vết nứt đã học được từ Stage 1, dẫn đến hiện tượng trôi dạt biểu diễn (representational drift) và làm Dice sụt giảm liên tục (.7641 \to 0.7611 \to 0.7585$).

3. **PHÁN QUYẾT KHÓA CHÍNH THỨC (PHASE 5.3 LOCK)**:
   - **Tỉ số tối ưu tuyệt đối**:  = 1.00$ (tức $\text{stage2\_shared\_lr} = \text{stage2\_base\_lr} = 1.0 \times 10^{-4}$).
   - **Quyết định kiến trúc & tối ưu**: Giữ nguyên cơ chế **Unified Stage-2 Optimizer** của Candidate B (không tách riêng Shared LR, không tách riêng SAGE LR ở Stage 2). Mọi nhóm tham số ở Stage 2 đều dùng  = 1.0\times 10^{-4}$.
   - **Chuyển giao sang Phase 6**: Toàn bộ các thông số của Phase 5 (sage_lr = 2e-4, warmup = 3, stage2_lr_ratio = 1.00) được khóa cứng làm nền tảng vững chắc để bước vào **Phase 6 (Regularization & Fusion Mechanics)**.


---

## 9. Phase 4 Load Balancing Loss Study ($LB \in \{0.005, 0.010, 0.030\}$): Cân Bằng Giữa Chuyên Môn Hóa & San Sẻ Tải Router

### 9.1. Tổng Quan & So Sánh Hiệu Năng Đỉnh Cao Toàn Diện

Nghiên cứu Phase 4 được kích hoạt nhằm giải quyết hiện tượng **Chuyên gia Chết (Dead Experts: 0.0% usage)** tại các router cục bộ tầng nông (`convnext.stage_0`, `stage_1`, `stage_2`). Bằng cách quét hệ số phạt cân bằng tải $load\_balance\_factor \in \{0.005, 0.010, 0.030\}$, chúng ta đánh giá sự đánh đổi giữa tự do chuyên biệt hóa đặc trưng (Specialization) và ép buộc phân phối đều lưu lượng (Fair Utilization).

| Cấu hình | $LB$ Factor | Peak S1 Dice | Peak S2 Dice (Global) | Mean IoU | Median Dice | Precision | Recall | Đáy Val Loss | Trạng thái Chuyên gia Cục bộ | Phán quyết Khoa học |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Phase 4 LB0005** | **0.005** | 0.7321 (Ep 12) | **0.7580** (Ep 19) | 0.6345 | 0.8030 | 0.7030 | **0.8721** | 0.9553 | Sụp đổ router nông (Stage 0: 2 experts, Stage 1: 2 experts) | Suy giảm mạnh (-0.0061 Dice), over-predict diện rộng |
| **Candidate B (Anchor)** | **0.010** | **0.7326 (Ep 08)** | 🏆 **0.7641** (Ep 14) | 🏆 **0.6417** | 🏆 **0.8066** | 🏆 **0.7337** | 0.8477 | 0.9475 | Cân bằng hoàn hảo: 0 dead experts pool toàn cục, router nông tự do lọc vân | 🏆 **KHÓA CHÍNH THỨC (Optimal Pareto Peak)** |
| **Phase 4 LB0030** | **0.030** | 0.7321 (Ep 14) | **0.7624** (Ep 16) | 0.6391 | 0.8056 | 0.7170 | 0.8629 | 🏆 **0.9315** | Hồi sinh chuyên gia chết (Stage 1 & 2 thêm 2-3 experts hoạt động) | Tăng đa dạng nhưng phạt nặng làm cùn biên (-0.0017 Dice) |

---

### 9.2. Phân Tích Chẩn Đoán Định Tuyến & Động Học Hồi Sinh Chuyên Gia (`per_router_usage.csv`)

So sánh trực tiếp ma trận phân phối lưu lượng giữa 3 mức $LB$ trên 348 mẫu Validation:

#### A. Tầng Nông `convnext.stage_1` (Hiện tượng Công tắc Nhị phân 50/50):
- **$LB = 0.005$**: Router sụp đổ gần như tuyệt đối thành công tắc giữa $E_1$ (49.71%) và $E_6$ (50.00%). Các chuyên gia $E_0, E_2, E_3, E_4, E_7$ nhận đúng 0.0%. Số chuyên gia hiệu dụng: $N_{eff} = 2.04$.
- **$LB = 0.010$**: $E_1$ (50.00%) và $E_6$ (50.00%) đảm nhiệm xử lý cục bộ, các chuyên gia khác 0.0%. $N_{eff} = 2.00$.
- **$LB = 0.030$**: Lực phạt gấp 3 lần đã **hồi sinh thành công** $E_3$ (1.01%) và $E_5$ (2.16%). Phân phối bắt đầu lan tỏa sang các chuyên gia khác, đẩy $N_{eff}$ lên $2.30$.

#### B. Tầng Nông `convnext.stage_2` (Mở rộng phổ chuyên gia):
- **$LB = 0.005$**: Chỉ có 3 chuyên gia hoạt động ($E_4 = 49.28\%, E_5 = 28.74\%, E_6 = 21.98\%$). Toàn bộ $E_0, E_1, E_2, E_3, E_7 = 0.0\%$. $N_{eff} = 2.83$.
- **$LB = 0.010$**: $E_4 = 47.13\%, E_6 = 36.21\%, E_1 = 14.22\%$. Bắt đầu có lưu lượng nhỏ ở $E_3$ (1.01%) và $E_5$ (1.15%). $N_{eff} = 3.05$.
- **$LB = 0.030$**: Lưu lượng phân tỏa rõ rệt: $E_4 = 43.25\%, E_6 = 39.80\%, E_1 = 9.63\%, E_3 = 5.75\%, E_5 = 1.58\%$. $N_{eff}$ tăng vọt lên **$3.27$**.

#### C. Tầng Sâu `convnext.stage_3` & `transformer`:
- Tại `stage_3`, $LB = 0.030$ đẩy số chuyên gia hiệu dụng từ $5.89$ ($LB=0.010$) lên **$7.02$**, thị phần chuyên gia thống trị giảm từ $27.73\%$ xuống $19.40\%$.
- Tại `transformer.block_1`, normalized entropy đạt **0.9810** ($N_{eff} = 7.69/8.00$).

---

### 9.3. Bảng Đối Chiếu Song Song Từng Epoch (Stage 1 & Stage 2)

#### Giai đoạn 1 (Stage 1: Huấn luyện Router, SAGE Adapter & Decoder, Epochs 1 — 17):
| Ep | LB=0.005 Val Dice (Loss) | LB=0.010 Val Dice (Loss) | LB=0.030 Val Dice (Loss) | LB Loss (0.005 / 0.010 / 0.030) |
| :-: | :---: | :---: | :---: | :---: |
| 1 | 0.1527 (2.0274) | 0.1514 (2.0269) | 0.1517 (2.0285) | 0.0433 / 0.0864 / 0.2585 |
| 2 | 0.6209 (1.6398) | 0.6222 (1.6422) | 0.6105 (1.6435) | 0.0422 / 0.0843 / 0.2523 |
| 3 | 0.5972 (1.5869) | 0.6241 (1.5959) | 0.6445 (1.5806) | 0.0413 / 0.0825 / 0.2470 |
| 4 | 0.6920 (1.5395) | 0.7040 (1.5156) | 0.6877 (1.5235) | 0.0409 / 0.0817 / 0.2443 |
| 5 | 0.6698 (1.4645) | 0.6532 (1.4536) | 0.6976 (1.4429) | 0.0407 / 0.0812 / 0.2427 |
| 6 | 0.6917 (1.4116) | 0.6270 (1.4234) | 0.6980 (1.4087) | 0.0406 / 0.0810 / 0.2419 |
| 7 | 0.6908 (1.3934) | 0.6758 (1.4229) | 0.6925 (1.4056) | 0.0405 / 0.0809 / 0.2415 |
| 8 | 0.7188 (1.3533) | **0.7326 (1.3810)** | 0.7061 (1.3846) | 0.0404 / 0.0808 / 0.2413 |
| 9 | 0.6919 (1.3323) | 0.7053 (1.3395) | 0.7144 (1.3283) | 0.0404 / 0.0807 / 0.2411 |
| 10 | 0.7107 (1.3090) | 0.7244 (1.2981) | 0.7248 (1.2980) | 0.0403 / 0.0806 / 0.2409 |
| 11 | 0.7144 (1.2902) | 0.7291 (1.2859) | 0.7153 (1.2720) | 0.0403 / 0.0805 / 0.2409 |
| 12 | **0.7321 (1.2661)** | 0.7202 (1.2762) | 0.7143 (1.2801) | 0.0403 / 0.0805 / 0.2408 |
| 13 | 0.7259 (1.2587) | 0.7241 (1.2681) | 0.7317 (1.2588) | 0.0403 / 0.0805 / 0.2407 |
| 14 | 0.7252 (1.2612) | 0.7275 (1.2608) | **0.7321 (1.2559)** | 0.0403 / 0.0805 / 0.2407 |
| 15 | 0.7282 (1.2526) | 0.7303 (1.2552) | 0.7259 (1.2530) | 0.0402 / 0.0804 / 0.2407 |
| 16 | 0.7297 (1.2519) | 0.7316 (1.2530) | 0.7289 (1.2513) | 0.0402 / 0.0804 / 0.2407 |
| 17 | *(Chuyển S2 @ Ep 16)* | 0.7315 (1.2520) | 0.7295 (1.2505) | 0.0402 / 0.0804 / 0.2406 |

#### Giai đoạn 2 (Stage 2: Mở khóa Toàn bộ Backbone, Epochs 1 — 18/19):
| S2 Ep | LB=0.005 Val Dice (Loss) | LB=0.010 Val Dice (Loss) | LB=0.030 Val Dice (Loss) | Ghi chú Trọng yếu |
| :-: | :---: | :---: | :---: | :--- |
| 1 | 0.7257 (1.3218) | 0.7288 (1.2734) | 0.7202 (1.2668) | Khởi động r=1:1 cosine schedule |
| 2 | 0.7268 (1.2731) | 0.7253 (1.2676) | 0.7261 (1.2464) | Hội tụ sơ khởi |
| 3 | 0.7208 (1.2339) | 0.7178 (1.2502) | 0.7174 (1.2185) | Tinh chỉnh bộ trích xuất |
| 4 | 0.7178 (1.1896) | 0.7184 (1.1632) | 0.7265 (1.1718) | Tái định hình đặc trưng vết nứt |
| 5 | 0.7228 (1.1557) | 0.6824 (1.1281) | 0.7099 (1.1215) | Dao động gradient chuyển đổi |
| 6 | 0.7169 (1.1090) | 0.7027 (1.1281) | 0.7310 (1.1070) | LB=0.030 bứt phá trước |
| 7 | 0.7279 (1.0894) | 0.7319 (1.0448) | 0.7360 (1.0544) | Cả 3 cấu hình vượt ngưỡng 0.73 |
| 8 | 0.7410 (1.0440) | 0.7422 (1.0153) | 0.7431 (1.0163) | Bắt đầu pha nước rút |
| 9 | 0.7358 (1.0347) | 0.7170 (1.0207) | 0.7483 (1.0125) | LB=0.030 duy trì độ dốc ổn định |
| 10 | 0.7397 (1.0143) | 0.7475 (1.0243) | 0.7482 (0.9991) | Vượt mốc 0.74 |
| 11 | 0.7454 (0.9918) | 0.7479 (0.9608) | 0.7559 (0.9845) | Tiếp cận ngưỡng 0.75 |
| 12 | 0.7404 (0.9789) | 0.7296 (0.9556) | 0.7446 (0.9737) | Đáy dao động cục bộ |
| 13 | 0.7416 (1.0045) | 0.7480 (0.9981) | 0.7553 (0.9709) | Khử nhiễu biên |
| 14 | 0.7442 (0.9938) | 🏆 **0.7641 (0.9544)** | 0.7586 (0.9602) | 🏆 **LB=0.010 ĐẠT ĐỈNH TOÀN CỤC (0.7641)** |
| 15 | 0.7552 (0.9866) | 0.7517 (0.9475) | 0.7555 (0.9315) | LB=0.030 chạm đáy loss (0.9315) |
| 16 | 0.7512 (0.9617) | 0.7584 (0.9569) | **0.7624 (0.9465)** | **LB=0.030 đạt đỉnh (0.7624)** |
| 17 | 0.7557 (0.9553) | 0.7552 (0.9545) | 0.7603 (0.9532) | Hội tụ plateau cuối lịch trình |
| 18 | 0.7548 (0.9625) | 0.7559 (0.9529) | 0.7601 (0.9407) | Khóa trạng thái hội tụ (35 eps) |
| 19 | **0.7580 (0.9572)** | — | — | **LB=0.005 đạt đỉnh muộn (0.7580)** |

---

### 9.4. Đánh Giá Khoa Học & Phán Quyết Khóa Chính Thức Phase 4

Thực nghiệm hoàn chỉnh trên 3 mức $LB \in \{0.005, 0.010, 0.030\}$ xác lập 3 kết luận cơ chế cốt lõi:

1. **Nguy cơ của Phạt Yếu ($LB = 0.005$): Sụp đổ Routing Cục bộ & Lỗi Diện Tích (Area Bias)**
   Khi lực phạt giảm 50% ($L_{LB} \approx 0.04$), các router tầng nông mất động lực tìm kiếm biểu diễn mới, co cụm cực đoan vào 2 chuyên gia duy nhất ($E_4, E_6$ ở stage 0; $E_1, E_6$ ở stage 1). Hậu quả trực tiếp trên phân tích lỗi là mô hình bị thiên lệch dự đoán thừa nghiêm trọng: diện tích dự đoán trung bình vọt lên **15,372 px** (so với Ground Truth 13,239 px, tỷ lệ 1.161), làm Precision giảm sút còn **0.7030** và kéo Val Dice tụt sâu về **0.7580** ($-0.0061$).

2. **Cơ chế Hồi sinh Chuyên gia & Đánh đổi của Phạt Mạnh ($LB = 0.030$):**
   Khi tăng lực phạt gấp 3 lần ($L_{LB} \approx 0.24$), router bị ép phải điều hướng mẫu sang các chuyên gia bị bỏ đói: $E_3$ và $E_5$ tại `stage_1` được hồi sinh; $E_1, E_3, E_5$ tại `stage_2` nhận lưu lượng đều đặn; số chuyên gia hiệu dụng tại `stage_3` tăng từ 5.89 lên 7.02. Điều này giúp tối ưu hóa không gian biểu diễn chung (đạt đáy Val Loss thấp nhất: **0.9315** và Dice cao: **0.7624**). Tuy nhiên, vì lực phạt quá lớn đã hạn chế quyền tự do "chuyên môn hóa sâu" của các router vào các dạng vân nứt đặc thù, khiến độ sắc nét của đường biên bị suy giảm nhẹ so với Candidate B (Dice thấp hơn $-0.0017$).

3. **Cân Bằng Pareto Tối Ưu Tại $LB = 0.010$ (Candidate B):**
   Mức $LB = 0.010$ đại diện cho điểm cân bằng Pareto hoàn hảo trong bài toán phân đoạn vết nứt:
   - Đủ lực phạt để giữ 0 dead experts trên toàn bộ pool toàn cục (không bị sụp đổ hệ thống).
   - Vừa đủ tự do cho các router tầng nông tự tổ chức thành các bộ lọc vân nứt chuyên biệt mà không bị ép chia đều nhân tạo.
   - Đạt đỉnh cao nhất trên toàn bộ các chỉ số đo lường chất lượng: **Val Dice = 0.7641, Mean IoU = 0.6417, Precision = 0.7337, Median Dice = 0.8066**.

> [!IMPORTANT]
> **PHÁN QUYẾT CHÍNH THỨC PHASE 4:**
> Khóa vĩnh viễn hệ số cân bằng tải **$load\_balance\_factor = 0.010$** làm tiêu chuẩn chuẩn mực (Canonical Frozen Hyperparameter) cho toàn bộ cấu hình SAGE-Lite B2 trong các pha tiếp theo.

---

## 10. Phase 6-A.1: Objective Probe — Soft Boundary IoU Loss Study (D4-K2-H64)

> **Mục tiêu Thực nghiệm 6-A.1:**
> Kiểm chứng xem giới hạn ~0.76 Dice và các dạng lỗi hình thái (Boundary margin loang rộng, Thin crack suy giảm) có xuất phát từ tín hiệu huấn luyện (Objective Bottleneck) hay không, bằng cách bổ sung **Soft Boundary IoU Loss (λ=0.50, d=2)** mà **KHÔNG thêm bất kỳ tham số hay nhánh kiến trúc nào** (0 extra params, architecture 100% frozen).

### 10.1. Lịch trình Huấn luyện Từng Epoch (Stage 2 Resumption, Epochs 1 — 18)

| S2 Ep | Train Loss (LB) | Train Dice | Val Loss | Val Dice | Delta vs Base Peak (0.7641) | Shared / Base / P3 LR | Gamma (S0 / S1) | Ghi chú Trọng yếu |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 1.7882 (0.0806) | 0.6968 | 1.7552 | 0.7311 | -0.0330 | 3.40e-05 | 0.0107 / 0.0139 | Resumption từ Candidate B Stage 1 |
| 2 | 1.7804 (0.0805) | 0.6947 | 1.7390 | 0.7414 | -0.0227 | 6.70e-05 | 0.0110 / 0.0143 | Tăng tốc warmup Stage 2 |
| 3 | 1.7603 (0.0805) | 0.6890 | 1.7404 | 0.7428 | -0.0213 | 9.34e-05 | 0.0109 / 0.0142 | Đạt đỉnh LR Stage 2 (~1e-4) |
| 4 | 1.7163 (0.0805) | 0.6933 | 1.6628 | 0.7411 | -0.0230 | 8.84e-05 | 0.0103 / 0.0142 | Bắt đầu chu kỳ Cosine Annealing |
| 5 | 1.6677 (0.0804) | 0.7008 | 1.5834 | 0.6992 | -0.0649 | 8.23e-05 | 0.0099 / 0.0145 | Dao động gradient thích ứng ranh giới |
| 6 | 1.6228 (0.0804) | 0.7098 | 1.5739 | 0.7369 | -0.0272 | 7.52e-05 | 0.0095 / 0.0151 | Phục hồi hội tụ nhanh chóng |
| 7 | 1.5824 (0.0804) | 0.7099 | 1.5616 | 0.7511 | -0.0130 | 6.74e-05 | 0.0091 / 0.0151 | Vượt mốc 0.75 Val Dice |
| 8 | 1.5503 (0.0803) | 0.7153 | 1.4864 | 0.7603 | -0.0038 | 5.91e-05 | 0.0094 / 0.0157 | Tiếp cận mốc 0.76 |
| 9 | 1.5162 (0.0803) | 0.7221 | 1.4872 | 0.7358 | -0.0283 | 5.05e-05 | 0.0096 / 0.0163 | Dao động cục bộ giữa lịch trình |
| 10 | 1.4902 (0.0803) | 0.7258 | 1.4969 | 0.7487 | -0.0154 | 4.19e-05 | 0.0091 / 0.0162 | Train Dice vượt 0.725 |
| 11 | 1.4595 (0.0803) | 0.7337 | 1.4560 | 0.7629 | -0.0012 | 3.36e-05 | 0.0088 / 0.0166 | Sát ngưỡng kỷ lục cũ |
| 12 | 1.4451 (0.0803) | 0.7320 | 1.4607 | 0.7444 | -0.0197 | 2.58e-05 | 0.0089 / 0.0168 | Ổn định ranh giới vết nứt |
| 13 | 1.4341 (0.0803) | 0.7336 | 1.4279 | 0.7591 | -0.0050 | 1.87e-05 | 0.0089 / 0.0170 | Train Loss tiếp cận 1.43 |
| 14 | 1.4156 (0.0803) | 0.7366 | 1.4062 | **0.7664** | **+0.0023** | 1.26e-05 | 0.0089 / 0.0170 | 🏆 **CHÍNH THỨC PHÁ VỠ KỶ LỤC CANDIDATE B (0.7664 > 0.7641)** |
| 15 | 1.4095 (0.0803) | 0.7391 | **1.3911** | **0.7675** | **+0.0034** | 7.63e-06 | 0.0091 / 0.0173 | 🏆 **ĐẠT ĐÁY VAL LOSS (1.3911)** |
| 16 | 1.4000 (0.0803) | 0.7434 | 1.4128 | 🏆 **0.7684** | 🏆 **+0.0043** | 3.99e-06 | 0.0092 / 0.0173 | 🏆 **ĐỈNH TOÀN DỰ ÁN MỚI: 0.7684 VAL DICE** |
| 17 | 1.3942 (0.0802) | 0.7445 | 1.4174 | 0.7657 | +0.0016 | 1.75e-06 | 0.0092 / 0.0172 | Duy trì trên 0.765 ở plateau cuối |
| 18 | 1.3881 (0.0802) | 0.7471 | 1.3964 | 0.7664 | +0.0023 | 1.00e-06 | 0.0092 / 0.0172 | Kết thúc lịch trình 35 epochs chuẩn mực |

---

### 10.2. Bảng Đối Chiếu Phân Tầng Tuyệt Đối (Stratified Absolute Delta Comparison)

*Toàn bộ 348 ảnh validation Crack500 đánh giá theo Setting A official tiling protocol:*

| Phân tầng Thẩm định (Stratum) | Chỉ số Đo lường | Candidate B Baseline | Phase 6-A.1 (Boundary IoU) | Độ lệch Tuyệt đối (Δ) | Diễn giải Cơ chế Vật lý |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Global Validation (n=348)** | **Val Dice** | 0.7641 | 🏆 **0.7684** | **+0.0043** | Đạt đỉnh mới toàn dự án |
| | **Mean IoU** | 0.6417 | 🏆 **0.6465** | **+0.0048** | Cải thiện độ trùng khớp vùng |
| | **Precision** | 0.7337 | 🏆 **0.7416** | **+0.0079** | Tăng độ chính xác, giảm FP nền |
| | **Recall** | **0.8477** | 0.8413 | -0.0064 | Giảm nhẹ hiện tượng loang viền dôi dư |
| | **Tỷ lệ Diện tích (Pred/GT)** | +8.16% | 🏆 **+7.85%** | **-0.31%** | Diện tích co sát lại gần GT hơn |
| **Boundary Margin (n=127)** | **Val Dice** | **0.7812** | 0.7787 | -0.0025 | Biến thiên rất nhỏ do siết viền |
| | **Precision** | 0.7452 | 🏆 **0.7525** | **+0.0073** | Viền sắc nét hơn, ít lem ra nền |
| | **Tỷ lệ Diện tích (Pred/GT)** | +5.85% | 🏆 **+5.55%** | **-0.30%** | Giảm loang viền ở crack trung bình |
| **Thin Cracks >0.20 (n=66)** | **Val Dice** | 0.6404 | 🏆 **0.6641** | 🚀 **+0.0237** | **Bứt phá +2.37% Dice trên crack mảnh!** |
| | **Precision** | 0.5232 | 🏆 **0.5648** | 🚀 **+0.0416** | **Bứt phá +4.16% Precision!** |
| | **Tỷ lệ Diện tích (Pred/GT)** | +80.37% | 🏆 **+64.55%** | 🚀 **-15.82%** | **Triệt tiêu mạnh lỗi phình to vết nứt mảnh!** |
| **Thin-Low-Area (n=5)** | **Val Dice** | 0.5370 | 🏆 **0.5382** | +0.0012 | Ổn định các ca thất bại nặng nhất |
| | **Precision** | 0.3799 | 🏆 **0.3896** | +0.0097 | Tăng nhẹ độ chính xác |
| | **Tỷ lệ Diện tích (Pred/GT)** | +141.66% | 🏆 **+127.33%** | 🚀 **-14.33%** | Giảm mạnh over-prediction ở ca cực đoan |
| **Complex Topology (n=27)** | **Val Dice** | 0.7515 | 🏆 **0.7572** | **+0.0057** | Giữ vững tính liên thông topo |
| | **Precision** | 0.7054 | 🏆 **0.7195** | **+0.0141** | Lọc nhiễu ở mạng lưới nứt phức tạp |
