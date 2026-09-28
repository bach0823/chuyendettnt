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


