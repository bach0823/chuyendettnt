# Phase 6: Contextual Spatial Dual-Stream Gate (T2-CSDG, K=3) Diagnostic Report

## 1. Executive Summary & System Invariants

This experiment executes a controlled, single-variable causal intervention at Decoder Block 1 (T2):
**Replacing Pointwise Gating (SDSG, $1 \times 1$) with Contextual Spatial Gating (CSDG, $3 \times 3$ Depthwise Context)**.

```
T2 = Concat[Bottleneck (192ch); Skip (96ch)]  (288 channels, 56x56)
  ↓
1x1 Conv(288 -> 64) -> BatchNorm -> ReLU
  ↓
3x3 DWConv(64 -> 64, groups=64, pad=1) -> BatchNorm -> ReLU  (Receptive Field = 24 px)
  ↓
1x1 Conv(64 -> 2)
  ↓
g = 2.0 * Sigmoid(a)                          (range: [0, 2])
  ↓
T2' = Concat[g_B * Bottleneck; g_S * Skip]
```

### Invariants & Experimental Rigor:
- **Strict Single-Variable Isolation**:
  - Training loss **pinned to Arm B**: $L = L_{\text{seg}} + 0.1 \cdot L_{\text{aux}}$ (using the identical 597 train false-bridge corridor coordinates).
  - Physical batch size = **14**, seed = 42, no gradient accumulation, AdamW ($lr=10^{-4}$), 8 epochs.
  - Candidate B checkpoint & parameters 100% frozen (`requires_grad = False`, modules kept in `eval()` mode).
  - Exact Identity Initialization: $W_{\text{gate}} = 0, b_{\text{gate}} = 0 \implies g_B = g_S \equiv 1.0$ bitwise (max feature diff $= 0.0$, max end-to-end logit diff $= 0.0$ at epoch 0).
  - Official test set ($N=1124$) strictly sealed.

---

## 2. Validation Subgroup Comparison Table

| Model | Cohort | $N$ | Dice | Recall | Precision | clDice | TSens | Br Imgs | $\Delta$Imgs | Br Evts | $\Delta$Evts | Break Evts | $\Delta$Break | Area Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Candidate B** | **Full Validation** | **348** | **0.7641** | **0.8477** | **0.7337** | **0.8499** | **0.8872** | **110** | +0 | **118** | +0 | **35** | +0 | **1.2829** |
| | Consensus 7/7 | 101 | 0.7371 | 0.8997 | 0.6538 | 0.8455 | 0.9233 | 101 | +0 | 109 | +0 | 11 | +0 | 1.5421 |
| | Consensus Resistant | 82 | 0.7575 | 0.9086 | 0.6752 | 0.8586 | 0.9301 | 82 | +0 | 88 | +0 | 8 | +0 | 1.4864 |
| | Consensus Sensitive | 19 | 0.6491 | 0.8616 | 0.5617 | 0.7892 | 0.8937 | 19 | +0 | 21 | +0 | 3 | +0 | 1.7821 |
| | Clean Control | 56 | 0.7540 | 0.7922 | 0.7432 | 0.8200 | 0.8227 | **0** | +0 | **0** | +0 | 8 | +0 | 1.1061 |
| **Arm B (T2-SDSG 1x1)** | **Full Validation** | **348** | **0.7629** | **0.8145** | **0.7520** | **0.8417** | **0.8635** | **102** | **-8** | **111** | **-7** | **82** | **+47** | **1.1828** |
| | Consensus 7/7 | 101 | 0.7453 | 0.8660 | 0.6819 | 0.8407 | 0.9012 | 99 | -2 | 108 | -1 | 22 | +11 | 1.3974 |
| | Consensus Resistant | 82 | 0.7639 | 0.8753 | 0.7024 | 0.8525 | 0.9081 | 80 | -2 | 87 | -1 | 15 | +7 | 1.3577 |
| | Consensus Sensitive | 19 | 0.6648 | 0.8258 | 0.5932 | 0.7896 | 0.8714 | 19 | +0 | 21 | +0 | 7 | +4 | 1.5686 |
| | Clean Control | 56 | 0.7465 | 0.7550 | 0.7603 | 0.8057 | 0.7960 | **0** | +0 | **0** | +0 | 16 | +8 | 1.0143 |
| **T2-CSDG (K=3 Spatial)** | **Full Validation** | **348** | **0.7654** | **0.8399** | **0.7366** | **0.8467** | **0.8761** | **104** | **-6** | **113** | **-5** | **70** | **+35** | **1.2498** |
| | Consensus 7/7 | 101 | 0.7415 | 0.8922 | 0.6620 | 0.8439 | 0.9157 | 99 | -2 | 108 | -1 | 22 | +11 | 1.4926 |
| | Consensus Resistant | 82 | 0.7615 | 0.9010 | 0.6830 | 0.8560 | 0.9224 | 80 | -2 | 86 | -2 | 14 | +6 | 1.4411 |
| | Consensus Sensitive | 19 | 0.6555 | 0.8541 | 0.5714 | 0.7917 | 0.8872 | 19 | +0 | 22 | +1 | 8 | +5 | 1.7145 |
| | Clean Control | 56 | 0.7511 | 0.7776 | 0.7507 | 0.8101 | 0.8085 | **0** | +0 | **0** | +0 | 15 | +7 | 1.0638 |

---

## 3. Spatial Gate Distribution Audit

To directly verify whether bridge suppression was caused by **true spatial selectivity** or merely **adaptive local erosion**, gate activations were extracted at $56 \times 56$ resolution across three distinct spatial compartments:
1. **True Crack Pixels** ($GT = 1$)
2. **False Bridge Corridor Pixels** ($\text{Pred} = 1 \land GT = 0$)
3. **Background Pixels** ($GT = 0 \land \text{Pred} = 0$)

```
------------------------------------------------------------------------
Compartment                g_B (Mean)   g_B (Median)   g_S (Mean)   g_S (Median)
------------------------------------------------------------------------
True Crack Region:           0.4423       0.4350         0.5820       0.5790
False Bridge Corridor:       0.4617       0.4580         0.5906       0.5880
Background:                  1.0302       1.0250         0.6508       0.6480
------------------------------------------------------------------------
Selectivity Ratio (Crack / Corridor):
  g_B Selectivity Ratio:     0.9580  (Target for true selectivity: >> 1.0)
  g_S Selectivity Ratio:     0.9855  (Target for true selectivity: >> 1.0)
------------------------------------------------------------------------
```

### Critical Empirical Finding: Zero Spatial Selectivity
- The gate modulation at True Cracks ($g_B = 0.442, g_S = 0.582$) is **virtually identical** to that at Bridge Corridors ($g_B = 0.462, g_S = 0.591$).
- In fact, the gate values at thin crack pixels are slightly **more suppressed** than at the bridge corridor (Selectivity Ratio $< 1.0$).
- **Conclusion**: CSDG ($K=3$) does NOT differentiate the false bridge neck corridor from genuine thin cracks. The suppression is entirely indiscriminate across narrow elongated structures.

---

## 4. Decision Tree Assessment

According to the pre-registered decision tree:

* **Trường hợp A** (Bridge giảm đáng kể, Breakage gần baseline, Recall giữ được):
  $\to$ **NOT OCCURRED**.
* **Trường hợp B** (Bridge giảm nhẹ, nhưng Breakage tăng mạnh, Selectivity Ratio $\approx 1.0$):
  $\to$ **CONFIRMED (CASE B)**.
  - False bridges dropped slightly: $118 \to 113$ events ($110 \to 104$ images).
  - But Breakage **doubled**: $35 \to 70$ events ($+100\%$).
  - Clean Control breakages **doubled**: $8 \to 15$ events ($+87.5\%$).
  - Resistant 82 bridges barely moved: $88 \to 86$ events, while breakages increased from $8 \to 14$.
* **Interpretation**:
  > **T2-CSDG ($K=3$) remains an adaptive erosion operator.**
  > A $3 \times 3$ depthwise spatial context (effective receptive field 24 px) is insufficient to break the fundamental trade-off: any suppression applied to bridge corridors equally damages narrow true crack continuity.

---

## 5. Epistemic Classification of Findings

| Claim | Epistemic Status | Evidence / Verification |
| :--- | :---: | :--- |
| CSDG ($K=3$) fails to achieve selective bridge suppression without severe continuity loss. | **Supported by Evidence** | Setting A Full Validation shows bridge events $118 \to 113$ ($-5$) accompanied by breakages $35 \to 70$ ($+35$, $+100\%$). |
| Bridge suppression in CSDG ($K=3$) is driven by erosion rather than geometric selectivity. | **Supported by Evidence** | Spatial gate audit reveals Crack/Corridor selectivity ratio $\approx 0.96 - 0.99$, indicating indiscriminate suppression of thin structures. |
| The CSDG family under $K=3$ fails to break the bridge-vs-breakage trade-off. | **Supported by Evidence** | Replicated under strictly pinned Arm B loss and physical batch 14. |
| Pointwise vs $3 \times 3$ spatial context comparison. | **Supported by Evidence** | $K=3$ retains slightly higher Recall ($0.8399$ vs $0.8145$) and fewer breakages ($70$ vs $82$) than pointwise SDSG, but both double baseline breakage ($35 \to 70/82$). |
| Whether $K=5$ (receptive field 40 px) or larger multi-scale context can solve this. | **Open Question / Next Hypothesis** | Needs empirical evaluation to differentiate "context was still too narrow" from "T2 representation is already irreparably entangled upstream". |
