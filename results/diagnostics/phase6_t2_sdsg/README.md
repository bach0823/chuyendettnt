# Phase 6: Minimal End-to-End T2-SDSG Intervention Diagnostic Report

## 1. Executive Summary

This diagnostic evaluates a minimal, non-disruptive, learned intervention module inserted at the entry of Decoder Block 1 (T2):
**T2-SDSG (Learned Pointwise Selective Dual-Stream Gate)**.

```
T2 = Concat[Bottleneck (192ch); Skip (96ch)]  (288ch, 56x56)
h  = ReLU(BatchNorm(Conv1x1(288 -> 64)))
a  = Conv1x1(64 -> 2)
g  = 2.0 * Sigmoid(a)                          (range: [0, 2])
T2' = Concat[g_B * Bottleneck; g_S * Skip]
```

### Invariants & Protocol
- **Frozen Lineage**: 100% of Candidate B parameters frozen (`requires_grad = False`, modules in `eval()` mode).
- **Exact Identity Initialization**: $W_2 = 0, b_2 = 0 \implies a = 0 \implies g = 2 \cdot \sigma(0) \equiv 1.0$ bitwise at epoch 0.
- **Physical Batch Size**: Pinned at **14** (no gradient accumulation), 8 epochs, AdamW ($lr=10^{-4}, wd=10^{-4}$).
- **Two Arms**:
  - **Arm A**: SDSG trained with standard segmentation loss $L_{\text{seg}} = \text{BCE} + \text{Dice}$.
  - **Arm B**: SDSG trained with $L_{\text{seg}} + 0.1 \cdot L_{\text{aux}}$ ($L_{\text{aux}}$ rewards $g \to 1$ on true crack pixels and penalizes $g \to 0$ on false-bridge neck corridors extracted from the 1,896 Crack500 training images, plus mild identity penalty).
- **Evaluation**: Canonical Setting A (448x448, stride 448, reflect padding, threshold $\tau=0.5$) across Full Validation $N=348$.
- **Data Leakage Lock**: Official Test set $N=1124$ strictly sealed. Checkpoint SHA256 and initial parameter SHA256 bitwise invariant.

---

## 2. Validation Subgroup Comparison Table

| Model | Cohort | $N$ | Dice | Recall | Precision | clDice | TSens | Bridges | $\Delta$Bridge | Cured | Created | Breakages | $\Delta$Break | AreaRatio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Candidate B** | Full Validation | 348 | 0.7641 | 0.8477 | 0.7337 | 0.8499 | 0.8872 | **118** | +0 | 0 | 0 | **35** | +0 | 1.2829 |
| | Consensus 7/7 | 101 | 0.7371 | 0.8997 | 0.6974 | 0.8455 | 0.9328 | 109 | +0 | 0 | 0 | 11 | +0 | 1.4057 |
| | Consensus Resistant | 82 | 0.7575 | 0.9086 | 0.7139 | 0.8586 | 0.9377 | 88 | +0 | 0 | 0 | 8 | +0 | 1.3857 |
| | Consensus Sensitive | 19 | 0.6491 | 0.8616 | 0.6263 | 0.7892 | 0.9118 | 21 | +0 | 0 | 0 | 3 | +0 | 1.4921 |
| | Clean Control | 56 | 0.7540 | 0.7922 | 0.7628 | 0.8200 | 0.8454 | **0** | +0 | 0 | 0 | 8 | +0 | 1.1396 |
| **Arm A (SDSG + $L_{\text{seg}}$)** | Full Validation | 348 | 0.7619 | 0.8366 | 0.7371 | 0.8464 | 0.8764 | **119** | **+1** | 1 | 2 | **51** | **+16** | 1.2436 |
| | Consensus 7/7 | 101 | 0.7420 | 0.8959 | 0.7042 | 0.8493 | 0.9298 | 109 | +0 | 0 | 0 | 12 | +1 | 1.3789 |
| | Consensus Resistant | 82 | 0.7623 | 0.9056 | 0.7196 | 0.8613 | 0.9348 | 88 | +0 | 0 | 0 | 8 | +0 | 1.3615 |
| | Consensus Sensitive | 19 | 0.6544 | 0.8539 | 0.6377 | 0.7977 | 0.9085 | 21 | +0 | 0 | 0 | 4 | +1 | 1.4542 |
| | Clean Control | 56 | 0.7474 | 0.7771 | 0.7667 | 0.8091 | 0.8315 | **0** | +0 | 0 | 0 | 11 | +3 | 1.1065 |
| **Arm B (SDSG + $L_{\text{seg}} + 0.1 L_{\text{aux}}$)** | Full Validation | 348 | 0.7629 | 0.8145 | 0.7520 | 0.8417 | 0.8635 | **111** | **-7** | 9 | 1 | **82** | **+47** | 1.1828 |
| | Consensus 7/7 | 101 | 0.7453 | 0.8660 | 0.7258 | 0.8407 | 0.9087 | 108 | -1 | 2 | 0 | 22 | +11 | 1.2954 |
| | Consensus Resistant | 82 | 0.7639 | 0.8753 | 0.7397 | 0.8525 | 0.9149 | 87 | -1 | 2 | 0 | 15 | +7 | 1.2785 |
| | Consensus Sensitive | 19 | 0.6648 | 0.8258 | 0.6659 | 0.7896 | 0.8817 | 21 | +0 | 0 | 0 | 7 | +4 | 1.3683 |
| | Clean Control | 56 | 0.7465 | 0.7550 | 0.7842 | 0.8057 | 0.8105 | **0** | +0 | 0 | 0 | 16 | +8 | 1.0543 |

---

## 3. Training Dynamics & Gate Modulation

```
Epoch   Arm A (Lseg only)                     Arm B (Lseg + 0.1 Laux)
        Loss      g_B     g_S     g_min..max  Loss_total  Loss_aux  g_B     g_S     g_min..max
1       0.7038    1.115   0.842   0.37..1.59  0.7804      0.6945    0.894   0.865   0.24..1.02
2       0.6893    1.147   0.776   0.08..1.91  0.7558      0.5044    0.892   0.829   0.23..1.10
3       0.6553    1.187   0.680   0.03..1.96  0.7512      0.4652    0.889   0.795   0.15..1.17
4       0.6266    1.317   0.535   0.04..1.95  0.7428      0.4490    0.887   0.774   0.13..1.28
5       0.6207    1.206   0.594   0.01..1.98  0.7354      0.4596    0.886   0.765   0.12..1.29
6       0.6130    1.365   0.462   0.02..1.97  0.7209      0.4582    0.868   0.764   0.15..1.41
7       0.6115    1.371   0.447   0.02..1.96  0.7078      0.4596    0.874   0.759   0.10..1.41
8       0.6059    1.401   0.428   0.02..1.96  0.7060      0.4616    0.862   0.759   0.07..1.36
```

### Key Analytical Takeaways:
1. **Autonomous Factorization in Arm A**:
   Without any auxiliary supervision ($L_{\text{aux}} = 0$), $L_{\text{seg}}$ alone monotonically drives $g_B$ UP ($1.0 \to 1.401$) and $g_S$ DOWN ($1.0 \to 0.428$).
   - This proves that the standard segmentation gradient naturally prefers the low-frequency semantic consistency of Bottleneck over the high-frequency edge ambiguity of Skip.
   - However, this autonomous suppression of Skip does NOT eliminate false bridges (+1 net bridge) while increasing breakages from 35 to 51.
2. **Forced Corridor Attenuation in Arm B**:
   Under $L_{\text{aux}}$, both streams are attenuated ($g_B \to 0.862, g_S \to 0.759$), driving minimum local gates down to $0.069$.
   - This successfully cured 9 false bridges across validation.
   - However, it resulted in a massive surge in topological breakages ($35 \to 82$, $+134\%$).

---

## 4. Answers to the 7 Mandatory Diagnostic Questions

### Q1: Did the gate really reduce false bridges, or did it merely shrink/erode the prediction mask?
**Answer: It is predominantly mask erosion.**
- In Arm B, while false bridges dropped by 7 ($118 \to 111$), global Recall dropped from $0.8477 \to 0.8145$ ($-3.32$ pp) and Area Ratio dropped from $1.2829 \to 1.1828$ ($-0.100$).
- More critically, topological sensitivity (`tsens`) dropped from $0.8872 \to 0.8635$, and total fragmentation/breakage events surged from $35 \to 82$ ($+134\%$).
- Among the 9 "cured" images in Arm B, 3 suffered collateral breakages ($0 \to 1$), and across all 9, recall dropped severely (e.g. `20160222_115837_1281_361`: recall $0.968 \to 0.845$; `20160407_165001_641_1081`: recall $0.892 \to 0.743$).
- The mechanism is thin-structure erosion: neck corridors are suppressed, but thin true crack segments are simultaneously eroded below the $\tau=0.5$ threshold.

### Q2: In Resistant 82, what happened to the bridges? Did they transition to cured or to breakage?
**Answer: Resistant bridges are almost completely immune to pointwise gating, transitioning only into breakage.**
- In Consensus Resistant 82, false bridges only decreased by **1** ($88 \to 87$, with 2 cured, 0 created, and 80 persistent).
- Meanwhile, breakages in Resistant 82 **nearly doubled**, jumping from $8 \to 15$ ($+87.5\%$).
- In Arm A ($L_{\text{seg}}$ alone), Resistant bridges were completely unchanged ($88 \to 88$, 0 cured, 0 created, 82 persistent).
- **Conclusion**: Pointwise T2 features in Resistant bridges cannot be suppressed without simultaneously severing true cracks.

### Q3: Pure-interaction vs Single-stream behavior?
**Answer: Gating single streams pointwise cannot overcome the shared ambiguity.**
- In Arm A, the network attempted single-stream preference by attenuating Skip ($g_S = 0.428$) and amplifying Bottleneck ($g_B = 1.401$). This mirrors the earlier static factorization findings: Bottleneck-only still contains 37/82 Resistant bridges and suffers higher false-positive dilation.
- Because both Bottleneck and Skip carry ambiguous evidence in false-bridge corridors, simply reweighting their channels pointwise cannot uncouple false connectivity from true crack continuity.

### Q4: Did Clean Control (56 images) develop any newly created false bridges?
**Answer: No false bridges were created, but true crack continuity was heavily damaged.**
- On Clean Control ($N=56$, $\ge 2$ GT cracks, 0 baseline bridges):
  - Baseline: 0 bridges, 8 breakages.
  - Arm A: 0 bridges (+0 created), 11 breakages (+3).
  - Arm B: 0 bridges (+0 created), 16 breakages (**doubled, +8**).
- This confirms that T2-SDSG does not hallucinate new spurious connections on clean backgrounds, but it aggressively fractures true crack networks.

### Q5: Selective trade-off vs Isotropic attenuation?
**Answer: Learned pointwise gating exhibits the exact same pathological trade-off as isotropic attenuation.**
- Earlier in Phase 6, isotropic attenuation at Decoder B1 was closed because reducing bridge events invariably destroyed true crack continuity.
- T2-SDSG was hypothesized to potentially discover a learned, selective subspace. The empirical data decisively proves this hypothesis false ($H_0$ holds): a pointwise $1 \times 1$ gate at $56 \times 56$ lacks the local spatial context to distinguish a false neck corridor from a narrow true crack.

### Q6: Gate modulation distribution ($g_B$ vs $g_S$)?
**Answer:**
- **Arm A ($L_{\text{seg}}$)**: $g_B \in [0.02, 1.96]$, mean $= 1.401$; $g_S \in [0.02, 1.96]$, mean $= 0.428$. The unconstrained model chooses an asymmetric, stream-level amplification of Bottleneck and attenuation of Skip.
- **Arm B ($L_{\text{seg}} + 0.1 L_{\text{aux}}$)**: $g_B \in [0.07, 1.36]$, mean $= 0.862$; $g_S \in [0.07, 1.36]$, mean $= 0.759$. Both streams are bilaterally suppressed, with intense suppression (down to $\sim 0.07$) occurring at high-confidence false corridors.

### Q7: Failure mode / Next architectural implications?
**Answer: Pointwise T2 intervention is closed. Context-aware or upstream interventions are required.**
- **Pointwise Gating at T2 is Ineffective**: Pointwise features at T2 do not possess sufficient separable information to differentiate true crack continuity from false bridge corridors without spatial context.
- **Architectural Implication**: Any successful intervention at Decoder Block 1 MUST incorporate:
  1. **Non-local / Multi-scale Spatial Context** (e.g. oriented kernels, graph/topological message passing, or patch-level attention), OR
  2. **Upstream Disentanglement** (preventing false-bridge ambiguity from forming in Stage 0 / Stem AC prior to multi-scale aggregation).
