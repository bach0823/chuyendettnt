# Phase 6: Minimal End-to-End T2-SDSG Intervention Diagnostic Report

## 1. Executive Summary & Verification of Invariants

This diagnostic investigates a minimal, non-disruptive, learned intervention module inserted at the entry of Decoder Block 1 (T2):
**T2-SDSG (Learned Pointwise Selective Dual-Stream Gate)**.

```
T2 = Concat[Bottleneck (192ch); Skip (96ch)]  (288 channels, 56x56)
h  = ReLU(BatchNorm(Conv1x1(288 -> 64)))
a  = Conv1x1(64 -> 2)
g  = 2.0 * Sigmoid(a)                          (range: [0, 2])
T2' = Concat[g_B * Bottleneck; g_S * Skip]
```

### Invariants & Hard Locks
- **Frozen Lineage**: 100% of Candidate B parameters frozen (`requires_grad = False`, modules kept in `eval()` mode with running stats frozen).
- **Exact Identity Initialization**: $W_2 = 0, b_2 = 0 \implies a = 0 \implies g = 2 \cdot \sigma(0) \equiv 1.0$ bitwise at epoch 0.
- **Pinned Training Protocol**: Physical batch size = **14** (no gradient accumulation), 8 epochs, AdamW ($lr=10^{-4}, wd=10^{-4}$), seed = 42. (Note: batch size 12 in repo history was strictly a Phase 0 runtime/preflight feasibility test, never the training protocol of Candidate B or T2-SDSG).
- **Two Ablation Arms**:
  - **Arm A**: SDSG trained with standard segmentation loss $L_{\text{seg}} = \text{BCE} + \text{Dice}$.
  - **Arm B**: SDSG trained with $L_{\text{seg}} + 0.1 \cdot L_{\text{aux}}$ ($L_{\text{aux}}$ rewards $g \to 1$ on true crack pixels and penalizes $g \to 0$ on false-bridge neck corridors extracted from the 1,896 Crack500 train images, plus mild identity penalty).
- **Evaluation**: Canonical Setting A (448x448, stride 448, reflect padding, threshold $\tau=0.5$) across Full Validation $N=348$.
- **Test Integrity**: Official Test set $N=1124$ strictly sealed.
- **Bitwise Checksum Audit**:
  - Candidate B Checkpoint SHA256: `147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66` (**BITWISE IDENTICAL**).
  - Candidate B Named Parameters SHA256: `4aeda58ce6d2fb32f5b772cdbf92dec7fb913f9510b392910efe6674b0adf5d6` (**BITWISE IDENTICAL**).

---

## 2. Audit: Provenance of Baseline Bridge Count (110 vs 118)

A critical discrepancy was investigated: why does some prior documentation cite **110 bridges / 348**, whereas the T2-SDSG diagnostic reports **118**?

### Audit Findings & Provenance Trace:
1. **Mathematical Counting Definition**:
   - **`bridge_images` = 110**: The count of validation images that contain **at least one** false-bridge event (`(bridge_events > 0).sum()`). This is the metric reported in `results/diagnostics/phase6_bridge_localization/README.md` ("$110 / 348$ Total Bridge Images").
   - **`bridge_events` = 118**: The total count of false-bridge events summed across all validation images (`bridge_events.sum()`). This is the metric recorded in `results/diagnostics/phase6_d2_topology/topology_7models_master_paired.csv` (`Base_bridge_events` sum = 118).
2. **Sample-by-Sample Value Distribution**:
   - In Candidate B Full Validation ($N=348$):
     - `bridge_events == 0`: 238 images
     - `bridge_events == 1`: 102 images
     - `bridge_events == 2`: 8 images
   - Total Bridged Images: $102 + 8 =$ **110 images**.
   - Total False Bridge Events: $102 \times 1 + 8 \times 2 =$ **118 events**.
3. **Exact Match Verification**:
   - A bitwise sample-by-sample check between `df_master['Base_bridge_events']` and `validation_baseline_per_sample.csv['bridge_events']` confirms **0 differences across all 348 validation images**.
   - The metric implementation, component labeling, threshold $\tau=0.5$, and checkpoint are **100% identical**.
   - To eliminate all future ambiguity, both columns (`Bridge Images` and `Bridge Events`) are explicitly reported side-by-side in all tables.

---

## 3. Validation Subgroup Comparison Table

| Model | Cohort | $N$ | Dice | Recall | Precision | clDice | TSens | Bridge Imgs | $\Delta$Imgs | Bridge Evts | $\Delta$Evts | Cured Imgs | Created Imgs | Break Evts | $\Delta$Break | Area Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Candidate B** | **Full Validation** | **348** | **0.7641** | **0.8477** | **0.7337** | **0.8499** | **0.8872** | **110** | +0 | **118** | +0 | 0 | 0 | **35** | +0 | **1.2829** |
| | Consensus 7/7 | 101 | 0.7371 | 0.8997 | 0.6538 | 0.8455 | 0.9233 | 101 | +0 | 109 | +0 | 0 | 0 | 11 | +0 | 1.5421 |
| | Consensus Resistant | 82 | 0.7575 | 0.9086 | 0.6752 | 0.8586 | 0.9301 | 82 | +0 | 88 | +0 | 0 | 0 | 8 | +0 | 1.4864 |
| | Consensus Sensitive | 19 | 0.6491 | 0.8616 | 0.5617 | 0.7892 | 0.8937 | 19 | +0 | 21 | +0 | 0 | 0 | 3 | +0 | 1.7821 |
| | Clean Control | 56 | 0.7540 | 0.7922 | 0.7432 | 0.8200 | 0.8227 | **0** | +0 | **0** | +0 | 0 | 0 | 8 | +0 | 1.1061 |
| **Arm A (SDSG + $L_{\text{seg}}$)** | **Full Validation** | **348** | **0.7619** | **0.8366** | **0.7371** | **0.8464** | **0.8764** | **111** | **+1** | **119** | **+1** | 1 | 2 | **51** | **+16** | **1.2436** |
| | Consensus 7/7 | 101 | 0.7420 | 0.8959 | 0.6620 | 0.8493 | 0.9202 | 101 | +0 | 109 | +0 | 0 | 0 | 12 | +1 | 1.5075 |
| | Consensus Resistant | 82 | 0.7623 | 0.9056 | 0.6830 | 0.8613 | 0.9268 | 82 | +0 | 88 | +0 | 0 | 0 | 8 | +0 | 1.4546 |
| | Consensus Sensitive | 19 | 0.6544 | 0.8539 | 0.5714 | 0.7977 | 0.8920 | 19 | +0 | 21 | +0 | 0 | 0 | 4 | +1 | 1.7359 |
| | Clean Control | 56 | 0.7474 | 0.7771 | 0.7507 | 0.8091 | 0.8085 | **0** | +0 | **0** | +0 | 0 | 0 | 11 | +3 | 1.0675 |
| **Arm B (SDSG + $L_{\text{seg}} + 0.1 L_{\text{aux}}$)** | **Full Validation** | **348** | **0.7629** | **0.8145** | **0.7520** | **0.8417** | **0.8635** | **102** | **-8** | **111** | **-7** | 9 | 1 | **82** | **+47** | **1.1828** |
| | Consensus 7/7 | 101 | 0.7453 | 0.8660 | 0.6819 | 0.8407 | 0.9012 | 99 | -2 | 108 | -1 | 2 | 0 | 22 | +11 | 1.3974 |
| | Consensus Resistant | 82 | 0.7639 | 0.8753 | 0.7024 | 0.8525 | 0.9081 | 80 | -2 | 87 | -1 | 2 | 0 | 15 | +7 | 1.3577 |
| | Consensus Sensitive | 19 | 0.6648 | 0.8258 | 0.5932 | 0.7896 | 0.8714 | 19 | +0 | 21 | +0 | 0 | 0 | 7 | +4 | 1.5686 |
| | Clean Control | 56 | 0.7465 | 0.7550 | 0.7603 | 0.8057 | 0.7960 | **0** | +0 | **0** | +0 | 0 | 0 | 16 | +8 | 1.0143 |

---

## 4. Training Dynamics & Gate Modulation

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

---

## 5. Scientific Findings & Core Interpretations

### 1. Arm A Core Finding: `Stream Selection ≠ Topology Selection`
- Without auxiliary bridge loss ($L_{\text{aux}} = 0$), standard segmentation loss ($L_{\text{seg}}$) autonomously drives the gate to amplify Bottleneck ($g_B \to 1.401$) and heavily attenuate Skip ($g_S \to 0.428$).
- This demonstrates that segmentation gradients naturally prefer the low-frequency stability of Bottleneck over the high-frequency edge noise of Skip.
- **However, stream selection does NOT solve topology**: False bridges remained unchanged/increased ($110 \to 111$ images, $118 \to 119$ events), while true crack breakages increased substantially ($35 \to 51$ events, $+45.7\%$).

### 2. Arm B Core Finding: `Suppression Learned ≠ Selective Disentanglement`
- Under auxiliary supervision ($L_{\text{aux}}$), the gate successfully learns spatial suppression at bridge corridors, pushing local gates down to $g_{\text{min}} \approx 0.069$ and reducing bridge events from $118 \to 111$ (102 bridged images, 9 cured).
- **However, pointwise suppression cannot distinguish false connectivity from thin true-crack continuity**:
  - Global Recall dropped from $0.8477 \to 0.8145$ ($-3.32$ pp).
  - Area Ratio shrank from $1.2829 \to 1.1828$.
  - True crack breakages **more than doubled**, surging from $35 \to 82$ events ($+134\%$).
  - Among the 9 cured images, 3 experienced new true crack breakages ($0 \to 1$).

### 3. Resistant-82 Cohort Resilience
- In Consensus Resistant 82 ($N=82$ images, 88 baseline bridge events):
  - Arm A: 82 bridged images, 88 bridge events (0 cured, 0 created, 82 persistent). Breakages: 8 (+0).
  - Arm B: 80 bridged images ($-2$), 87 bridge events ($-1$ event, 2 cured, 0 created, 80 persistent). Breakages: **surged from $8 \to 15$ (+87.5%)**.
- The false bridges in Resistant 82 are almost entirely unyielding to pointwise gating: attempting to suppress them pointwise breaks legitimate cracks without clearing the bridge corridor.

### 4. Clean Control ($N=56$) Behavior
- Both Arm A and Arm B generated **zero new false bridges** on Clean Control (`created = 0`), maintaining $0/56$ bridge events.
- However, true crack breakages on Clean Control increased from $8 \to 11$ in Arm A and jumped to **16 in Arm B (doubled)**.

---

## 6. Answers to Mandatory Diagnostic Questions (Bounded Claims)

### Q1: Did the gate really reduce false bridges, or did it merely shrink/erode the prediction mask?
**Status: [Supported by Evidence]**
- The reduction in bridges in Arm B is predominantly driven by non-selective thinning and erosion.
- Evidence: Bridge reduction is accompanied by a substantial decline in Recall ($0.8477 \to 0.8145$), a reduction in Area Ratio ($1.2829 \to 1.1828$), a drop in Topological Sensitivity ($0.8872 \to 0.8635$), and a $+134\%$ surge in Breakage events ($35 \to 82$).

### Q2: In Resistant 82, what happened to the bridges?
**Status: [Supported by Evidence]**
- Resistant 82 bridges were almost completely immune to T2-SDSG gating.
- Out of 88 bridge events, 87 persisted in Arm B (80/82 bridged images persisted). Meanwhile, true crack breakages in this cohort increased from $8 \to 15$.

### Q3: Single-stream vs Pure-interaction behavior?
**Status: [Supported by Evidence]**
- Pointwise reweighting of streams cannot resolve bridge ambiguity. Arm A autonomously attenuated Skip ($g_S = 0.428$) and amplified Bottleneck ($g_B = 1.401$), yet failed to eliminate bridges because Bottleneck itself contains bridge-compatible activations (corroborating earlier findings where Bottleneck-only retained 37/82 Resistant bridges).

### Q4: Did Clean Control develop new false bridges?
**Status: [Supported by Evidence]**
- No new false bridges were created on Clean Control ($0/56$ images, 0 events in all arms). However, crack breakage doubled ($8 \to 16$ in Arm B).

### Q5: How does this trade-off compare to earlier Isotropic Attenuation?
**Status: [Supported by Evidence]**
- Like isotropic attenuation at Decoder B1, T2-SDSG suffers from a severe trade-off where any reduction in false bridges is coupled with unacceptable breakage of true cracks.

### Q6: What was the gate modulation distribution?
**Status: [Supported by Evidence]**
- Arm A: $g_B \in [0.02, 1.96]$, mean $= 1.401$; $g_S \in [0.02, 1.96]$, mean $= 0.428$.
- Arm B: $g_B \in [0.07, 1.36]$, mean $= 0.862$; $g_S \in [0.07, 1.36]$, mean $= 0.759$.

### Q7: Mechanism & Architectural Conclusion (Bounded Formulation)
**Status: [Carefully Bounded & Calibrated]**
- **Direct Conclusion [Supported by Evidence]**:
  > T2-SDSG pointwise, trong formulation và protocol đã kiểm thử, không đạt được mục tiêu bridge-selective suppression: mọi mức giảm bridge quan sát được đều đi kèm tổn thất continuity thật quá lớn.
- **Hypothesis on Receptive Field [Consistent with Hypothesis, but Not Directly Proven]**:
  > Kết quả phù hợp với giả thuyết rằng pointwise gating thiếu spatial context cần thiết để phân biệt false bridge corridor với đoạn crack thật mảnh, nhưng causal mechanism này chưa được chứng minh trực tiếp.
- **Architectural Implication [Not Proven / Open Question]**:
  > Liệu một cơ chế có trường tiếp nhận lớn hơn (ví dụ: oriented kernels, multi-scale / topological context) hoặc can thiệp ngược dòng tại Stage 0 / Stem AC có thể bóc tách được bridge mà không gây đứt gãy hay không vẫn là câu hỏi mở cần kiểm chứng bằng thực nghiệm độc lập. Không thể ngoại suy rằng "mọi can thiệp tại T2 đều không khả thi" chỉ từ kết quả của mô-đun pointwise 1x1 này.

---

## 7. Epistemic Classification of Conclusions

| Category | Claim | Status | Proof / Evidence |
| :--- | :--- | :---: | :--- |
| **Provenance** | 110 represents `bridge_images` ($\ge 1$ bridge), while 118 represents `bridge_events` (total event count due to 8 double-bridge images). | **Supported by Evidence** | Sample-by-sample exact match between master paired CSV and baseline evaluation across all 348 validation images. |
| **Optimization** | Standard segmentation loss autonomously suppresses Skip ($g_S \approx 0.43$) and amplifies Bottleneck ($g_B \approx 1.40$). | **Supported by Evidence** | `sdsg_training_log.csv` monotonic evolution across 8 epochs under $L_{\text{seg}}$ alone. |
| **Topology** | Stream selection does not equal topology selection: suppressing Skip does not reduce bridges (118 $\to$ 119) and increases breakages (35 $\to$ 51). | **Supported by Evidence** | Full Validation Setting A evaluation of Arm A. |
| **Trade-off** | T2-SDSG pointwise with auxiliary supervision reduces bridges (118 $\to$ 111) but causes severe collateral breakage (35 $\to$ 82). | **Supported by Evidence** | Full Validation Setting A evaluation of Arm B. |
| **Resistant 82** | Resistant 82 false bridges are essentially unaffected by pointwise gating (88 $\to$ 87 events), while breakages jump from 8 $\to$ 15. | **Supported by Evidence** | Subgroup analysis of 82 consensus resistant images. |
| **Causality** | Pointwise gating fails because it lacks 2D spatial context / geometric orientation. | **Consistent with Hypothesis, Not Directly Proven** | Empirical behavior matches spatial erosion, but receptive-field causality has not been isolated via a controlled kernel-size ablation. |
| **Generalization** | All possible interventions at T2 are incapable of solving bridges. | **Not Proven / Rejected Overclaim** | Only the specific $1 \times 1$ pointwise SDSG formulation was evaluated; context-aware or upstream methods remain uncharacterized. |
