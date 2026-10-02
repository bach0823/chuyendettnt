# Phase 6 Counterfactual Shallow Skip Diagnostic Report (v3)

**Status:** Completed Diagnostic Investigation  
**Scope:** Strictly Diagnostic Only (No training, no optimizer, no checkpoint modification, sealed test set untouched, Setting A preserved)  
**Evaluation Target:** Crack500 Official Validation Split ($N=25$ Representative Stratified Samples) under Setting A Tiling Protocol ($448 \times 448$, stride 448)  
**Core Model:** Candidate B Official Checkpoint (`P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth`, $10,118,955$ parameters)  
**Methodology:** Controlled inference-time PyTorch forward pre-hooks neutralizing skip connections with zero weight or architecture changes.

---

## Executive Summary: Causal vs Correlational Verdict

$$
\boxed{
\text{Activation Localization } \neq \text{ Causal Localization}
}
$$

In the Phase 6-v2 diagnostic, intermediate decoder activation contrast between the connector neck and background was found to surge at Stage S1 ($56 \times 56$: $3.17\times$) and peak at S0 ($112 \times 112$: $4.22\times$). This counterfactual experiment was designed to resolve whether those shallow skip features **cause** the bridge, or merely **reflect/amplify** an upstream decision.

| Counterfactual Condition | Consensus Bridges Remaining | Neck Median Prob (Consensus 7/7) | Breakage Cases ($N=25$) | Mean Dice ($N=25$) | Mechanistic Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **C0 — Normal Baseline** | **10/10** | **0.9522** | 1/25 | 0.7750 | Canonical reference (exact reproduction, $\Delta=0.000$). |
| **C2_zero — S0-off (Zero S0 skip)** | **10/10** | **0.9406** | 4/25 | 0.7809 | **S0 skip removal does NOT reduce bridges.** Bridge probability remains $>0.94$. |
| **C2_mean — S0-mean (Mean ref)** | **10/10** | **0.9499** | 1/25 | 0.7741 | On-manifold neutralization: **100% of consensus bridges persist.** |
| **C1_mean — S1-mean (Mean ref)** | **10/10** | **0.9155** | 4/25 | 0.7601 | Neutralizing S1 spatial features leaves **all 10 consensus bridges intact.** |
| **C3_mean — S1+S0-mean (Both)** | **10/10** | **0.9118** | 4/25 | 0.7550 | Neutralizing both S1 and S0 spatial skips **fails to dislodge consensus bridges.** |
| **C1_zero — S1-off (Zero S1 skip)** | 5/10 | 0.4379 | 17/25 | 0.4855 | **Catastrophic off-manifold collapse.** Recall drops from $85.8\% \to 36.9\%$. |
| **C3_zero — S1+S0-off (Zero both)**| 3/10 | 0.3041 | 19/25 | 0.3401 | Extreme foreground collapse. Not a selective bridge cure. |

**Primary Mechanistic Conclusion:**
> The evidence **disfavors** the hypothesis that shallow high-resolution skip features (S1/S0) are the causal origin of the false bridges.
> When S0 and S1 spatial features are neutralized on-manifold (C1_mean, C2_mean, C3_mean), **10 out of 10 consensus bridges persist**, and the connector neck probability remains above **$0.91$**.
> Therefore, the high activation contrast observed at S1/S0 in v2 represents **shallow decoder amplification of an upstream semantic decision** that was already committed prior to S0, rather than an error injected de novo by the high-resolution skip connections.

---

## 1. Architectural Source Code Inspection & Tensor Contracts

Inspection of [`SAGE_LITE/sage/networks/decoder_block.py`](file:///D:/truong/SpecialSubjectTTNT/SAGE_LITE/sage/networks/decoder_block.py#L220-L295):

1. **Tensor Definitions & Entry Points:**
   - **S2 Skip (`reversed_skips[0]`):** ConvNeXt Stage 2 feature map, shape $(B, 192, 28, 28)$. Enters `decoder_blocks[0]`.
   - **S1 Skip (`reversed_skips[1]`):** ConvNeXt Stage 1 feature map, shape $(B, 96, 56, 56)$. Enters `decoder_blocks[1]`.
   - **S0 Skip (`reversed_skips[2]`):** ConvNeXt Stage 0 feature map, shape $(B, 48, 112, 112)$. Enters `decoder_blocks[2]`.
2. **Merge Mechanics:**
   In every `DecoderBlock.forward(x, skip)`:
   ```python
   x = self.upsample(x)  # ConvTranspose2d 2x
   x = torch.cat([x, skip], dim=1)  # Channel concatenation
   x = self.conv1(x)     # 3x3 Conv + BatchNorm2d + ReLU
   x = self.conv2(x)     # 3x3 Conv + BatchNorm2d + ReLU
   ```
   At Block 1 (S1): $(192 \text{ up} + 96 \text{ skip}) \to 288 \to 96$.  
   At Block 2 (S0): $(96 \text{ up} + 48 \text{ skip}) \to 144 \to 48$.
3. **Intervention Implementation:**
   Forward pre-hooks registered on `decoder_blocks[1]` and `decoder_blocks[2]` intercept `(x, skip)` and substitute `skip` dynamically without touching model weights or persistent graph structure.

---

## 2. Quantitative Results Across Conditions

**Artifacts Generated:**
- [`counterfactual_conditions_per_sample.csv`](file:///D:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_bridge_localization_v3/counterfactual_conditions_per_sample.csv)
- [`counterfactual_bridge_probability.csv`](file:///D:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_bridge_localization_v3/counterfactual_bridge_probability.csv)
- [`counterfactual_stage_activation.csv`](file:///D:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_bridge_localization_v3/counterfactual_stage_activation.csv)
- [`counterfactual_summary.csv`](file:///D:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_bridge_localization_v3/counterfactual_summary.csv)

### 2.1 Summary Metrics Across All 25 Representative Cases

| Condition | Description | Bridges (Total 16) | Bridges (7/7 Cons.) | Breakage ($N=25$) | Dice | Precision | Recall | clDice | Neck Med. Prob (7/7) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C0** | Normal Baseline | 16/16 | 10/10 | 1/25 | 0.7750 | 0.7236 | 0.8577 | 0.8490 | 0.9522 |
| **C1_zero** | Zero S1 Skip ($96\text{ch}, 56^2$) | 5/16 | 5/10 | 17/25 | 0.4855 | 0.9058 | 0.3693 | 0.5653 | 0.4379 |
| **C2_zero** | Zero S0 Skip ($48\text{ch}, 112^2$) | 14/16 | 10/10 | 4/25 | 0.7809 | 0.7557 | 0.8283 | 0.8440 | 0.9406 |
| **C3_zero** | Zero S1 + S0 Skips | 3/16 | 3/10 | 19/25 | 0.3401 | 0.8770 | 0.2397 | 0.4102 | 0.3041 |
| **C1_mean** | Mean Ref S1 Skip | 14/16 | 10/10 | 4/25 | 0.7601 | 0.7408 | 0.7995 | 0.8264 | 0.9155 |
| **C2_mean** | Mean Ref S0 Skip | 16/16 | 10/10 | 1/25 | 0.7741 | 0.7246 | 0.8544 | 0.8477 | 0.9499 |
| **C3_mean** | Mean Ref S1 + S0 Skips | 13/16 | 10/10 | 4/25 | 0.7550 | 0.7406 | 0.7894 | 0.8232 | 0.9118 |

---

## 3. Detailed Answers to Core Diagnostic Questions

### Q1: Does S1/S0 counterfactual removal reduce bridge probability at the original Base connector sites?
* **For S0-off:** **NO.**
  - Under `C2_zero`, the median probability at the connector neck across the 10 consensus cases is **0.9406** (compared to $0.9522$ in C0). $89.11\%$ of the neck area remains $\ge 0.75$.
  - Under `C2_mean`, the median probability is **0.9499**, with $92.90\%$ of the neck area $\ge 0.75$.
* **For S1-off:**
  - Under on-manifold spatial neutralization (`C1_mean`), the median neck probability decreases only slightly from **0.9522 to 0.9155**, remaining overwhelmingly positive ($87.02\%$ remains $\ge 0.75$).
  - Under `C1_zero`, the probability drops to $0.4379$, but this is accompanied by a collapse of true crack foreground across the entire image (Recall drops to $0.3693$).

### Q2: Does it actually reduce false-bridge topology?
* **On Consensus 7/7 Cases:** **NO.**
  - Under `C2_zero`, **10/10 consensus bridges persist**.
  - Under `C2_mean`, **10/10 consensus bridges persist**.
  - Under `C1_mean`, **10/10 consensus bridges persist**.
  - Under `C3_mean`, **10/10 consensus bridges persist**.
* **On Non-Consensus / Cured Cases:**
  - The 2 cases where bridges disappeared in `C2_zero` (`20160307_145059_1281_721` and `IMG_2941_1297_969`) were precisely cases that were already fragile and easily cured by D2. Consensus bridges showed zero topological reduction.

### Q3: What happens to breakage and thin-crack behavior?
* Neutralizing S0 (`C2_mean`) produces **zero collateral damage**: breakage remains at $1/25$ (identical to C0), and clDice remains $0.8477$ (vs $0.8490$ in C0).
* Neutralizing S1 spatial features (`C1_mean`) causes modest breakage increase from $1/25 \to 4/25$ ($+12\text{ pp}$), with clDice dropping from $0.8490 \to 0.8264$.
* Abruptly zeroing S1 (`C1_zero`) causes **catastrophic fragmentation**: breakage explodes to **$17/25$ ($68\%$)**, Recall collapses by **$48.84\text{ pp}$**, and Dice plummets from $0.7750 \to 0.4855$.

### Q4: Which is the more plausible interpretation?
* The data **strongly supports Interpretation B/C**:
  > **Shallow skip amplification, but NOT causal origin.**
* The bridge decision is **already encoded in the upsampled deep representation** coming from S2 ($28 \times 28$) into S1 ($56 \times 56$).
* The high-resolution skip features at S0 ($112 \times 112$) and S1 ($56 \times 56$) provide high-frequency sharpening and boundary localization, which amplifies the contrast of whatever foreground signal is delivered to them. Because the upsampling path already carries a strong positive crack hallucination across the gap, S0/S1 faithfully sharpens that bridge.
* Removing S0/S1 skip features does not remove the bridge because the upsampled stream itself is already positive enough to cross the classification threshold.

---

## 4. Normalization and Distribution-Shift Verification

Norm tracking across conditions:

| Tensor | Normal (C0) L2 Norm | C1_zero L2 Norm | C1_mean L2 Norm | C2_zero L2 Norm | C2_mean L2 Norm |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **S1 Skip (entering Block 1)** | 1.6112 | 0.0000 | 0.8845 | 1.6112 | 1.6112 |
| **S1 Out (merged Block 1)** | 0.7616 | **1.5783** *(+107%)* | **0.6878** *(-9.7%)* | 0.7616 | 0.7616 |
| **S0 Skip (entering Block 2)** | 1.7311 | 1.7311 | 1.7311 | 0.0000 | 0.9512 |
| **S0 Out (merged Block 2)** | 0.7476 | 0.7453 | 0.7425 | **0.8295** *(+11%)* | **0.7425** *(-0.7%)* |

**Significance:**
- `C1_zero` creates an extreme BatchNorm activation anomaly at Block 1 ($1.5783$ vs $0.7616$), explaining why zeroing S1 causes catastrophic foreground collapse.
- In contrast, `C1_mean`, `C2_mean`, and `C2_zero` remain within regular operational norms, providing a reliable, undistorted counterfactual readout.

---

## 5. Next Highest-Information Scientific Question

Since the false bridge is **already committed upstream of S0/S1**, what is the next critical question?

> **Q5: Is the false bridge committed in the bottleneck / deep transformer representation (S3 / Stage 3 + ViT tokens at $14 \times 14$), or does it emerge during the first deep upsampling step in Decoder Block 0 (S3 $\to$ S2, $14 \times 14 \to 28 \times 28$)?**

At $14 \times 14$, the patch token size is $32 \times 32$ pixels. A gap of $8$ pixels between two cracks falls entirely inside a single ViT patch token. If the ViT self-attention or patch token embedding merges nearby crack features because they co-occur in the same receptive field, the deep bottleneck representation itself is already merged before the decoder ever begins. Testing token-level separability at $14 \times 14$ is the exact next step.
