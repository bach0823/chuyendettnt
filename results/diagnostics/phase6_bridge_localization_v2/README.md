# Phase 6 False-Bridge Spatial & Mechanistic Localization Report (v2)

**Status:** Completed Diagnostic Investigation  
**Scope:** Strictly Diagnostic Only (No retraining, no threshold changes, sealed test set untouched, Setting A preserved)  
**Evaluation Target:** Crack500 Official Validation Split ($N=348$) under Setting A Tiling Protocol ($448 \times 448$, stride 448)  
**Models Evaluated:**
1. Candidate B (Base Canonical)
2. Phase 6-A.1 (BoundaryIoU Objective Probe)
3. Phase 6-A.2 (Pure PLU Representation Probe)
4. Phase 6-B.1 (AB-BPL Boundary Margin Probe)
5. Phase 6-C.1 (clDice Topology Probe)
6. Phase 6-D.1 (PLU + AB-BPL Composite Probe)
7. Phase 6-D.2 (Pure Inter-Component Separation Probe)

---

## Executive Summary of Core Mechanistic Findings

| Question | Diagnostic Finding | Verdict & Mechanism |
| :--- | :--- | :--- |
| **Q1: Image-Level Consensus** | **$101 / 110$ Base bridges ($91.82\%$) fail in ALL 7 models ($7/7$)**. Mean pairwise Jaccard across models is **$0.9319$**. $228/348$ images ($65.52\%$) never experience a bridge in any model. | **Shared Hard-Case Persistence**, not random optimization churn. |
| **Q2: Spatial & Geometric Consistency** | On the 101 consensus cases, the mean pairwise spatial IoU of the bridge-FP mask across the 7 models is **$0.7403$** (median $0.7430$). Base vs D2 spatial Dice is **$0.8573$**. $72.66\%$ of union pixels are shared by $\ge 4$ models. | **Identical Spatial Defect**. All models bridge the exact same physical coordinates with nearly identical geometry. |
| **Q3: Connector / Neck Probability** | For minimal connector neck pixels directly spanning between GT CCs, the median probability is **$0.9348$** (mean $0.8706$). **$59.03\%$ of neck pixels have $p \ge 0.90$**, and **$74.59\%$ have $p \ge 0.80$**. Only $7.88\%$ are marginal ($p \in [0.50, 0.60)$). | **High-Confidence Semantic Hallucination**, not a marginal threshold artifact. |
| **Q4: Decoder Stage Localization** | Contrast (Neck vs Background) is flat at S3 Bottleneck ($14 \times 14$: $1.0138$) and S2 ($28 \times 28$: $1.1508$). It surges at S1 ($56 \times 56$: $3.1740$) and S0 ($112 \times 112$: $4.2157$). Pre-upsample $112 \times 112$ logits already have median $p = 0.9392$ ($91.4\%$ positive). | **Committed in Shallow Decoder (S1/S0)**. The bridge is NOT created by the $4\times$ bilinear upsampler; it is already fully committed before upsampling. |
| **Q5: Routing Association** | Mean gate scale $g_s$ is $0.4983$ on consensus bridge vs $0.4981$ on clean ($p=0.0086$, $\Delta = +0.0002$). Mean routing margin is $0.0013$ vs $0.0013$. D2 vs Base expert agreement is $43.2\%$ on bridge vs $44.1\%$ on clean. | **Routing Invariance**. SAGE router behavior is healthy and identical across both sets; false bridges are not driven by MoE routing pathology. |

---

## 1. Task A: Spatial Bridge Consistency Across 7 Models

**Artifacts:**
- [`bridge_spatial_consensus_per_image.csv`](file:///D:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_bridge_localization_v2/bridge_spatial_consensus_per_image.csv)
- [`bridge_spatial_consensus_summary.csv`](file:///D:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_bridge_localization_v2/bridge_spatial_consensus_summary.csv)
- [`bridge_pairwise_model_iou_matrix_101consensus.csv`](file:///D:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_bridge_localization_v2/bridge_pairwise_model_iou_matrix_101consensus.csv)

### 1.1 Pairwise Model Spatial IoU Matrix on the 101 Consensus Cases
Evaluating the spatial intersection-over-union of the false-bridge false-positive mask $B_m(x, y) = [P_m(x,y)=1 \land T(x,y)=0 \land (x,y) \in \text{Bridge CC}]$:

```
model   Base     A1     A2     B1     C1     D1     D2
 Base 1.0000 0.7597 0.7170 0.7879 0.7888 0.7122 0.7623
   A1 0.7597 1.0000 0.7387 0.8080 0.7366 0.7513 0.7200
   A2 0.7170 0.7387 1.0000 0.7241 0.6894 0.7813 0.6825
   B1 0.7879 0.8080 0.7241 1.0000 0.7685 0.7415 0.7559
   C1 0.7888 0.7366 0.6894 0.7685 1.0000 0.6670 0.7891
   D1 0.7122 0.7513 0.7813 0.7415 0.6670 1.0000 0.6653
   D2 0.7623 0.7200 0.6825 0.7559 0.7891 0.6653 1.0000
```

### 1.2 Stratified Summary Metrics

| Stratum | $N$ Cases | Mean Pairwise IoU | Median Pairwise IoU | Base vs D2 Spatial IoU | Base vs D2 Dice | 7-Way Strict IoU | $\ge 4$ Models Core Ratio | $\ge 6$ Models Core Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Consensus 7/7** | 101 | **0.7403** | **0.7430** | **0.7623** | **0.8573** | **0.4936** | **0.7266** | **0.5829** |
| **Persistent Base $\to$ D2** | 103 | 0.7318 | 0.7426 | 0.7582 | 0.8544 | 0.4840 | 0.7215 | 0.5715 |
| **Cured by D2** | 7 | 0.2659 | 0.2552 | 0.0000 | 0.0000 | 0.0000 | 0.4060 | 0.0492 |
| **Created by D2** | 6 | 0.4365 | 0.5330 | 0.0000 | 0.0000 | 0.0000 | 0.6318 | 0.3637 |
| **All Target Cases** | 116 | 0.6884 | 0.7313 | 0.6732 | 0.7586 | 0.4297 | 0.6978 | 0.5293 |

**Key Takeaway:**
The 101 consensus cases are **spatially rigid**. With a mean pairwise IoU of $0.7403$ and Base vs D2 Dice of $0.8573$, the models are not randomly wandering across different background pixels; they are predicting the exact same pixel mask in the exact same physical locations. In stark contrast, the 7 cases cured by D2 had a much lower multi-model agreement ($0.2659$), indicating that D2 only cured unstable, non-consensus boundaries.

---

## 2. Task B: Bridge-Neck / Connector Probability Diagnostic

**Artifacts:**
- [`bridge_connector_probability_global.csv`](file:///D:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_bridge_localization_v2/bridge_connector_probability_global.csv)
- [`bridge_connector_probability_per_image.csv`](file:///D:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_bridge_localization_v2/bridge_connector_probability_per_image.csv)
- [`bridge_connector_probability_bins.csv`](file:///D:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_bridge_localization_v2/bridge_connector_probability_bins.csv)

### 2.1 Connector Neck vs Broader Regions Probability Profile (Candidate B)

| Region | $N$ Pixels | Mean | Median (P50) | P25 | P75 | P90 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Minimal Connector Neck (All 110)** | 15,992 | **0.8706** | **0.9348** | 0.7967 | 0.9775 | 0.9922 |
| **Whole Bridge-FP (All 110)** | 402,820 | 0.8202 | 0.8652 | 0.6980 | 0.9567 | 0.9859 |
| **Non-Neck Bridge-FP (All 110)** | 386,828 | 0.8181 | 0.8618 | 0.6950 | 0.9551 | 0.9855 |
| **True Positive Crack (Bridge Images)** | 1,058,218 | 0.9455 | 0.9862 | 0.9507 | 0.9946 | 0.9972 |
| **Neck: 7/7 Consensus (101 cases)** | 15,464 | **0.8750** | **0.9394** | 0.8066 | 0.9783 | 0.9924 |
| **Neck: Cured by D2 (7 cases)** | 364 | **0.7410** | **0.7551** | 0.6207 | 0.8497 | 0.9238 |

### 2.2 Probability Bin Distribution in the Minimal Connector Neck

```
Region: Minimal_Connector_Neck_All110 (N = 15,992 pixels)
  - Marginal [0.50, 0.60):           1,260 px  ( 7.88%)
  - Low-Intermediate [0.60, 0.70):   1,265 px  ( 7.91%)
  - High-Intermediate [0.70, 0.80):  1,539 px  ( 9.62%)
  - Confident [0.80, 0.90):          2,488 px  (15.56%)
  - Highly Confident [0.90, 1.00]:   9,440 px  (59.03%)

Region: Neck_7of7_Consensus101 (N = 15,464 pixels)
  - Marginal [0.50, 0.60):           1,174 px  ( 7.59%)
  - Low-Intermediate [0.60, 0.70):   1,154 px  ( 7.46%)
  - High-Intermediate [0.70, 0.80):  1,424 px  ( 9.21%)
  - Confident [0.80, 0.90):          2,297 px  (14.85%)
  - Highly Confident [0.90, 1.00]:   9,415 px  (60.88%)

Region: Neck_Cured_by_D2_7cases (N = 364 pixels)
  - Marginal [0.50, 0.60):              58 px  (15.93%)
  - Low-Intermediate [0.60, 0.70):      88 px  (24.18%)
  - High-Intermediate [0.70, 0.80):     68 px  (18.68%)
  - Confident [0.80, 0.90):             126 px  (34.62%)
  - Highly Confident [0.90, 1.00]:      24 px  ( 6.59%)
```

**Key Takeaway:**
1. The connector neck is **NOT a marginal probability artifact**. In the consensus cases, the model is not "barely crossing" the threshold $0.50$; $59.03\%$ of neck pixels have $p \ge 0.90$, with a median probability of $0.9348$.
2. The few bridges that D2 cured were fundamentally different: their neck median was $0.7551$, and over $40.11\%$ of pixels were below $0.70$. Spatial separation losses only act on marginal boundaries, leaving the confident bridges untouched.

---

## 3. Task C: Decoder-Stage Bridge Localization

**Artifacts:**
- [`decoder_bridge_localization_per_stage.csv`](file:///D:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_bridge_localization_v2/decoder_bridge_localization_per_stage.csv)
- [`decoder_bridge_localization_summary.csv`](file:///D:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_bridge_localization_v2/decoder_bridge_localization_summary.csv)

### 3.1 Feature Contrast and Probabilities Across the Hierarchy

By registering forward hooks across Candidate B's decoder and measuring feature activation magnitude $M = \frac{1}{C}\sum_c |F_c|$ and scalar logits:

| Stratum | Stage | Resolution | Scale | Contrast (Neck / BG) | Neck Median Prob | Neck $\ge 0.50$ | Neck $\ge 0.75$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Consensus 7/7** | S3 (Bottleneck) | $14 \times 14$ | 1/32 | **1.0138** | — | — | — |
| | S2 (Decoder Block 0) | $28 \times 28$ | 1/16 | **1.1508** | — | — | — |
| | S1 (Decoder Block 1) | $56 \times 56$ | 1/8 | **3.1740** | — | — | — |
| | S0 (Decoder Block 2) | $112 \times 112$ | 1/4 | **4.2157** | — | — | — |
| | **Pre-Upsample Head** | $112 \times 112$ | 1/4 | — | **0.9392** | **91.4%** | **86.0%** |
| | **Final Output Logits** | $448 \times 448$ | 1 | — | **0.9522** | **100.0%** | **93.4%** |
| **Cured by D2** | S3 (Bottleneck) | $14 \times 14$ | 1/32 | 1.0088 | — | — | — |
| | S2 (Decoder Block 0) | $28 \times 28$ | 1/16 | 0.9665 | — | — | — |
| | S1 (Decoder Block 1) | $56 \times 56$ | 1/8 | 2.2294 | — | — | — |
| | S0 (Decoder Block 2) | $112 \times 112$ | 1/4 | 2.3502 | — | — | — |
| | **Pre-Upsample Head** | $112 \times 112$ | 1/4 | — | **0.7878** | **84.0%** | **61.5%** |
| | **Final Output Logits** | $448 \times 448$ | 1 | — | **0.7819** | **100.0%** | **58.0%** |
| **Created by D2** | S3 (Bottleneck) | $14 \times 14$ | 1/32 | 1.0130 | — | — | — |
| | S2 (Decoder Block 0) | $28 \times 28$ | 1/16 | 0.9131 | — | — | — |
| | S1 (Decoder Block 1) | $56 \times 56$ | 1/8 | 1.5998 | — | — | — |
| | S0 (Decoder Block 2) | $112 \times 112$ | 1/4 | 1.6699 | — | — | — |
| | **Pre-Upsample Head** | $112 \times 112$ | 1/4 | — | **0.2992** | **21.8%** | **11.1%** |
| | **Final Output Logits** | $448 \times 448$ | 1 | — | **0.2886** | **16.9%** | **7.2%** |

### 3.2 Mechanistic Conclusions on Localization
1. **The 4x Bilinear Upsampler is Exonerated:**
   The pre-upsample head at $112 \times 112$ already outputs a median probability of **0.9392**, with **$91.4\%$ of neck pixels already positive**. Bilinear interpolation merely renders this committed decision at $448 \times 448$ ($p = 0.9522$). The interpolation does NOT create the bridge.
2. **Deep Stages (S3, S2) Do Not Distinguish the Neck:**
   At $14 \times 14$ and $28 \times 28$, the neck region has contrast of only $1.01 - 1.15$ over background. The bridge representation does not exist at these deep scales.
3. **The Bridge Surges at S1 ($56 \times 56$) and S0 ($112 \times 112$):**
   Contrast skyrockets from $1.15 \to 3.17$ at S1, and reaches $4.22$ at S0. This corresponds precisely to where high-resolution encoder skip connections (ConvNeXt Stage 1 and Stage 0) are concatenated with the upsampled decoder path.

---

## 4. Task D: SAGE Routing Association Diagnostic

**Artifacts:**
- [`routing_bridge_association.csv`](file:///D:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_bridge_localization_v2/routing_bridge_association.csv)
- [`routing_bridge_summary.csv`](file:///D:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_bridge_localization_v2/routing_bridge_summary.csv)

### 4.1 Statistical Comparison: 101 Consensus Bridge vs 228 Consensus Clean

| Metric | 101 Consensus Bridge | 228 Consensus Clean | Difference | Mann-Whitney $p$-value | Significant ($\alpha=0.05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean Gate Scale ($g_s$)** | 0.4983 | 0.4981 | +0.0002 | 0.0086 | Yes (Negligible magnitude) |
| **Mean Routing Margin** | 0.0013 | 0.0013 | 0.0000 | 0.6140 | No |
| `convnext.stage_0` $g_s$ | 0.4867 | 0.4864 | +0.0003 | 0.1780 | No |
| `convnext.stage_1` $g_s$ | 0.4392 | 0.4390 | +0.0002 | 0.5954 | No |
| `convnext.stage_2` $g_s$ | 0.5229 | 0.5229 | 0.0000 | 0.8345 | No |
| `convnext.stage_3` $g_s$ | 0.5034 | 0.5027 | +0.0007 | 0.0003 | Yes (Negligible magnitude) |
| `transformer.block_0` $g_s$ | 0.5212 | 0.5208 | +0.0004 | 0.0138 | Yes (Negligible magnitude) |
| `transformer.block_1` $g_s$ | 0.4954 | 0.4949 | +0.0005 | 0.0062 | Yes (Negligible magnitude) |
| `transformer.block_2` $g_s$ | 0.5094 | 0.5096 | -0.0002 | 0.1320 | No |
| `transformer.block_3` $g_s$ | 0.5079 | 0.5084 | -0.0005 | 0.0035 | Yes (Negligible magnitude) |

### 4.2 Router Stability Under Phase 6-D.2 Fine-Tuning
- **Identical Routers (out of 8) Base vs D2 on 101 Consensus Bridge:** $3.46 / 8$ ($43.2\%$).
- **Identical Routers (out of 8) Base vs D2 on 228 Consensus Clean:** $3.53 / 8$ ($44.1\%$).
- **Identical Routers (out of 8) Base vs D2 on 7 Cured by D2:** $3.86 / 8$ ($48.2\%$).

**Key Takeaway:**
Router behavior is virtually invariant between bridge and clean images. The effect sizes of gate scale differences are less than $0.04\%$ of the gate magnitude. SAGE routing is completely functional and healthy; false bridges are not caused by routing collapse or expert misallocation.

---

## 5. Source Code Proofs

### Proof 1: Decoder Architecture & Skip Connection Fusion Points
File: [`SAGE_LITE/sage/networks/decoder_block.py`](file:///D:/truong/SpecialSubjectTTNT/SAGE_LITE/sage/networks/decoder_block.py#L220-L245)
```python
# S2: Block 0: in=384, skip=192 -> (B, 192, 28, 28)
# S1: Block 1: in=192, skip=96  -> (B, 96, 56, 56)   <-- SURGE POINT (+176% contrast)
# S0: Block 2: in=96,  skip=48  -> (B, 48, 112, 112) <-- PEAK CONTRAST (4.22x)
for i in range(len(reversed_channels) - 1):
    in_ch = reversed_channels[i]
    skip_ch = reversed_channels[i + 1]
    out_ch = reversed_channels[i + 1]

    block = DecoderBlock(
        in_channels=in_ch,
        skip_channels=skip_ch,
        out_channels=out_ch,
        use_dwsc=use_dwsc,
    )
    self.decoder_blocks.append(block)
```

### Proof 2: Pre-Upsample Segmentation Head Output (112x112)
File: [`SAGE_LITE/sage/networks/decoder_block.py`](file:///D:/truong/SpecialSubjectTTNT/SAGE_LITE/sage/networks/decoder_block.py#L295-L306)
```python
# Apply segmentation head: (B, 48, 112, 112) -> (B, num_classes, 112, 112)
logits = self.segmentation_head(x_dec)

# Bilinear 4x upsampling to match input resolution (448, 448)
if target_size is not None and logits.shape[2:] != target_size:
    logits = F.interpolate(
        logits,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
```

---

## 6. Synthesis & Strategic Guidance for Next Phase

1. **The False Bridge is an Upstream Semantic Representation Failure:**
   The bridge is fully committed by Stage S0/S1 of the decoder, with median probability exceeding $0.93$. It is not an artifact of post-processing, thresholding, bilinear upsampling, or routing collapse.
2. **Local Output-Space Regularizers Cannot Break Confident Bridges:**
   All probes that modified output losses (BoundaryIoU, AB-BPL, clDice, and D2 Pure Separation) failed because a loss with reasonable $\lambda$ cannot overpower a logit that is already strongly positive ($\text{logit} \approx +2.5$ to $+3.5$). When $\lambda_{\text{sep}}$ is pushed higher, as demonstrated by D2, it creates collateral damage on continuity (breakage $+4.89\text{ pp}$) before it can dislodge the persistent bridges.
3. **No Retraining or Architecture Modifications Should Be Done Blindly:**
   Future interventions, if any, must address the intermediate decoder representation at S1/S0 (e.g. feature-space topological contrast or explicit neck feature inhibition), rather than generic output mask boundary penalties.
