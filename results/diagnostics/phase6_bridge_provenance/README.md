# Phase 6: Bridge Provenance Across Scales Diagnostic Report

**Target Checkpoint:** Candidate B (`P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth`, 10,118,955 params)  
**Evaluation Protocol:** Official Setting A ($448 \times 448$, stride 448, non-overlapping, reflect padding, exact cropping, $\tau = 0.5$)  
**Cohort Analyzed:** All $N=110$ validation images exhibiting Base False Bridges  
- **Consensus 7/7:** $N=101$ samples (persists across all 7 Phase 6 models: Base, A1, A2, B1, C1, D1, D2)
- **Cured by D2:** $N=7$ samples (cured under D.2 InterComponentSeparationLoss)
- **Other Base Bridges:** $N=2$ samples (intermediate model consensus: 5 or 6 models)
**Status:** Diagnostic-only counterfactual analysis complete. Frozen and reproducible.

---

## 1. Audit Resolutions (Consistency & Methodology Verification)

Before analyzing bridge provenance, two potential discrepancies in earlier reports were audited and fully accounted for:

### A. Breakage Baseline Discrepancy (33 vs 34 Cases)
- **Historical Master paired:** $33/348 = 9.48\%$
- **V5 Full-Validation run:** $34/348 = 9.77\%$
- **Discrepancy Source:** Pairwise sample diff across all 348 samples identified exactly 3 divergent cases:
  - `20160307_162438_1921_1081`: Master = 0, V5 = 1 (area diff: $-21$ px out of 15,056 px, $-0.14\%$)
  - `20160326_141808_641_721`: Master = 0, V5 = 1 (area diff: $+116$ px out of 3,917 px, $+2.9\%$)
  - `20160330_172309_1921_721`: Master = 1, V5 = 0 (area diff: $+25$ px out of 19,275 px, $+0.13\%$)
- **Root Cause:** Historical master diagnostics were executed on Google Colab GPU (Linux) with `torch.amp.autocast('cuda')` (FP16), whereas local validation was executed on Windows CUDA in full FP32 (preventing ViT-Tiny attention float16 underflow/overflow). The miniscule logit variation ($\sim 10^{-3}$) along razor-thin 1-pixel boundary necks flipped connectivity on exactly 3 borderline samples (net $+1$ sample). 
- **Conclusion:** 345/348 ($99.14\%$) samples are bitwise identical; False Bridge count is identical ($110/348$ exact); Global Dice is identical ($0.7641$ exact). This is normal numerical precision tolerance.

### B. Area Excess Discrepancy (+8.16% vs +28.29%)
- **Aggregate Area Excess:** $\frac{\sum_{i=1}^N \text{pred}_i - \sum_{i=1}^N \text{gt}_i}{\sum_{i=1}^N \text{gt}_i} \times 100\% = \mathbf{+8.16\%}$
- **Macro Sample-Mean Area Excess:** $\frac{1}{N}\sum_{i=1}^N \frac{\text{pred}_i - \text{gt}_i}{\text{gt}_i} \times 100\% = \mathbf{+28.29\%}$
- **Conclusion:** Both numbers are numerically exact on the same data. Historical reports cited the dataset-wide Aggregate Area Excess ($+8.16\%$), while the V5 per-sample table averaged per-image percentages ($+28.29\%$).

---

## 2. Multi-Scale Hierarchy & Metric Definitions

To localize where the false bridge originates and where it is amplified, hooks were placed at 8 architectural stages of Candidate B:

| Stage ID | Architectural Location | Spatial Res | Channels | Functional Role |
| :--- | :--- | :---: | :---: | :--- |
| `bottleneck` | Block 0 Input (ViT / ConvNeXt output) | $14 \times 14$ | 384 | Deepest semantic representation |
| `S2` | Block 0 Output (post-upsample & convs) | $28 \times 28$ | 192 | Deep decoder feature map |
| `T1` | Block 1 Post-Upsample (`upsample(S2)`) | $56 \times 56$ | 192 | 2x spatial expansion before skip |
| `T2` | Block 1 Post-Skip (`cat([T1, skip1])`) | $56 \times 56$ | 288 | Fusion with encoder Stage 1 skip |
| `T3` | Block 1 Post-Conv1 (`conv1(T2)`) | $56 \times 56$ | 96 | Spatial mixing stage 1 |
| `T4` (`S1`) | Block 1 Post-Conv2 (`conv2(T3)`) | $56 \times 56$ | 96 | Spatial mixing stage 2 (Block 1 out) |
| `S0` | Block 2 Output | $112 \times 112$ | 48 | Shallowest decoder stage |
| `Final_Head` | Segmentation Head Output | $448 \times 448$ | 1 | Final logits / sigmoid probability |

### Region of Interest (ROI) Definitions
For each bridged sample:
- **`Neck` ($E_{\text{neck}}$):** The narrow false-positive corridor spanning between separate GT components.
- **`Crack` ($E_{\text{crack}}$):** The legitimate ground-truth crack components merged by the bridge.
- **`Background` ($E_{\text{bg}}$):** The local non-crack, non-bridge background in the dilated perimeter ($12$–$25$ px) of the corridor.

### Metrics Computed
1. **Spatial Energy Map:** $E(y, x) = \|F(y, x, :)\|_2$ (L2 norm across channels), resized via bilinear interpolation to image dimensions.
2. **Neck / Crack Contrast:** $C_{\text{neck/crack}} = \frac{E_{\text{neck}}}{E_{\text{crack}} + \epsilon}$
3. **Neck / Background Contrast:** $C_{\text{neck/bg}} = \frac{E_{\text{neck}}}{E_{\text{bg}} + \epsilon}$
4. **Normalized Emergence Index ($R_{\text{norm}}$):**
   $$R_{\text{norm}} = \frac{E_{\text{neck}} - E_{\text{bg}}}{E_{\text{crack}} - E_{\text{bg}} + \epsilon}$$
   - $R_{\text{norm}} \approx 0.0$: Bridge neck feature energy is indistinguishable from background.
   - $R_{\text{norm}} \ge 0.50$: Bridge neck has crossed $50\%$ of the gap between background and genuine crack.
   - $R_{\text{norm}} \approx 1.0$: Bridge neck is fully crack-like in feature intensity.

---

## 3. Empirical Results Across Scales

### Table 1: Multi-Scale Trajectory Summary Across Cohorts

| Stage | Cohort | Median $R_{\text{norm}}$ | Mean $R_{\text{norm}}$ | Median $\frac{E_{\text{neck}}}{E_{\text{crack}}}$ | $\ge 50\%$ Emerged | $\ge 70\%$ Emerged |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`bottleneck`** ($14 \times 14$) | **Consensus 7/7** ($N=101$) | **0.6718** | 0.8528 | **1.0000** | **61.39%** | **47.52%** |
| | Cured by D2 ($N=7$) | **0.1751** | 0.0539 | 0.9998 | 14.29% | 0.00% |
| **`S2`** ($28 \times 28$) | **Consensus 7/7** ($N=101$) | **0.5773** | 5.4227 | **0.9590** | **54.46%** | **39.60%** |
| | Cured by D2 ($N=7$) | **0.2984** | 0.2641 | 0.8582 | 14.29% | 0.00% |
| **`T1`** ($56 \times 56$ up) | **Consensus 7/7** ($N=101$) | **0.5194** | 0.8086 | **0.9512** | **51.49%** | **34.65%** |
| | Cured by D2 ($N=7$) | **0.0548** | -0.3615 | 0.8216 | 14.29% | 0.00% |
| **`T2`** ($56 \times 56$ skip) | **Consensus 7/7** ($N=101$) | **0.4009** | 0.1144 | **0.9936** | **46.53%** | **37.62%** |
| | Cured by D2 ($N=7$) | **0.6220** | 1.7261 | 0.9509 | 57.14% | 42.86% |
| **`T3`** ($56 \times 56$ conv1)| **Consensus 7/7** ($N=101$) | **0.6986** | 0.8486 | **0.8935** | **78.22%** | **49.50%** |
| | Cured by D2 ($N=7$) | **0.3872** | 0.3806 | 0.7387 | 28.57% | 0.00% |
| **`T4`** ($56 \times 56$ conv2)| **Consensus 7/7** ($N=101$) | **0.7193** | 0.8433 | **0.8679** | **79.21%** | **53.47%** |
| | Cured by D2 ($N=7$) | **0.2940** | 0.2396 | 0.6167 | 0.00% | 0.00% |
| **`S0`** ($112 \times 112$) | **Consensus 7/7** ($N=101$) | **0.7525** | 0.7959 | **0.8514** | **83.17%** | **59.41%** |
| | Cured by D2 ($N=7$) | **0.2585** | 0.2522 | 0.5240 | 0.00% | 0.00% |
| **`Final_Head`** ($448 \times 448$)| **Consensus 7/7** ($N=101$) | **0.9891** | 1.0244 | **0.9909** | **100.0%** | **99.01%** |
| | Cured by D2 ($N=7$) | **0.8185** | 0.7853 | 0.8425 | 100.0% | 71.43% |

---

### Table 2: First Emergence Point Distribution ($R_{\text{norm}} \ge 0.50$ and $\ge 0.70$)

| Earliest Stage Reaching Threshold | Consensus 7/7 ($\ge 50\%$) | Consensus 7/7 ($\ge 70\%$) | Cured by D2 ($\ge 50\%$) | Cured by D2 ($\ge 70\%$) |
| :--- | :---: | :---: | :---: | :---: |
| **`bottleneck`** ($14 \times 14$) | **62 (61.39%)** | **48 (47.52%)** | 1 (14.29%) | 0 (0.00%) |
| **`S2`** ($28 \times 28$) | **13 (12.87%)** | **12 (11.88%)** | 1 (14.29%) | 0 (0.00%) |
| **`T1`** ($56 \times 56$ post-up) | 4 (3.96%) | 2 (1.98%) | 0 (0.00%) | 0 (0.00%) |
| **`T2`** ($56 \times 56$ post-skip)| 12 (11.88%) | 15 (14.85%) | **3 (42.86%)** | **3 (42.86%)** |
| **`T3`** ($56 \times 56$ post-conv1)| 6 (5.94%) | 7 (6.93%) | 0 (0.00%) | 0 (0.00%) |
| **`T4`** ($56 \times 56$ post-conv2)| 0 (0.00%) | 2 (1.98%) | 0 (0.00%) | 0 (0.00%) |
| **`S0`** ($112 \times 112$) | 1 (0.99%) | 4 (3.96%) | 0 (0.00%) | 0 (0.00%) |
| **`Final_Head`** ($448 \times 448$) | 3 (2.97%) | 11 (10.89%) | **2 (28.57%)** | **3 (42.86%)** |
| **Cumulative By S2 ($28 \times 28$)** | **75 / 101 (74.26%)** | **60 / 101 (59.41%)** | **2 / 7 (28.57%)** | **0 / 7 (0.00%)** |
| **Cumulative By T2 ($56 \times 56$)** | **91 / 101 (90.10%)** | **77 / 101 (76.24%)** | **5 / 7 (71.43%)** | **3 / 7 (42.86%)** |

---

## 4. Addressing Core Scientific Questions

### Question A: At what scale does the false bridge first emerge as crack-like?
**Answer: For the core consensus population, the bridge emerges in the deep bottleneck and S2 representations.**
- In **$61.39\%$** (62/101) of Consensus 7/7 bridges, the normalized emergence index $R_{\text{norm}}$ is already $\ge 0.50$ at the **bottleneck** ($14 \times 14$).
- Cumulatively by **`S2`** ($28 \times 28$), **$74.26\%$** (75/101) have emerged at the $\ge 50\%$ level, and **$59.41\%$** (60/101) have emerged at the $\ge 70\%$ level.
- By contrast, only $9.9\%$ of consensus cases require Decoder Block 1 convs or subsequent stages to first cross the $50\%$ emergence threshold.
- Therefore, the bridge signal is **not initiated** in the shallow decoder; its spatial semantics are already fused before entering Decoder Block 1.

### Question B: How do Consensus 7/7 bridges compare with D2-Cured bridges across scales?
**Answer: They exhibit fundamentally distinct provenance trajectories.**
1. **Consensus 7/7 Trajectory (Deep-Seeded & Amplified):**
   - Originates strong at bottleneck ($R_{\text{norm}} = 0.67$) and S2 ($0.58$).
   - Passes into Block 1, where Conv1 and Conv2 progressively solidify it ($R_{\text{norm}}: 0.40 \to 0.70 \to 0.72$).
   - Continues climbing through S0 ($0.75$) to near-unity at the head ($0.99$).
   - At every single stage, its neck-to-crack contrast remains above $85\%$.
2. **D2-Cured Trajectory (Shallow, Transient & Fragile):**
   - Weak at bottleneck ($R_{\text{norm}} = 0.18$) and S2 ($0.30$).
   - Near zero at T1 post-upsample ($0.05$).
   - Exhibits an isolated transient peak at T2 ($0.62$), driven purely by the Stage 1 skip connection.
   - However, unlike consensus bridges, Block 1 convs **suppress** rather than amplify this signal ($R_{\text{norm}}: 0.62 \to 0.39 \to 0.29$).
   - At S0, it remains fully subdued ($0.26$, with $0/7$ samples $\ge 50\%$).
   - It only flips to a positive prediction at the final sigmoid thresholding in the head ($0.82$).
- **Conclusion:** D2-cured cases are shallow, boundary-diffusion errors that are easily dissolved by boundary penalties. Consensus 7/7 cases are deep structural mergers that resist downstream penalties.

### Question C: Epistemic Interpretation (Origin vs Mechanism)
**Answer: Deep Encoder Representation with Decoder Spatial-Mixing Amplification.**
- **Origin:** The false bridge is **deeply seeded** in the encoder/bottleneck representations ($14 \times 14$ and $28 \times 28$), where low spatial resolution and large effective receptive fields merge nearby parallel crack branches into a single unresolved activation blob.
- **Amplification/Mediation:** Decoder Block 1 ($56 \times 56$) does **not** invent the error. Instead, its off-center $3 \times 3$ convolutional spatial mixing acts as an **essential mediator/amplifier**: it propagates the low-resolution bridge activation across spatial channels, converting a fuzzy deep blob into a crisp, high-probability crack segment.
- This explains the V5 paradox: attenuating spatial mixing ($\alpha=0.5$) slightly reduces neck probability, but cannot break the bridge ($99.01\%$ persistence) because the deep representation ($S2$) is already strongly fused. Meanwhile, attenuating spatial mixing breaks legitimate crack continuity elsewhere, doubling Breakage from $9.77\% \to 20.98\%$.

### Question D: Recommended Next Research Step
**Answer: Shift focus from Decoder-Loss Probes to Deep-Representation Interventions.**
- All previous Phase 6 probes (A1 Boundary IoU, A2 PLU-Head, B1 Thinness Reweighting, C1 clDice, D1 Composite, D2 Separation Loss) operated **downstream** at the decoder output ($448 \times 448$ or $112 \times 112$).
- Because the core consensus bridge is already fully formed at $14 \times 14$ and $28 \times 28$, downstream penalties force the network into an unsolvable trade-off: penalizing the neck at $448 \times 448$ without altering the deep feature map causes tearing along real cracks (Breakage surging $9.5\% \to 14.4\% \to 21.0\%$) while leaving $93\%$ to $99\%$ of false bridges intact.
- **Single Recommended Research Direction:**
  If a training intervention is ever to be considered, it must target the **deep multi-scale representation** directly:
  1. **Bottleneck / S2 Topological Moat Regularization:** Enforce topological component separation or repulsive contrastive loss directly on the ViT/ConvNeXt bottleneck ($14 \times 14$) or Decoder Block 0 ($28 \times 28$) feature embeddings before they enter the upsampling stream.
  2. **High-Resolution Bypass / Direct Thin Path:** Enhance the capacity of shallow encoder features (Stage 0, Stage 1) to gate or veto deep unresolved blobs, preventing deep merged semantics from overpowering high-resolution geometry.
