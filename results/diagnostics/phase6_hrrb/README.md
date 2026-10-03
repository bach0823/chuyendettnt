# Phase 6: High-Resolution Residual Bypass (HRRB) Diagnostic Report

## 1. Scientific Objective & Research Hypothesis
Following the failure of Stem learnable spatial overlap ($U_0\text{-C1}$) and Stage-0 LayerNorm affine adaptation ($U_0\text{-C2-A}$), this experiment tests the core hypothesis:
> **Hypothesis:** Does Candidate B's persistent false bridge phenomenon stem from the total lack of an independent high-resolution feature pathway bypassing the aggressive stride-4 Stem downsampling to the final prediction?
> 
> **Specific Question:** Can a tiny independent high-resolution residual pathway bypass the main $/4$ pathway and correct false connectivity without paying for that correction with true-crack breakage?

## 2. Architectural Design & Invariants
- **Main Model:** Candidate B 100% strictly frozen (`torch.no_grad()`).
- **HRRB Branch:**
  $$\text{RGB } [B, 3, 448, 448] \xrightarrow{\text{Conv3x3 } s=1} [B, 8, 448, 448] \xrightarrow{\text{GELU}} \xrightarrow{\text{Conv3x3 } s=2} [B, 16, 224, 224] \xrightarrow{\text{Conv1x1}} [B, 1, 224, 224] \xrightarrow{\text{Bilinear } \times 2} \text{detail\_residual } [B, 1, 448, 448]$$
  $$\text{final\_logits} = \text{main\_logits}_{448} + \text{detail\_residual}$$
- **Zero-Init Identity:** Final $1\times 1$ projection initialized with `weight = 0`, `bias = 0`. At $t=0$, $\text{detail\_residual} \equiv 0$, $\text{final\_logits} \equiv \text{main\_logits}$ (Bitwise difference $= 0.00\text{e+00}$).
- **Active Trainable Parameters:** Exactly **1,409** parameters ($0.0104\%$ of Candidate B).
- **Protocol:** Physical batch 14, seed 42, FP32 ONLY, no GA, 8 epochs, AdamW($\text{lr}=10^{-4}, \text{wd}=10^{-2}$), CosineAnnealingLR, Loss = BCE + Dice. Validation $N=348$ Setting A ($\tau=0.5$). Test set $N=1124$ SEALED.

## 3. Experimental Results

### Subgroup Topology & Segmentation Metrics
| Metric | Candidate B (C0) | HRRB | Net Delta ($\Delta_{\text{HRRB vs C0}}$) | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Bridge Events (Full Val, $N=348$)** | 118 | **108** | **-10** | **-8.47%** |
| **Bridge Images (Full Val, $N=348$)** | 110 | **101** | **-9** | **-8.18%** |
| **Cured Images / Created Images** | — | **9 / 0** | **Net +9 cured** | — |
| **Break Events (Full Val, $N=348$)** | 35 | **75** | **+40** | **+114.29%** |
| **Recall (Full Val, $N=348$)** | 0.8477 | **0.7822** | **-0.0655** | **-7.72%** |
| **Precision (Full Val, $N=348$)** | 0.7337 | **0.7801** | **+0.0464** | **+6.32%** |
| **Dice (Full Val, $N=348$)** | 0.7641 | **0.7593** | -0.0048 | -0.62% |
| **clDice (Full Val, $N=348$)** | 0.8499 | **0.8264** | **-0.0234** | -2.76% |
| **Resistant Bridge Events ($N=82$)** | 88 | **83** | **-5** | -5.68% |
| **Resistant Bridge Images ($N=82$)** | 82 | **78** | **-4** | -4.88% |
| **Resistant Break Events ($N=82$)** | 8 | **13** | **+5** | +62.50% |
| **Clean Control Break Events ($N=56$)** | 8 | **16** | **+8** | +100.00% |

### Residual Magnitude & Mechanism Audit
| Region | Mean $|\text{detail\_residual}|$ | Mean $|\text{main\_logits}|$ | Mean Ratio $\frac{|\text{res}|}{|\text{main}| + \epsilon}$ |
| :--- | :---: | :---: | :---: |
| **Global Full Image** | 1.9250 | 2.7466 | 0.9512 |
| **True Crack ($GT == 1$)** | 1.0873 | 3.5783 | 1.6376 |
| **Background ($GT == 0$)** | 1.9750 | 2.6994 | 0.9512 |
| **Near Decision Boundary ($|\text{main}| < 0.5$)** | **1.2584** | 0.2500 | **21.4888** |

### Standalone Residual Diagnostics
- Standalone Residual AUC: **0.6447**
- Standalone Residual Dice: **0.0001**
- Mean Positive Residual Area per Image: **118.2 px** (vs GT crack area: 13,239.7 px)
- Fraction of Images with **Zero** Positive Residual Pixels: **93.39%**

## 4. Epistemic Readout & Scientific Conclusion
1. **Did HRRB cure false bridges?** 
   - Yes, for the first time in Phase 6, bridge events dropped from $118 \to 108$ (-10 events), with 9 images cured and 0 created. In the Resistant cohort, 4 persistently bridged images were cured ($88 \to 83$ events).
2. **What was the cost?**
   - **Severe fragmentation and true crack breakage:** Break events surged from $35 \to 75$ (+114%), and Recall plunged by $-6.55\%$ ($84.77\% \to 78.22\%$).
3. **What is the underlying mechanism?**
   - In 93.4% of validation images, the residual output is **completely negative** across all pixels.
   - At the decision boundary, the residual magnitude ($\approx 1.26$) is **21.5 times larger** than the marginal main logits ($\approx 0.05 - 0.25$), violently dragging weak boundary logits below zero ($p < 0.5$).
   - **Conclusion:** HRRB did **not** learn an independent feature discriminator that separates cracks from necks. Instead, it learned a **global prediction-space erosion/suppression field**, trading true-crack recall for false-bridge precision.
   - Therefore, the question *"Can a tiny independent high-resolution residual pathway bypass the main /4 information pathway and correct false connectivity without paying for that correction with true-crack breakage?"* is answered empirically: **NO. A simple un-modulated residual bypass acts as a blunt erosion operator, failing to resolve the fine topological trade-off.**
