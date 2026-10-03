# Phase 6: Upstream Genesis — Condition U0-C2-A (Stem Halo + Stage-0 LayerNorm2d Affine Adaptation)

## 1. Experimental Overview & Scientific Question
Following condition $U_0\text{-C1}$ (Learnable Spatial Overlap Stem Halo), where false bridge topology remained virtually unchanged ($118 \to 116$ bridge events) despite improved representation separation at the Stem output ($0.0125 \to 0.0319$), a key confounder emerged:
> **Confounder Hypothesis:** Did $U_0\text{-C1}$ fail because spatial overlap carries no useful anti-bridging signal ($H_1$ false), or because the newly learned Stem representations were incompatible with the downstream frozen Stage 0 distribution?

To test this hypothesis cleanly with minimal intervention:
- **Condition $U_0\text{-C2-A}$:** Initialize from $U_0\text{-C1}$ weights. Continue learning the Stem halo ($4,752$ active degrees of freedom), while unfreezing **only** the `LayerNorm2d` affine parameters ($\gamma, \beta$) in Stage 0 (`blocks.0.norm` and `blocks.1.norm`, exactly $192$ parameters). Total active trainable parameters: **$4,944$**.
- All other components remain strictly frozen: Stem center 4x4, Stem bias, Stem LayerNorm, Stage 0 depthwise convolutions, Stage 0 MLPs, Stage 0 GRN, SAGE modules, Stages 1–3, ViT, Decoder, and Seg Head.

## 2. Invariants & Verification
- Checkpoint SHA256: `147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66`
- Parameters SHA256: `4aeda58ce6d2fb32f5b772cdbf92dec7fb913f9510b392910efe6674b0adf5d6`
- Preflight identity at $t=0$: $U_0\text{-C2-A} \equiv U_0\text{-C1}$ (`max_abs_diff = 0.00e+00`)
- Sealed Test Set: $N=1124$ strictly untouched. Validation $N=348$ Setting A ($\tau=0.5$).
- Training protocol: FP32 / AMP matching Phase 6 standard, seed 42, 8 epochs, batch size 14, lr 1e-4, wd 1e-2.

## 3. Results & Comparative Summary

### End-to-End Segmentation & Topology Metrics
| Metric | $U_0\text{-C0}$ (Candidate B) | $U_0\text{-C1}$ (Stem Halo) | $U_0\text{-C2-A}$ (Halo + Stage0 LN) | $\Delta_{\text{C2A vs C1}}$ | $\Delta_{\text{C2A vs C0}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Bridge Events (Full Val, $N=348$)** | 118 | 116 | **120** | **+4** | **+2** |
| **Bridge Images (Full Val, $N=348$)** | 110 | 110 | **113** | **+3** | **+3** |
| **Break Events (Full Val, $N=348$)** | 35 | 38 | **45** | **+7** | **+10** |
| **Dice (Full Val, $N=348$)** | 0.7641 | 0.7478 | **0.7453** | -0.0025 | -0.0188 |
| **clDice (Full Val, $N=348$)** | 0.8499 | 0.8342 | **0.8295** | -0.0047 | -0.0204 |
| **Consensus 7/7 Bridge Events ($N=101$)** | 109 | 107 | **107** | 0 | -2 |
| **Consensus Resistant Bridge Events ($N=82$)** | 88 | 86 | **86** | 0 | -2 |
| **Consensus Resistant Bridge Images ($N=82$)** | 82 | 82 | **81** | -1 | -1 |
| **Consensus Resistant Break Events ($N=82$)** | 8 | 7 | **8** | +1 | 0 |

### Dual Representation Audit (Stem vs Stage 0)
| Level | Metric | Cohort | $U_0\text{-C0}$ | $U_0\text{-C1}$ | $U_0\text{-C2-A}$ |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Stem** | Separation Margin | Resistant ($N=82$) | 0.0125 | 0.0319 | 0.0265 |
| **Stem** | $\cos(\text{neck}, \text{crack})$ | Resistant ($N=82$) | 0.8397 | 0.8256 | 0.8346 |
| **Stage 0** | Separation Margin | Resistant ($N=82$) | — | 0.0092 | **0.0304** (+230%) |
| **Stage 0** | $\cos(\text{neck}, \text{crack})$ | Resistant ($N=82$) | — | 0.9781 | **0.9534** |

## 4. Epistemic Readout & Decision Verdict
1. **Did Stage 0 LayerNorm adapt?** Yes. Unfreezing 192 affine parameters allowed Stage 0 to widen its feature separation margin from $0.0092$ to $0.0304$ (+230%) and slightly reduce cosine correlation ($0.9781 \to 0.9534$).
2. **Did this cure false bridges?** **NO.** 
   - Across the entire validation set ($N=348$), false bridge events actually *increased* from $116 \to 120$ (+4).
   - In the Resistant cohort ($N=82$), bridge events remained completely flat at 86 (81 out of 82 images remain persistently bridged).
   - Crack fragmentation and breakage increased notably from $38 \to 45$ (+7 vs C1, +10 vs C0).
   - Dice and clDice degraded further ($-1.88\%$ Dice and $-2.04\%$ clDice vs baseline C0).
3. **Verdict:** **Refuted.** The distribution shift between the Stem and Stage 0 at the channel affine normalization level ($\gamma, \beta$) is **NOT** the bottleneck preventing bridge cures. 
   - Neither single-layer spatial overlap ($U_0\text{-C1}$) nor overlap paired with Stage-0 affine adaptation ($U_0\text{-C2-A}$) provides the topological resolution required to resolve sub-4px inter-crack gap ambiguities.
   - The aggressive stride-4 operation in a single convolution step fundamentally compresses fine spatial topological details beyond recovery by local affine adjustments.
