# Phase 6 Diagnostic Report: 2x2 Factorial Decoder Causal Pathway Ablation

**Target Cohort:** Exactly the 43 wider-gap bridge events ($D_{\text{gap}} > 5.0\text{ px}$) from Phase 6 Diagnostic D  
**Target Model:** Candidate B Baseline (`B2ConvNeXtViTUNet`, checkpoint `P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth`)  
**Evaluation Setting:** Canonical Setting A (Tile $448 \times 448$, Stride $448$, non-overlap, threshold $0.5$)  
**Methodology:** Zero-training 2x2 Factorial Pathway Intervention (`u` vs `skip` before `torch.cat`)  
**Date:** October 3, 2026  

---

## 1. Executive Summary & Problem Framing

In the preceding localization diagnostic, we observed that false bridge representations first emerge predominantly at `decoder_28` (32.6%) and `decoder_56` (20.9%), with 100% of bridges already having positive logits at `head_112`. However, correlational localization does not establish **causality**: in each `DecoderBlock`, two distinct information streams merge before two sequential $3 \times 3$ convolutions:
```python
# SAGE_LITE/sage/networks/decoder_block.py, lines 99-108
u = self.upsample(x)              # ConvTranspose2d (upsampled deeper feature)
s = skip                          # Lateral skip connection from encoder
x = torch.cat([u, s], dim=1)      # Concatenation
x = self.conv1(x)                 # 3x3 Conv -> BN -> ReLU
x = self.conv2(x)                 # 3x3 Conv -> BN -> ReLU
return x
```

To isolate the true causal drivers without retraining or modifying model weights, we performed a **2x2 factorial intervention** at both `decoder_28` (DecoderBlock 0) and `decoder_56` (DecoderBlock 1):
- `Cond 00`: Baseline (`u` + `s` unperturbed)
- `Cond 10`: Mask corridor on `u`, keep `s`
- `Cond 01`: Keep `u`, mask corridor on `s`
- `Cond 11`: Mask corridor on both `u` and `s`

Each event was evaluated alongside a matched **True-Crack Control Segment** of identical length ($\approx D_{\text{gap}}$) within the same image to quantify selective versus generic crack suppression.

---

## 2. Quantitative Evidence

### 2.1 2x2 Factorial Causal Summary Table ($N = 43$ events)

| Decoder Stage | Resolution & Stride | Baseline Logit | Cond 10 Logit (Mask U) | Cond 01 Logit (Mask S) | Cond 11 Logit (Mask Both) | Median $\Delta U$ | Median $\Delta S$ | Median $\Delta US$ | Median Interaction $I_{US}$ | Median Selective $\Delta U$ | Median Selective $\Delta S$ | Direct Cured Count (Cond 11) | Direct Cure Rate (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `decoder_28` (Block 0) | $28 \times 28$ ($S=16$) | +2.4556 | +2.3576 | +2.5342 | +2.5275 | **0.0000** | **0.0000** | **0.0000** | **0.0000** | 0.0000 | 0.0000 | 1 / 43 | 2.3% |
| `decoder_56` (Block 1) | $56 \times 56$ ($S=8$) | +2.4556 | +2.1507 | +1.5306 | **+0.9696** | **0.2307** | **0.6427** | **0.7795** | **+0.0030** | **0.2302** | **0.6427** | **11 / 43** | **25.6%** |

*Definitions:*
- $\Delta U = z00 - z10$ (effect of removing upsampled stream)
- $\Delta S = z00 - z01$ (effect of removing skip stream)
- $\Delta US = z00 - z11$ (joint effect of removing both streams)
- $I_{US} = \Delta US - \Delta U - \Delta S$ (super-additive interaction / non-linear fusion effect)
- $\Delta_{\text{selective}} = \Delta_{\text{corridor}} - \Delta_{\text{control}}$ (net suppression specific to false corridor)

---

### 2.2 Causal Driver Classification Distribution

| Driver Category | Criteria | `decoder_28` ($28 \times 28$) | `decoder_56` ($56 \times 56$) |
|---|---|---|---|
| **Skip Driver** | $\Delta S > \Delta U + 0.3$ | 0 (0.0%) | **27 (62.8%)** |
| **Negligible Effect** | $|\Delta US| < 0.2$ | **35 (81.4%)** | 10 (23.3%)* |
| **Co-contributed** | Both $\Delta U, \Delta S > 0.2$ and $|\Delta U - \Delta S| \le 0.3$ | 5 (11.6%) | 4 (9.3%) |
| **Upsample Driver** | $\Delta U > \Delta S + 0.3$ | 3 (7.0%) | 2 (4.7%) |
| **Interaction / Fusion** | $I_{US} > 0.5$ and $I_{US} > \max(\Delta U, \Delta S)$ | 0 (0.0%) | 0 (0.0%) |

*\*Note on Negligible events at Stage 56: Exactly 10 events showed negligible effect at Stage 56. All 10 correspond to the sub-cohort with $D_{\text{gap}} \le 8.0\text{ px}$ (median gap $7.0\text{ px}$), where a $8 \times 8$ cell still encompasses true crack endpoints.*

---

### 2.3 Robustness Check: Natural Background Replacement vs Zeroing

To verify that the causal effect is not an operator-specific artifact of zeroing activations, we executed a robustness probe on all 21 events with strong causal effect ($\Delta US > 0.8$):
- Instead of setting corridor feature cells to $0.0$, we replaced them with the channel-wise mean feature vector of local background cells within the same tile:
  $$x_{\text{corridor}} \leftarrow \mu_{\text{local\_bg}}$$
- **Results:**
  - Mean $\Delta US_{\text{zero}} = \mathbf{2.3919}$
  - Mean $\Delta US_{\text{bg\_replacement}} = \mathbf{1.6690}$
  - Pearson Correlation: $\mathbf{r = 0.8227}$ ($p < 10^{-5}$)
- **Conclusion:** The concordance between zeroing and local-background replacement provides robustness against the hypothesis that the observed effect is specific to one perturbation operator. It confirms that the effect is driven by removing crack evidence from the corridor, rather than an artifact of zero-weight singular activations.

### 2.4 Mask Selectivity & Collateral-Damage Control
The matched true-crack control segments yielded $\Delta_{\text{selective}} \approx \Delta$. Note that because the mask projection specifically protects cells containing true crack pixels ($w = \max(0, (n_{\text{corr}} - n_{\text{crack}})/S^2)$), this measurement primarily functions as a **mask selectivity and collateral-damage control** (proving the intervention does not damage true crack representation), rather than absolute proof that skip signal is selective only for bridge.

---

## 3. Key Causal Findings & Interpretation

### Finding 1: The Skip Connection at Stage 1 ($56 \times 56$) is the Dominant Causal Driver
- At `decoder_56`, the median effect of removing the skip connection ($\Delta S = \mathbf{0.6427}$) is approximately **$2.8\times$ the median upsample effect** ($\Delta U = \mathbf{0.2307}$). (In terms of mean values, the skip effect is $0.9851$ vs $0.3219$, or approximately $3.1\times$).
- In $62.8\%$ of all wider-gap events (27/43), the skip stream is classified as the primary causal driver of the false bridge.
- When both are masked at `decoder_56`, the median corridor logit drops by **$0.7795$ points** (mean $1.3153$ points), directly curing **$25.6\%$ of all wider-gap false bridges (11/43 events)** with **zero retraining**.

### Finding 2: Reconciling Localization vs Causality (First Appearance != Causal Source)
- In the localization diagnostic, `decoder_28` had the largest number of first-emergence events (14/43, 32.6%).
- However, the causal ablation reveals that intervening at `decoder_28` produces **$\Delta US = 0.0000$** for $81.4\%$ of events (cure rate only $2.3\%$).
- **Explanation:**
  1. *Decoder 28 = where the bridge becomes observable:* At $28 \times 28$ ($S=16\text{ px}$), gaps of $8-15\text{ px}$ occupy less than 1 cell. Feature correlations begin shifting here.
  2. *Decoder 56 = where Stage-1 skip provides localized causal evidence:* Even if wiped out at $28 \times 28$, the encoder skip connection at Stage 1 ($56 \times 56$) independently carries the high-frequency false bridge evidence directly into Decoder Block 1!
  3. Thus, **first appearance $\neq$ causal source**. Stage 28 represents early feature correlation, but **Stage 56 is the actual causal commitment bottleneck**.

### Finding 3: No Substantial Non-Linear Interaction Detected
- The median interaction effect $I_{US}$ at `decoder_56` is $+0.0030$ (mean $+0.0083$).
- Under the tested interventions and operating point, **no substantial nonlinear interaction was detected between the two branches**.
- The combined effect is approximately additive ($\Delta US \approx \Delta U + \Delta S$), indicating that the false bridge is not generated by an unexpected non-linear synergy or constructive resonance between the two streams.

---

## 4. Negative Evidence & What Remains Unproven

1. **The Sub-8px Ceiling:**
   For the 11 sub-8px events ($D_{\text{gap}} \le 8.0\text{ px}$), the current spatial corridor intervention at stride 8 does not provide a viable selective causal mechanism for bridge removal ($|\Delta US| < 0.2$). This reflects two intertwined mechanisms: (1) spatial representation limitation (a single $8 \times 8$ cell may encompass both endpoints) and (2) intervention-mask limitation (soft protection dampens masking when true crack is present). This does not prove that spatial intervention below 8 px is fundamentally impossible in all architectures, but bounds the capability of the current stride-8 grid.

2. **Scope Limitation:**
   These findings apply strictly to the **43 wider-gap bridge events ($D_{\text{gap}} > 5.0\text{ px}$)** of Candidate B on Crack500. They do not claim that the decoder is the "sole cause" of all errors in the network, but they conclusively prove that:
   $$\boxed{\text{For wider-gap bridges, the Stage-1 skip branch entering Decoder Block 1 is the dominant measured causal pathway.}}$$

---

## 5. Updated Phase 6 Hypothesis Status & Decision

| Hypothesis | Status |
|---|---|
| Final 4x bilinear creates bridge | **Closed** (disproved: 100% committed at head_112) |
| Tile boundary / context truncation dominant | **Strongly weakened** (disproved: 98.2% persist in overlap) |
| Scalar-probability saddle | **Strongly refuted** (Diagnostic A: plateau at p ~ 0.90) |
| Router / gating magnitude dominant | **Closed** (disproved by routing audit) |
| Decoder 28 causal source | **Weak / not supported** (ablation delta = 0.0) |
| Decoder 56 upsample branch dominant | **Not supported** (delta U = 0.23) |
| Decoder 56 skip branch dominant | **Strongly supported for 43 wider-gap events** (delta S = 0.64) |
| Strong U x S nonlinear interaction | **Not supported** (I_US ~ 0.0) |
| Stage-1 skip itself is the upstream source of erroneous feature | **Still open -> Target of next diagnostic** |
| SAGE injection is source of erroneous Stage-1 skip | **Still open -> Target of next diagnostic** |

### Next Step: Stage-1 Skip Provenance Ablation (Zero-Training)
Trace upstream:
```text
Encoder Stage 1 -> SAGE injection -> Stage-1 Skip -> Decoder Block 1 -> Final Logits
```
- **Probe A (pre- vs post-SAGE):** Measure whether bridge corridor feature is already crack-like pre-SAGE or created by SAGE.
- **Probe B (activation patching on Stage-1 skip):** Patch corridor with local background vs matched true-crack vs shuffled feature.
- **Probe C (endpoint-preserving gap replacement):** Preserve crack endpoints and replace strictly interior gap activation.
