# Phase 6 Diagnostic Report: Decoder-Stage False Bridge Localization & Progression

**Target Cohort:** 43 wider-gap bridge events ($D_{\text{gap}} > 5.0\text{ px}$) from Phase 6 Diagnostic D  
**Target Model:** Candidate B Baseline (`B2ConvNeXtViTUNet`, checkpoint `P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth`)  
**Evaluation Setting:** Canonical Setting A (Tile $448 \times 448$, Stride $448$, non-overlap, threshold $0.5$)  
**Execution Time:** 43.16 seconds (CuDNN disabled for numerical precision)  
**Date:** October 3, 2026  

---

## 1. Executive Summary & Problem Framing

Following the completion of the Setting A vs Setting B audit (which conclusively disproved the tile-boundary hypothesis by showing $98.2\%$ bridge persistence across overlap blending), this zero-training diagnostic locates the internal representation and decoder stages where false bridges actually form and commit.

Rather than averaging over the entire 110-image validation failure set, this investigation strictly examines the **43 wider-gap bridge events ($D_{\text{gap}} > 5.0\text{ px}$)** identified in Phase 6 Diagnostic D. These events represent genuine non-local topological hallucinations (gaps up to $130.3\text{ px}$, median $12.0\text{ px}$) that cannot be excused as $1$-pixel boundary rounding artifacts.

We hooked intermediate feature activations and logits across six discrete stages of Candidate B:
1. `bottleneck_14`: Deep ViT/CNN bottleneck ($14 \times 14$, $384\text{ channels}$)
2. `decoder_28`: Decoder Block 0 output ($28 \times 28$, $192\text{ channels}$, fuses encoder Stage 2 skip)
3. `decoder_56`: Decoder Block 1 output ($56 \times 56$, $96\text{ channels}$, fuses encoder Stage 1 skip)
4. `decoder_112`: Decoder Block 2 output ($112 \times 112$, $48\text{ channels}$, fuses encoder Stage 0 skip)
5. `head_112`: Segmentation head output ($112 \times 112$, $1\text{ channel}$ logit before upsampling)
6. `final_448`: Final output logits ($448 \times 448$, $1\text{ channel}$ after bilinear $\times 4$)

Every bridge event is paired with a matched **Control Group** within the same image consisting of an actual true crack continuation segment of identical length ($\approx D_{\text{gap}}$).

---

## 2. Quantitative Evidence

### 2.1 Stage-by-Stage Emergence & Metric Summary ($N = 43$ events)

| Decoder Stage | Resolution | Channels | First Emergence Count | First Emergence Pct (%) | Cumulative Emergence (%) | Median Contrast vs Crack | Median Cosine Sim to Crack | Median Cosine Sim to BG | Control Crack Sim to Crack | Median Logit (Corr) | Median Prob (Corr) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `bottleneck_14` | $14 \times 14$ | 384 | 7 | 16.3% | 16.3% | 1.4370 | 0.9244 | **0.9714** | 0.9999 | N/A | N/A |
| `decoder_28` | $28 \times 28$ | 192 | **14** | **32.6%** | **48.8%** | 0.5615 | 0.9121 | 0.8319 | 0.9990 | N/A | N/A |
| `decoder_56` | $56 \times 56$ | 96 | 9 | 20.9% | 69.8% | 0.6285 | 0.9610 | 0.6204 | 0.9991 | N/A | N/A |
| `decoder_112` | $112 \times 112$ | 48 | 4 | 9.3% | 79.1% | 0.6210 | 0.9878 | 0.5418 | 0.9991 | N/A | N/A |
| `head_112` | $112 \times 112$ | 1 | 6 | 14.0% | 93.0%* | 0.7433 | 1.0000 | -1.0000 | 1.0000 | **+2.1986** | **0.9001** |
| `final_448` | $448 \times 448$ | 1 | **0** | **0.0%** | **100.0%** | 0.7433 | 1.0000 | -1.0000 | 1.0000 | **+2.1986** | **0.9001** |

*\*Note: 3 events (7.0%) did not strictly exceed the cosine threshold $\text{Sim}(\text{Corr}, \text{Crack}) > \text{Sim}(\text{Corr}, \text{BG}) + 0.05$ at feature stages due to low feature amplitude, but at `head_112` all 43 events (100.0%) were committed with positive logits.*

### 2.2 Invariant Verification: Head Interpolation vs Logit Commitment

A crucial architectural invariant in Candidate B is the relationship between `head_112` and `final_448`:
```python
# SAGE_LITE/sage/networks/decoder_block.py, lines 296-305
logits = self.segmentation_head(x_dec)
if target_size is not None and logits.shape[2:] != target_size:
    logits = F.interpolate(
        logits,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
```

**Empirical Result:**
- First positive logit ($> 0$) emergence:
  - `head_112`: **43 / 43 (100.0%)**
  - `final_448`: **0 / 43 (0.0%)**
- Median corridor logit at `head_112`: **+2.1986** (corresponding to probability $p = 0.9001$).
- Median corridor logit at `final_448`: **+2.1986** (identical).
- **Direct Proof:** Bilinear $4\times$ interpolation is **100% innocent** of creating false bridges. All 43 wider-gap false bridges are already fully committed as positive predictions in the low-resolution $112 \times 112$ logit map before any spatial interpolation occurs.

---

## 3. Observations & Stage Progression Dynamics

### Observation 1: The Bottleneck ($14 \times 14$) Does NOT View Corridors as Cracks
At the deepest representation level (`bottleneck_14`):
- Only 7 out of 43 events (16.3%) exhibit crack-dominant feature similarity.
- For the remaining 36 events, the corridor feature vector is actually **closer to Background than to True Crack**:
  $$\text{Median Sim to BG} = 0.9714 \quad > \quad \text{Median Sim to Crack} = 0.9244$$
- In contrast, the matched Control Crack segments at $14 \times 14$ have a median similarity to crack of **0.9999** and contrast of **0.7879**.
- **Implication:** The deep Transformer/ConvNeXt bottleneck representation does not fundamentally mistake empty pavement gaps for cracks. The false bridge representation is predominantly synthesized downstream.

### Observation 2: The Critical Transition Occurs at Decoder 28 & Decoder 56
- `decoder_28` ($28 \times 28$) is the single largest point of origin: **14 events (32.6%)** first become crack-dominated here.
- By `decoder_56` ($56 \times 56$), an additional 9 events emerge, bringing the cumulative total to **30 out of 43 (69.8%)**.
- Simultaneously, feature similarity between the corridor and the background collapses rapidly:
  - Bottleneck 14: $\text{Sim}(\text{Corr}, \text{BG}) = 0.9714$
  - Decoder 28: $\text{Sim}(\text{Corr}, \text{BG}) = 0.8319$ (down $0.1395$)
  - Decoder 56: $\text{Sim}(\text{Corr}, \text{BG}) = 0.6204$ (down $0.2115$)
  - Decoder 112: $\text{Sim}(\text{Corr}, \text{BG}) = 0.5418$
- Meanwhile, similarity between corridor and crack rises to **0.9610** at Stage 56 and **0.9878** at Stage 112, approaching the control crack baseline ($0.9991$).

### Observation 3: Pre-Upsample Head Projection Confirms High Confidence
At `head_112`, the $1 \times 1$ conv projection (`head_mid_channels=24 -> num_classes=1`) maps the $112 \times 112$ feature map into scalar logits:
- $100\%$ of corridor pixels across all 43 events produce positive logits (median $+2.1986$, $75^{\text{th}}$ percentile $+3.26$).
- The classifier head is not "hesitating" or operating near the $0.0$ boundary; it treats the corridor with essentially the same certainty as the adjoining true crack components (median crack logit $+4.16$).

---

## 4. Interpretation & Hypotheses

Based strictly on the measured tensor activations:

1. **Rejection of the "Final Upsampling Artifact" Hypothesis:**
   The hypothesis that false bridges are generated by blur or bleeding during the final $112 \to 448$ bilinear upsampling is completely disproven. Zero bridges originate at `final_448`.

2. **The "Cross-Scale Diffusion / Feature Smearing" Phenomenon:**
   In each `DecoderBlock`:
   ```python
   # SAGE_LITE/sage/networks/decoder_block.py, lines 99-107
   x = self.upsample(x)  # ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
   x = torch.cat([x, skip], dim=1)
   x = self.conv1(x)     # Conv3x3 + BN + ReLU
   x = self.conv2(x)     # Conv3x3 + BN + ReLU
   ```
   At $28 \times 28$, each pixel corresponds to a $16 \times 16$ receptive patch on the input image. A gap of $12\text{ px}$ occupies less than 1 pixel at $28 \times 28$ ($12 / 16 = 0.75\text{ px}$) and only 2 pixels at $56 \times 56$ ($12 / 8 = 1.5\text{ px}$).
   When the two disjoint crack ends enter Decoder Block 0 and Block 1, their receptive fields overlap significantly. The two sequential $3 \times 3$ convolutions across concatenated skip and upsampled streams perform spatial feature smoothing that bridges the sub-pixel or 1-pixel gap before higher resolutions are reached.

---

## 5. What Remains Unproven

While we have precisely pinned the transition point to **Decoder 28 and Decoder 56 (accounting for 70% of emergence)**, the current diagnostic cannot yet establish:

1. **Skip vs Upsample Pathway Attribution:**
   In `DecoderBlock`, does the bridge originate from:
   - (A) The **upsampled deeper stream** (`self.upsample(x)` spreading across the gap)?
   - (B) The **encoder skip connection** (`skip` tensor carrying false continuity from the backbone)?
   - (C) The **non-linear fusion** (`conv1` and `conv2` combining disjoint signals)?
2. **Receptive Field Overlap vs Semantic Hallucination:**
   Is the decoder bridging gaps because the coarse resolution physically merges the features (effective resolution limit of $28 \times 28$ / $56 \times 56$), or because the learned weights actively reward continuous curvilinear structures (inductive bias toward connected cracks)?

---

## 6. Recommended Next Step: Zero-Training Skip vs Upsample Causal Ablation

To resolve the unproven question without retraining:
- Perform a **pathway ablation probe** on the 43-event cohort:
  1. Hook `DecoderBlock 0` ($28 \times 28$) and `DecoderBlock 1` ($56 \times 56$).
  2. Evaluate corridor activation when zeroing/masking:
     - Only the `skip` tensor in the corridor bounding box.
     - Only the `upsample` tensor in the corridor bounding box.
  3. Measure which stream delivers the dominant energy driving the logit positive.
- This will decisively pinpoint whether false bridging is fed by encoder skip leakage or decoder upsampling dilation.
