# Phase 6: Contextual Spatial Dual-Stream Gate (T2-CSDG, K=3 & K=5) Diagnostic Report

## 1. Executive Summary & Experimental Framework

This experiment systematically resolves the central hypothesis left open after the failure of pointwise gating (T2-SDSG):
> **Did pointwise gating fail merely due to a lack of 2D spatial context (H2), or are T2 representations already irreparably entangled upstream (H1)?**

To provide a decisive, clean causal answer, we evaluated the **Contextual Spatial Dual-Stream Gate (T2-CSDG)** across two spatial receptive fields:
- **$K=3$**: $3 \times 3$ Depthwise Conv (Receptive Field = $3 \times 8 = \mathbf{24\text{ px}}$, covering the $20.6\text{ px}$ median false-bridge neck corridor).
- **$K=5$**: $5 \times 5$ Depthwise Conv (Receptive Field = $5 \times 8 = \mathbf{40\text{ px}}$, almost double the median neck width, fully encompassing adjacent crack bodies).

```
T2 = Concat[Bottleneck (192ch); Skip (96ch)]  (288 channels, 56x56)
  ↓
1x1 Conv(288 -> 64) -> BatchNorm -> ReLU
  ↓
KxK DWConv(64 -> 64, groups=64, pad=K//2) -> BatchNorm -> ReLU  (K=3: RF=24px; K=5: RF=40px)
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

## 2. Validation Subgroup Comparison Table (All 4 Architectural Arms)

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
| **T2-CSDG (K=3, RF=24px)** | **Full Validation** | **348** | **0.7654** | **0.8399** | **0.7366** | **0.8467** | **0.8792** | **104** | **-6** | **113** | **-5** | **70** | **+35** | **1.2498** |
| | Consensus 7/7 | 101 | 0.7415 | 0.8922 | 0.6621 | 0.8439 | 0.9173 | 99 | -2 | 108 | -1 | 22 | +11 | 1.4926 |
| | Consensus Resistant | 82 | 0.7615 | 0.9010 | 0.6834 | 0.8560 | 0.9233 | 80 | -2 | 86 | -2 | 14 | +6 | 1.4411 |
| | Consensus Sensitive | 19 | 0.6555 | 0.8541 | 0.5703 | 0.7917 | 0.8915 | 19 | +0 | 22 | +1 | 8 | +5 | 1.7145 |
| | Clean Control | 56 | 0.7511 | 0.7776 | 0.7462 | 0.8101 | 0.8129 | **0** | +0 | **0** | +0 | 15 | +7 | 1.0638 |
| **T2-CSDG (K=5, RF=40px)** | **Full Validation** | **348** | **0.7657** | **0.8405** | **0.7366** | **0.8476** | **0.8801** | **105** | **-5** | **114** | **-4** | **65** | **+30** | **1.2526** |
| | Consensus 7/7 | 101 | 0.7411 | 0.8924 | 0.6619 | 0.8455 | 0.9185 | 100 | -1 | 109 | +0 | 20 | +9 | 1.4975 |
| | Consensus Resistant | 82 | 0.7612 | 0.9008 | 0.6834 | 0.8577 | 0.9243 | 81 | -1 | 87 | -1 | 13 | +5 | 1.4425 |
| | Consensus Sensitive | 19 | 0.6542 | 0.8562 | 0.5692 | 0.7927 | 0.8932 | 19 | +0 | 22 | +1 | 7 | +4 | 1.7347 |
| | Clean Control | 56 | 0.7516 | 0.7795 | 0.7451 | 0.8133 | 0.8151 | **0** | +0 | **0** | +0 | 14 | +6 | 1.0691 |

---

## 3. Spatial Gate Distribution Audit ($K=3$ vs $K=5$)

To evaluate the mechanism of action, spatial gate activations ($g_B, g_S$) were audited on all 101 Consensus 7/7 bridged images across 3 disjoint pixel compartments:
1. **True Crack** ($GT = 1$)
2. **False Bridge Corridor** ($\text{Pred} = 1 \land GT = 0$)
3. **Background** ($GT = 0 \land \text{Pred} = 0$)

```
----------------------------------------------------------------------------------
Compartment                        CSDG (K=3, RF=24px)       CSDG (K=5, RF=40px)
                                    g_B          g_S          g_B          g_S
----------------------------------------------------------------------------------
True Crack Region (Mean):          0.4423       0.5820       0.4576       0.5718
False Bridge Corridor (Mean):      0.4617       0.5906       0.4796       0.5821
Background (Mean):                 1.0302       0.6508       0.9474       0.6692
----------------------------------------------------------------------------------
Selectivity Ratio (Crack / Corridor):
  g_B Selectivity Ratio:           0.9580       -            0.9541       -
  g_S Selectivity Ratio:           -            0.9855       -            0.9823
----------------------------------------------------------------------------------
```

### Empirical Finding: The Selectivity Ratio Remains Invariant and Defective
- For both $K=3$ and $K=5$, the selectivity ratio is strictly **$< 1.0$** ($\sim 0.95 - 0.98$).
- Widening the receptive field from $24\text{ px} \to 40\text{ px}$ **does not produce spatial discrimination**: the gate values at genuine crack pixels remain slightly *lower* than the gate values at false bridge corridors.
- Both spatial kernels exhibit the exact same failure mode: **indiscriminate suppression of narrow foreground structures**.

---

## 4. Evaluation Against Decision Tree & Scientific Synthesis

### 1. Decision Tree Conclusion: The Entire CSDG Family Remains in Case B
- **Trường hợp B Confirmed**:
  - False bridges decrease marginally (by 4 to 5 events out of 118).
  - But Breakage **almost doubles** ($35 \to 65/70$, an increase of $+85\% \text{ to } +100\%$).
  - Clean Control breakages **nearly double** ($8 \to 14/15$, an increase of $+75\% \text{ to } +87\%$).
  - Resistant 82 bridges are essentially untouched ($88 \to 86/87$).

### 2. Disentangling the Hypotheses:
- **Was the failure of $K=3$ due to insufficient receptive field?**
  - **NO**. Expanding the kernel size to $K=5$ (40 px, $2\times$ the median corridor width) shifted the selectivity ratio from $0.958 \to 0.954$ for $g_B$ and $0.985 \to 0.982$ for $g_S$. Receptive field expansion did not move the selectivity metric towards separation.
- **Root Conclusion**:
  - The hypothesis that "T2 possesses separable bridge-vs-crack signals that only require moderate spatial context to decode" is **conclusively unsupported** within the convolutional gating family.
  - At the entry of Decoder Block 1, the representations of narrow true cracks and false bridge corridors are **already deeply entangled and mutually indistinguishable** across both Skip (Stage 0) and Bottleneck streams.
  - Any downstream gate or attenuation at T2 acts merely as an adaptive morphological erosion filter.

---

## 5. Epistemic Classification of Conclusions

| Category | Claim | Status | Proof / Evidence |
| :--- | :--- | :---: | :--- |
| **Intervention Scope** | The CSDG family ($K=3, K=5$), within this formulation and protocol, fails to achieve selective bridge suppression without severe true crack breakage. | **Supported by Evidence** | Setting A Full Validation shows bridge reduction of only 4-5 events accompanied by a 85-100% surge in breakages ($35 \to 65/70$). |
| **Spatial Mechanism** | Increasing spatial context from 24 px to 40 px does not improve gate selectivity (ratio remains ~0.95-0.98). | **Supported by Evidence** | Spatial gate audit on 101 consensus bridge images in `gate_spatial_distribution_audit_k5.csv`. |
| **Resistant 82** | Consensus Resistant 82 false bridges are invariant to spatial context gating (88 $\to$ 86/87). | **Supported by Evidence** | Subgroup analysis across 82 consensus resistant images. |
| **Causality** | T2 representation is already irreparably corrupted upstream prior to Decoder Block 1. | **Consistent with Evidence, Strongest Hypothesis** | Neither linear probes (AUC 0.69), pointwise gates, nor 24px/40px spatial gates can separate neck from crack without breaking thin cracks. |
| **Generalization** | Every conceivable intervention at T2 is impossible. | **Not Proven / Bounded** | Non-local graph neural networks or topological persistence modules remain uncharacterized, but convolutional gating at T2 is definitively closed. |

---

## 6. Next Steps: Closing T2 Intervention & Moving Upstream

With both Pointwise SDSG and Spatial CSDG ($K=3, K=5$) conclusively evaluated and exhibiting the exact same erosion pathology, **Phase 6 exploration at T2 is officially closed**.

The logical next step in the investigation framework is to pivot to **Upstream Genesis**:
- Investigate **Stem AC / Stage 0 aliasing and downsampling** to determine where and why the false bridge corridor is first hallucinated before reaching T2.
