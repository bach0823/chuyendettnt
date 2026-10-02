# Phase 6 V5 Counterfactual Full-Validation Generalization Test (Both-Half: $\alpha_1=0.5, \alpha_2=0.5$)

**Protocol:** Strictly Diagnostic-Only Full-Validation Generalization Test.  
- Zero training, zero gradient updates, zero optimizer steps.  
- Zero permanent modifications; checkpoints on disk remain untouched.  
- Inference Setting A ($448 \times 448$, stride 448, non-overlapping, reflect padding, exact cropping, fixed threshold $\tau=0.5$).  
- Full population: $N=348$ validation images. Sealed test set ($N=1124$) strictly untouched.  
- Evaluated condition: $\mathbf{W}' = \mathbf{W}_{\text{center}} + 0.5\,\mathbf{W}_{\text{off}}$ for Conv1 and Conv2 in Decoder Block 1.

---

## 1. Protocol

This experiment tests whether the promising spatial-mixing attenuation observed in the V5 25-case sample generalizes across the complete validation distribution ($N=348$).
The intervention is applied strictly in memory during inference:
- Center taps $[1, 1]$: $100\%$ preserved.
- Biases, BatchNorm affine weights ($\gamma, \beta$), running stats ($\mu, \sigma^2$), skip tensors, S2 inputs, and all subsequent layers: $100\%$ preserved.
- Both weights are restored bitwise and verified via tensor equality checks immediately upon run completion.

> **Crucial Disclaimer:** *C3 Both-Half is a counterfactual inference manipulation of the trained Candidate B checkpoint, not a trained model.* It has no learned adaptations and no post-intervention BatchNorm calibration.

---

## 2. C0 Baseline Reproduction Sanity Check

Before running the counterfactual, Candidate B was evaluated on all $N=348$ images under exact Setting A:

| Metric | Official Candidate B Reference | C0 Reproduction | Discrepancy | Status |
|:---|:---:|:---:|:---:|:---:|
| **Global Mean Dice** | $0.7641$ | $0.7641$ | $< 0.0001$ | **PASSED** |
| **Global Mean Precision** | $0.7338$ | $0.7337$ | $-0.0001$ | **PASSED** |
| **Global Mean Recall** | $0.8477$ | $0.8477$ | $0.0000$ | **PASSED** |
| **False Bridge Count** | **$110$** | **$110$** | $0$ | **PASSED (100% exact)** |
| **Breakage Count** | $33$ ($9.48\%$) | $34$ ($9.77\%$) | $+1$ case | **PASSED** |
| **Model Parameters** | $10,118,955$ | $10,118,955$ | $0$ | **PASSED** |

---

## 3. Full-348 Global Metrics (C0 vs Both-Half)

*(Extracted from [`full348_bothhalf_summary.csv`](file:///d:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_spatial_mixing_v5_full348/full348_bothhalf_summary.csv))*

| Metric | Base C0 ($\alpha=1.0$) | Both-Half ($\alpha=0.5$) | Absolute Delta |
|:---|:---:|:---:|:---:|
| **Global Mean Dice** | $0.7641$ | $0.7635$ | $-0.0006$ |
| **Global Median Dice** | $0.8041$ | $0.8040$ | $-0.0001$ |
| **Win / Tie / Loss (Dice)** | — | $157\text{ Wins} / 23\text{ Ties} / 168\text{ Losses}$ | — |
| **Precision** | $0.7337$ | $0.7611$ | **$+0.0274$ (+2.74 pp)** |
| **Recall** | $0.8477$ | $0.8118$ | **$-0.0359$ (-3.59 pp)** |
| **Area Excess (%)** | $+28.29\%$ | $+14.91\%$ | **$-13.38$ pp** |
| **clDice** | $0.8499$ | $0.8477$ | $-0.0022$ |
| **Tprec (Topology Precision)** | $0.8473$ | $0.8707$ | $+0.0234$ |
| **Tsens (Topology Recall)** | $0.8872$ | $0.8562$ | $-0.0310$ |
| **FalseBridge Count** | **$110$** | **$102$** | **$-8$ (7.27% cure)** |
| **FalseBridge Rate (%)** | $31.61\%$ | $29.31\%$ | **$-2.30$ pp** |
| **Breakage Count** | **$34$** | **$73$** | **$+39$ (+11.21 pp)** |
| **Breakage Rate (%)** | $9.77\%$ | $20.98\%$ | $+11.21$ pp |
| **Spurious Islands** | $111$ | $153$ | $+42$ |
| **Mean Abs CC Error** | $1.1092$ | $1.2443$ | $+0.1351$ |

---

## 4. Bridge Transitions Breakdown

*(Extracted from [`full348_bridge_transition.csv`](file:///d:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_spatial_mixing_v5_full348/full348_bridge_transition.csv))*

```
Base Bridge = 110
BothHalf Bridge = 102

Cured = 8
Created = 0
Persistent = 102
NetChange = -8
Cure Fraction = 8 / 110 = 7.27%
```

### Transition Breakdown by Population:
- **Consensus 7/7 ($N=101$):**
  - Base bridges: $101$ $\to$ BothHalf bridges: $100$
  - Cured: **$1$**, Persistent: **$100$**, Created: $0$
  - **Persistence Rate: $99.01\%$** (Cure Fraction: $0.99\%$).
- **D2-Cured Group ($N=7$):**
  - Base bridges: $7$ $\to$ BothHalf bridges: $1$
  - Cured: **$6$**, Persistent: **$1$**, Created: $0$
  - **Cure Fraction: $85.71\%$** (Persistence Rate: $14.29\%$).
- **Low-Consensus Bridges ($N=2$):**
  - Base bridges: $2$ $\to$ BothHalf bridges: $1$
  - Cured: **$1$**, Persistent: **$1$**, Created: $0$
  - **Cure Fraction: $50.00\%$**.
- **Clean Base Population ($N=238$):**
  - Created bridges: **$0$** (zero spurious bridges created).

---

## 5. 7/7 Consensus Population Behavior

The core hard-case population ($N=101$) demonstrates extreme stability:
- **$100 / 101 = 99.01\%$** of bridges remain completely intact after 50% spatial-mixing attenuation.
- **Why did consensus bridges persist?**
  - Base connector median probability was **$0.9484$** ($100\%$ of connector pixels $\ge 0.50$, $81.19\% \ge 0.75$).
  - Attenuating off-center spatial mixing reduced median probability to **$0.8677$** ($\Delta = -0.0807$).
  - Although the confidence softened by $\approx 8$ pp, **$96.04\%$ of connector pixels remained $\ge 0.50$**, keeping the binary topological bridge unbroken under the $\tau=0.5$ threshold.

---

## 6. D2-Cured / D2-Created Subgroup Behavior

- **D2-Cured Cases ($N=7$):**
  - Both-Half cured **$6$ out of the $7$ cases** without any loss function retraining!
  - In this subpopulation, connector probability median collapsed from **$0.8002 \to 0.4392$** ($\Delta = -0.3610$), dropping below the $0.5$ threshold.
  - Only $14.29\%$ of connector pixels remained above $0.50$, completely dissolving the false bridge.
- **D2-Created Cases ($N=6$):**
  - Base bridges = $0$, Both-Half bridges = $0$. Zero bridges created.

---

## 7. Safety Guardrails & Thin Crack Analysis

*(Extracted from [`full348_subgroup_summary.csv`](file:///d:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_spatial_mixing_v5_full348/full348_subgroup_summary.csv))*

| Target / Subgroup | Base C0 | Both-Half ($\alpha=0.5$) | Pre-registered Guardrail | Guardrail Status |
|:---|:---:|:---:|:---:|:---:|
| **False Bridge Count** | $110$ | **$102$** | $N_{\text{bridge}} \le 105$ | **MET** |
| **Bridge Net Change** | — | **$-8$** | $\text{NetChange} \le -5$ | **MET** |
| **Global Mean Dice** | $0.7641$ | **$0.7635$** | $\text{GlobalDice} \ge 0.7640$ | Borderline ($-0.0006$) |
| **Global Recall** | $0.8477$ | **$0.8118$** | $\text{Recall} \ge 0.8400$ | **VIOLATED (-3.59 pp)** |
| **Breakage Rate** | $9.77\%$ ($34$ ca) | **$20.98\%$ ($73$ ca)** | $\text{Breakage} \le 9.5\%$ | **VIOLATED (+11.21 pp)** |
| **Thin Cracks Dice ($n=66$)** | $0.6404$ | **$0.6601$** | $\text{ThinDice} \ge 0.6400$ | **MET (+1.97 pp)** |
| **Thin Cracks Breakage** | $8 / 66$ ($12.12\%$) | **$17 / 66$ ($25.76\%$)** | — | Doubled (+9 cases) |
| **Thin-Low-Area Dice ($n=5$)** | $0.5370$ | **$0.6042$** | — | Strong gain (+6.72 pp) |

---

## 8. Connector Probability Shift

*(Extracted from [`full348_connector_probability.csv`](file:///d:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_spatial_mixing_v5_full348/full348_connector_probability.csv))*

| Subgroup | $N$ | Base Median Prob | Both-Half Median Prob | $\Delta \text{Median}$ | Base $\% \ge 0.5$ | Both-Half $\% \ge 0.5$ | Both-Half $\% \ge 0.75$ | Both-Half $\% \ge 0.90$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Base 110 Bridges** | 110 | $0.9437$ | $0.8585$ | **$-0.0852$** | $100.0\%$ | $90.00\%$ | $74.55\%$ | $34.55\%$ |
| **Consensus 7/7** | 101 | $0.9484$ | $0.8677$ | **$-0.0807$** | $100.0\%$ | $96.04\%$ | $81.19\%$ | $37.62\%$ |
| **D2-Cured** | 7 | $0.8002$ | $0.4392$ | **$-0.3610$** | $100.0\%$ | **$14.29\%$** | $0.00\%$ | $0.00\%$ |
| **Low Consensus** | 2 | $0.8049$ | $0.5554$ | **$-0.2495$** | $100.0\%$ | $50.00\%$ | $0.00\%$ | $0.00\%$ |

---

## 9. Scientific Interpretation: Scenario B Confirmed

Connecting all empirical evidence yields an unambiguous conclusion:

$$\boxed{\text{SCENARIO B IS CONCLUSIVELY CONFIRMED}}$$

> **Core Scientific Finding:**  
> Spatial mixing in Decoder Block 1 is the primary enabler of a **weaker, sensitive bridge subpopulation** (accounting for $6/7 = 85.7\%$ of D2-cured bridges), where connector probability drops from $0.80 \to 0.44$.  
> However, spatial mixing attenuation alone **CANNOT cure the core 7/7 consensus bridge population** ($100 / 101 = 99.01\%$ remain persistent). Even though connector probability softens from $0.95 \to 0.87$, it remains well above the $0.50$ binary threshold.  
> Furthermore, attenuating spatial mixing by 50% at inference time causes **significant collateral damage to crack continuity**, more than doubling the Breakage rate from **$9.77\% \to 20.98\%$**.

---

## 10. Decision for Next Research Step

> [!CAUTION]
> ### Definitive Conclusion & Guardrail Decision:
> 1. **Do NOT adopt inference-time kernel attenuation as a bridge solution:** It violates the pre-registered Breakage guardrail ($20.98\% > 9.5\%$) and leaves $99\%$ of consensus bridges untouched.
> 2. **Scientific Insight for Future Training Probes:** The persistent bridge population is not a shallow decoding artifact that can be surgically turned off by scaling down spatial mixing. It is deeply supported by the combined representations entering Decoder Block 1. Any future architectural probe (e.g. directional anisotropic routing or moat-preserving attention) must tackle the core representation without blindly reducing spatial receptive field, which destroys true crack continuity.
