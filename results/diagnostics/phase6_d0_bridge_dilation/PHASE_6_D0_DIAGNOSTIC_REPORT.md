# Phase 6-D.0 Diagnostic Report: Deconfounding Over-dilation and False Bridge Failures

**Evaluation Split:** Crack500 Official Validation Split ($N=348$) under Canonical Setting A Tiling ($448 \times 448$)  
**Artifact Directory:** `results/diagnostics/phase6_d0_bridge_dilation/`  
**Generated Date:** 2026-10-01  

---

## 1. Executive Summary & Core Research Questions

Phase 6-D.0 investigates the central topological question facing Phase 6:
> **Is False Bridge (component merge) merely a downstream symptom of Over-dilation ($\text{AreaExcess} > 0$), or does it constitute an independent topology-separation bottleneck?**

### The Key Finding: A Dual Phenotype

1. **Strong Conditional Dependency (Dilation-driven Bridges):**
   - Having $\text{AreaExcess} > 0$ increases the odds of a False Bridge by **$5.67\times$** ($p < 0.0001$).
   - The False Bridge rate escalates monotonically with dilation severity:
     * $\text{AreaExcess} \le 0\%$ (Under-segmented / tight): **$10.7\%$** ($11 / 103$)
     * $\text{AreaExcess} \in (0, 10\%]$ (Tight boundary): **$29.0\%$** ($18 / 62$)
     * $\text{AreaExcess} \in (10, 30\%]$ (Moderate dilation): **$34.5\%$** ($29 / 84$)
     * $\text{AreaExcess} > 30\%$ (Severe dilation): **$52.5\%$** ($52 / 99$)
   - Of the $110$ False Bridge samples in Candidate B, **$99$ samples ($90.0\%$) occur when $\text{AreaExcess} > 0$**.

2. **Intrinsic Separation Residual (Separation Bottleneck):**
   - Even when $\text{AreaExcess} \le 0\%$, **$10.7\%$ of samples ($11$ samples) still suffer from False Bridges**.
   - In `Complex_Topology` ($n=27$), the False Bridge rate remains **$66.7\%$** even when $\text{AreaExcess} \le 0\%$!
   - This proves that for closely spaced cracks or junction topologies, reducing area alone cannot resolve topological confusion.

3. **Cross-Model Transition Barrier:**
   - Shifting from Candidate B to A1, A2, B1, or C1 leaves $\sim 105 - 108$ of the $110$ bridges completely **persistent**.
   - Curing a False Bridge purely through thinning required an extreme local area reduction ($\Delta \text{AreaExcess} = -40.16\%$ in A2).
   - Concurrently, A2 and B1 created $6 - 7$ new bridges elsewhere due to local artifacts or fragmentation.

---

## 2. Quantitative Evidence

### 2.1 Contingency & Conditional Probabilities ($N=348$ Global)

| Model | $P(\text{Bridge} \mid \text{Excess} \le 0)$ | $P(\text{Bridge} \mid \text{Excess} > 0)$ | Odds Ratio | Fisher $p$-value | Point-Biserial $r$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Candidate B (Base)** | **10.7%** ($11/103$) | **40.4%** ($99/245$) | **5.67** | $p < 0.0001$ | $+0.251$ ($p < 0.0001$) |
| **Phase 6-A.1 (B-IoU)** | **14.7%** ($14/95$) | **39.5%** ($100/253$) | **3.78** | $p < 0.0001$ | $+0.213$ ($p = 0.0001$) |
| **Phase 6-A.2 (PLU)** | **16.1%** ($19/118$) | **40.0%** ($92/230$) | **3.47** | $p < 0.0001$ | $+0.180$ ($p = 0.0007$) |
| **Phase 6-B.1 (AB-BPL)** | **12.0%** ($12/100$) | **41.5%** ($103/248$) | **5.21** | $p < 0.0001$ | $+0.248$ ($p < 0.0001$) |
| **Phase 6-C.1 (clDice)** | **12.2%** ($10/82$) | **38.7%** ($103/266$) | **4.55** | $p < 0.0001$ | $+0.252$ ($p < 0.0001$) |

---

### 2.2 Quantile Stratification on Candidate B ($N=348$)

| Area Excess Stratum | Sample Count ($N$) | % of Dataset | Bridge Count | False Bridge Rate | Mean GT Components |
| :--- | :---: | :---: | :---: | :---: | :---: |
| $\le 0\%$ (Under-segmented) | 103 | 29.6% | 11 | **10.7%** | 1.81 |
| $(0\%, 10\%]$ (Tight boundary) | 62 | 17.8% | 18 | **29.0%** | 1.98 |
| $(10\%, 30\%]$ (Moderate dilation) | 84 | 24.1% | 29 | **34.5%** | 2.44 |
| $> 30\%$ (Severe dilation) | 99 | 28.4% | 52 | **52.5%** | 2.68 |

---

### 2.3 Paired Cross-Model Transition Dynamics

| Transition Pair | Persistent Bridges | Cured Bridges | Created Bridges | Net Delta | Mean $\Delta \text{Excess}$ (Cured) | Mean $\Delta \text{Excess}$ (Persistent) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base $\to$ A1 (B-IoU)** | 107 | 3 | 7 | $+4$ | $-19.70\%$ | $-8.65\%$ |
| **Base $\to$ A2 (PLU)** | 105 | 5 | 6 | $+1$ | $-40.16\%$ | $-14.56\%$ |
| **Base $\to$ B1 (AB-BPL)** | 108 | 2 | 7 | $+5$ | $-24.42\%$ | $-3.96\%$ |
| **Base $\to$ C1 (clDice)** | 108 | 2 | 5 | $+3$ | $+6.15\%$ | $+5.00\%$ |

---

## 3. Conclusions for Phase 6-D Design

1. **False Bridge is NOT purely an over-dilation artifact:**
   While over-dilation substantially raises bridge probability ($10.7\% \to 52.5\%$), a residual $10.7\%$ of under-segmented samples still bridge. In complex topologies, this residual is $66.7\%$.
2. **Mild boundary objectives cannot cure bridges:**
   Previous interventions (A1, B1) only marginally shifted global area excess and were completely ineffective at severing false bridges (curing only 2–3 bridges while creating 7 new ones).
3. **Implication for 6-D:**
   A naive combination of $A1 + B1$ will merely intensify boundary pressure without providing a repulsive or gap-preserving mechanism.
