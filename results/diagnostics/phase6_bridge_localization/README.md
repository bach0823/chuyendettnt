# Phase 6 Bridge Localization Diagnostic Report

**Evaluation Split:** Crack500 Official Validation Split ($N=348$) under Canonical Setting A Tiling ($448 \times 448$)  
**Artifact Directory:** `results/diagnostics/phase6_bridge_localization/`  
**Generated Date:** 2026-10-02  
**Investigation Mode:** **DIAGNOSTIC-ONLY (No training, no model/evaluator modifications)**

---

## 1. Executive Summary & Core Diagnostic Answers

This diagnostic investigation was executed to localize the mechanistic causal origin of the persistent **False Bridge bottleneck** (~109–115 failure cases across Phase 6 probes) before considering any further training interventions.

| Question | Diagnostic Finding | Decisive Implication |
| :--- | :--- | :--- |
| **1. Consensus Stability:** *Do the same images fail across models, or do different models fail randomly?* | **Same hard images fail repeatedly.**<br>• $101 / 110$ Base bridges ($91.8\%$) fail in **all 7 models (7/7)**.<br>• $106 / 110$ ($96.4\%$) fail in **$\ge 6/7$ models**.<br>• Mean pairwise Jaccard similarity across all 7 models is **$0.9319$** ($0.8879 - 0.9741$). | Total bridge count stability is **NOT** a numerical coincidence. It is driven by an invariant, shared failure locus across all architectures and objectives. |
| **2. Probability Phenotype:** *Are bridge pixels marginally positive ($\approx 0.5$) or confidently positive?* | **Confidently positive.**<br>• Bridge-FP pixel median probability is **$0.8652$** (mean: $0.8202$).<br>• **$42.55\%$** of all bridge pixels have probability $> 0.90$.<br>• **$106 / 110$ ($96.36\%$)** bridge images are *Confident Bridges* ($\text{median} \ge 0.75$).<br>• **$0 / 110$ ($0.00\%$)** are *Marginal Bridges* ($\text{median} < 0.60$). | False bridge is **NOT** an inference threshold artifact. Threshold shifts cannot cure bridges without severely truncating legitimate crack pixels. |
| **3. 112-Grid Representability:** *Does the 112×112 decoder grid resolution collapse GT component separation?* | **Mostly preserved.**<br>• In **$67.3\%$ ($74/110$)** of Base bridge images, GT components remain **fully separated** on the stride-4 decoder grid.<br>• Only **$25.5\%$ ($28/110$)** suffer a grid-induced merge. | The 112-grid resolution is **NOT** the primary ceiling. In over two-thirds of bridge cases, the grid provides sufficient capacity to separate components, yet the model actively predicts foreground in the gap. |

---

## 2. Task A — 7-Model Bridge Consensus Diagnostic

### 2.1 Sanity Check Verification
Across all 348 validation images in canonical Setting A, official model bridge counts reproduce exactly:
- **Candidate B (Base):** $110 / 348$ ($31.61\%$)
- **Phase 6-A.1 (BoundaryIoU):** $114 / 348$ ($32.76\%$)
- **Phase 6-A.2 (Pure PLU):** $111 / 348$ ($31.90\%$)
- **Phase 6-B.1 (AB-BPL):** $115 / 348$ ($33.05\%$)
- **Phase 6-C.1 (clDice):** $113 / 348$ ($32.47\%$)
- **Phase 6-D.1 (PLU+AB-BPL):** $112 / 348$ ($32.18\%$)
- **Phase 6-D.2 (Pure Sep):** $109 / 348$ ($31.32\%$)

### 2.2 Consensus Distribution Across 7 Models ($N=348$)
Trích xuất từ [`bridge_consensus_summary.csv`](file:///d:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_bridge_localization/bridge_consensus_summary.csv):

| Consensus Stratum ($k/7$) | Image Count ($N$) | % of Dataset ($N=348$) | Mean Base Dice | Median Base Dice | Mean Min Gap | Median Min Gap |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 / 7 (Consistently Clean)** | 228 | 65.52% | 0.7794 | 0.8213 | 79.58 px | 43.30 px |
| **1 / 7 (Model-Isolated)** | 3 | 0.86% | 0.5460 | 0.7456 | 26.75 px | 32.00 px |
| **2 / 7** | 2 | 0.57% | 0.7182 | 0.7182 | 8.50 px | 8.50 px |
| **3 / 7** | 1 | 0.29% | 0.7867 | 0.7867 | 11.00 px | 11.00 px |
| **4 / 7** | 3 | 0.86% | 0.8021 | 0.8536 | 24.82 px | 7.00 px |
| **5 / 7** | 5 | 1.44% | 0.6396 | 0.7140 | 17.92 px | 8.25 px |
| **6 / 7** | 5 | 1.44% | 0.8560 | 0.8416 | 13.20 px | 7.00 px |
| **7 / 7 (Consensus Hard Core)** | **101** | **29.02%** | 0.7371 | 0.7611 | 7.21 px | **2.24 px** |

### 2.3 Cumulative Intersections
- **All 7 models ($7/7$):** **$101$ images ($29.02\%$)** — Base Dice: $0.7371$, Median Min Gap: $2.24\text{ px}$.
- **$\ge 6$ models ($\ge 6/7$):** **$106$ images ($30.46\%$)** — Base Dice: $0.7427$, Median Min Gap: $2.24\text{ px}$.
- **$\ge 5$ models ($\ge 5/7$):** **$111$ images ($31.90\%$)** — Base Dice: $0.7381$, Median Min Gap: $2.24\text{ px}$.
- **$\ge 4$ models ($\ge 4/7$):** **$114$ images ($32.76\%$)** — Base Dice: $0.7398$, Median Min Gap: $2.24\text{ px}$.
- **$\ge 1$ model (Any Bridge):** **$120$ images ($34.48\%$)** — only 120 images ever experience a bridge event across all 7 interventions!

### 2.4 Pairwise Model Jaccard Overlap Matrix
Trích xuất từ [`bridge_model_pairwise_jaccard.csv`](file:///d:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_bridge_localization/bridge_model_pairwise_jaccard.csv):

| Model Pair | Intersection | Union | Jaccard Similarity $J(A, B)$ |
| :--- | :---: | :---: | :---: |
| **Base ∩ A1** | 107 | 117 | 0.9145 |
| **Base ∩ A2** | 105 | 116 | 0.9052 |
| **Base ∩ B1** | 108 | 117 | 0.9231 |
| **Base ∩ C1** | 108 | 115 | 0.9391 |
| **Base ∩ D1** | 105 | 117 | 0.8974 |
| **Base ∩ D2** | 103 | 116 | **0.8879** |
| **A1 ∩ B1** | 113 | 116 | **0.9741** |
| **C1 ∩ B1** | 112 | 116 | 0.9655 |
| **D1 ∩ A1** | 110 | 116 | 0.9483 |
| **D2 ∩ C1** | 108 | 114 | 0.9474 |
| **All Distinct Pairs Mean** | — | — | **0.9319** (Range: $0.8879 - 0.9741$) |

---

## 3. Task B — Base Bridge-Pixel Probability Diagnostic

### 3.1 Sanity Check Verification
Under official Setting A tiling on Candidate B:
- **Sample-Mean Dice:** $0.7641$ (Exact match to official baseline $0.7641$)
- **Sample-Mean Precision:** $0.7338$ (Exact match to official baseline $0.7338$)
- **Sample-Mean Recall:** $0.8477$ (Exact match to official baseline $0.8477$)
- **Total Bridge Images:** **$110 / 348$** (Exact match)

### 3.2 Global Pixel Probability Statistics
Trích xuất từ [`bridge_probability_global_stats.csv`](file:///d:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_bridge_localization/bridge_probability_global_stats.csv):

| Pixel Group | Pixel Count ($N$) | Mean Prob | Median Prob | Std | P10 | P25 | P75 | P90 | Min | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TP-crack** | 3,915,789 | 0.9425 | **0.9861** | 0.1005 | 0.8103 | 0.9489 | 0.9945 | 0.9972 | 0.5000 | 0.9999 |
| **Bridge-FP** | 402,820 | 0.8202 | **0.8652** | 0.1513 | 0.5771 | 0.6980 | 0.9567 | 0.9835 | 0.5000 | 0.9995 |
| **Other-FP** | 664,593 | 0.8044 | **0.8407** | 0.1571 | 0.5614 | 0.6674 | 0.9514 | 0.9852 | 0.5000 | 0.9997 |

### 3.3 Separation Statistics
- $\text{mean}(\text{Bridge-FP}) - \text{mean}(\text{TP-crack}) = -0.1223$
- $\text{median}(\text{Bridge-FP}) - \text{median}(\text{TP-crack}) = -0.1209$
- $\text{mean}(\text{Bridge-FP}) - \text{mean}(\text{Other-FP}) = +0.0157$
- $\text{median}(\text{Bridge-FP}) - \text{median}(\text{Other-FP}) = +0.0245$

### 3.4 Probability Histogram Distribution
Trích xuất từ [`bridge_probability_histogram.csv`](file:///d:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_bridge_localization/bridge_probability_histogram.csv):

| Probability Bin | Bridge-FP Pixels | % of Bridge-FP | TP-crack Pixels | % of TP-crack | Other-FP Pixels | % of Other-FP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $[0.50, 0.55)$ | 26,442 | 6.56% | 49,547 | 1.27% | 54,949 | 8.27% |
| $[0.55, 0.60)$ | 25,460 | 6.32% | 50,075 | 1.28% | 49,121 | 7.39% |
| $[0.60, 0.65)$ | 24,818 | 6.16% | 53,716 | 1.37% | 46,222 | 6.95% |
| $[0.65, 0.70)$ | 25,031 | 6.21% | 60,314 | 1.54% | 45,533 | 6.85% |
| $[0.70, 0.75)$ | 26,211 | 6.51% | 70,497 | 1.80% | 45,740 | 6.88% |
| $[0.75, 0.80)$ | 29,217 | 7.25% | 86,825 | 2.22% | 48,039 | 7.23% |
| $[0.80, 0.90)$ | 74,229 | 18.43% | 282,141 | 7.21% | 117,656 | 17.70% |
| **$[0.90, 1.00]$** | **171,412** | **42.55%** | **3,262,674** | **83.32%** | **257,333** | **38.72%** |

### 3.5 Per-Image Bridge Phenotype Distribution ($N=110$ Base Bridge Images)
- **Confident Bridge ($\text{median} \ge 0.75$):** **$106 / 110$ images ($96.36\%$)**
- **Intermediate Bridge ($0.60 \le \text{median} < 0.75$):** **$4 / 110$ images ($3.64\%$)**
- **Marginal Bridge ($\text{median} < 0.60$):** **$0 / 110$ images ($0.00\%$)**

---

## 4. Task C — 112-Grid Representability Diagnostic

### 4.1 Objective & Experimental Protocol
Evaluates whether downsampling to the decoder's stride-4 feature grid ($112 \times 112$ for a $448 \times 448$ tile) collapses distinct Ground Truth connected components into single merged components under nearest-neighbor interpolation.

Trích xuất từ [`bridge_112grid_representability.csv`](file:///d:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_bridge_localization/bridge_112grid_representability.csv):

| Subgroup | $N$ Images | Stride-4 Merged % | Preserved Separation % | Fixed 112×112 Merged % | Original Min Gap (Med / Min / Max) | Stride-4 Median Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Multi-CC Images** | 176 | 33 / 176 (**18.8%**) | **73.3%** | 34 / 176 (19.3%) | 4.7 / 2.0 / 480.1 px | 3.2 px |
| **All Base Bridge Images** | 110 | 28 / 110 (**25.5%**) | **67.3%** | 29 / 110 (26.4%) | 2.2 / 2.0 / 130.3 px | 2.2 px |
| **Base $\to$ D2 Persistent** | 103 | 28 / 103 (**27.2%**) | **65.0%** | 29 / 103 (28.2%) | 2.2 / 2.0 / 130.3 px | 2.2 px |
| **Base $\to$ D2 Cured** | 7 | 0 / 7 (**0.0%**) | **100.0%** | 0 / 7 (0.0%) | 8.2 / 3.0 / 56.1 px | 3.2 px |
| **Base Clean $\to$ D2 Created** | 6 | 1 / 6 (**16.7%**) | **83.3%** | 2 / 6 (33.3%) | 6.2 / 2.0 / 65.5 px | 3.0 px |
| **7/7 Consensus Bridges** | 101 | 28 / 101 (**27.7%**) | **64.4%** | 29 / 101 (28.7%) | 2.2 / 2.0 / 130.3 px | 2.2 px |

### 4.2 Key Representability Observation
1. Only **$28$ out of $110$ Base bridge images ($25.5\%$)** experience a purely geometric collision where the stride-4 downsampling merges distinct components.
2. In **$74$ out of $110$ Base bridge images ($67.3\%$)**, the 112-grid **fully preserves component separation**.
3. Therefore, false bridging cannot be dismissed as a mere resolution quantization bottleneck of the 112-grid. The decoder retains sufficient grid resolution to preserve separation in over two-thirds of bridge cases, yet repeatedly fails to separate them.

---

## 5. Neutral Decision Support & Mechanism Synthesis

Combining the findings from Tasks A, B, and C provides clear boundaries on candidate hypotheses:

### 1. Can thresholding or simple post-processing solve False Bridge?
**NO.**
- $96.36\%$ of bridge images are *Confident Bridges* ($\text{median} \ge 0.75$), and $42.55\%$ of bridge pixels have probability $> 0.90$.
- Shifting the probability threshold upward cannot eliminate bridge connections without heavily carving away legitimate crack pixels (mean TP probability is $0.9425$).

### 2. Is False Bridge an artifact of the 112×112 resolution grid?
**NO (in $\ge 67\%$ of cases).**
- While very narrow gaps ($\le 2\text{ px}$) can collide on a stride-4 grid, $67.3\%$ of bridge cases have adequate spatial resolution to remain separate. The model actively generates high-probability foreground where background should be.

### 3. Is False Bridge an invariant shared representation failure?
**YES.**
- The $101$ consensus bridge cases ($91.8\%$ of Base bridges, $84.2\%$ of all bridge-prone images) fail universally across:
  * Objective loss changes (BoundaryIoU, AB-BPL, clDice, Separation Moat)
  * Feature head changes (PLU vs standard Base decoder)
- The high pairwise Jaccard ($0.9319$) demonstrates that intervening on losses and boundary margins does not alter the underlying feature confusion.

### 4. Scientific Recommendation for Next Steps
Before deciding whether a new probe (such as D3 Feature Discrimination) is justified:
- **Spatial / appearance ambiguity at the bridge locus:** The bridge pixels are predicted with extreme confidence ($p > 0.86$), strongly suggesting that the model perceives bridge pixels as authentic crack texture (e.g. low-contrast asphalt boundaries, dark seams, aggregate edges).
- **Decoder vs Representation Locus:** Because all probes shared the frozen Candidate B Stage 1 encoder representation and modified only Stage 2 loss / head tuning, the persistent consensus indicates that **the representation itself lacks feature discriminability** between genuine crack connectivity and pseudo-crack surface textures in narrow inter-component gaps.
