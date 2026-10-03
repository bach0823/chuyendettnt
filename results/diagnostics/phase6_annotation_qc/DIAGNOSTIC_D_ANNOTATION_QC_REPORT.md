# Phase 6 — Diagnostic D: Annotation QC & Gap Stratification Report

**Date:** 2026-10-03  
**Target:** Candidate B False Bridge Cohort (118 Events across 110 Validation Images on Crack500)  
**Objective:** Disentangle genuine architectural false bridges from dataset annotation artifacts and discretization boundaries via rigorous Euclidean gap measurement ($D_{\text{gap}}$) and visual crop inspection.

---

## 1. Executive Summary & Core Findings

1. **Major Annotation Ambiguity Component Identified:**
   - **52.54%** (62 / 118) of all false bridge events have a gap distance **$D_{\text{gap}} \le 3.0$ pixels** (median $D_{\text{gap}} = 2.12$ px).
   - An additional **11.02%** (13 / 118) have $3.0 < D_{\text{gap}} \le 5.0$ pixels (median $D_{\text{gap}} = 4.24$ px).
   - In total, **63.56%** (75 / 118) of all bridge events occur across gaps of **$\le 5.0$ pixels**.

2. **Visual Inspection Confirms Physical Continuity (Panel Verification):**
   - High-resolution visual crops (4-panel verification strips: Raw RGB, GT Components, Pred Overlay, Widest Path Localization) generated for all 118 events reveal that in the majority of $D_{\text{gap}} \le 3$ px cases (e.g. `ev001`), the crack in the raw RGB image is **physically continuous**.
   - The human annotator labeled the continuous crack as two disconnected segments separated by a 1-to-2 pixel gap (likely due to brush strokes, thresholding noise, or manual tracing discretization).
   - Candidate B correctly detects the continuous physical crack features, causing the prediction to span the 2-pixel gap. The evaluation protocol penalizes this as a "false bridge event".

3. **Genuine Architectural False Bridges ($D_{\text{gap}} > 5.0$ px):**
   - **36.44%** (43 / 118) of bridge events have $D_{\text{gap}} > 5.0$ pixels (median $D_{\text{gap}} = 12.00$ px, max = $130.31$ px).
   - These 43 events represent **genuine network errors**:
     - Diagonal shortcuts between two parallel crack branches (e.g. `ev002`, gap = 15.65 px).
     - Hallucinated connectivity across dark asphalt aggregate texture (e.g. `ev004`, gap = 7.21 px).
   - Even on these 43 wider-gap events, Candidate B's median bottleneck probability is **$0.9306$** (88.37% have $P_{\text{bottleneck}} \ge 0.75$). The network has learned high-confidence representational shortcuts.

---

## 2. Gap Stratification Table (N = 118 Events)

| Stratification Category | Definition | Count | % of 118 | Median $D_{\text{gap}}$ | Median $P_{\text{bottleneck}}$ | $P_{\text{bottleneck}} \ge 0.75$ | Median Path Length |
|---|---|---|---|---|---|---|---|
| **Narrow Ambiguity** | $D_{\text{gap}} \le 3.0\text{ px}$ | **62** | **52.54%** | **2.12 px** | **0.9670** | **88.71%** | 4.0 px |
| **Intermediate Ambiguity** | $3.0 < D_{\text{gap}} \le 5.0\text{ px}$ | **13** | **11.02%** | **4.24 px** | **0.9797** | **92.31%** | 7.0 px |
| **Wider Gap (Real Bridge)** | $D_{\text{gap}} > 5.0\text{ px}$ | **43** | **36.44%** | **12.00 px** | **0.9306** | **88.37%** | 15.0 px |
| **Total Cohort** | All bridge events | **118** | **100.00%** | **2.91 px** | **0.9607** | **88.98%** | 7.0 px |

---

## 3. Detailed Profile by Gap Category

### 3.1 Category 1: Narrow Ambiguity ($D_{\text{gap}} \le 3$ px, N = 62)
- **Gap Distribution:** Min 2.00 px, 25th percentile 2.00 px, Median 2.12 px, 75th percentile 2.24 px. (Dominated by immediate 8-neighbor diagonal / 2-pixel Manhattan adjacency).
- **Probability Distribution:** $P_{\text{bottleneck}}$ 25th percentile = 0.9123, Median = 0.9670, 75th percentile = 0.9897.
- **Path Length:** Median 4.0 px.
- **Visual Characterization:** The vast majority exhibit visual continuity in the RGB channel. The annotator left a tiny 1-2 pixel seam between two strokes.

### 3.2 Category 2: Intermediate Ambiguity ($3 < D_{\text{gap}} \le 5$ px, N = 13)
- **Gap Distribution:** Min 3.16 px, Median 4.24 px, Max 5.00 px.
- **Probability Distribution:** $P_{\text{bottleneck}}$ 25th percentile = 0.9692, Median = 0.9797, 75th percentile = 0.9902.
- **Path Length:** Median 7.0 px.
- **Visual Characterization:** Often corresponds to faint crack necks or crack tips where the crack width drops below 1 pixel and annotator confidence became ambiguous.

### 3.3 Category 3: Wider Gap / Genuine Architectural Bridge ($D_{\text{gap}} > 5$ px, N = 43)
- **Gap Distribution:** Min 5.10 px, 25th percentile 8.00 px, Median 12.00 px, 75th percentile 18.03 px, Max 130.31 px.
- **Probability Distribution:** $P_{\text{bottleneck}}$ 25th percentile = 0.8729, Median = 0.9306, 75th percentile = 0.9616.
- **Path Length:** Median 15.0 px (P75 = 26.0 px, Max = 135.0 px).
- **Correlation with Interventions:**
  All 7 cases cured by Phase 6-D.2 belong strictly to this category (median gap of cured cases = 13.2 px). D2's boundary separation loss had sufficient leverage only when the gap was wide enough ($> 5$ px) and the bottleneck probability was lower ($P \approx 0.82$).

---

## 4. Methodological Conclusion & Recommendations

1. **Re-assessing the "118 Bridge Events" Metric:**
   Over **52%** of the recorded bridge events are artifacts of the strict connected-component metric applied to imperfect human annotations at the 1-2 pixel scale. The true number of structural false bridges where Candidate B hallucinates connections across empty pavement ($D > 5$ px) is **43 events** across the validation set.
   
2. **Implication for Model Architecture & Loss:**
   - Any global penalty targeting bridges (such as Boundary IoU, AB-BPL, or HRRB suppression) inevitably suppresses the model on real cracks because in 63.6% of cases, the "bridge" is visually identical to a real crack.
   - For the 43 genuine wider-gap bridges, scalar thresholding is ineffective because $P_{\text{bottleneck}} = 0.9306$. Addressing these requires either:
     (a) **Context-aware long-range discrimination** (preventing shortcuts between distinct parallel branches), or
     (b) **Curvature/Orientation graph pruning** (penalizing unnatural orthogonal/sharp turns that connect parallel cracks).
