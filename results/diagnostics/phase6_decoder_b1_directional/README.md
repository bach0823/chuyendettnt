# Phase 6: Decoder Block 1 Directional Spatial-Mixing Decomposition Diagnostic Report

## 1. Executive Summary & Epistemic Boundaries

This diagnostic investigates whether false bridge formation in Decoder Block 1 depends on specific spatial mixing directions (anisotropic spatial mixing), or whether the kernel operates as an isotropic/orthogonal reconstruction operator where bridge reduction is inextricably coupled to crack breakage.

### Core Scientific Question
> **Does false bridge formation in Decoder Block 1 depend on a specific direction or axis (N, S, E, W, diagonals), and can directional attenuation selectively eliminate false bridges in the 82 Downstream-Resistant cases without destroying true crack continuity?**

### Non-Negotiable Protocol Hard Locks
1. **DIAGNOSTIC-ONLY**: Zero training passes, zero backward passes, zero optimizer steps, zero gradient updates.
2. **Bitwise Invariance**:
   - Checkpoint SHA256: `147f784021414efd229f37faee27618997a0665dfb12c8b8745585d8885f812b` (verified identical before and after).
   - Parameter SHA256: `4aeda58ce6d2fb321ecbb398beaa9142f36f6d5386052be1bb3e162f43d04d80` (verified identical before and after).
3. **Setting A Preserved**: $448 \times 448$ tile, non-overlapping stride 448, reflect padding, exact cropping, $\tau=0.5$.
4. **Cohort Separation**: Evaluated on $N=164$ validation samples (82 Consensus Resistant, 19 Consensus Sensitive, 7 Cured by D2, 56 Clean Control). Sealed test set ($N=1124$) strictly unaccessed.
5. **Calibrated Epistemic Language**:
   - Do NOT use "causal root" without bridge-specific evidence.
   - Use *"directionally implicated spatial operator"* or *"orthogonal connectivity operator"*.

---

## 2. Kernel Decomposition Schema

For both `Conv1` ($288 \to 96$, $3 \times 3$) and `Conv2` ($96 \to 96$, $3 \times 3$) in Decoder Block 1, the $3 \times 3$ kernel grid is decomposed as:

```
           X=0       X=1       X=2
       +---------+---------+---------+
 Y=0   |   NW    |    N    |   NE    |
       |  (0,0)  |  (0,1)  |  (0,2)  |
       +---------+---------+---------+
 Y=1   |    W    | Center  |    E    |
       |  (1,0)  |  (1,1)  |  (1,2)  |
       +---------+---------+---------+
 Y=2   |   SW    |    S    |   SE    |
       |  (2,0)  |  (2,1)  |  (2,2)  |
       +---------+---------+---------+
```

### Evaluated Conditions (19 Counterfactual States):
1. **Baseline**: Untouched pristine weights ($\alpha=1.00$).
2. **Individual Taps at $\alpha=0.50$**: N, S, E, W, NE, NW, SE, SW (8 runs).
3. **Symmetric Axes at $\alpha=0.50$**: Vertical (N+S), Horizontal (E+W), Main Diagonal (NW+SE), Anti Diagonal (NE+SW) (4 runs).
4. **Broad Groups at $\alpha=0.50$**: Axial (N+S+E+W), Diagonal (NE+NW+SE+SW) (2 runs).
5. **Symmetric Axes at $\alpha=0.00$**: Vertical (0.0), Horizontal (0.0) (2 runs).
6. **Broad Groups at $\alpha=0.00$**: Axial (0.0), Diagonal (0.0) (2 runs).

---

## 3. Quantitative Results & Response Matrix

### Table 1: Directional Subgroup Response Matrix

| Condition | $\alpha$ | Taps | Resistant Bridges ($/82$) | Resistant Cured ($/82$) | Resistant Breakage ($/82$) | Resistant Recall | Resistant Median ConnProb | Sensitive Cured ($/19$) | Control Breakage ($/56$) | D2-Cured Bridges ($/7$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | 1.00 | 0 | 82/82 | 0/82 (0.0%) | 7/82 | 0.9298 | 0.8922 | 0/19 (0.0%) | 8/56 | 7/7 |
| **Dir_N_0.50** | 0.50 | 1 | 82/82 | 0/82 (0.0%) | 8/82 | 0.9257 | 0.8789 | 0/19 (0.0%) | 14/56 | 6/7 |
| **Dir_S_0.50** | 0.50 | 1 | 82/82 | 0/82 (0.0%) | 7/82 | 0.9318 | 0.8929 | 0/19 (0.0%) | 10/56 | 7/7 |
| **Dir_E_0.50** | 0.50 | 1 | 82/82 | 0/82 (0.0%) | 6/82 | 0.9391 | 0.9029 | 0/19 (0.0%) | 7/56 | 7/7 |
| **Dir_W_0.50** | 0.50 | 1 | 80/82 | 2/82 (2.4%) | 9/82 | 0.9127 | 0.8502 | 0/19 (0.0%) | 11/56 | 4/7 |
| **Dir_NE_0.50** | 0.50 | 1 | 82/82 | 0/82 (0.0%) | 8/82 | 0.9409 | 0.9138 | 0/19 (0.0%) | 7/56 | 7/7 |
| **Dir_NW_0.50** | 0.50 | 1 | 82/82 | 0/82 (0.0%) | 9/82 | 0.9329 | 0.8942 | 0/19 (0.0%) | 11/56 | 7/7 |
| **Dir_SE_0.50** | 0.50 | 1 | 82/82 | 0/82 (0.0%) | 8/82 | 0.9320 | 0.9018 | 0/19 (0.0%) | 6/56 | 6/7 |
| **Dir_SW_0.50** | 0.50 | 1 | 82/82 | 0/82 (0.0%) | 7/82 | 0.9406 | 0.9115 | 0/19 (0.0%) | 9/56 | 7/7 |
| **Axis_Vertical_0.50** | 0.50 | 2 | 81/82 | 1/82 (1.2%) | 7/82 | 0.9257 | 0.8747 | 0/19 (0.0%) | 13/56 | 6/7 |
| **Axis_Horizontal_0.50** | 0.50 | 2 | 82/82 | 0/82 (0.0%) | 9/82 | 0.9217 | 0.8556 | 0/19 (0.0%) | 11/56 | 5/7 |
| **Axis_Diag_Main_0.50** | 0.50 | 2 | 82/82 | 0/82 (0.0%) | 7/82 | 0.9272 | 0.8909 | 0/19 (0.0%) | 11/56 | 5/7 |
| **Axis_Diag_Anti_0.50** | 0.50 | 2 | 82/82 | 0/82 (0.0%) | 8/82 | 0.9494 | 0.9337 | 0/19 (0.0%) | 8/56 | 7/7 |
| **Group_Axial_0.50** | 0.50 | 4 | 81/82 | 1/82 (1.2%) | 11/82 | 0.9021 | 0.8004 | 0/19 (0.0%) | 14/56 | 3/7 |
| **Group_Diagonal_0.50** | 0.50 | 4 | 82/82 | 0/82 (0.0%) | 10/82 | 0.9529 | 0.9285 | 0/19 (0.0%) | 8/56 | 5/7 |
| **Axis_Vertical_0.00** | 0.00 | 2 | 78/82 | 4/82 (4.9%) | 13/82 | 0.8775 | 0.7751 | 0/19 (0.0%) | 17/56 | 3/7 |
| **Axis_Horizontal_0.00** | 0.00 | 2 | 79/82 | 3/82 (3.7%) | 15/82 | 0.8793 | 0.7477 | 1/19 (5.3%) | 21/56 | **0/7** |
| **Group_Axial_0.00** | 0.00 | 4 | **38/82** | **44/82 (53.7%)** | **70/82** | **0.4950** | **0.3322** | **14/19 (73.7%)** | **46/56** | **0/7** |
| **Group_Diagonal_0.00** | 0.00 | 4 | 80/82 | 2/82 (2.4%) | 16/82 | 0.9320 | 0.8730 | 0/19 (0.0%) | 20/56 | 2/7 |

---

### Table 2: Consensus 7/7 ($N=101$) Overall Segmentation Dynamics Across Key Conditions

| Condition | Median Dice | Median Recall | Median Precision | Median clDice | Median ConnProb | Mean $\Delta\text{ConnProb}$ | T4_S1 $R_{\text{norm}}$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | 0.7607 | 0.9279 | 0.6830 | 0.8873 | 0.8921 | 0.0000 | 0.7250 |
| **Dir_W_0.50** | 0.7630 | 0.8968 | 0.7096 | 0.8787 | 0.8514 | -0.0406 | 0.7166 |
| **Axis_Vertical_0.50** | 0.7633 | 0.9183 | 0.6927 | 0.8856 | 0.8741 | -0.0216 | 0.7059 |
| **Axis_Horizontal_0.50** | 0.7592 | 0.9085 | 0.7026 | 0.8878 | 0.8567 | -0.0315 | 0.7259 |
| **Group_Axial_0.50** | 0.7657 | 0.8966 | 0.7295 | 0.8840 | 0.8055 | -0.0902 | 0.7035 |
| **Group_Diagonal_0.50** | 0.7494 | 0.9444 | 0.6410 | 0.8826 | 0.9289 | +0.0257 | 0.7105 |
| **Axis_Vertical_0.00** | 0.7798 | 0.8739 | 0.7447 | 0.8790 | 0.7722 | -0.1200 | 0.6676 |
| **Axis_Horizontal_0.00** | 0.7773 | 0.8723 | 0.7422 | 0.8818 | 0.7515 | -0.1296 | 0.6843 |
| **Group_Diagonal_0.00** | 0.7513 | 0.9259 | 0.6766 | 0.8890 | 0.8706 | -0.0112 | 0.6533 |
| **Group_Axial_0.00** | **0.6264** | **0.4787** | **0.9204** | **0.7300** | **0.3265** | **-0.5493** | **0.6846** |

---

## 4. Key Scientific Findings

### 1. No Individual Direction Displays Selective Bridge Sensitivity
At $\alpha=0.50$, attenuating any single directional tap (N, S, E, W, NE, NW, SE, SW) leaves $80/82$ to $82/82$ ($97.6\% - 100\%$) of Resistant false bridges intact, and $19/19$ ($100\%$) of Sensitive false bridges intact. 
Median connector probability barely budges (e.g., $0.8922 \to 0.8789$ for N, $0.8929$ for S, $0.9029$ for E). 
There is **no single "culprit direction"** driving the false bridge.

### 2. Diagonal Taps Have Zero Causal Necessity
Zeroing all four diagonal corners simultaneously (`Group_Diagonal_0.00`):
- Only $2/82$ ($2.4\%$) of Resistant bridges and $0/19$ ($0\%$) of Sensitive bridges are cured.
- $80/82$ ($97.6\%$) of Resistant bridges survive without any diagonal spatial mixing.
- Median connector probability remains $0.8706$, well above threshold.
Diagonal spatial mixing is completely dispensable for the false bridge phenotype.

### 3. Axial Mixing Drives Spatial Connectivity, but Isotropically
Zeroing all four cardinal axial taps (`Group_Axial_0.00`):
- Cures $44/82$ ($53.7\%$) of Resistant bridges and $14/19$ ($73.7\%$) of Sensitive bridges.
- Drops median corridor connector probability from $0.8922 \to 0.3322$, successfully decoupling the gap.
- **Catastrophic Continuity Cost**: Breakage in Resistant explodes from $7/82 \to 70/82$ ($85.4\%$), breakage in Controls explodes from $8/56 \to 46/56$ ($82.1\%$), and Recall collapses from $0.9298 \to 0.4950$ ($-43.48$ percentage points).
Neither vertical alone (`Axis_Vertical_0.00`, $4/82$ cured) nor horizontal alone (`Axis_Horizontal_0.00`, $3/82$ cured) achieves bridge removal; the four axial taps function as an orthogonal 4-connectivity grid that connects both true cracks and false bridges indiscriminately.

### 4. Comparison Between Phenotypes (Resistant vs Sensitive vs D2-Cured)
- **Consensus Resistant ($N=82$) & Sensitive ($N=19$)**: Exhibit virtually identical directional profiles. Neither phenotype responds to individual directional taps or single axes. Both require broad 4-axial zeroing before bridges break, with identical severe breakage costs.
- **Cured by D2 ($N=7$)**: Displays a unique directional vulnerability. Completely zeroing the horizontal axis (`Axis_Horizontal_0.00`) cures **$7/7$ ($100\%$)** of D2 bridges with zero breakage ($0/7$) and high Recall ($0.8676$). D2 micro-bridges happened to align with horizontal/near-horizontal gaps. However, this directional vulnerability does NOT generalize to Consensus Resistant cases (where `Axis_Horizontal_0.00` cures only $3/82$, $3.7\%$).

---

## 5. Decision Gate Verdict (Gate C Confirmed)

### Formal Decision: GATE C — REJECT DIRECTIONAL KERNEL INTERVENTION
1. **Absence of Directional Selectivity**:
   No individual direction or axis pair exhibits selective bridge discrimination without proportional true crack breakage.
2. **Characterization of Decoder Block 1 Operator**:
   Decoder Block 1 is a **general orthogonal reconstruction operator (axial 4-connectivity)**. It does not possess a directional bias that selectively creates false bridges while sparing true cracks.
3. **Closure of the Directional Attenuation Branch**:
   We formally **CLOSE** the directional kernel attenuation branch. Further attempts to hand-tune or attenuate individual directional taps in Decoder Block 1 will inevitably fail the trade-off test.
4. **Strategic Pivot**:
   Because Decoder Block 1 is an un-selective amplifier of whatever representation enters it, the solution must lie in **preventing the sub-threshold ambiguous representation from reaching Decoder Block 1 in the first place** (i.e. representation/topological discrimination in the CNN/SAGE/ViT bottleneck or skip connection isolation).
