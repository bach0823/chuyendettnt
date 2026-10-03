# Phase 6D: Context-Guided Stage-1 Skip Refinement (CGSR) Evaluation Report

**Date:** 2026-10-03 17:40:10

## Final Scientific Verdict: `H2 NOT SUPPORTED`

> CGSR fails to cure wider-gap false bridges (cure rate: 11.6%, median Delta z_bridge: +0.1595), gate is either non-selective (contrast: +0.0004) or inactive, or causes excessive degradation to true cracks.

---

## 1. Official Setting A Benchmark (Full Val N=348)

| Model | Val Dice | Val IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| **Control (Stage 2 Locked Base)** | 0.7641 | 0.6417 | 0.7337 | 0.8477 |
| **CGSR (Phase 6D Intervention)** | 0.7667 | 0.6429 | 0.7337 | 0.8458 |
| **Delta (CGSR - Control)** | **+0.0027** | **+0.0012** | **+0.0000** | **-0.0019** |

## 2. False Bridge Cohort Performance

| Cohort | N | Cured Events | Cure Rate (%) | Median Delta z_bridge | Mean Delta z_bridge |
|---|---:|---:|---:|---:|---:|
| **43 Wider-Gap (D_gap > 5 px)** | 43 | 5 | **11.6%** | +0.1595 | +0.2170 |
| Group A (5 < D <= 8 px) | 11 | 1 | 9.1% | +0.1595 | +0.2322 |
| Group B (D > 8 px) | 32 | 4 | 12.5% | +0.1467 | +0.2117 |
| All 118 Bridge Events | 118 | 6 | 5.1% | +0.0165 | +0.0284 |

## 3. CGSR Gate Mechanistic Diagnostics

| Metric | Median | Mean |
|---|---:|---:|
| Bridge Gate ($G_{bridge}$) | 0.9549 | 0.9547 |
| True Crack Gate ($G_{crack}$) | 0.9555 | 0.9552 |
| Gate Contrast ($G_{crack} - G_{bridge}$) | **+0.0004** | **+0.0006** |

## 4. Negative Controls & Collateral Damage

- Clean Crack Segments Tested: 118
- Clean Crack Breakage Events: 13 (11.0%)
- Mean Delta z on Clean Cracks: +0.0557

