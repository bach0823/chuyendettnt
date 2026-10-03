# Phase 6D: Context-Guided Stage-1 Skip Refinement (CGSR) Evaluation Report

**Date:** 2026-10-03 15:45:55

## Final Scientific Verdict: `H2 NOT SUPPORTED`

> CGSR fails to cure wider-gap false bridges (cure rate: 9.3%, median Delta z_bridge: -0.1386), gate is either non-selective (contrast: +0.0003) or inactive, or causes excessive degradation to true cracks.

---

## 1. Official Setting A Benchmark (Full Val N=348)

| Model | Val Dice | Val IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| **Control (Stage 2 Locked Base)** | 0.7641 | 0.6417 | 0.7337 | 0.8477 |
| **CGSR (Phase 6D Intervention)** | 0.7589 | 0.6359 | 0.7235 | 0.8517 |
| **Delta (CGSR - Control)** | **-0.0052** | **-0.0058** | **-0.0102** | **+0.0040** |

## 2. False Bridge Cohort Performance

| Cohort | N | Cured Events | Cure Rate (%) | Median Delta z_bridge | Mean Delta z_bridge |
|---|---:|---:|---:|---:|---:|
| **43 Wider-Gap (D_gap > 5 px)** | 43 | 4 | **9.3%** | -0.1386 | +0.0014 |
| Group A (5 < D <= 8 px) | 11 | 1 | 9.1% | -0.3078 | -0.0553 |
| Group B (D > 8 px) | 32 | 3 | 9.4% | +0.0286 | +0.0210 |
| All 118 Bridge Events | 118 | 5 | 4.2% | -0.1793 | -0.1035 |

## 3. CGSR Gate Mechanistic Diagnostics

| Metric | Median | Mean |
|---|---:|---:|
| Bridge Gate ($G_{bridge}$) | 0.9556 | 0.9554 |
| True Crack Gate ($G_{crack}$) | 0.9562 | 0.9560 |
| Gate Contrast ($G_{crack} - G_{bridge}$) | **+0.0003** | **+0.0006** |

## 4. Negative Controls & Collateral Damage

- Clean Crack Segments Tested: 118
- Clean Crack Breakage Events: 12 (10.2%)
- Mean Delta z on Clean Cracks: -0.0356

