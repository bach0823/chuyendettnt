# Phase 6-C Pre-Intervention Topology & Connectivity Diagnostic Report

**Evaluation Split:** Crack500 Official Validation Split ($N=348$) under Setting A Tiling Protocol

## 1. Global Topology Profile across 4 Models

| Metric | Candidate B (Base) | Phase 6-A.1 (B-IoU) | Phase 6-A.2 (Pure PLU) | Phase 6-B.1 (AB-BPL) |
| :--- | :---: | :---: | :---: | :---: |
| **Global Dice** | 0.7641 | 0.7683 | 0.7664 | 0.7684 |
| **Centerline Dice (clDice Mean)** | 0.8498 | 0.8445 | 0.8315 | 0.8520 |
| **clDice Median** | 0.9087 | 0.9087 | 0.8873 | 0.9126 |
| **Skeleton Precision (Tprec)** | 0.8473 | 0.8348 | 0.8178 | 0.8460 |
| **Skeleton Sensitivity (Tsens)**| 0.8872 | 0.8875 | 0.8746 | 0.8887 |
| **Mean GT Connected Components**| 2.239 | 2.239 | 2.239 | 2.239 |
| **Mean Pred Connected Components**| 1.615 | 1.894 | 3.348 | 1.707 |
| **Mean Absolute CC Error** | 1.106 | 1.201 | 2.293 | 1.101 |
| **Pred CC < GT CC (Merged/Missed)** | 131 (37.6%) | 112 (32.2%) | 88 (25.3%) | 123 (35.3%) |
| **Pred CC == GT CC (Exact Match)** | 162 (46.6%) | 160 (46.0%) | 107 (30.7%) | 164 (47.1%) |
| **Pred CC > GT CC (Fragment/Island)**| 55 (15.8%) | 76 (21.8%) | 153 (44.0%) | 61 (17.5%) |

## 2. Disentangled Topological Failure Events

| Event Type | Candidate B (Base) | Phase 6-A.1 (B-IoU) | Phase 6-A.2 (Pure PLU) | Phase 6-B.1 (AB-BPL) |
| :--- | :---: | :---: | :---: | :---: |
| **Samples with False Bridge** | 110 (31.6%) | 114 (32.8%) | 111 (31.9%) | 115 (33.0%) |
| **Total Bridge Events** | 118 | 120 | 115 | 120 |
| **Total Merged GT Components** | 357 | 364 | 340 | 364 |
| **Samples with Breakage/Frag**| 33 (9.5%) | 34 (9.8%) | 74 (21.3%) | 29 (8.3%) |
| **Total Fragmented GT Components**| 33 | 35 | 79 | 29 |
| **Total Extra Fragments** | 35 | 56 | 116 | 41 |
| **Samples with Spurious Islands** | 69 (19.8%) | 103 (29.6%) | 176 (50.6%) | 75 (21.6%) |
| **Total Spurious Islands Count** | 108 | 176 | 614 | 127 |
| **Mean Spurious Island Area (px)**| 134.0 | 155.3 | 179.0 | 174.1 |

## 3. Disentangled Phenotype Distribution (Mutual Exclusive Categories)

| Phenotype | Candidate B (Base) | Phase 6-A.1 (B-IoU) | Phase 6-A.2 (Pure PLU) | Phase 6-B.1 (AB-BPL) |
| :--- | :---: | :---: | :---: | :---: |
| **Clean** | 175 (50.3%) | 149 (42.8%) | 99 (28.4%) | 170 (48.9%) |
| **Merge** | 77 (22.1%) | 71 (20.4%) | 41 (11.8%) | 81 (23.3%) |
| **Fragment** | 20 (5.7%) | 19 (5.5%) | 22 (6.3%) | 17 (4.9%) |
| **Island** | 40 (11.5%) | 62 (17.8%) | 87 (25.0%) | 43 (12.4%) |
| **Mixed** | 36 (10.3%) | 47 (13.5%) | 99 (28.4%) | 37 (10.6%) |
