# Phase 6-D.2 Diagnostic Report: Pure Inter-Component Separation Supervision (Negative Moat)

**Evaluation Split:** Crack500 Official Validation Split ($N=348$) under Canonical Setting A Tiling ($448 \times 448$)  
**Artifact Directory:** `results/diagnostics/phase6_d2_topology/`  
**Master Dataset:** `results/diagnostics/phase6_d2_topology/topology_7models_master_paired.csv`  
**Generated Date:** 2026-10-02  
**Final Verdict:** **CLOSED (Negative for Primary Objective)**

---

## 1. Executive Summary & Official Verdict

Phase 6-D.2 investigated whether **explicit inter-component spatial separation supervision** ($L_{\text{sep}}$ with $G_{\max} = 8.0\text{ px}, \lambda_{\text{sep}} = 0.010$) can resolve the persistent False Bridge bottleneck without sacrificing crack continuity:

$$
\boxed{\text{Phase 6-D.2: Negative for the Primary Objective}}
$$

### Core Findings
1. **False Bridge Persistence:**
   - Base False Bridge count barely moved: **$110 \to 109$** (Net change: **$-1$** vs pre-registered target $\le -5$).
   - $103/110$ ($93.64\%$) bridges from Base remained completely persistent.
   - Out of the $105$ bridges persistent under D.1, D.2 resolved only **$4/105$ ($3.81\%$)**, while **$101/105$ ($96.19\%$)** remained persistent.
   - $7$ cured bridges were immediately offset by $6$ newly created bridges on previously clean images.
2. **Severe Breakage Inflation (Guardrail Breach):**
   - Global Breakage surged from **$9.48\%$ to $14.37\%$** ($+4.89\text{ pp}$ over Base, breaching the $\le 9.5\%$ guardrail).
   - In Thin Cracks ($n=66$), Breakage climbed from $12.12\%$ to $16.67\%$.
   - In Thin-low-area cracks ($n=5$), Breakage jumped from $0.0\%$ to **$40.0\%$**.
3. **Degradation in Thin Crack Recovery:**
   - Thin Crack Dice dropped from $0.6404$ to **$0.6300$** ($\Delta = -0.0104$).
   - Thin-low-area Dice dropped from $0.5370$ to **$0.5120$** ($\Delta = -0.0250$).
4. **Action Directive:**
   - **D.2 is permanently closed.**
   - No $\lambda_{\text{sep}}$ sweeps, no moat radius tuning, no attempts to rescue this probe.

---

## 2. Pre-Registered Guideposts & Guardrails Evaluation

| Guidepost / Guardrail | Pre-Registered Target | Actual Result (D.2) | Status |
| :--- | :---: | :---: | :---: |
| **Primary: False Bridge Count** | $\le 105$ ($\text{Net} \le -5$) | **$109$** ($\text{Net} = -1$) | ❌ **FAIL** |
| **Guardrail: Breakage Rate** | $\le 9.50\%$ | **$14.37\%$** ($50/348$) | ❌ **FAIL** |
| **Guardrail: Thin Crack Dice** | $\ge 0.6400$ | **$0.6300$** | ❌ **FAIL** |
| **Guardrail: Global Dice** | $\ge 0.7640$ | **$0.7589$** | ❌ **FAIL** |
| **Guardrail: Global Recall** | $\ge 0.8400$ | **$0.8591$** | ✅ **PASS** |

---

## 3. Transition Dynamics vs Candidate B and Phase 6-D.1

```text
Base Bridge Count:         110/348 (31.61%)
D.1 Bridge Count:          112/348 (32.18%)
D.2 Bridge Count:          109/348 (31.32%)

D.2 vs Base:
  - Cured Bridges:         7 cases
  - Created Bridges:       6 cases
  - Persistent Bridges:    103/110 (93.64%)
  - Net Change vs Base:    -1 case

D.2 on D.1-Persistent Bridges (105 cases):
  - Resolved by D.2:       4/105 (3.81%)
  - Still persistent:      101/105 (96.19%)
```

---

## 4. Multi-Cohort Cross-Tabulation (7-Model Canonical Setting A)

Trích xuất từ `topology_7models_master_paired.csv`:

### 4.1 Global Cohort ($N=348$)
| Model | Dice (Mean/Med) | $\Delta$ Dice | Win vs Base | Prec | Rec | AreaExcess | clDice | Bridge Count (%) | Breakage (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate B (Base)** | 0.7641 / 0.8041 | ref | --- | 0.7338 | 0.8477 | $+8.16\%$ | 0.8498 | 110 (31.6%) | 9.48% |
| **Phase 6-A.1 (B-IoU)** | 0.7683 / 0.8114 | $+0.0043$ | 55.5% | 0.7416 | 0.8413 | $+7.85\%$ | 0.8445 | 114 (32.8%) | 9.77% |
| **Phase 6-A.2 (PLU)** | 0.7664 / 0.8092 | $+0.0023$ | 46.8% | 0.7491 | 0.8251 | $+4.93\%$ | 0.8315 | 111 (31.9%) | 21.26% |
| **Phase 6-B.1 (AB-BPL)** | **0.7684** / 0.8122 | $+0.0043$ | 56.3% | 0.7376 | 0.8458 | $+9.46\%$ | 0.8520 | 115 (33.0%) | **8.33%** |
| **Phase 6-C.1 (clDice)** | 0.7613 / 0.8002 | $-0.0028$ | 38.5% | 0.7210 | 0.8591 | $+11.82\%$ | **0.8525** | 113 (32.5%) | 8.91% |
| **Phase 6-D.1 (PLU+AB-BPL)** | 0.7669 / **0.8132** | $+0.0028$ | 52.6% | **0.7525** | 0.8238 | **$+5.07\%$** | 0.8309 | 112 (32.2%) | 20.69% |
| **Phase 6-D.2 (Pure Sep)** | 0.7589 / 0.8024 | $-0.0052$ | 40.5% | 0.7140 | **0.8591** | $+13.11\%$ | 0.8470 | **109 (31.3%)** | 14.37% |

### 4.2 Thin Cracks ($n=66$)
| Model | Thin Dice | $\Delta$ vs Base | Win vs Base | AreaExcess | Bridge Count (%) | Breakage (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate B (Base)** | 0.6404 | ref | --- | $+80.38\%$ | 44 (66.7%) | 12.12% |
| **Phase 6-A.1 (B-IoU)** | 0.6640 | $+0.0236$ | 74.2% | $+64.56\%$ | 42 (63.6%) | 12.12% |
| **Phase 6-A.2 (PLU)** | **0.6674** | **$+0.0270$** | 75.8% | **$+57.00\%$** | 41 (62.1%) | 30.30% |
| **Phase 6-B.1 (AB-BPL)** | 0.6554 | $+0.0150$ | 71.2% | $+73.02\%$ | 42 (63.6%) | **9.09%** |
| **Phase 6-C.1 (clDice)** | 0.6305 | $-0.0099$ | 24.2% | $+88.13\%$ | 43 (65.2%) | 12.12% |
| **Phase 6-D.1 (PLU+AB-BPL)** | 0.6655 | $+0.0251$ | **80.3%** | $+64.22\%$ | 43 (65.2%) | 21.21% |
| **Phase 6-D.2 (Pure Sep)** | 0.6300 | $-0.0104$ | 33.3% | $+84.97\%$ | **39 (59.1%)** | 16.67% |

---

## 5. Mechanistic Synthesis

1. **Separation vs Continuity Dilemma:**
   - Moat loss cục bộ tạo ra trường đẩy không gian trong khoảng cách $\le 8\text{ px}$ giữa các component.
   - Hiệu ứng này giảm được 5 false bridges ở nhóm thin cracks ($44 \to 39$), nhưng đồng thời gây ra đứt gãy trên các cấu trúc nứt thật chạy gần nhau (Breakage tăng $+4.89\text{ pp}$ toàn cục, và tăng lên $40\%$ ở thin-low-area).
2. **Ambiguity of Appearance / Texture:**
   - Kết quả phù hợp với giả thuyết rằng các vùng bridge có sự mơ hồ lớn về appearance/texture (vệt nhựa đường, khe tiếp giáp mặt đường), khiến geometric moat supervision thuần túy không đủ để phân biệt false bridge với crack thật gần kề.
3. **Checkpoint Provenance Verification (Epoch 17 vs 18):**
   - Checkpoint được đánh giá là Epoch 18 (`best_model_b2_global.pth`, Dice $0.758900$, Loss $0.960802$).
   - Sự lưu trữ này tuân theo đúng quy tắc tie-break trong `sage_lite/scripts/train_crack.py` (dòng 1238–1274): khi $|\Delta \text{Dice}| \le 10^{-4}$, mô hình có Validation Loss thấp hơn được chọn làm best checkpoint.

---

## 6. Synthesis across Phase 6 Probes (A1 through D2)

$$
\boxed{\text{Boundary tightening alone is insufficient (A1, B1, D1)}}
$$
$$
\boxed{\text{Topology preservation alone is insufficient (C1)}}
$$
$$
\boxed{\text{Pure spatial separation is insufficient (D2)}}
$$
$$
\boxed{\text{False Bridge is NOT reducible to a single geometric error mode}}
$$
