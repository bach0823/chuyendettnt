# Phase 6 Framework: Failure-Driven Investigation
**SAGE-Lite D4-K2-H64 | Canonical Base Locked @ Dice 0.7641 | 2026-09-30**

> [!IMPORTANT]
> **Config đã locked trước Phase 6 — không thay đổi:**
> D=4, K=2, H=64, LB=0.010, stage2_shared_lr ratio r=1.0, sage_lr=2e-4 (Stage 1).
> Phase 6 hoàn toàn về crack-structure failure, không phải SAGE routing tuning.

---

## 0. Câu hỏi trung tâm

> **Với Base SAGE-Lite hiện tại (~0.76 Dice), giới hạn chính nằm ở đâu: representation, crack-boundary/thin-crack recovery, topology, hay objective?**

---

## 1. Diagnostic Evidence — Xác minh từ source thực

*Tất cả số liệu dưới đây đọc trực tiếp từ `per_sample_metrics.csv` và `error_summary.json` của Candidate B best checkpoint (348 val samples).*

### 1.1 Overall Performance

| Metric | Mean | Median | Std |
|:---|:---:|:---:|:---:|
| Val Dice | 0.7641 | 0.8042 | 0.165 |
| IoU | 0.6417 | 0.6725 | 0.180 |
| Precision | **0.7337** | 0.7822 | 0.192 |
| Recall | **0.8477** | 0.8907 | 0.167 |
| Pred area / GT area | **+8.16%** | — | — |

**Pattern:** Model recall-biased — dự đoán rộng hơn GT ~8%. Mọi intervention nào tăng FN-penalty phải được kiểm tra cẩn thận để không làm FP tệ hơn.

### 1.2 Failure Taxonomy (primary_error_category)

| Category | Count | % | Mean Dice |
|:---|:---:|:---:|:---:|
| boundary_margin_error | **127** | **36.5%** | 0.781 |
| high_quality | 122 | 35.1% | 0.891 |
| moderate_general_error | 30 | 8.6% | — |
| complex_topology | 27 | 7.8% | 0.751 |
| false_crack_high_fp | 13 | 3.7% | — |
| missed_crack_high_fn | 12 | 3.4% | **0.187** |
| over_segmentation | 12 | 3.4% | — |
| thin_low_area_failure | 5 | 1.4% | **0.537** |

### 1.3 Morphology → Performance Correlation

| Predictor | Pearson r | Spearman ρ | Interpretation |
|:---|:---:|:---:|:---|
| thinness_score ↔ Dice | **-0.470** | **-0.618** | Crack càng mảnh → Dice càng thấp |
| gt_area ↔ Dice | **+0.420** | **+0.546** | Crack càng nhỏ → Dice càng thấp |

### 1.4 Phát hiện mới: `boundary_margin_error` KHÔNG phải thin-crack chain

**Đây là phát hiện quan trọng nhất để tránh suy luận sai trong Phase 6.**

| Category | Mean Thinness | thinness > 0.5 | Mean GT area | B/A Ratio |
|:---|:---:|:---:|:---:|:---:|
| high_quality | 0.090 | 0 | 18,577 | 0.181 |
| **boundary_margin_error** | **0.138** | **0** | **11,909** | **0.276** |
| thin_low_area_failure | 0.283 | 5/5 | 3,745 | 0.567 |
| complex_topology | 0.163 | 0 | 17,090 | 0.326 |
| missed_crack_high_fn | 0.150 | 0 | 4,915 | 0.300 |

> [!WARNING]
> **`boundary_margin_error` (n=127) có mean thinness = 0.138 — không có sample nào thinness > 0.5.** Đây là crack có diện tích TB và boundary rộng, không phải hairline crack. Do đó **boundary_margin_error và thin_low_area_failure là HAI failure chains khác nhau**, không phải một chain.

**Kết luận:** 
- `thin_low_area_failure` (n=5, mean thinness=0.283, mean Dice=0.537) = thin-crack failure chain → **volume nhỏ nhưng severity cao**.
- `boundary_margin_error` (n=127, mean thinness=0.138, mean Dice=0.781) = prediction rộng hơn GT ở crack medium-size → **volume lớn, moderate severity**.
- Hai chains này cần **hai can thiệp khác nhau** nếu muốn address riêng.

---

## 2. So sánh SAGE-Lite vs hướng SOTA Crack Segmentation

| Thành phần | SAGE-Lite hiện tại | SOTA crack-specific | Khoảng trống |
|:---|:---:|:---:|:---|
| CNN representation | ✅ ConvNeXtV2-Femto | ✅ | Nhỏ |
| Transformer context | ✅ ViT-Tiny (D=4) | ✅ | Nhỏ |
| Heterogeneous expert routing | ✅ SAGE | ❌ (rare) | Đây là điểm riêng |
| Multi-scale features (stages) | ✅ | ✅ | Ổn |
| High-res / thin-crack explicit branch | ❌ | ✅ phổ biến | **Đáng nghiên cứu** |
| Edge / high-frequency branch | ❌ | ✅ crack-specific | **Đáng nghiên cứu** |
| Boundary supervision/loss | ❌ | ✅ thường có | **Cần kiểm chứng** |
| Topology loss (clDice/skeleton) | ❌ | ✅ trong connectivity-critical | Evidence hiện tại yếu hơn |
| Crack-specific objective | BCE+Dice+LB hiện tại | Variants + auxiliary | Cần kiểm chứng |
| Error-stratified diagnostics | ✅ sâu | Rare | Điểm mạnh |
| Routing health | ✅ (entropy ~1.0) | N/A | Không phải bottleneck |

> [!NOTE]
> Routing đã được xác nhận không phải bottleneck: routing entropy = 0.999993 ± 0.000005 trên mọi sample → model không differentiate routing theo input. CNN shallow uniform affinity nhất quán với SAGE gốc (hình paper). Phase 6 không đụng routing.

---

## 3. Research Questions

### RQ6.1 — Failure Mechanism
> **Tại sao Candidate B fail trên thin/small cracks?**

Failure chain cần kiểm chứng:
- **Chain A (thin-crack):** thinness → thiếu high-frequency/edge representation → prediction miss hoặc break thin structures → Dice thấp.
- **Chain B (boundary):** crack medium-size → prediction rộng hơn GT → boundary_margin_error → precision thấp (đã có Precision=0.734 < Recall=0.848).

### RQ6.2 — Representation vs Objective
> **~0.76 ceiling do representation thiếu hay objective sai hướng?**

Kiểm tra:
- Nếu thêm loss mà không cải thiện → representation bottleneck.
- Nếu auxiliary objective cải thiện thin/boundary metrics mà không tăng FP → objective bottleneck.
- Nếu cả hai → cần cả hai loại can thiệp.

### RQ6.3 — Minimal SOTA-Inspired Intervention
> **Can thiệp tối thiểu nào phù hợp trực tiếp với failure SAGE-Lite đang có?**

Không phải: *"SOTA dùng gì?"*  
Mà là: *"SOTA dùng gì phù hợp với failure chain A hoặc B?"*

---

## 4. Decision Tree Can thiệp

```
BASE SAGE-LITE (Dice ~0.76, Precision < Recall)
│
├─[Q0] Evaluation protocol đúng canonical chưa?
│       └─ YES → tiếp tục
│
├─[Phase 6-A] THIN CRACK / HIGH-FREQUENCY [PRIORITY 1]
│   Evidence: thinness-Dice r=-0.618 (mạnh nhất)
│   Câu hỏi: Thin crack fail do thiếu high-freq detail hay thiếu objective pressure?
│   Candidate intervention:
│     - High-resolution skip / refinement branch
│     - Edge/boundary feature extraction
│     - Focal/Tversky với FN bias (nhưng phải watch FP)
│   Risk: model đã recall-biased → tăng FN pressure dễ tăng FP
│
├─[Phase 6-B] BOUNDARY MARGIN [PRIORITY 2]
│   Evidence: 127/348 samples (36.5%), mean Dice 0.781 (moderate)
│   Câu hỏi: Là boundary localization độc lập hay hậu quả của
│             model predict rộng hơn GT (+8.16%)?
│   Candidate intervention:
│     - Boundary supervision loss (auxiliary head)
│     - Boundary-aware feature
│   Risk: Nếu là hậu quả của recall bias → fixing root cause là đủ
│
├─[Phase 6-C] TOPOLOGY [PRIORITY 3 — evidence yếu]
│   Evidence: complex_topology 27/348 (7.8%), Dice 0.751
│             thin_low_area_failure 5/348 (1.4%)
│   Fragmentation ratio: mean=1.0 nhưng median của frag>0.5 = 238 samples
│   → fragmentation_ratio metric có thể đang count connected components
│     theo cách khác với "broken crack" topology
│   Candidate: clDice / skeleton loss
│   Risk: Không có evidence mạnh; thêm vội dễ làm FP tăng
│
└─[Phase 6-D] OBJECTIVE TUNING [SAU KHI biết failure chain]
    Không bắt đầu từ đây — phải biết failure mechanism trước
```

---

## 5. Thứ tự ưu tiên điều tra

| Pha | Focus | Ưu tiên | Lý do |
|:---|:---|:---:|:---|
| **6-A** | Thin crack / high-frequency | **1** | Correlation mạnh nhất (r=-0.618) |
| **6-B** | Boundary margin | **2** | Volume lớn nhất (36.5%) nhưng severity vừa |
| **6-C** | Topology | **3** | Evidence hiện tại chưa đủ mạnh |
| **6-D** | Loss/objective | **Sau 6-A/B** | Phải biết failure chain trước khi chọn loss |

> [!CAUTION]
> **Không nhảy sang 6-D (loss tuning) trước khi làm rõ 6-A và 6-B.** Nếu thêm loss mà không đúng failure mode → waste compute, có thể làm tệ hơn do model đã recall-biased.

---

## 6. Locked Config cho Phase 6

```
D=4 (ViT blocks)          ← locked Phase 1
K=2 (top_k)               ← locked Phase 2  
H=64 (router hidden)      ← locked Phase 3
LB=0.010                  ← locked Phase 4
sage_lr=2e-4 (Stage 1)    ← locked Phase 5.1
r=1.00 (shared LR ratio)  ← locked Phase 5.3
```

**Phase 6 chỉ được thêm/sửa:** auxiliary loss, refinement branch, decoder head, feature extraction module — **không đụng SAGE routing mechanism hoặc bất kỳ locked hyperparameter nào trên.**

---

## 7. Một câu chốt

> **"Base locked → identify the dominant crack-structure failure → introduce one targeted refinement mechanism → test whether the ~0.76 ceiling is caused by missing fine-structure representation or by the training objective."**

SAGE-Lite đã đầu tư mạnh vào **heterogeneous representation + dynamic routing** (và routing đã confirmed không phải bottleneck). Phase 6 vì thế có giá trị nhất nếu kiểm tra khoảng trống **crack-structure-specific refinement ở đầu ra/feature**, thay vì tiếp tục tối ưu SAGE routing.

---

*Dữ liệu nguồn:*
- `results/P3_C_Routing_Diagnostics_D4_K2_H64_Phase5_SAGELR2e-4/diagnostics/error_analysis/per_sample_metrics.csv` (348 samples)
- `results/P3_C_Routing_Diagnostics_D4_K2_H64_Phase5_SAGELR2e-4/diagnostics/error_analysis/error_summary.json`
- `results/p3_abc_epoch_by_epoch_metrics.json`
