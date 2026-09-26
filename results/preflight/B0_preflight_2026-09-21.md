# B0 Preflight — Colab T4 (2026-09-21)

**Notebook:** `notebooks/B0_Baseline_Colab_T4.ipynb`  
**Hardware:** Google Colab — NVIDIA Tesla T4 (14.56 GB VRAM)  
**Status:** ✅ PASS

---

## Model Info

| Field | Value |
|---|---|
| model_name | B0ConvNeXtViTUNet (Baseline Ladder B0) |
| total_parameters | 10,232,593 |
| trainable_parameters | 10,232,593 (100%) |
| num_classes | 1 |
| img_size | 448 |
| num_shared_experts | 4 |

---

## Shape Contract ✅

| Check | Shape | Expected | Status |
|---|---|---|---|
| Bottleneck 2D | (2, 384, 14, 14) | (2, 384, 14, 14) | ✅ |
| Tokens sequence | (2, 196, 192) | (2, 196, 192) | ✅ CLS discarded |
| Skip 0 | (2, 48, 112, 112) | — | ✅ |
| Skip 1 | (2, 96, 56, 56) | — | ✅ |
| Skip 2 | (2, 192, 28, 28) | — | ✅ |
| Output Logits | (2, 1, 448, 448) | (2, 1, 448, 448) | ✅ |

---

## AMP FP16 ✅

| Metric | Value |
|---|---|
| Total Loss | 1.7929 |
| BCE | 0.7245 |
| Dice | 0.7123 |
| GradScaler scale | 65536.0 |
| Model weights | Giữ nguyên khởi tạo (không step optimizer trước VRAM probe) |

---

## VRAM Probe ✅

Candidates: `[4, 8, 12, 16, 20]`  
Loss: `BCE + 1.5 * SoftDice`  
Strategy: Sandbox (fresh optimizer/scaler per candidate, gc.collect + empty_cache)

| Batch Size | Peak VRAM (MB) | Peak VRAM (GB) | Status |
|---|---|---|---|
| 4 | 798.9 MB | 0.78 GB | ✅ SUCCESS |
| 8 | 1498.5 MB | 1.46 GB | ✅ SUCCESS |
| 12 | 2194.4 MB | 2.14 GB | ✅ SUCCESS |
| 16 | 2886.0 MB | 2.82 GB | ✅ SUCCESS |
| 20 | 3570.0 MB | 3.49 GB | ✅ SUCCESS |

**Auto-select:** Batch Size = **20**  
`Peak 3.49 GB ≤ Safe threshold 13.56 GB (Total 14.56 GB − Margin 1.0 GB)`

---

## Smoke Test (5 Steps, BS=20) ✅

| Step | Loss | BCE | DiceLoss | Finite | Step |
|---|---|---|---|---|---|
| 1/5 | 1.8789 | 0.7277 | 0.7674 | ✅ | ✅ |
| 2/5 | 1.8727 | 0.7212 | 0.7677 | ✅ | ✅ |
| 3/5 | 1.8662 | 0.7146 | 0.7678 | ✅ | ✅ |
| 4/5 | 1.8597 | 0.7072 | 0.7683 | ✅ | ✅ |
| 5/5 | 1.8533 | 0.6993 | 0.7693 | ✅ | ✅ |

Loss giảm nhẹ monotone (1.8789 → 1.8533) với random data — Forward, Backward, Optimizer Step ổn định.  
**Không NaN/Inf. Không OOM.**

---

## Verdict

> ✅ **B0 Baseline sẵn sàng cho Full Training trên Colab T4 với Batch Size = 20.**
