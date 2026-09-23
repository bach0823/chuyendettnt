# B2 SAGE-Lite Independent Cross-Check Audit Report

**Branch**: `crack500-audit`
**Date**: 2026-09-23
**Scope**: READ-ONLY verification — implementation vs. SAGE-Lite specification

---

## Executive Verdict: ✅ READY WITH DOCUMENTATION

The B2 implementation is a **faithful SAGE-Lite adaptation**, not merely a cosmetic copy. All critical SAGE mechanisms are present, and the key intentional deviations from Original SAGE are well-motivated and consistent with the user-approved design. Two documentation-level findings (noted below) should be addressed before full training.

---

## 1. Architecture Audit

### 1.1 ConvNeXtV2-Femto Backbone

| Requirement | Status | Evidence |
|---|---|---|
| Model: `convnextv2_femto.fcmae` | ✅ | `convnextv2_vit_hybrid.py` L59 |
| 4 stages | ✅ | L95-97: `assert self.encoder_channels == CONVNEXT_FEMTO_CHANNELS` |
| Channels `[48, 96, 192, 384]` | ✅ | L34: `CONVNEXT_FEMTO_CHANNELS: List[int] = [48, 96, 192, 384]` |
| `stem_channels=48` | ✅ | L88: profiled via dummy forward pass |

### 1.2 ViT-Tiny

| Requirement | Status | Evidence |
|---|---|---|
| Model: `vit_tiny_patch16_224` | ✅ | `convnextv2_vit_hybrid.py` L60 |
| `transformer_dim=192` | ✅ | L115: `self.transformer_dim = vit_full.embed_dim  # 192` |
| Configurable depth | ✅ | L61 constructor param; L126-128 sliced from full 12 blocks |
| Depth 12 = 12 blocks | ✅ | b2_unet passes `num_transformer_layers=12` from config |
| Depth 6 = 6 blocks | ✅ | Verified by verify_b2.py Test 8 |
| CLS token discarded | ✅ | L140-142: explicit CLS separation, only 196 patch tokens kept |

### 1.3 Interface Layers (Newly Initialized → decoder LR tier)

| Layer | Definition |
|---|---|
| `convnext_to_transformer` | `nn.Linear(384, 192)` — L177 |
| `transformer_to_decoder` | `nn.Linear(192, 384)` — L181 |
| `pre_transformer_norm` | `nn.LayerNorm(192)` — L178 |
| `post_transformer_norm` | `nn.LayerNorm(384)` — L182 |

### 1.4 Decoder

| Requirement | Status | Evidence |
|---|---|---|
| 3 decoder stages | ✅ | `decoder_block.py` L150: `range(len(reversed_channels) - 1)` = 3 |
| Progressive: 384→192→96→48 | ✅ | L147-149 |
| Skip connections | ✅ | L105: `torch.cat([x, skip], dim=1)` |
| Seg head 48→24→1 + 4x upsample | ✅ | L168-173 |

---

## 2. SAGE Injection Audit

### 2.1 Expert Pool Construction

| Requirement | Status | Evidence |
|---|---|---|
| All 4 CNN stages injected | ✅ | `sage_injection.py` L50-68 |
| All N_vit ViT blocks injected | ✅ | L71-89 |
| Depth 12 → 16 experts (4+12) | ✅ | L29: `total_experts_to_wrap = len(convnext.stages) + len(transformer_blocks)` |
| Depth 6 → 10 experts (4+6) | ✅ | Same formula |
| `expert_pool = ModuleList` of `.main_block` refs | ✅ | L92-99 |
| `expert_pool` linked to all SageLayers | ✅ | L103-108 |
| Single shared SAHub instance | ✅ | L43: `sa_hub = SAHub()` shared across all layers |

### 2.2 `my_index` Assignment

| Layer | Index | Evidence |
|---|---|---|
| CNN Stage i | i (0–3) | `sage_injection.py` L65: `my_index=stage_idx` |
| ViT Block j | 4+j | L86: `my_index=len(convnext.stages) + layer_idx` |

### 2.3 Self-Selection Bypass

`sage_layer.py` L237-238:
```python
if self.my_index is not None and expert_idx == self.my_index:
    adapted_expert_output = main_output[original_batch_indices]
```
✅ Zero-cost: reuses `main_output` directly, no redundant forward pass.

### 2.4 SA-Hub Pre-population

- O(D²) pairwise adapters created at init: `sage_injection.py` L135-148
- Active dimensions: {48, 96, 192, 384} (CNN) + {192} (ViT) — duplicates merged
- No dynamic adapter creation in forward: `sa_hub.py` L280-285 raises `RuntimeError` if adapter missing

---

## 3. Router Audit

### 3.1 SAGE Equations Implementation

| Paper Eq. | Implementation | Status |
|---|---|---|
| **Eq. 6** g_s = σ(linear(x)) | `router.py` L236: `g_s = torch.sigmoid(self.shared_expert_gate(...))` | ✅ |
| **Eq. 7** SAR = Q·K^T/√d + noise | L239-247: `matmul(query, keys.T) / temperature` + training noise | ✅ |
| **Eq. 8** Logit modulation | L251-263: `base_logits + mask·log(g_s) + (1-mask)·log(1-g_s)` in FP32 with eps=1e-5 | ✅ |
| **Eq. 9** Top-K | L268-272: `torch.topk(modulated_logits, self.top_k, dim=-1)` | ✅ |

### 3.2 DEFAULT_SAGE_CONFIG (`b2_unet.py` L42-53)

```python
DEFAULT_SAGE_CONFIG = {
    "top_k": 4,
    "gating_type": "sigmoid",
    "shared_expert_indices": [0, 1, 2, 3],  # 4 CNN stages = shared experts
    "router_hidden_dim": 64,
    "load_balance_factor": 0.01,
    "logit_modulation": True,
    "expert_dropout": 0.1,
    "fusion_type": "residual",
    "residual_scale": 0.1,
    "adaptive_alpha": 0.9,
}
```

### 3.3 Load Balance Loss — Verified Identical to SAGE Reference

Both SAGE-Lite and Original SAGE (`SAGE/sage/components/router.py` L306-335):
```python
P = expert_probs.mean(dim=0)   # avg routing prob per expert
f = P                           # soft approximation
load_balance_loss = expert_pool_size * (f * P).sum()
return self.load_balance_factor * load_balance_loss
```
✅ Character-for-character identical.

---

## 4. SageLayer Fusion Audit

### 4.1 Residual Fusion (Default)

`sage_layer.py` L135-136:
```python
final_output = main_output + self.expert_dropout(self.residual_scale * expert_output)
```
- `residual_scale` = **plain `float`**, NOT `nn.Parameter` — L78: `self.residual_scale = float(residual_scale)` ✅

### 4.2 Adaptive Fusion (Variant, matches SAGE paper Eq. 3)

`sage_layer.py` L132-134:
```python
alpha = torch.clamp(self.alpha, 0.1, 1.0)
final_output = alpha * main_output + (1.0 - alpha) * self.expert_dropout(expert_output)
```
- `self.alpha = nn.Parameter(torch.tensor(adaptive_alpha))` — L76 ✅
- Intentional improvement over Original SAGE: SAGE-Lite wraps expert_output with dropout in both fusion types

### 4.3 Mutual Exclusion

- Residual variant: NO `alpha` parameter in named_parameters ✅
- Adaptive variant: NO `residual_scale` attribute ✅
- Verified by verify_b2.py Test 8.5

---

## 5. Exception Handling & OOM Propagation

### Expert Path (`sage_layer.py` L274-281):
```python
except torch.cuda.OutOfMemoryError as e:
    self.logger.error("OOM encountered in expert path, propagating...")
    raise e                          # ← explicit re-raise
except Exception as e:
    ...
    return torch.zeros_like(main_output), routing_info   # ← recoverable fallback
```
✅ OOM placed BEFORE generic except. Verified by Test 10.

**Comparison**: Original SAGE has NO OOM re-raise — SAGE-Lite added this as an improvement.

---

## 6. Optimizer Groups & LR Tiers

`train_crack.py` L50-64:
```python
if 'router' in name or 'sa_hub' in name or 'alpha' in name:
    tier = 'sage'        # lr = base_lr (1e-4)
elif name.startswith('decoder') or any(k in name for k in interface_keys):
    tier = 'decoder'     # lr = base_lr (1e-4)
else:
    tier = 'backbone'    # lr = base_lr * 0.1 (1e-5)
```

| Tier | LR | WD (weights) | WD (norm/bias) |
|---|---|---|---|
| backbone | 1e-5 | 0.05 | 0.0 |
| decoder | 1e-4 | 0.05 | 0.0 |
| sage | 1e-4 | 0.05 | 0.0 |

6 physical groups (3 tiers × 2 decay modes). All 485 trainable params individually verified in Test 7.

---

## 7. Training Pipeline Audit

| Component | Implementation | Status |
|---|---|---|
| Loss | `seg_loss + 1.0 * lb_loss` | ✅ |
| Seg loss | `CrackBinaryLoss`: BCE(w=1.0) + Dice(w=1.5) | ✅ |
| AMP | `torch.amp.autocast` + `GradScaler` | ✅ |
| Optimizer | `AdamW` with 6 param groups | ✅ |
| Scheduler | `CosineLRScheduler`, lr_min=1e-6, warmup=3 | ✅ |
| Early stopping | patience=6, dice primary / loss tiebreaker | ✅ |
| GTX 1650 guard | `cudnn.enabled = False` | ✅ |
| Checkpoint | Saves `sage_config` in dict | ✅ |

---

## 8. Evaluation Pipeline Audit

| Component | Status |
|---|---|
| B2 model loading from checkpoint | ✅ `pretrained=False`, loads from `.pth` |
| Forward during eval uses `model(tensor)` not `forward_with_routing_info` | ✅ |
| Setting A (non-overlapping tiling 448×448) | ✅ |
| Setting B (50% overlap, stride=224, avg blending) | ✅ |
| Direct (DeepCrack full-image) | ✅ |

---

## 9. Config Audit (`b2_crack500_depth12.yaml`)

| Key | Value | Status |
|---|---|---|
| model | B2 | ✅ |
| num_transformer_layers | 12 | ✅ |
| batch_size | 20 | ⚠️ OOM on T4 — update after Phase 0 probe |
| lr | 1e-4 | ✅ |
| sage_config.top_k | 4 | ✅ |
| sage_config.gating_type | sigmoid | ✅ |
| sage_config.shared_expert_indices | [0,1,2,3] | ✅ |
| sage_config.router_hidden_dim | 64 | ✅ |
| sage_config.load_balance_factor | 0.01 | ✅ |
| sage_config.expert_dropout | 0.1 | ✅ |
| fusion_type | **absent** (defaults correctly) | ⚠️ doc |
| residual_scale | **absent** (defaults correctly) | ⚠️ doc |

---

## 10. Verification Coverage (`verify_b2.py`)

| Test | Scope | Status |
|---|---|---|
| 1 | Depth 12: 16 routers, 16 experts | ✅ Structural |
| 2 | Zero dynamic params (Lock #2) | ✅ Semantic |
| 3 | `forward()` pure Tensor (Lock #1) | ✅ Contract |
| 4 | `forward_with_routing_info()` + my_index | ✅ Semantic |
| 5 | LB loss finite, non-negative | ✅ Semantic |
| 6 | AMP FP16 backward, 100% core grad flow | ✅ Semantic |
| 7 | 485 params individually verified | ✅ Semantic |
| 8 | Depth 6 compatibility (10 routers) | ✅ Structural |
| 8.5 | Fusion type isolation | ✅ Semantic |
| 9 | Checkpoint round-trip, 0 missing keys | ✅ Integration |
| 10 | OOM propagation + recoverable fallback | ✅ Critical Safety |

---

## 11. Review Points Recheck (7 items)

| RP | Finding | Resolution |
|---|---|---|
| RP-1 | OOM propagation | ✅ Fixed — explicit re-raise before `except Exception` |
| RP-2 | LB loss `f=P` (soft approx) | ✅ Matches SAGE reference — by design |
| RP-3 | Interface layers LR | ✅ Decoder tier (1e-4) — user-approved |
| RP-4 | `expert_pool_size` consistency | ✅ Both derived from same formula |
| RP-5 | Parameter sharing via pool references | ✅ Implicit by reference identity |
| RP-6 | Logit modulation FP32 stability | ✅ FP32 cast + eps=1e-5 present |
| RP-7 | TupleSafeWrapper attribute delegation | ✅ `__getattr__` delegates to `self.module` |

---

## 12. Confirmed Issues

| # | Severity | Location | Description |
|---|---|---|---|
| 1 | LOW (Doc) | `b2_crack500_depth12.yaml` | `fusion_type` và `residual_scale` không khai báo tường minh. Defaults từ code là đúng. |
| 2 | LOW (Runtime) | `b2_crack500_depth12.yaml` | `batch_size: 20` OOM trên T4. Cần cập nhật sau Phase 0 probe. |

---

## 13. Requirement Matrix (25/25 ✅)

| # | Requirement | Status |
|---|---|---|
| 1 | ConvNeXtV2-Femto backbone, 4 stages [48,96,192,384] | ✅ |
| 2 | ViT-Tiny, configurable depth, CLS dropped | ✅ |
| 3 | SAGE injection into ALL 4 CNN stages | ✅ |
| 4 | SAGE injection into ALL N_vit ViT blocks | ✅ |
| 5 | Expert Pool = 4+N_vit experts | ✅ |
| 6 | my_index + zero-cost self-selection bypass | ✅ |
| 7 | Parameter sharing via expert pool references | ✅ |
| 8 | SA-Hub with O(D²) pre-populated adapters | ✅ |
| 9 | SageRouter: SAR + g_s + logit modulation + top-K | ✅ |
| 10 | LB loss = N * sum(f·P), f=P, factor=0.01 | ✅ |
| 11 | Residual fusion: main + dropout(scale * expert) | ✅ |
| 12 | Adaptive fusion: α·main + (1−α)·dropout(expert) | ✅ |
| 13 | residual_scale = fixed float, NOT nn.Parameter | ✅ |
| 14 | Fusion type configurable (residual / adaptive) | ✅ |
| 15 | OOM propagation (not swallowed) | ✅ |
| 16 | forward() returns pure Tensor (Lock #1) | ✅ |
| 17 | Zero dynamic params in forward (Lock #2) | ✅ |
| 18 | 3-tier optimizer (backbone/decoder/sage) | ✅ |
| 19 | WD=0 for norms/biases | ✅ |
| 20 | AMP FP16 + GradScaler | ✅ |
| 21 | CrackBinaryLoss (BCE + Dice) | ✅ |
| 22 | Evaluation: Setting A, B, Direct | ✅ |
| 23 | Checkpoint round-trip integrity | ✅ |
| 24 | GTX 1650 cuDNN safeguard | ✅ |
| 25 | UNet Decoder 3 stages + seg head | ✅ |

---

## Final Recommendation

**Q1: B2 hiện tại có thực sự là SAGE-Lite theo thiết kế?**
Có. Tất cả cơ chế SAGE đều khớp specification. Các sai lệch so với Original SAGE là intentional adaptations.

**Q2: Có finding nào cần fix trước khi training?**
Không có blocking issue. 2 documentation findings có workaround rõ ràng.

**Q3: Verification suite có đủ coverage?**
Có. 10 tests bao phủ structural, semantic, contract, integration, và safety.

**Q4: Sẵn sàng cho Phase 0 OOM Probe?**
Có, sau khi xác định batch_size feasible qua probe B18→B16→B14→B12.
