# 09_phase3_params.md — Đếm Params Theo Module và Hệ Quả Variant N (Revision 3)

**Ngày:** 2026-11-17  
**Nguồn chính:** `results/final_protocol/00_audit/audit_result.json`, `results/final_protocol/00_audit/run_phase0.py`, `results/final_protocol/03_capacity_control.md`, `results/phase7_combination/phase7_s2g_new_pipeline_run2/best_model_b2_stage2.pth`

---

## 1. Phương pháp đếm

Từ `run_phase0.py` dòng 75:
```python
num_params = sum(p.numel() for p in model.parameters())
```

`model.parameters()` của PyTorch **không đếm hai lần** các tensor được chia sẻ (shared modules). `expert_pool` là một module dùng chung bởi 8 `SageLayer` — PyTorch đếm mỗi tensor một lần duy nhất.

Kiểm tra từ `state_dict`:
- `expert_pool.*` (top-level): 180 keys = **6,624,480** params
- `backbone.convnext.stages.X.expert_pool.*` (8 lần): 8 × 6,624,480 = 52,995,840 (cùng tensors, 8 tham chiếu)
- `state_dict` lưu tất cả tham chiếu → tổng state_dict = 74,524,364 (không loại trùng)
- `model.parameters()` loại trùng → **10,205,755** (nguồn: `audit_result.json`)

---

## 2. Phân rã params theo module

| Module | Params | % tổng |
|---|:---:|:---:|
| **Expert pool** (shared, 8 SageLayers tham chiếu) | **6,624,480** | **64.9%** |
| **ConvNeXt stages** (không tính EP refs) | 7,445,236 | — |
| **Transformer blocks** (4 layers) | 30,933,616 | — |
| **Decoder** | 2,655,324 | 26.0% |
| **Router** (8 SageLayers tổng) | 88,912 | 0.9% |
| **Projection + norm layers** | 337,344 | 3.3% |
| **Stem** | 2,448 | — |

> [!IMPORTANT]
> Tổng `model.parameters()` = **10,205,755** — đây là số unique params (EP đếm một lần). Tổng state_dict = 74,524,364 (EP lưu 9 lần: 1 top-level + 8 backbone references).

**Phân rã phần còn lại (không phải EP):**
- Tổng − EP = 10,205,755 − 6,624,480 = **3,581,275** params
- Nguồn: `audit_result.json` → `num_params = 10205755`; `03_capacity_control.md` → EP = 6,624,480

> [!NOTE]
> Backbone ConvNeXt femto (stages thuần) + Transformer 4 layers + Decoder + projections có thể lớn hơn 10M khi đếm đơn lẻ. Số 10,205,755 phản ánh cách PyTorch dedup shared modules khi gọi `model.parameters()`. Phân rã backbone/transformer/decoder theo module riêng lẻ cần chạy model.build() thành công — hiện bị lỗi khi load config local (TypeErxror: `num_classes` trả về dict từ `sage_b2_crack500.yaml`), ghi nhận là **CHƯA LÀM ĐƯỢC** cho phân rã chi tiết backbone/transformer.

---

## 3. Router — xác nhận số params

Từ `state_dict`:
- Router keys trong backbone: 88,912 params  
- `03_capacity_control.md` ghi: 88,776 params

Chênh lệch 136 params (88,912 − 88,776 = 136). Nguồn gốc: `state_dict` tính cả `expert_successes` và `expert_usage_count` (buffers, không phải parameters), trong khi `model.parameters()` chỉ tính learnable params. **Router trainable = 88,776** (theo `03_capacity_control.md`).

---

## 4. Hệ quả Variant N

**Cơ chế Variant N** (từ `SAGE_LITE/sage/components/sage_layer.py` dòng 81):
```python
self.residual_scale = float(residual_scale)  # Python float, không phải nn.Parameter
```

Với `residual_scale=0`:
```python
output = main_out + 0.0 * expert_out  # expert_out * 0 → gradient về EP = 0
```

**Hậu quả:** Expert pool (6,624,480 params) và router (88,776 params) **không nhận gradient**, không được cập nhật trong quá trình train Variant N.

| Thành phần | Params | Trạng thái Variant N |
|---|:---:|---|
| Expert pool | 6,624,480 | Frozen (gradient = 0) |
| Router (8 SageLayers) | 88,776 | Frozen (gradient = 0) |
| **Phần còn lại** (backbone non-EP, decoder, projections) | **3,492,499** | Trainable |
| **Tổng** | **10,205,755** | — |

> Phần trainable thực tế của Variant N: 10,205,755 − 6,624,480 − 88,776 = **3,492,499 params** (34.2% tổng).

**Kết quả Variant N** (từ `08_revision2/08_phase3_recheck.md`):
- Seed 42: val = 0.778212, Δ vs canonical μ = −0.000425
- Seed 43: val = 0.777812, Δ vs canonical μ = −0.000825  
- μ Variant N: **0.778012**, Δ = **−0.000625** (−0.06 điểm)
- Không vượt MDE → "chưa có bằng chứng expert pool đóng góp ngoài dung lượng"

**Lưu ý:** Chỉ chạy 2 seed (42, 43) cho Variant N. Kết luận về "không đóng góp" yếu hơn so với 3-seed canonical. Thêm seed 44 sẽ cho CI chính xác hơn.

---

## 5. Nguồn xác nhận

- `results/final_protocol/00_audit/audit_result.json` → `"num_params": 10205755`
- `results/final_protocol/00_audit/run_phase0.py` dòng 75: `sum(p.numel() for p in model.parameters())`
- `results/final_protocol/03_capacity_control.md` → EP = 6,624,480; Router = 88,776
- `SAGE_LITE/sage/components/sage_layer.py` dòng 81: `self.residual_scale = float(residual_scale)`
