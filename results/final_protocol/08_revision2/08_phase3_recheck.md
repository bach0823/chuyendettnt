# 08_phase3_recheck.md — Kiểm tra lại Phase 3 (Revision 2)

**Nguồn:** `results/final_protocol/02_phase2_summary.json`, `03_capacity_control.md`, `phase3_variantN_seed42/run_metadata.json`, `phase3_variantN_seed43/run_metadata.json`, `SAGE_LITE/sage/components/sage_layer.py`  
**Ngày viết:** 2026-11-16

---

## 1. Baseline khớp seed cho Phase 3

**Vấn đề:** Phase 3 Variant D và R chỉ dùng seed 42 và 43, không dùng seed 44. Baseline Phase 2 (3 seed) có thể không khớp.

**Seed 42 và 43 (Phase 2):**
- Nguồn: `results/final_protocol/02_phase2_summary.json`
  - seed42: val_dice = 0.7804085131466723
  - seed43: val_dice = 0.7829318620367279
  - seed44: val_dice = 0.7725724757772799

**Baseline khớp seed (seed 42 + 43 only):**
- Mean = (0.7804085131466723 + 0.7829318620367279) / 2 = **0.7816701875917001**

| Baseline | Val Dice μ | Áp dụng cho |
|---|:---:|---|
| Phase 2 μ (3 seed: 42/43/44) | 0.778638 | Tham chiếu chính thức |
| Phase 2 μ (2 seed: 42/43 khớp với Variant D/R/N) | **0.781670** | So sánh khớp seed |

---

## 2. So sánh Δ vs cả hai baseline

| Cấu hình | Seed | Best Val Dice | Δ vs μ 3-seed | Vượt MDE? | Δ vs μ 2-seed (42/43) | Vượt MDE? |
|---|---|:---:|:---:|---|:---:|---|
| Canonical (Phase 2) | 42 | 0.780409 | +0.001771 | — | — | — |
| Canonical (Phase 2) | 43 | 0.782932 | +0.004294 | — | — | — |
| Canonical (Phase 2) | 44 | 0.772572 | −0.006066 | — | — | — |
| **Canonical μ (3 seed)** | — | **0.778638** | — | — | — | — |
| **Canonical μ (2 seed: 42/43)** | — | **0.781670** | — | — | — | — |
| Variant D | 42 | 0.773668 | −0.004970 | Không | −0.008002 | Không |
| Variant D | 43 | 0.782093 | +0.003455 | Không | +0.000423 | Không |
| **Variant D μ** | — | **0.777881** | **−0.000757** | **Không** | **−0.003789** | **Không** |
| Variant R | 42 | 0.779086 | +0.000448 | Không | −0.002584 | Không |
| Variant R | 43 | 0.777913 | −0.000725 | Không | −0.003757 | Không |
| **Variant R μ** | — | **0.778500** | **−0.000138** | **Không** | **−0.003170** | **Không** |
| Variant N | 42 | 0.778212 | −0.000425 | Không | −0.003458 | Không |
| Variant N | 43 | 0.777812 | −0.000825 | Không | −0.003858 | Không |
| **Variant N μ** | — | **0.778012** | **−0.000625** | **Không** | **−0.003658** | **Không** |

MDE = 0.010804 (nguồn: `02_phase2_summary.json`).  
Mọi |Δ| đều < MDE, với cả hai baseline. Kết luận về "không phát hiện" không thay đổi.

---

## 3. Sửa "rộng hơn 12×" — MDE CI

Từ `07a_cluster_bootstrap_results.json` → `mde_analysis`:
- MDE nominal = 0.010804
- CI 95% lo = 0.005625, hi = 0.067924
- Tỉ lệ cận trên / cận dưới = 0.067924 / 0.005625 = **12.08×**

**Cách diễn đạt sửa:** "MDE CI 95% = [0.0056, 0.0679]: cận trên gấp khoảng 12 lần cận dưới (12.1×), do n=3 seed."  
Không phải "CI rộng hơn 12× so với ước lượng điểm 0.0108".

---

## 4. Gradient check Variant N — residual_scale là Python float

**Nguồn:** `SAGE_LITE/sage/components/sage_layer.py`, dòng 81:
```python
elif self.fusion_type == "residual":
    self.residual_scale = float(residual_scale)
```

`self.residual_scale` là **Python `float`**, không phải `nn.Parameter`. Không tạo gradient.

**Hệ quả khi residual_scale=0:**  
Forward pass của `SageLayer` với `fusion_type="residual"`:
```
output = main_out + residual_scale * expert_out
       = main_out + 0.0 * expert_out
       = main_out
```
Expert pool output bị nhân 0 → gradient ngược đến expert params = 0 → **expert pool params không được cập nhật** trong quá trình train.

**Xác nhận:** `run_metadata.json` Variant N seed42/43:
- `method: "residual_scale=0 on all SageLayers after model creation"`
- `n_zeroed_sagelayers: 8`
- `total_params_after_injection: 10,205,755` (bằng canonical vì chỉ thay đổi runtime scalar, không đổi params)

**Kết quả:** Variant N huấn luyện với expert pool frozen (gradient=0). Không phải "expert pool không tồn tại" mà là "expert pool không được cập nhật và không đóng góp vào output".

---

## 5. Tổng số tham số

Từ `run_metadata.json` Variant N: `total_params_after_injection = 10,205,755`  
Bằng canonical (xác nhận từ `PROGRESS.md`: "Params: 10,205,755 (10.206M)").

---

## 6. Hành vi canonical router chưa được đo đối chiếu

Routing diagnostics có cho Variant R (seed42: phân phối 8 expert gần đều 11.66%–13.04%, kỳ vọng 12.50%). Canonical router chưa có diagnostics tương tự để so sánh phân phối. Hạn chế này cần nêu trong báo cáo.
