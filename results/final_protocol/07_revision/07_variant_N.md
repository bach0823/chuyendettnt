# Việc 3 — Đối chứng Không Nhánh Expert (Variant N)

**Trạng thái:** 🔄 **CẦN GPU — Chuẩn bị sẵn, chưa chạy**  
**Ngày chuẩn bị:** 2026-11-16  
**Notebook Colab:** [`phase3_variant_N_colab.ipynb`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/07_revision/phase3_variant_N_colab.ipynb)  
**Script đã được cập nhật vào repo:** Commit `110d652` (SAGE_LITE main)

---

## 1. Thiết kế Variant N

**Phương pháp (ít can thiệp nhất):**  
Sau khi tạo model canonical, duyệt qua tất cả `SageLayer` và set `residual_scale = 0.0`.

```python
# Fusion của SageLayer (residual type):
final_output = main_output + residual_scale * expert_output
# Với residual_scale = 0.0:
final_output = main_output  # expert branch zeroed
```

Kiến trúc giữ nguyên hoàn toàn, số tham số = canonical (**10,205,755**). Expert pool tồn tại trong RAM nhưng contribution = 0 ở forward pass. Tham số expert pool vẫn được cập nhật bởi optimizer (gradient từ expert path = 0 → không ảnh hưởng gì).

> [!NOTE]
> Gradient của expert pool = 0 vì contribution = 0. Optimizer cập nhật AdamW momentum nhưng bước cập nhật thực = 0 cho expert params (do gradient = 0). Effectively: expert pool không học được gì.

**So sánh tham số kiến trúc:**

| Variant | Tham số tổng | Expert Pool | Router | Nhánh Expert Contribution |
|---|:---:|:---:|:---:|:---:|
| **Canonical** | 10,205,755 | 6,624,480 | 88,776 | `0.1 × expert_output` |
| **Variant D** | 16,918,747 | Thay bằng DenseAdapter | — | `0.1 × adapter_output` |
| **Variant R** | 10,116,979 | 6,624,480 | 0 (Random) | `0.1 × random_expert_output` |
| **Variant N** | **10,205,755** | 6,624,480 | 88,776 | **0** (zeroed) |

---

## 2. Quy tắc Diễn giải Đặt Trước

| Kết quả Variant N | Diễn giải |
|---|:---|
| $\mu_N < \mu_{\text{canonical}} - \text{MDE}$ | Có bằng chứng nhánh expert (dung lượng) đóng góp. Kết hợp Phase 3 (D ≈ R ≈ canonical): "dung lượng giúp, router không quan trọng" |
| $|\mu_N - \mu_{\text{canonical}}| < \text{MDE}$ | Không phát hiện đóng góp của nhánh expert; toàn bộ hiệu năng đến từ phần còn lại (backbone lai, P3, S2-Gate, pipeline) |
| $\mu_N > \mu_{\text{canonical}} + \text{MDE}$ | Ghi rõ — kết quả phủ định đối với SAGE trong đề tài này |

**Cảnh báo về MDE:** MDE = 0.0108 là ước lượng điểm từ $n=3$ seed. Khoảng tin cậy 95% thực cho MDE là $[0.0056, 0.0679]$ — rất rộng. Kết luận từ Variant N cần kèm cảnh báo này.

---

## 3. Trạng thái Thực hiện

**ĐÃ CHUẨN BỊ:**
- Đã thêm `--variant N` vào `train_phase3_ablation.py` (commit `110d652`)
- Đã viết notebook Colab với 3-cell structure theo SKILL invariants
- Cờ diễn giải đặt trước đã được ghi nhận ở trên

**CẦN CHẠY TRÊN COLAB T4:**
- Seed 42: ~35 epoch × ~3 phút/epoch ≈ 1.75 giờ
- Seed 43: ~1.75 giờ
- Tổng: ~3.5 giờ

**Khi có kết quả:** Điền số liệu vào bảng dưới đây và cập nhật `FINAL_REPORT_REVISED.md`.

---

## 4. Bảng Kết quả (Chờ Điền)

| Seed | Stage 1 Val Dice | Stage 2 Val Dice | Global Best | Δ vs Canonical | Vượt MDE? |
|---|:---:|:---:|:---:|:---:|:---:|
| **N-42** | *[chờ]* | *[chờ]* | *[chờ]* | *[chờ]* | *[chờ]* |
| **N-43** | *[chờ]* | *[chờ]* | *[chờ]* | *[chờ]* | *[chờ]* |
| **N mean** | — | — | *[chờ]* | *[chờ]* | *[chờ]* |
| **Canonical mean** | — | — | 0.778638 | — | — |
| **MDE (nominal)** | — | — | — | 0.010804 | — |
