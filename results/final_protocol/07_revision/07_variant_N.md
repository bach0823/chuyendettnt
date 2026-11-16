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

**ĐÃ HOÀN THÀNH (2026-11-16):**
- Đã thêm `--variant N` vào `train_phase3_ablation.py` (commit `110d652`)
- Đã chạy trên Colab T4: Seed 42 và Seed 43 — kết quả đầy đủ
- Thư mục kết quả: [`phase3_variantN_seed42/`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/phase3_variantN_seed42/) và [`phase3_variantN_seed43/`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/phase3_variantN_seed43/)

**Kiểm tra tính toàn vẹn:**
- N-42 `best_model_b2_global.pth` SHA-256: `161C9EF30C031602466E86E9412DF3E316BC63ECC16EF93DEB169BD3B9CC6FD7`
- N-43 `best_model_b2_global.pth` SHA-256: `756CC7A00378554A73D898ACEFAC2574F960C64B0D42957077ED5CBE4FC5CE9A`

---

## 4. Bảng Kết quả

| Seed | Stage 1 Val Dice | Stage 2 Val Dice | Global Best | Δ vs Canonical μ | Vượt MDE (0.010804)? |
|---|:---:|:---:|:---:|:---:|:---:|
| **N-42** | 0.758173 | 0.778212 | **0.778212** | −0.000425 | Không |
| **N-43** | 0.759447 | 0.777812 | **0.777812** | −0.000825 | Không |
| **N mean** | — | — | **0.778012** | **−0.000625** | **Không** |
| **Canonical mean** | — | — | 0.778638 | — | — |
| **MDE (nominal)** | — | — | — | 0.010804 | — |

> [!IMPORTANT]
> `n_zeroed_sagelayers = 8` — tất cả 8 SageLayer đã được đặt `residual_scale = 0.0`. Xác nhận bởi `run_metadata.json` cả hai seed.

---

## 5. Diễn giải Kết quả

**Áp dụng bảng diễn giải đặt trước (Mục 2):**

$|\Delta_{\mu_N}| = 0.000625 \ll \text{MDE} = 0.010804$ → **Không phát hiện đóng góp của nhánh expert pool.**

Kết hợp với Phase 3 (D ≈ R ≈ Canonical), chuỗi Variant D → R → N đều không phân biệt được với Canonical trong tầm MDE:

| Variant | Δ vs Canonical μ | Thành phần bị loại bỏ |
|---|:---:|---|
| **R** | −0.000138 | Router (giữ expert pool, router ngẫu nhiên) |
| **D** | −0.000757 | Router + Expert Pool (thay bằng DenseAdapter |
| **N** | −0.000625 | Expert Pool contribution (router giữ nhưng contribution=0) |

**Kết luận giới hạn theo bằng chứng:**

> Với 2 seed mỗi biến thể và MDE = 0.0108 điểm (CI 95%: [0.0056, 0.0679]), **không phát hiện chênh lệch** giữa Canonical và bất kỳ biến thể nào (D, R, N). Không thể kết luận SAGE router hay nhánh expert *hoàn toàn* vô ích — chỉ có thể nói: với thiết lập này, hiệu năng không phụ thuộc có thể phát hiện vào cơ chế routing hay expert contribution, trong tầm MDE của thí nghiệm.
