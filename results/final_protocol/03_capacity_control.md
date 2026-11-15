# Phase 3 — Đối chứng dung lượng (Capacity Control: Variant D & Variant R)

**Ngày hoàn thành:** 2026-11-15  
**Baseline tham chiếu:** Phase 2 Multi-seed Canonical ($\mu = 0.778638$, $s = 0.005402$, $\text{MDE} = 0.010804$)  
**Quy tắc diễn giải đặt trước:**
- $|\Delta| < \text{MDE}$: *"Chưa có bằng chứng router/expert đóng góp ngoài dung lượng"*.
- $\text{Canonical} > D \text{ và } R \text{ vượt MDE}$: *"Ủng hộ cơ chế SAGE"*.
- $D > \text{Canonical vượt MDE}$: *"Dense adapter hiệu quả hơn routing"*.

---

## 1. Thiết kế thực nghiệm và Khớp tham số

### Biến thể D (Dense Capacity Control)
- **Mục tiêu:** Thay thế toàn bộ expert pool dùng chung và 8 router trong các `SageLayer` bằng khối dày (`DenseAdapter` 2-layer MLP bottleneck) có số tham số tương đương trên từng layer.
- **Tham số gốc cần thay thế:**
  - `expert_pool` (shared, đếm 1 lần): $6,624,480$ params.
  - `router` (8 SageLayers tổng cộng): $88,776$ params.
  - **Tổng mục tiêu:** **$6,713,256$ params**.
- **Khớp kiến trúc DenseAdapter:**
  - Mỗi `SageLayer` áp dụng: $\text{out} = \text{main\_out} + 0.1 \times \text{Linear}(H \to C)(\text{GELU}(\text{Linear}(C \to H)(\text{main\_out})))$
  - Stage 0 ($C=48, H=8651$): $839,195$ params.
  - Stage 1 ($C=96, H=4347$): $839,067$ params.
  - Stage 2 ($C=192, H=2179$): $839,107$ params.
  - Stage 3 ($C=384, H=1091$): $839,363$ params.
  - Transformer 0–3 ($C=192, H=2179 \times 4$): $3,356,428$ params.
  - **Tổng tham số thực tế:** **$6,713,160$ params** (sai lệch: $-96$ params, tức $-0.0014\%$).

### Biến thể R (Random Routing Control)
- **Mục tiêu:** Giữ nguyên toàn bộ `expert_pool` và kiến trúc SAGE, nhưng thay thế `SageRouter` có học bằng `RandomFixedRouter`.
- **Cơ chế:** Chọn $k=2$ expert ngẫu nhiên đều cho mỗi mẫu dữ liệu theo seed cố định, trọng số gán đều $0.5$, không có tham số huấn luyện (0 trainable parameters).

---

## 2. Bảng số liệu thực nghiệm

| Cấu hình | Seed | Stage 1 Val Dice | Stage 2 Val Dice | Global Best Val Dice | $\Delta$ vs Phase 2 $\mu$ | Vượt MDE ($0.010804$)? |
|---|---|---|---|---|---|---|
| **Canonical (Phase 2)** | 42 | — | — | 0.780409 | $+0.001771$ | — |
| **Canonical (Phase 2)** | 43 | — | — | 0.782932 | $+0.004294$ | — |
| **Canonical (Phase 2)** | 44 | — | — | 0.772572 | $-0.006066$ | — |
| **Canonical Trung bình** | — | — | — | **0.778638** | Baseline | — |
| **Biến thể D** | 42 | 0.757519 | 0.773668 | 0.773668 | $-0.004970$ | Không ($|\Delta| < \text{MDE}$) |
| **Biến thể D** | 43 | 0.759103 | 0.782093 | 0.782093 | $+0.003456$ | Không ($|\Delta| < \text{MDE}$) |
| **Biến thể D Trung bình** | — | — | — | **0.777881** | **$-0.000757$** | **Không** |
| **Biến thể R** | 42 | 0.759251 | 0.779086 | 0.779086 | $+0.000448$ | Không ($|\Delta| < \text{MDE}$) |
| **Biến thể R** | 43 | 0.756005 | 0.777913 | 0.777913 | $-0.000725$ | Không ($|\Delta| < \text{MDE}$) |
| **Biến thể R Trung bình** | — | — | — | **0.778500** | **$-0.000138$** | **Không** |

---

## 3. Kiểm định Giả thuyết và Diễn giải Đặt trước

1. **So sánh Biến thể D vs Canonical:**
   - $\Delta_{\mu} = 0.777881 - 0.778638 = -0.000757$.
   - $|\Delta| = 0.000757 \ll \text{MDE} = 0.010804$.
   - Thay thế toàn bộ expert pool + router bằng dense adapter với cùng số lượng tham số cho kết quả tương đương hoàn toàn (chênh lệch dưới $0.08\%$ Dice, nằm trọn trong nhiễu seed).

2. **So sánh Biến thể R vs Canonical:**
   - $\Delta_{\mu} = 0.778500 - 0.778638 = -0.000138$.
   - $|\Delta| = 0.000138 \ll \text{MDE} = 0.010804$.
   - Thay router động SAGE bằng router chọn ngẫu nhiên đều (Random Top-2) cho kết quả tương đương hoàn toàn (chênh lệch dưới $0.014\%$ Dice).

3. **Kết luận khoa học chính thức:**
   > **Theo đúng tiêu chí định trước: $|\Delta| < \text{MDE}$. Kết luận: CHƯA CÓ BẰNG CHỨNG ROUTER/EXPERT ĐÓNG GÓP NGOÀI DUNG LƯỢNG (CAPACITY).**
   > Hiệu năng của mô hình đến từ tổng dung lượng tham số và kiến trúc backbone/decoder, không đến từ cơ chế định tuyến chuyên hóa hay phân hóa cục bộ của SAGE router.

---

## 4. Chẩn đoán Định tuyến Biến thể R (Routing Diagnostics Audit)

Chạy trên 348 mẫu tập Val (`tools/analyze_routing.py`), tổng số lượt chọn: $5,568$ lượt ($348 \times 8 \times 2$):

| Expert ID | Tên Expert | Loại | Lượt chọn (Seed 42) | Tỷ lệ thực tế | Tỷ lệ kỳ vọng đều | Độ lệch so với kỳ vọng |
|---|---|---|---|---|---|---|
| E0 | CNN_Stage_0 | CNN | 712 | 12.79% | 12.50% | $+0.29\%$ |
| E1 | CNN_Stage_1 | CNN | 649 | 11.66% | 12.50% | $-0.84\%$ |
| E2 | CNN_Stage_2 | CNN | 722 | 12.97% | 12.50% | $+0.47\%$ |
| E3 | CNN_Stage_3 | CNN | 701 | 12.59% | 12.50% | $+0.09\%$ |
| E4 | ViT_Block_0 | ViT | 726 | 13.04% | 12.50% | $+0.54\%$ |
| E5 | ViT_Block_1 | ViT | 667 | 11.98% | 12.50% | $-0.52\%$ |
| E6 | ViT_Block_2 | ViT | 689 | 12.37% | 12.50% | $-0.13\%$ |
| E7 | ViT_Block_3 | ViT | 702 | 12.61% | 12.50% | $+0.11\%$ |

*Nhận xét:* Tần suất kích hoạt của cả 8 expert phân bố đều tuyệt đối xung quanh mức $12.5\%$ (dao động trong khoảng $11.66\% - 13.04\%$), xác nhận cơ chế router ngẫu nhiên hoạt động hoàn hảo và không tạo ra bất kỳ độ chệch nào.

---

## 5. Danh mục Checkpoint & Tính toàn vẹn (Integrity Audit)

| Run | Tập tin | SHA-256 Checksum |
|---|---|---|
| `phase3_variantD_seed42` | `best_model_b2_global.pth` | `38383C9EBF8E5BAFF3553BEA1F7524D61E109214446184C02A2CB58A30C71C3B` |
| `phase3_variantD_seed42` | `best_model_b2_stage1.pth` | `8C082DC6080AB214988FA7DC44F6804FF70A3629F0BF135A09C5A8C3B35A1250` |
| `phase3_variantD_seed43` | `best_model_b2_global.pth` | `A24A905D709AA8B44ECA25842942230B432A366F42A37AA6CC1F1D2FCDBBF68B` |
| `phase3_variantD_seed43` | `best_model_b2_stage1.pth` | `71A2C2E545639A5FCE4A3E186008272D15806020A25751B8E6D11E903B35494D` |
| `phase3_variantR_seed42` | `best_model_b2_global.pth` | `F3043B6EA57804648FD49CA00E10914A66F794E2AFBC4B25CC321BFB8C5BC76C` |
| `phase3_variantR_seed42` | `best_model_b2_stage1.pth` | `75C727B70BD8DFB14CEDD3C14753280A7A114AD9D02BEC3B5914879F832F5298` |
| `phase3_variantR_seed43` | `best_model_b2_global.pth` | `CB9CF335DF663D8EDB24CDF49B9A0FE1F214A95E64CD920DA66C27B2266D311F` |
| `phase3_variantR_seed43` | `best_model_b2_stage1.pth` | `E8D20A1AF3F15D732F6C7132C2BEE178C6A4DAD137D293FC69D91A01183AA789` |
