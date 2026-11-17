# 10_param_structure.md — Cấu Trúc Tham Số Thực Tế và Phân Tích Cơ Chế Variant N (Revision 4)

**Ngày:** 2026-11-17  
**Mục tiêu:** Xác minh quan hệ chia sẻ trọng số giữa `expert_pool` và `main_block`, phân loại tham số theo quyền sở hữu độc quyền (không trùng lặp), và đo lường gradient thực tế sau 1 batch trên Variant N.  
**Căn cứ mã nguồn:** `sage/networks/sage_injection.py` (dòng 102–135), `sage/components/sage_layer.py` (dòng 60–200), `configs/p3_ablation/phase7/phase7_b2_s2g_b1_b2_end_to_end.yaml`.  
**Script thực nghiệm:** `results/final_protocol/10_revision4/scratch/verify_param_structure.py`.

---

## 1. Bằng chứng Mã nguồn và Định danh Tham số (Memory ID Matching)

Trong tệp `sage/networks/sage_injection.py` (dòng 102–120):
```python
expert_pool = nn.ModuleList()
for stage_idx, stage_wrapper in enumerate(convnext.stages):
    expert_mod = stage_wrapper.main_block
    expert_pool.append(expert_mod)

for block_idx, block_wrapper in enumerate(transformer_blocks):
    expert_mod = block_wrapper.main_block
    expert_pool.append(expert_mod)
```

`expert_pool` **không sở hữu bất kỳ tham số độc lập nào**. Danh sách `expert_pool` chỉ chứa các tham chiếu trỏ trực tiếp đến `main_block` của các tầng ConvNeXt và các khối Transformer ViT.

### Kết quả kiểm tra bằng Python (`id(p)` matching):
Kiểm tra từng tham số trong 8 phần tử của `expert_pool` đối chiếu với `main_block` tương ứng:
- **Expert 0** (`convnext_stage_0`, 20 tensors): `id(p)` trùng khớp 100% với `convnext.stages[0].main_block`.
- **Expert 1** (`convnext_stage_1`, 24 tensors): `id(p)` trùng khớp 100% với `convnext.stages[1].main_block`.
- **Expert 2** (`convnext_stage_2`, 64 tensors): `id(p)` trùng khớp 100% với `convnext.stages[2].main_block`.
- **Expert 3** (`convnext_stage_3`, 24 tensors): `id(p)` trùng khớp 100% với `convnext.stages[3].main_block`.
- **Expert 4** (`transformer_block_0`, 12 tensors): `id(p)` trùng khớp 100% với `transformer_blocks[0].main_block`.
- **Expert 5** (`transformer_block_1`, 12 tensors): `id(p)` trùng khớp 100% với `transformer_blocks[1].main_block`.
- **Expert 6** (`transformer_block_2`, 12 tensors): `id(p)` trùng khớp 100% với `transformer_blocks[2].main_block`.
- **Expert 7** (`transformer_block_3`, 12 tensors): `id(p)` trùng khớp 100% với `transformer_blocks[3].main_block`.

> [!IMPORTANT]
> **Toàn bộ 8 expert trong `expert_pool` có địa chỉ vùng nhớ trùng 100% với các khối của nhánh chính.** Không có việc tồn tại một "kho expert pool độc lập 6.62M tham số" bị đóng băng như các phiên bản báo cáo trước đã nhầm lẫn.

---

## 2. Phân loại Tham số theo Sở hữu Độc quyền (Không Trùng lặp)

Tổng số tham số duy nhất của mô hình (`sum(p.numel() for p in model.parameters())`) là **10,205,755**.

Phân chia theo 3 nhóm sở hữu độc quyền:

| Phân nhóm thành phần | Các module cấu thành | Số lượng tham số | Tỉ lệ (%) |
|---|---|:---:|:---:|
| **(a) Backbone Pretrain** | ConvNeXt Stem (2,448) + 4 ConvNeXt Stages `main_block` (4,845,024) + 4 ViT Blocks `main_block` (1,779,456) | **6,626,928** | **64.93%** |
| **(b) Thành phần riêng của SAGE** | 8 Router SAGE (88,776) + Hệ thống chuyển đổi định dạng tensor SA-Hubs / Format Adapters (649,440) | **738,216** | **7.23%** |
| **(c) Decoder, Head & Projections** | UNet Decoder (2,653,795) + ViT to Decoder proj (74,112) + ConvNeXt to ViT proj (73,920) + PE (37,632) + LayerNorms (1,152) | **2,840,611** | **27.83%** |
| **TỔNG CỘNG** | **Đầy đủ toàn bộ mô hình (không đếm trùng)** | **10,205,755** | **100.00%** |

*(Ghi chú: P3 refinement được truyền là `None` trong mã nguồn hiện hành tại `sage_injection.py:70, 97`, chiếm 0 tham số).*

---

## 3. Kiểm chứng Gradient Norm sau 1 Batch: Canonical vs Variant N

Trong `sage/components/sage_layer.py`:
$$\text{final\_output} = \text{main\_output} + \text{self.residual\_scale} \times \text{expert\_output}$$
- **Canonical:** `residual_scale = 0.1`
- **Variant N:** `residual_scale = float(0.0)`

Đo lường Gradient Norm sau 1 lượt lan truyền ngược (BCE loss trên batch thực tế):

| Nhóm tham số | Canonical (`scale = 0.1`) | Variant N (`scale = 0.0`) | Trạng thái ở Variant N |
|---|:---:|:---:|---|
| **(a) Backbone Pretrain** (6,626,928 params) | Grad Norm = **0.020769** (0 grad = 0) | Grad Norm = **0.024597** (0 grad = 0) | **Vẫn nhận gradient bình thường qua nhánh chính** |
| **(b) Thành phần riêng SAGE** (738,216 params) | Grad Norm = **0.000144** | Grad Norm = **0.000000** (738,216/738,216 grad = 0) | **Bị đóng băng 100% (gradient = 0 tuyệt đối)** |
| **(c) Decoder + Head** (2,840,611 params) | Grad Norm = **0.045406** | Grad Norm = **0.068322** | **Vẫn nhận gradient và tối ưu hóa bình thường** |

> [!CAUTION]
> **Đính chính sai lệch của Revision 3:**  
> Revision 3 từng nhận định rằng *"Variant N làm đóng băng 65.78% tham số (6.71M params)"* và *"mô hình chỉ tối ưu trên 3.49M params"*.  
> **Thực tế:** Variant N chỉ đóng băng duy nhất **738,216 tham số (7.23%)** thuộc về Router và SA-Hub. Toàn bộ Backbone (6.63M) và Decoder/Head (2.84M) — tức **9,467,539 tham số (92.77%)** — **vẫn được huấn luyện và cập nhật trọng số liên tục qua nhánh chính**!

---

## 4. Diễn giải lại Ý nghĩa của Thử nghiệm Phase 3

1. **Bản chất của Variant N:**  
   Variant N tắt bỏ hoàn toàn nhánh phụ SAGE (cắt đứt đường truyền qua Router và SA-Hub), chuyển toàn bộ luồng tính toán về dạng Backbone thuần túy cấp thẳng sang Decoder. Do đó, Variant N chính là cấu hình **"B2 không SAGE" (B2 without SAGE)** trong cùng một pipeline huấn luyện mới.
2. **Hiệu năng của "B2 không SAGE":**  
   - Canonical 2-seed trung bình: **0.781670** (Seed 42: 0.780409, Seed 43: 0.782932)
   - Variant N 2-seed trung bình: **0.778012** (Seed 42: 0.778212, Seed 43: 0.777812)
   - Độ chênh lệch: **−0.003658 (−0.37 điểm)** so với canonical 2-seed (hoặc **−0.06 điểm** so với canonical 3-seed $\mu = 0.778638$).
3. **Kết luận khoa học:**  
   Khi loại bỏ hoàn toàn 738K tham số của cơ chế định tuyến và điều phối expert liên tầng, hiệu năng mô hình trên tập Val vẫn nằm trọn trong khoảng chênh lệch nhỏ (~0.06 – 0.37 điểm), cách xa ngưỡng phát hiện MDE (1.08 điểm). Điều này chứng minh rằng hiệu quả của B2 chủ yếu đến từ sức mạnh kết hợp của Backbone ConvNeXt + Transformer Blocks và kiến trúc Decoder S2-Gate, chứ không đến từ cơ chế định tuyến chuyên biệt hóa của SAGE.
