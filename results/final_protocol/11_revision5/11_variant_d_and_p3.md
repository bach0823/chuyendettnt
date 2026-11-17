# 11_variant_d_and_p3.md — Cấu Trúc Tham Số Variant D và Xác Minh Toàn Diện P3 Refinement (Revision 5)

**Ngày lập:** 2026-11-17  
**Nguồn dữ liệu:**
- Mã nguồn Variant D: `sage_lite/scripts/train_phase3_ablation.py` (dòng 49–185).
- Tệp metadata Variant D: `results/final_protocol/phase3_variantD_seed42/run_metadata.json` và `phase3_variantD_seed43/run_metadata.json`.
- Mã nguồn khởi tạo SAGE: `sage_lite/sage/networks/sage_injection.py` (dòng 20, 70, 97).
- Mã nguồn tầng SAGE: `sage_lite/sage/components/sage_layer.py` (dòng 55, 97).
- Cấu hình huấn luyện: `configs/p3_ablation/phase7/phase7_b2_s2g_b1_b2_end_to_end.yaml`.

---

## 1. Cấu Trúc Tham Số Thật Của Variant D (DenseAdapter)

### A. Cơ chế khởi tạo và số lượng tham số thực tế
Trong kịch bản `sage_lite/scripts/train_phase3_ablation.py`:
- Tác giả ban đầu giả định rằng toàn bộ cụm router + expert pool chiếm 6,713,256 tham số, nên đã đặt:
  `_TOTAL_DENSE_TARGET = 6_713_256`
- Lớp `DenseAdapter` được thiết kế dưới dạng MLP nút thắt cổ chai 2 lớp (`Linear(C, H) -> GELU -> Linear(H, C)`), với số chiều ẩn $H$ tương ứng cho từng kênh đầu ra $C$:
  - ConvNeXt Stage 0 ($C=48, H=8651$): $(48 \times 8651 + 8651) + (8651 \times 48 + 48) = 839,195$ tham số.
  - ConvNeXt Stage 1 ($C=96, H=4347$): $(96 \times 4347 + 4347) + (4347 \times 96 + 96) = 839,067$ tham số.
  - ConvNeXt Stage 2 ($C=192, H=2179$): $(192 \times 2179 + 2179) + (2179 \times 192 + 192) = 839,107$ tham số.
  - ConvNeXt Stage 3 ($C=384, H=1091$): $(384 \times 1091 + 1091) + (1091 \times 384 + 384) = 839,363$ tham số.
  - 4 khối Transformer ($C=192, H=2179$): $4 \times 839,107 = 3,356,428$ tham số.
  - **Tổng tham số DenseAdapter thêm mới:** $\mathbf{6,713,160}$ tham số.
- Khi hàm `_inject_variant_D(model)` được gọi:
  Mô hình gốc Canonical B2 (10,205,755 tham số) **không bị xóa bỏ bất kỳ tầng nào**. Bộ thích ứng `model.phase3_dense_adapters` gồm 8 mô-đun DenseAdapter được đăng ký thêm vào danh sách tham số huấn luyện của mô hình.
- Trong `run_metadata.json` của cả seed 42 và seed 43 ghi nhận rõ:
  - `"dense_adapter_params": 6713160`
  - `"total_params_after_injection": 16918915`

### B. Bảng đối chiếu tham số Canonical vs Variant D

| Thành phần | Canonical B2 | Variant D | Khác biệt |
|---|:---:|:---:|:---:|
| **(a) Backbone pretrain** (ConvNeXt + ViT) | 6,626,928 | 6,626,928 | 0 |
| **(b) Thành phần riêng của SAGE** (Routers + Adapters) | 738,216 | 738,216 | 0 (Vẫn tồn tại trong mô hình nhưng bị bypass ở forward) |
| **(c) Decoder + Head + S2-Gates + Projections** | 2,840,611 | 2,840,611 | 0 |
| **(d) Nhánh DenseAdapter thêm mới** | 0 | **6,713,160** | **+6,713,160** |
| **TỔNG THAM SỐ CỦA MÔ HÌNH** | **10,205,755** (10.21M) | **16,918,915** (16.92M) | **+6,713,160 (+65.78%)** |

### C. Hiệu chỉnh mô tả và diễn giải lại kết quả Variant D
- **Hiệu chỉnh mô tả:**
  Bác bỏ hoàn toàn mô tả ở Revision 1–3 cho rằng Variant D là *"kiến trúc cùng dung lượng với canonical"*.
  **Định nghĩa chuẩn xác:** Variant D là **"mô hình Canonical B2 cộng thêm một nhánh dày DenseAdapter gồm 6.71M tham số huấn luyện"** (tổng dung lượng tăng từ 10.21M lên 16.92M tham số, tương đương tăng +65.78%).
- **Diễn giải lại kết quả thực nghiệm:**
  - Val Dice của Canonical B2 (10.21M tham số): Seed 42 = **0.7793**, Seed 43 = **0.7783** (Trung bình: **0.7788**).
  - Val Dice của Variant D (16.92M tham số): Seed 42 = **0.7737**, Seed 43 = **0.7821** (Trung bình: **0.7779**).
  - Mặc dù được bổ sung thêm tới **6.71 triệu tham số huấn luyện**, hiệu năng của Variant D trên tập Val không hề vượt qua Canonical B2 (chênh lệch trung bình $-0.09$ điểm, nằm hoàn toàn trong nhiễu ngẫu nhiên giữa các seed).
  - Điều này chứng minh rằng việc tăng thêm dung lượng thuần túy dưới dạng các tầng MLP dày (dense adapters) không mang lại giá trị gia tăng nào cho việc phân đoạn vết nứt.

---

## 2. Xác Minh Toàn Diện P3 Refinement Trong Toàn Bộ Các Biến Thể

### A. Kiểm tra cấu hình và mã nguồn
1. **Trong tệp cấu hình:** Tệp cấu hình gốc `configs/p3_ablation/phase7/phase7_b2_s2g_b1_b2_end_to_end.yaml` không khai báo trường `p3`, `p3_mode` hay `p3_refinement`.
2. **Trong mã nguồn khởi tạo:**
   - Trong `sage_lite/sage/networks/sage_injection.py`:
     Hàm `inject_sage_into_model(..., p3_mode=None, ...)` nhận tham số mặc định là `None`. Khi gọi `create_sage_layer` cho các stage ConvNeXt (dòng 70) và các khối Transformer (dòng 97), giá trị truyền vào được cố định là:
     `p3_refinement=None`
3. **Trong mã nguồn tầng SAGE (`SageLayer`):**
   - Trong `sage_lite/sage/components/sage_layer.py`:
     Biến `self.p3_refinement` chỉ được gán trong `__init__` (dòng 97). Trong hàm `forward` và `_execute_expert_path`, hoàn toàn **không có bất kỳ dòng lệnh nào gọi tới `self.p3_refinement`**.
4. **Số lượng tham số của P3:** Bằng **0 tham số** trên tất cả các mô hình.

### B. Đối chiếu trạng thái P3 trên Canonical, Variant N, D, R

| Mô hình / Biến thể | Cấu hình `p3_mode` | Tham số P3 | Trạng thái hoạt động của P3 | Variant N có tắt P3 không? |
|---|:---:|:---:|:---:|:---:|
| **Canonical B2** | `None` | 0 | **Không kích hoạt** (Không tồn tại trong luồng xử lý) | Không áp dụng |
| **Variant N** (`residual_scale=0.0`) | `None` | 0 | **Không kích hoạt** | P3 vốn bằng 0 và không hoạt động; nhánh expert bị tắt hoàn toàn |
| **Variant D** (DenseAdapter) | `None` | 0 | **Không kích hoạt** | Không áp dụng |
| **Variant R** (RandomFixedRouter) | `None` | 0 | **Không kích hoạt** | Không áp dụng |

### C. Kết luận về phân rã tham số
- Số lượng tham số thuộc phần riêng của SAGE (b) giữ nguyên là **738,216 tham số** (gồm 8 Router và các adapter của SA-Hub; P3 = 0 tham số).
- Tổng số tham số của mô hình Canonical B2 vẫn khớp tuyệt đối:
  $$6,626,928 \text{ (Backbone)} + 738,216 \text{ (SAGE)} + 2,840,611 \text{ (Decoder/Head)} = \mathbf{10,205,755} \text{ tham số}.$$
- Khẳng định: Tuyên bố *"P3 không được kích hoạt"* trong các báo cáo là hoàn toàn chính xác dựa trên mã nguồn thực tế.
