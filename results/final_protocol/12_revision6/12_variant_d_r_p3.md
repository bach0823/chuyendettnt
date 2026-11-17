# 12_variant_d_r_p3.md — Cơ Chế Forward Variant D, Phân Tích Tham Số Variant R Và Trạng Thái P3 (Revision 6)

**Ngày lập:** 2026-11-17  
**Nguồn dữ liệu:**
- Mã nguồn thực thi: `sage_lite/scripts/train_phase3_ablation.py` (dòng 97–185 cho Variant D, dòng 189–253 cho Variant R).
- Tệp metadata Phase 3: `results/final_protocol/phase3_variantD_seed42/run_metadata.json`, `phase3_variantR_seed42/run_metadata.json`.
- Tệp checkpoint đối chứng: `results/final_protocol/phase2_seed42/best_model_b2_stage2.pth` và `phase3_variantR_seed42/best_model_b2_stage2.pth`.
- Cấu hình huấn luyện: `configs/p3_ablation/phase7/phase7_b2_s2g_b1_b2_end_to_end.yaml`.

---

## 1. Cơ Chế Hoạt Động Của Variant D (DenseAdapter)

### A. Mô tả chính xác luồng Forward
Trong `sage_lite/scripts/train_phase3_ablation.py`, hàm `_inject_variant_D` thực hiện monkey-patch hàm forward của mọi `SageLayer`:
```python
def _variant_D_forward(self_layer, x, expert_pool=None):
    self_layer.forward_calls += 1
    main_out = self_layer.main_block(x)
    if isinstance(main_out, tuple):
        main_out = main_out[0]
    # Áp dụng phần dư DenseAdapter
    out = self_layer._variant_D_adapter(main_out)
    self_layer._last_lb_loss = None
    self_layer._last_routing_info = {
        "variant": "D",
        "load_balance_loss": torch.tensor(0.0, device=x.device),
    }
    return out
```

### B. Trạng thái hoạt động của nhánh SAGE và số tham số thêm vào
1. **Trong luồng tính toán Forward:**
   - Luồng chính `main_block` chạy bình thường.
   - Đầu ra `main_out` được đưa qua `DenseAdapter` (MLP 2 lớp: `Linear -> GELU -> Linear`).
   - Nhánh SAGE (gồm `router`, `sa_hub` và các lời gọi chuyên gia trong `expert_pool`) **không được gọi** trong forward pass.
2. **Trong cấu trúc mô hình bộ nhớ:**
   - Toàn bộ 8 Router và SA-Hub **vẫn tồn tại nguyên vẹn** trong cấu trúc của `SageLayer` (chiếm 738,216 tham số) vì kịch bản chỉ ghi đè hàm `forward` chứ không xóa bỏ các mô-đun con.
   - Nhánh DenseAdapter được khởi tạo thêm mới gồm **6,713,160 tham số** và đăng ký vào `model.phase3_dense_adapters`.
3. **Kết luận về Variant D:**
   - Tổng tham số của Variant D tăng từ 10,205,755 lên **16,918,915 tham số** (+65.78%).
   - **Xác nhận chuẩn xác:** **Variant D là "Canonical B2 cộng thêm nhánh dày DenseAdapter 6.71M tham số", không phải là một mô hình đối chứng thay thế ngang dung lượng.**

---

## 2. Giải Mã Số Tham Số Của Variant R (RandomFixedRouter)

### A. Đối chiếu số liệu giữa Tệp Nguồn Gốc và Báo Cáo Cũ
- Trong báo cáo Revision 5 cũ, Variant R được ghi là có `9,824,315` tham số (ít hơn Canonical $10,205,755 - 9,824,315 = 381,440$ tham số).
- **Kiểm tra trực tiếp tệp nguồn gốc `results/final_protocol/phase3_variantR_seed42/run_metadata.json`:**
  ```json
  {
    "variant": "R",
    "seed": 42,
    "random_routers_replaced": 8,
    "total_params_after_injection": 10116979
  }
  ```
  Số tham số thực tế sau khi inject Variant R được ghi nhận chính xác trong tệp nguồn là **10,116,979 tham số**.
- Hiệu số tham số thực tế giữa Canonical và Variant R:
  $$10,205,755 - 10,116,979 = \mathbf{88,776} \text{ tham số}.$$

### B. Danh sách các mô-đun bị loại bỏ khi thay thế bằng `RandomFixedRouter`
Hàm `_inject_variant_R` duyệt qua 8 tầng `SageLayer` và thay thế trường `m.router = RandomFixedRouter(...)` (lớp này chỉ có buffer thống kê, **0 tham số huấn luyện**). Đối chiếu trực tiếp `state_dict` giữa checkpoint Canonical và Variant R xác nhận đúng **64 tensor trọng số của 8 Router bị loại bỏ hoàn toàn**, gồm:
1. `backbone.convnext.stages.0.router`: `query_projection` (3,136), `noise_projection` (392), `shared_expert_gate` (49), bias/weights (520) $\to$ 4,097 params.
2. `backbone.convnext.stages.1.router`: `query_projection` (3,136), `noise_projection` (392), `shared_expert_gate` (49), bias/weights (520) $\to$ 4,097 params.
3. `backbone.convnext.stages.2.router`: `query_projection` (6,208), `noise_projection` (776), `shared_expert_gate` (97), bias/weights (520) $\to$ 7,601 params.
4. `backbone.convnext.stages.3.router`: `query_projection` (12,352), `noise_projection` (1,544), `shared_expert_gate` (193), bias/weights (520) $\to$ 14,609 params.
5. 4 khối Transformer (`backbone.transformer_blocks.0..3.router`): mỗi router gồm 14,609 params $\to$ tổng 58,436 params.
- **Tổng số tham số của 8 Router bị loại bỏ:** **88,776 tham số** (khớp chính xác với chênh lệch $10,205,755 - 10,116,979 = 88,776$).
- **Giải thích về con số 381,440:** Con số $381,440$ ở Revision 5 là từ một bản thảo sơ bộ giả định trừ thêm một số adapter của SA-Hub. Tuy nhiên, mã nguồn thực thi `train_phase3_ablation.py` **chỉ thay router và giữ nguyên toàn bộ SAHub cùng các adapter**, do đó số tham số chính thức theo file nguồn là **10,116,979** (chỉ giảm đúng **88,776** tham số của 8 Router).

---

## 3. Trạng Thái Của P3 Refinement Trong Các Tệp Cấu Hình

Kiểm tra tệp cấu hình huấn luyện `configs/p3_ablation/phase7/phase7_b2_s2g_b1_b2_end_to_end.yaml` và các tệp metadata của Canonical, Variant N, D, R:

| Mô hình / Biến thể | Giá trị trường `p3_mode` | Danh sách mô-đun P3 được khởi tạo | Số tham số của P3 |
|---|:---:|:---:|:---:|
| **Canonical B2** | `None` (Không khai báo) | Không có (`None`) | **0** |
| **Variant N** (`residual_scale=0`) | `None` (Không khai báo) | Không có (`None`) | **0** |
| **Variant D** (DenseAdapter) | `None` (Không khai báo) | Không có (`None`) | **0** |
| **Variant R** (RandomRouter) | `None` (Không khai báo) | Không có (`None`) | **0** |

Mã nguồn `sage_injection.py` luôn truyền `p3_refinement=None` vào `create_sage_layer`, và `SageLayer.forward` hoàn toàn không có lệnh gọi nào tới `p3_refinement`. Mô-đun P3 có 0 tham số và không tham gia vào bất kỳ cấu hình nào trong số các biến thể trên.
