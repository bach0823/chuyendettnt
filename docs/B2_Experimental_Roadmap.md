# B2 Experimental Roadmap (SAGE-Lite)

Cập nhật kế hoạch thực nghiệm B2 theo roadmap sau. Mục tiêu là khảo sát có kiểm soát, không mở grid HPO rộng ngay từ đầu.

## Nguyên tắc chung

* Không dùng Test set để chọn hyperparameter.
* Validation là tiêu chí chọn cấu hình.
* Mỗi phase chỉ thay đổi nhóm biến cần khảo sát; các biến còn lại giữ nguyên baseline.
* Không tự ý sửa architecture/pipeline chỉ để cải thiện kết quả.
* Ghi lại đầy đủ config, checkpoint, Val Dice và các routing diagnostics cho từng run để có thể truy nguyên thí nghiệm.

## Phase 0 — Runtime preflight (HOÀN TẤT ✅)

Kết quả đo đạc thực tế trên Tesla T4 (14.56 GB usable, AMP FP16, 448×448, ViT Depth 12):
* **Batch 20 & Batch 14+**: ❌ **OOM** (Vượt ngưỡng VRAM 14.56 GB T4).
* **Batch 12 (Real Crack500 Data)**: ✅ **PASS** (12 training batches thật: Peak Alloc 13.84 GB, Peak Res 14.05 GB, **Free 0.51 GB**, Throughput 0.55 samples/s, 0 memory leak từ batch 2).
* **Batch 12 (Synthetic)**: ✅ **PASS** (12 iters: Peak Alloc 13.77 GB, Peak Res 14.10 GB, Free 0.47 GB, memory drift +0.08 MB).
* **Batch 10 (Synthetic)**: ✅ **PASS** (12 iters: Peak Alloc 13.43 GB, Peak Res 13.89 GB, Free 0.68 GB).
* **Batch 8 (Synthetic)**: ✅ **PASS** (12 iters: Peak Alloc 12.07 GB, Peak Res 12.50 GB, Free 2.07 GB).
* **Kết luận Runtime Batch Size**: **`batch_size: 12`** chính thức được xác nhận khả thi và an toàn cho Full Training trên dữ liệu Crack500 thật.
* Chi tiết log xem tại: `docs/B2_Phase0_Preflight_Log.md`.

| Hạng mục | Kết quả |
|---|---|
| Real Crack500 pipeline | PASS |
| B2-D12 / top-k=4 | PASS |
| Batch 12 OOM | Không |
| Forward/backward/optimizer | PASS |
| Numerical stability | PASS |
| 16 experts active | PASS |
| VRAM | **Rất sát giới hạn** |
| Throughput | Chạy được nhưng biến động cao |
| Có cần sửa architecture? | **Không** |

*Lưu ý:* Việc điều chỉnh batch size từ 20 xuống 12 là **hardware-constrained runtime setting**, không phải HPO/tuning result.





## Phase 1 — Khảo sát ViT depth (Baseline Scale) (ĐÃ SẴN SÀNG CẤU HÌNH ✅)

Do thay depth ảnh hưởng đến architecture và routing scale, Phase 1 chạy đầu tiên để lock base architecture.

* **Cố định dùng chung**:
  - `batch_size = 12` (runtime setting đã xác thực qua Real-Data Preflight trên T4)
  - `img_size = 448`, `seed = 42`, `lr = 1e-4`, `epochs = 30`, `patience = 6`
  - Canonical preprocessing (Crack500 random crop 448x448, smart filter `fg_pixels >= 20`, reflect pad)
  - SAGE config: `top_k = 4`, `hidden = 64`, `gating = sigmoid`, `noise = ON`, `logit_mod = ON`, `LB = 0.01`, `dropout = 0.1`, `fusion_type = residual`, `residual_scale = 0.1`.
* **Bộ cấu hình thực nghiệm Phase 1**:
  * **Run 1: Depth 12** (`configs/b2_crack500_depth12.yaml` → `output_dir: .../B2_Crack500_Depth12`): 16 routers (4 CNN + 12 ViT), 16 experts, 13.9M params.
  * **Run 2: Depth 6** (`configs/b2_crack500_depth6.yaml` → `output_dir: .../B2_Crack500_Depth6`): 10 routers (4 CNN + 6 ViT), 10 experts, 12.0M params.
  * **Run 3: Depth 4** (`configs/b2_crack500_depth4.yaml` → `output_dir: .../B2_Crack500_Depth4`): 8 routers (4 CNN + 4 ViT), 8 experts, 10.1M params.
* **Quy tắc Quyết định**: Chọn 1 depth tốt nhất **dựa duy nhất trên Validation Dice (tập Val)** để làm base cho Phase 2 (top_k). **TUYỆT ĐỐI KHÔNG DÙNG TEST SET ĐỂ CHỌN DEPTH**.


## Phase 2 — Khảo sát Routing Capacity (top_k)

* Fix: Base depth từ Phase 1.
* Thử nghiệm (thay đổi top_k):
  * Run 4: top_k = 2
  * Run 5: top_k = 4 (đã chạy ở Phase 1)
* Decision: Chọn top_k tốt nhất. (Nếu top_k=2 gần bằng top_k=4, ưu tiên top_k=2 vì FLOPS thấp hơn).

## Phase 3 — Khảo sát Router Hidden Dim

* Fix: Base depth (P1) + Best top_k (P2).
* Thử nghiệm:
  * Run 6: router_hidden_dim = 32
  * Run 7: router_hidden_dim = 64 (từ P1)
  * Run 8: router_hidden_dim = 128
* Decision: Chọn hidden_dim.

## Phase 4 — Cân bằng tải (Load Balancing)

Chỉ mở phase này nếu log cho thấy expert bị "dead" hoặc mất cân bằng nghiêm trọng. Chú ý: LB loss scale theo số lượng routers, nên Depth 12 vs Depth 6 sẽ có total LB loss khác nhau. Phải soi trung bình LB/router.

* Fix: Base config từ P3.
* Thử nghiệm (LB weight):
  * Run 9: LB = 0.005
  * Run 10: LB = 0.01 (từ P1)
  * Run 11: LB = 0.03

## Phase 5 — Optimization Stability (SAGE LR & Warmup)

Nhóm parameters của SAGE (Routers + Adapters) thường cần LR khác backbone.
Mặc định hiện tại: Backbone=1e-5, Decoder=1e-4.

* Thử nghiệm (SAGE LR):
  * Run 12: SAGE LR = 5e-5
  * Run 13: SAGE LR = 1e-4
  * Run 14: SAGE LR = 2e-4
* Thử nghiệm Warmup epochs (nếu thấy training bị spike loss đầu epoch):
  * Run 15: Warmup = 2
  * Run 16: Warmup = 3
  * Run 17: Warmup = 5

## Phase 6 — Regularization (Nghiên cứu thêm)

* Thử nghiệm dropout cho adapter:
  * expert_dropout = {0.0, 0.1, 0.2}
* Thử nghiệm pure residual scale:
  * residual_scale = {0.05, 0.1, 0.2}

Quy trình: Đưa cho tôi script preflight và file kế hoạch này. Khi nào chuẩn bị xong Colab tôi sẽ chạy Phase 0 và gửi log.

## Phase A — Khảo sát Residual Scale
* Fix: fusion_type = "residual"
* Cố định các biến khác (dropout=0.1, top_k, depth, router, v.v.)
* Thử nghiệm residual_scale:
  * Run A1: residual_scale = 0.05
  * Run A2: residual_scale = 0.10 (Baseline từ trước)
  * Run A3: residual_scale = 0.20
* Decision: Chọn residual_scale tốt nhất trên Validation.

## Phase B — So sánh Fusion Type (Residual vs Adaptive)
* Fix: Chọn residual_scale tốt nhất từ Phase A cho Residual variant.
* Adaptive Fusion variant: Sử dụng `alpha` (learnable scalar).
* Đảm bảo cách ly (isolation): giữ nguyên cùng ViT depth, expert pool, top_k, router hidden, LB factor, learning rate, preprocessing, random seed, v.v.
* Thử nghiệm so sánh:
  * Run B1: Residual Fusion (với scale tốt nhất Phase A)
  * Run B2: Original SAGE Adaptive Fusion (với learnable alpha)
* Decision: Chọn Fusion mechanism.
