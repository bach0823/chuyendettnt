# B2 Experimental Roadmap (SAGE-Lite)

Cập nhật kế hoạch thực nghiệm B2 theo roadmap sau. Mục tiêu là khảo sát có kiểm soát, không mở grid HPO rộng ngay từ đầu.

## Nguyên tắc chung

* Không dùng Test set để chọn hyperparameter.
* Validation là tiêu chí chọn cấu hình.
* Mỗi phase chỉ thay đổi nhóm biến cần khảo sát; các biến còn lại giữ nguyên baseline.
* Không tự ý sửa architecture/pipeline chỉ để cải thiện kết quả.
* Ghi lại đầy đủ config, checkpoint, Val Dice và các routing diagnostics cho từng run để có thể truy nguyên thí nghiệm.

## Phase 0 — Runtime preflight (HOÀN TẤT ✅)

Kết quả đo đạc thực tế trên Tesla T4 (14.56 GB usable, AMP FP16, 448×448, ViT Depth 12, Stress-test 12 Iterations):
* **Batch 20 & Batch 14+**: ❌ **OOM** (Vượt ngưỡng VRAM 14.56 GB T4).
* **Batch 12** (12 iters): ✅ **PASS** (Peak Alloc 13.77 GB, Peak Res 14.10 GB, Free 0.47 GB, memory drift +0.08 MB).
* **Batch 10** (12 iters): ✅ **PASS** (Peak Alloc 13.43 GB, Peak Res 13.89 GB, Free 0.68 GB, memory drift +1.37 MB) — **Khuyến nghị cân bằng (Throughput / Headroom)**.
* **Batch 8** (12 iters): ✅ **PASS** (Peak Alloc 12.07 GB, Peak Res 12.50 GB, Free 2.07 GB, memory drift -0.22 MB) — **Khuyến nghị an toàn tuyệt đối**.
* Đã verify 100%: 16/16 experts được route đồng đều (5.7% - 6.8%), checkpoint round-trip exact match, 0 missing/unexpected keys.
* Chi tiết log xem tại: `docs/B2_Phase0_Preflight_Log.md`.

*Lưu ý:* Việc giảm batch size xuống 10 hoặc 8 là **hardware-constrained runtime setting**, không phải HPO/tuning result.



## Phase 1 — Khảo sát ViT depth (Baseline Scale)

Do thay depth ảnh hưởng đến architecture và routing scale, phải chạy đầu tiên để lock base architecture.

* Fix: top_k=4, hidden=64, sigmoid, noise=ON, logit_mod=ON, LB=0.01, dropout=0.1, residual_scale=0.1.
* Thử nghiệm:
  * Run 1: Depth 12 (16 routers)
  * Run 2: Depth 6 (10 routers)
  * Run 3: Depth 4 (8 routers) – Tuỳ chọn nếu compute cho phép.
* Decision: Chọn 1 depth tốt nhất trên Validation để làm base cho các Phase sau.

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
