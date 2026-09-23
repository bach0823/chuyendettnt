# SAGE-lite Development Notes & Feedback

## 1. Phản hồi Kiến trúc (Ngày 2026-09-17)
- **Backbone & Đầu ra:** THAY THẾ (xoá bỏ) backbone cũ bằng backbone mới (ConvNeXtV2-Femto/ViT-Tiny), không phải chỉ bổ sung.
- **Positional Embedding (2D):** Đồng ý hướng xử lý nhưng cần đánh giá kỹ tác dụng phụ.
- **Decoder (Depthwise Separable):** Đúng hướng nhưng rủi ro. Cần đánh giá xem có đáng đánh đổi không. Nếu model train bị bất ổn (không hội tụ), đây là nơi đầu tiên cần kiểm tra và rollback.
- **Router Noise & Fusion:** Giữ nguyên Exploration Noise làm baseline (sẽ làm ablation sau) và triển khai Residual Fusion.
- **SA-Hub Lazy Init:** Cần xem xét thêm, phân tích rõ lợi/hại và cách triển khai cụ thể trước khi chốt.
- **Granularity & Audit (Injection):** Vẫn còn mơ hồ, cần được giải thích rõ ràng và trực quan hơn.
- **TupleSafeWrapper:** Cần test thực tế trước khi sửa phức tạp.

## 2. Global Rules & Nguyên tắc làm việc (Cần tuân thủ)
- **Show Diff:** Mọi thay đổi vào mã nguồn (chỉnh sửa, xoá module) bắt buộc phải show diff cho User xem sau khi làm xong.
- **Objective Subagent Prompting:** Khi ra lệnh cho subagent review một ý tưởng, phải dùng câu lệnh khách quan (ví dụ: "so sánh A và B", "đánh giá ưu nhược điểm") thay vì hỏi chung chung "có tốt không".
- **Dedicated Subagents:** Khi User chỉ đích danh một điểm cần kiểm tra/giải thích (ví dụ: đánh giá rủi ro decoder, giải thích SA-Hub), phải gọi hẳn một subagent riêng biệt chuyên trách cho tác vụ đó.

## 3. Cập nhật Chiến lược (Ngày 2026-09-17)
- **Decoder DWSC Strategy:** Ưu tiên số 1 là xây dựng bản Decoder dùng Standard 3x3 Conv truyền thống trước để đảm bảo mốc cơ sở (baseline) hội tụ. DWSC chỉ được implement dưới dạng cờ (toggle) để chạy smoke test đối chứng sau.

## 4. Bí mật Hệ thống định tuyến toàn cục (Ngày 2026-09-17)
- **SA-Hub O(D²) Initialization:** Thiết kế đẻ ra toàn bộ tổ hợp Adapter (O(D²)) không phải là viết code lười biếng, mà là yêu cầu bắt buộc (Hard constraint) của mạng SAGE. Do SAGE sử dụng Global Dynamic Routing (định tuyến theo dữ liệu đầu vào), một Tensor ở tầng nông có thể vọt lên Expert ở tầng sâu nhất. Không thể quét tĩnh (static scan) vì đường đi thay đổi theo từng ảnh. BẮT BUỘC giữ nguyên cơ chế O(D²) của tác giả gốc.

## 5. Đánh giá Kiến trúc SA-Hub Pairwise vs Shared Latent (Ngày 2026-09-17)
- **Insight:** O(D²) Adapter không phải là bản chất toán học của SAGE, mà là hệ quả của cách thiết kế Pairwise (1-1). Hoàn toàn có thể dùng Shared Latent Dimension D để giảm tham số về O(N).
- **Trade-off (Đánh đổi):** Mặc dù Shared Latent giúp giảm tham số, nó lại làm TĂNG số lượng FLOPs lúc inference (phải tính qua 2 bước chiếu: {in} \to D \to C_{out}$ thay vì 1 bước như Pairwise).
- **Nguy cơ Information Bottleneck:** Nếu ép qua không gian trung gian D quá nhỏ, mạng sẽ bị mất rank (low-rank bottleneck), hoạt động như một bộ lọc thông thấp (low-pass filter). Đối với bài toán vết nứt, điều này sẽ nghiền nát các dải tần số không gian cao (high-frequency spatial harmonics). Do đó, giữ nguyên thiết kế O(D²) Pairwise gốc là sự hy sinh xứng đáng để bảo toàn trọn vẹn tín hiệu topo học của vết nứt.

## 6. Đánh giá Granularity: Stage-level vs Block-level (Ngày 2026-09-17)
- **Vấn đề:** Ban đầu có ý tưởng xé nhỏ SAGE Injection từ cấp Stage xuống cấp Block để giống MoE.
- **Phân tích rủi ro:** 
  1. **Routing Jitter:** Nhảy cóc expert ở mỗi block làm vỡ vụn tính liên tục của luồng feature.
  2. **Overhead:** ConvNeXt-Femto có cấu trúc block là [2, 2, 6, 2] = 12 blocks. Nếu bơm SAGE ở cấp Block thay vì cấp Stage riêng phần CNN, tổng số injection point của toàn mạng sẽ **tăng thêm 8** (bất kể baseline của ViT đang là bao nhiêu blocks). Việc này làm bùng nổ tham số và FLOPs một cách không cần thiết.
  3. **Phá hủy hình thái nứt:** Bức ảnh bị cắt vụn qua các expert khác nhau ở mỗi block sẽ làm vết nứt bị đứt đoạn, mất tính liên kết topo.
- **Kết luận:** CHỐNG CHỈ ĐỊNH dùng Block-level. BẮT BUỘC giữ nguyên Stage-level Granularity như thiết kế gốc.

### C. Hàm Loss & Tăng cường dữ liệu (Augmentation)
> **[CẬP NHẬT GHI ĐÈ]**: Toàn bộ kết luận sơ bộ ở Mục C này (cả Loss và Augmentation) đã bị phủ quyết và thay thế bởi Mục 7 và Mục 9 ở phía dưới.

- **Loss Function:** SAGE natively uses 0.5*BCE + 0.5*Dice + lb_loss. For crack segmentation, BCE + Dice is extremely robust. The proposed Laplacian Boundary Loss is too brittle due to human annotation noise (1-2 pixel shift causes massive gradient spikes). Keep lb_loss to balance the routers (Full Injection baseline đầu tiên; N=2 = sparse candidate; chưa chốt N cuối cùng).
- **Augmentation:** Simple rigid flips (Horizontal/Vertical) perfectly preserve crack topological continuity. Complex geometric (Elastic/Grid) warp and resample binary masks, breaking 1-pixel cracks into dashed dots. Photometric (CLAHE) amplifies concrete background noise causing False Positives.

## 7. Phân tích & Đánh giá Các Ứng viên Loss Function (Ngày 2026-09-18)
*(Chi tiết quá trình thẩm định 4 ứng viên Loss Function cho mô hình SAGE-lite. Cập nhật thay thế cho ghi chú sơ bộ ở mục C).*

### ❌ Candidate 1 (BCE + Dice thông thường)
- **Công thức:** `BCE_with_logits + DiceLoss`
- **Tình trạng:** **Loại bỏ hoàn toàn.**
- **Lý do:** Mắc lỗi kiến trúc chí mạng là không có Load Balance Loss (`lb_loss`). Thiếu thành phần này chắc chắn sẽ gây ra Router Collapse trong mạng MoE của SAGE. Ngoài ra, việc để tỷ lệ 1:1 mặc định không có tác dụng đẩy mạnh Recall (chống bỏ sót vết nứt) so với tỷ lệ gốc của SAGE.

### ❌ Candidate 3 (BCE + Dice + Tversky)
- **Công thức:** `0.5*BCE + 0.3*Dice + 0.2*Tversky(alpha=0.3, beta=0.7)`
- **Tình trạng:** **Loại bỏ hoàn toàn.**
- **Lý do:** Tương tự Candidate 1, thiếu `lb_loss`. Về mặt logic, mục đích đưa Tversky vào (chỉnh alpha, beta) để ưu tiên Recall là có cơ sở. Tuy nhiên ở giai đoạn này làm như vậy là dư thừa và phức tạp hóa không cần thiết, vì chỉ riêng việc nâng trọng số Dice (`dice_weight = 1.5`) đã đủ sức gánh vác mục tiêu này. Dice vốn dĩ là trường hợp đặc biệt của Tversky, việc dùng song song sẽ gây lãng phí tài nguyên tính toán (thêm forward pass).

### ⏳ Candidate 2 (Weighted BCE + Dice)
- **Công thức:** `Weighted_BCE(pos_weight=20.0) + DiceLoss`
- **Tình trạng:** **Loại bỏ ở hiện tại — Đưa vào chế độ chờ (Standby Option).**
- **Lý do loại bỏ hiện tại:** Thiếu `lb_loss` và vướng lỗi gán tĩnh (hard-code) `pos_weight = 20.0` dựa trên ước đoán.
- **Điều kiện hồi sinh:** Trở thành một option hợp lệ để nâng cấp trong tương lai **khi và chỉ khi** đo đạc được thông số chính xác từ dữ liệu thực tế bằng công thức: `pos_weight = total_background_pixels / total_crack_pixels` (đo trên toàn bộ tập train, ước tính rơi vào khoảng 15-20 đối với bộ dữ liệu crack hiện tại).

### ✅ Candidate 4 (BCE + Dice + L_balance)
- **Công thức:** `BCE + Dice + λ_aux * L_balance`
- **Tình trạng:** **TRÚNG TUYỂN (Tinh chỉnh thành Golden Candidate).**
- **Lý do được chọn:** Là kiến trúc duy nhất thỏa mãn triết lý SAGE-lite, bảo vệ được mạng MoE thông qua Switch Transformer Loss (`L_balance`). Được giữ lại để tinh chỉnh (`1.0 * BCE + 1.5 * Soft Dice + 1.0 * L_balance`) và đưa vào Pipeline triển khai chính thức.

## 8. Chuẩn Kích thước Ảnh (Ngày 2026-09-18)
- **Quyết định:** Thiết lập `img_size = 448x448`. 
- **Kích thước chuẩn:** Thiết lập img_size = 448x448 cho mọi luồng dữ liệu. Với kiến trúc ConvNeXt hiện tại, ảnh đầu vào được giảm kích thước không gian tổng cộng **32 lần** qua các bước downsampling (stride 4 ở stem và ba lần stride 2 ở các stage tiếp theo), nên kích thước feature map ở cuối ConvNeXt là 448 / 32 = 14, tức **14x14 = 196 token**. Các token này được tạo bằng cách flatten feature map cuối của ConvNeXt trước khi đưa vào các Transformer blocks. Vì vậy, 196 là số token của bottleneck hiện tại; không phải 784 từ phép chia trực tiếp 448 / 16 của một ViT patch16 áp dụng lên ảnh đầu vào. Vừa vặn để batch size 8-16 chạy trên GPU T4 (16GB VRAM) mà không làm rách các vết nứt siêu mảnh.

## 9. Chiến lược Tiền xử lý & Augmentation (Ngày 2026-09-18)
- **Hàm Loss Chốt:** `1.0 * BCE + 1.5 * Soft Dice + 1.0 * L_balance` (Bỏ hẳn pos_weight, dựa vào 1.5x Soft Dice để chống imbalance. Sẽ review lại sau 3-5 epoch nếu mô hình predict all-zero).
- **Augmentation Chốt:** ĐÃ LOẠI BỎ HOÀN TOÀN ElasticTransform, GridDistortion (vì làm đứt gãy vết nứt) và CLAHE (khuếch đại nhiễu bê tông/False Positive). Chỉ giữ lại các phép biến đổi an toàn: Flip, Rotate90, ShiftScaleRotate, RandomBrightnessContrast, HueSaturationValue, và GaussianBlur.
- **Xử lý Tập DeepCrack:** Dùng **Pad + Resize về 448x448**. Giữ nguyên tỷ lệ khung hình (aspect ratio) để đảm bảo hình thái vết nứt không bị kéo giãn biến dạng.
- **Xử lý Tập Crack500 (Ảnh siêu lớn):**
  - **Lúc Train:** Không Resize, dùng `RandomCrop(448, 448)`.
  - **Lọc Mảng Trống (Smart Filtering):** Dùng hàm kiểm tra số lượng pixel nứt (`min_pixels >= 20`) và độ dài thành phần liên thông (`min_component_length >= 15`). Cân bằng: Giữ lại toàn bộ (80-85%) các patch thỏa mãn điều kiện, và khoảng 15-20% patch pure-background ngẫu nhiên để mô hình học cách không over-predict (nhìn đâu cũng thấy nứt). Ngưỡng 20px sẽ được tinh chỉnh (tune) bằng cách chạy script phân tích Histogram mật độ pixel trước khi chốt số cứng.
  - **Lúc Test/Eval:** Dùng Non-overlapping tiling 448x448 (nếu không chia hết thì pad). Predict từng ô tile rồi ghép lại nguyên bản, cuối cùng tính metric trên toàn bộ ảnh full.


## 11. Số lượng Transformer Blocks của ViT (Ngày 2026-09-18)
- **Quyết định (Thực nghiệm):** Bắt đầu với 6 blocks (điểm giữa).
- **Lý do cụ thể:**
  - ViT-Tiny (patch16) pretrained có 12 blocks, lấy nửa đầu giữ được phần lớn knowledge đã học từ ImageNet.
  - Đủ để self-attention lan truyền thông tin toàn cục qua nhiều "hop".
  - Nếu 6 blocks underperform thì tăng lên 8-12, nếu overfit thì giảm xuống 4.
## 12. Khắc phục lỗi cấu trúc luồng Huấn luyện của SAGE gốc (Ngày 2026-09-18)
- **Vấn đề từ bản gốc (Structural Bug):** Hàm `train_one_epoch` và `validate_one_epoch` gọi `outputs = model(images)` rồi ngay lập tức ghi đè bằng `model.forward_with_routing_info(images)`. Điều này lãng phí 100% thời gian tính toán và làm sai lệch thống kê của các lớp BatchNorm do cập nhật `running_mean`/`running_var` hai lần trên cùng 1 lượng dữ liệu.
- **Giải pháp (SAGE-lite):** Bọc điều kiện `if/else` tường minh để mỗi batch chỉ đi qua duy nhất 1 lần forward pass.

```python
# Thay vì:
outputs = model(images)
if hasattr(model, "forward_with_routing_info"):
    detailed = model.forward_with_routing_info(images)
    outputs = detailed["logits"]
    # ...

# Sửa thành:
if hasattr(model, "forward_with_routing_info"):
    detailed = model.forward_with_routing_info(images)
    outputs = detailed["logits"]
    routing_infos = detailed.get("routing_infos", {})
    lb_loss = compute_lb_loss(routing_infos).to(device)
else:
    outputs = model(images)
    routing_infos = None
    lb_loss = torch.tensor(0.0, device=device)
```

## 13. Tối ưu hóa Training Pipeline của SAGE gốc (Ngày 2026-09-18)
Đánh giá 6 ý tưởng điều chỉnh Pipeline gốc cho SAGE-lite (ConvNeXt-Femto + ViT-Tiny 6 blocks):

### ✅ 1. AMP — Áp dụng, nhưng fix phải nằm trong `router.py`
- **Vấn đề**: `eps = 1e-9` tại `router.py` dòng 253-260. Phép `g_s + eps` chạy dưới FP16 *trước khi* `torch.log` kịp upcast → nếu `g_s` cực nhỏ vẫn ra `log(0) = -inf`. Đưa `lb_loss` ra ngoài `autocast` ở training loop KHÔNG giải quyết được root cause.
- **Fix đúng — phải nằm trong `router.py`**:
```python
with torch.autocast(device_type='cuda', enabled=False):
    g_s_fp32 = g_s.float()
    log_g_s = torch.log(g_s_fp32 + 1e-5)
    log_one_minus_g_s = torch.log(1 - g_s_fp32 + 1e-5)
```
- **Lý do quan trọng**: `sage_colon.yaml` dòng 50 xác nhận tác giả gốc dùng **BF16** (`use_amp: true`) trên **A100**. BF16 có dải số mũ 8-bit giống FP32 nên không bị underflow ở `1e-9`. Trên T4 của ta dùng **FP16** (Tensor Cores) nên nhất định phải fix.

### ❌→✅ 2. LR `stage2_base_lr` / `stage2_shared_lr` — Bắt đầu 1:1, quan sát rồi scale
- **Thực tế từ 3 config gốc** (không có "tỷ lệ chuẩn"):
  - `sage_colon.yaml` (dòng 52-53): `base_lr=0.0001`, `shared_lr=0.0005` → tỷ lệ **1:5** (shared cao hơn)
  - `sage_glas.yaml` (dòng 48-49): `base_lr=0.0001`, `shared_lr=0.00005` → tỷ lệ **2:1** (shared thấp hơn)
  - `sage_ebhi.yaml` (dòng 48-49): `base_lr=0.0001`, `shared_lr=0.00005` → tỷ lệ **2:1** (shared thấp hơn)
- **Câu hỏi thực chất**: Với bài toán crack segmentation trên T4 (~1500 ảnh), shared experts có nên học nhanh hơn (để ổn định routing sớm) hay chậm hơn (để non-shared có thời gian explore)?
  - Nếu `shared_lr` cao (1:3 hay 1:5): Shared experts hội tụ nhanh, cung cấp nền tảng ổn định. Nhưng rủi ro Router lock vào shared experts sớm, không explore được expert còn lại.
  - Nếu `shared_lr` thấp: Non-shared/router explore tự do hơn, nhưng thiếu nền tảng ổn định sớm.
- **Quyết định**: Bắt đầu **1:1** (`stage2_base_lr = stage2_shared_lr`). Chỉ tăng `shared_lr` lên 1:3 nếu quan sát thấy routing collapse qua `gs_tracker` (g_s sụp về 0 hoặc 1 quá sớm). Không giả định trước tỷ lệ nào.

### ✅ 3. `select_shared_experts` AttributeError — Áp dụng
- Xác nhận bug 100% từ code: `train_sage.py` dòng 304 hard-code `len(model.convnext.stages)`.
- **Fix**: Thêm `@property num_sage_experts` vào `ConvNeXtV2ViTTinyUNet`. Hàm `select_shared_experts` ưu tiên gọi property này trước, fallback về `len(model.convnext.stages)` cho tương thích ngược.

### ✅ 4. Giảm `warmup_epochs` xuống 2-3 — An toàn, áp dụng
- **Bằng chứng code (`train_sage.py` dòng 256-277)**: `warmup_epochs` chỉ đơn giản thay đổi tham số `total_iters` của `LinearLR` — LR tăng tuyến tính từ 1% lên 100% base_lr trong số epoch đó.
- **Áp dụng cả 2 Stage**: Hàm `create_scheduler` được gọi cả ở Stage 1 (`train_sage.py` dòng 381) và Stage 2 (`train_sage.py` dòng 455). Cùng giá trị `warmup_epochs` được dùng cho cả hai.
- **Kết luận**: Giảm warmup từ 5 xuống 2-3 chỉ làm LR đạt đỉnh sớm hơn, không bỏ qua bất kỳ khởi tạo quan trọng nào. Hoàn toàn an toàn.

### ✅ 5. Giảm `EarlyStopping patience` xuống 5-7 — An toàn, áp dụng
- **Stage 2 luôn luôn chạy** (`train_sage.py` dòng 433): EarlyStopping chỉ `break` ra khỏi vòng lặp Stage 1, sau đó code tiếp tục xuống Stage 2 bình thường. Không có gì chặn Stage 2.
- **Model truyền vào Stage 2 là best Stage 1** (`train_sage.py` dòng 435-439): Ngay sau khi vòng lặp Stage 1 kết thúc (dù bởi EarlyStopping hay hết epoch), code reload checkpoint tốt nhất trước khi Stage 2 khởi động:
```python
stage1_best_path = stage1_es.path
if os.path.exists(stage1_best_path):
    model.load_state_dict(torch.load(stage1_best_path, map_location=device))
    logging.info(f"Loaded best Stage 1 weights from {stage1_best_path}")
```
- **Chỉ theo dõi mIoU** (`train_sage.py` dòng 383-389): EarlyStopping không theo dõi `lb_loss`. Concern về "Deceptive Plateau" từ lần trước là suy diễn không có cơ sở trong code này.
- **Kết luận**: Giảm patience xuống 5-7 là an toàn. Stage 2 luôn nhận đúng model tốt nhất từ Stage 1.

### ❌ 6. `gs_tracker.py` nhãn "may mắn" — Sai lý luận, không cần action
- `visualization.py` gán nhãn 4 items đầu là CNN là **đúng có chủ đích**, không phải trùng hợp.
- Mọi ConvNeXt variant (atto/femto/pico/nano/base) đều cố định **đúng 4 stages** về mặt kiến trúc — chỉ `channel dims` thay đổi, không phải số stage. Vì vậy, hardcode 4 là invariant, không cần sửa.

## 14. Xác nhận Kiến trúc: 4 Shared Experts (0–3) là 4 CNN Stages là Thiết kế CÓ CHỦ ĐÍCH (Ngày 2026-09-18)

Việc 4 chuyên gia đầu tiên (indices `[0, 1, 2, 3]`) trở thành **Shared Experts** trong SAGE ở Stage 2 hoàn toàn là một chủ đích kiến trúc đã được tính toán kỹ lưỡng, không phải sự ngẫu nhiên hay gán nhãn trùng hợp.

### Bằng chứng mã nguồn (Source Code Proofs)

1. **Thứ tự đóng gói Expert Pool bị khóa cứng trong `sage_injection.py` (lines 32–42 & 92–98):**
   Mô hình luôn duyệt qua các stage của ConvNeXt trước, sau đó mới nạp Transformer blocks:
   ```python
   # sage/networks/sage_injection.py
   for i in range(len(convnext.stages)):
       expert_infos.append({"type": "cnn", "name": f"convnext_stage_{i}", "index": i})
   for i in range(len(transformer_blocks)):
       expert_infos.append({"type": "transformer", "name": f"transformer_block_{i}", "index": len(convnext.stages) + i})
   ...
   for stage_wrapper in convnext.stages:
       expert_pool.append(stage_wrapper.main_block)  # 4 stages CNN luôn chiếm indices 0, 1, 2, 3
   for block_wrapper in transformer_blocks:
       expert_pool.append(block_wrapper.main_block)  # Các ViT blocks nằm sau từ index 4 trở đi
   ```

2. **Cơ chế chọn Shared Experts mặc định trong `train_sage_ddp.py` (lines 445–452):**
   ```python
   # scripts/train_sage_ddp.py
   def select_shared_experts(config: Dict, base_model: nn.Module) -> List[int]:
       sage_cfg = config.get("sage", {})
       explicit  = sage_cfg.get("shared_expert_indices", [])
       if explicit:
           return explicit
       k     = config["training"]["num_shared_experts"]  # = 4 từ file config
       total = len(base_model.convnext.stages) + len(base_model.transformer_blocks)
       return list(range(min(k, total)))  # Luôn trả về [0, 1, 2, 3] tương ứng đúng 4 CNN stages
   ```

3. **Comment và khẳng định trực tiếp từ tác giả gốc trong tools phân tích:**
   - Trong `tools/visualize_gs_evolution.py` (lines 66–68):
     ```python
     # CNN layers: high gs (favor shared experts)
     cnn_mean = 0.65 + 0.03 * np.sin(epochs / 8)
     cnn_mean = np.clip(cnn_mean, 0.6, 0.75)
     ```
   - Trong `tools/visualize_expert_routing_decisions.py` (lines 290–296):
     ```python
     # Labels - CNN vs Transformer
     # CNN: L0-L3 (4 layers from ConvNeXt-Large)
     # ViT: L4-L27 (24 layers from Vision Transformer)
     # First 4 are CNN, rest are Transformer
     ```

### Ý nghĩa thiết kế (Design Rationale)
- **Prior-Guided Locality:** CNN (ConvNeXt) có tính chất bất biến tịnh tiến (translation invariance) và trường tiếp nhận cục bộ (local receptive field), chuyên tóm bắt mép viền, texture và chi tiết vi mô — đây là các đặc trưng phổ quát mà **mọi patch ảnh** (bất kể mô thường hay mô bệnh, background hay crack) đều bắt buộc phải dùng đến. Do đó, ép 4 stage CNN làm Shared Experts giúp mô hình luôn có nền móng không gian ổn định.
- **Phân định vai trò:** 
  - **4 Shared Experts (CNN):** Đóng vai trò anchor point trích xuất cấu trúc hình học chung.
  - **Non-shared / Fine-grained Experts (ViT):** Tận dụng self-attention toàn cục để mô hình hóa ngữ cảnh phức tạp và các bất thường topo tùy theo từng đặc trưng riêng của patch.



### 7. Chiến lược N_injection (Full vs Sparse)

Bỏ qua các vấn đề implementation trước, xét thuần trade-off thì **full injection không nhất thiết là lựa chọn tối ưu**. Trong bối cảnh model nhỏ + dataset nhỏ, có hai lý do chính:

1. **Diminishing returns giữa các injection point liền kề**: Feature giữa các layer gần nhau thường có mức tương quan cao. Thêm injection point làm tăng thêm router parameters và routing decisions, nhưng lượng thông tin mới đóng góp (marginal value) có thể khá nhỏ.
2. **Router calibration với dataset nhỏ (Điểm quan trọng nhất)**: Mỗi injection point có Router/gating riêng cần calibration. Với dataset ~1500–2400 ảnh, càng nhiều injection point đồng nghĩa càng nhiều routing decisions cần đủ dữ liệu để học ổn định. Việc ép full injection mang risk là một số Router dư thừa học pattern kém ổn định hoặc nhiễu. (Lưu ý: Không có nghĩa full injection chắc chắn overfit, chỉ là N_injection không có quan hệ monotonic với performance).

**Tóm tắt Lộ trình Thực nghiệm (6 Mốc):**
- **Mốc 1 (Giai đoạn chính - Full Injection)**: Giữ nguyên **full injection mặc định** của `sage_injection.py` để tuân thủ Minimal Migration, tạo baseline reference ổn định. Đối với SAGE-lite, số lượng Router cho Full Injection phụ thuộc trực tiếp vào số block của backbone: `4 (ConvNeXt stages) + num_transformer_layers`. (Ví dụ: với baseline 6 blocks thì Full Injection = 10 Routers). Không nhầm lẫn với 16 routers (nếu dùng 12 blocks) hay 28 routers của SAGE gốc.
- **Mốc 2 (Diagnostic sau baseline)**: Kiểm tra `expert_usage_ratio` và load-balance behavior (Lưu ý: Không dùng `gs_tracker` vì nó chỉ đo `g_s` giữa shared/dynamic, không phải tần suất expert trong Top-K). "Phân tán đều thì tốt" (nên hiểu là "không có dấu hiệu routing collapse rõ rệt"), lệch nặng là tín hiệu điều tra routing collapse (dù usage khá đều cũng chưa chắc đã tối ưu). Validation performance vẫn là tiêu chí cuối.
- **Mốc 3 (Chốt reference)**: Giữ nguyên toàn bộ cấu hình khác làm reference. Từ đây, mọi thay đổi sparse chỉ thay đổi `N_injection` và vị trí để isolate tác động.
- **Mốc 4 (Giai đoạn phụ - Full vs Sparse N=2)**: Chạy đúng 1 phép so sánh: Full vs N=2 (Candidate: CNN→Transformer boundary + Mid-Transformer). Mục tiêu kiểm chứng: Có thực sự cần routing ở toàn bộ layer, hay một số point có marginal value cao đã đủ giữ lợi ích SAGE?
- **Mốc 5 (Quy tắc dừng sau N=2)**: Không sweep ngay N=3/4/6.
  - N=2 ≈ Full: Point bị bỏ đi có marginal contribution nhỏ.
  - N=2 > Full: Sparse routing tập trung phù hợp hơn.
  - N=2 < Full: Full vẫn mang lại marginal benefit đáng kể.
- **Mốc 6 (Chỉ khi N=2 thắng rõ)**: Thử đúng 1 run **N=3** (N=2 + Stage1 Early-mid CNN). Stage1 quan trọng để trị "hairline crack" vì giữ local detail tốt (lưu ý: đây là giả thuyết mang tính domain-specific, không phải chân lý). Tùy kết quả N=3 so với N=2 để quyết định dừng hay đi tiếp.

**Priority hypothesis của các injection point (Nếu tăng dần):**
1. **CNN→Transformer boundary**: Ứng viên mạnh nhất do thay đổi bản chất rõ rệt giữa local và global representation.
2. **Mid-Transformer**: Point refinement ở giữa.
3. **Stage1 (early-mid CNN)**: Cân bằng giữa detail preservation (trị hairline crack) và semantic representation.
4. **Early-Transformer block**: Nằm gần điểm bắt đầu của attention.
5. **Stage2 (mid CNN)**: Lấp khoảng trống giữa Stage1 và Boundary, nhưng có thể bị bão hòa tương quan.
6. **Stage0 (very early CNN)**: Feature thô, local detail cao (cạnh tranh trực tiếp với Stage 1).
7. **Late-Transformer**: Gần decoder interface.

*(Lưu ý: Khi implement sparse injection trong code, phải map chính xác tên vị trí với tensor thực tế mà Router nhìn thấy trước khi `main_block` thực thi).*

### 8. Chiến lược Inference Tiling (Crack500)
- **Baseline (Mặc định cho Ablation):** Bắt buộc bắt đầu bằng **Non-overlapping tiling 448x448** cho toàn bộ các thí nghiệm ablation. Mục đích là để giữ pipeline inference cố định, đơn giản và chạy nhanh, giúp chu kỳ thực nghiệm không bị nghẽn ở khâu test.
- **Xử lý Boundary Artifacts:** Sau khi đã chốt được cấu hình (config) tốt nhất từ tập ablation, tiến hành kiểm tra prediction error (sai số dự đoán) dọc theo các ranh giới tile 448x448. 
- **Quy tắc kích hoạt Overlap:** Chỉ khi phát hiện xuất hiện pattern lỗi tập trung rõ rệt ở tile boundary, ta mới chạy thêm một vòng inference sử dụng **Overlap + Average blending** cho riêng config đã chọn đó.
- **Báo cáo kết quả:** Bắt buộc báo cáo song song cả kết quả Non-overlap và Overlap của config cuối cùng. Điều này cung cấp minh chứng định lượng về việc Overlap thực sự đóng góp bao nhiêu % vào hiệu năng cuối cùng, tránh ảo giác rằng model tốt lên nhờ kiến trúc trong khi thực chất là nhờ inference trick.

### 9. Đính chính Kiến thức và Quản trị Siêu tham số

* **Phần cứng & SA-Hub:** Tesla T4 thực chất có **16GB VRAM**. Adapter của SA-Hub scale theo số lượng **unique channel dimensions** (`O(D²)`) chứ không scale theo số lượng Router/injection point (`N`). Với 4 mức channel dims (48/96/192/384), tập các cặp chuyển đổi là cố định (~12 cặp có hướng; implementation tạo cả `Conv2d` và `Linear` cho mỗi cặp), nên overhead SA-Hub về mặt tham số/VRAM gần như không đổi khi thay đổi số lượng Transformer blocks hoặc `N_injection`.

* **Hiểu lầm về Gradient Accumulation:** Gradient Accumulation **KHÔNG giải quyết được nhiễu thống kê của BatchNorm**, vì BN statistics được tính ngay trong forward pass trên từng micro-batch, chứ không phải khi `optimizer.step()`. Vì vậy GA chỉ tăng effective batch size đối với gradient update; nếu physical batch vẫn nhỏ thì BatchNorm vẫn nhìn thấy batch nhỏ. Decoder hiện tại thực sự sử dụng `BatchNorm2d`.

* **Trình tự Thực nghiệm:** Probe Batch Size (vài chục giây) → LR Range Test (vài phút) → Chốt Batch Size và LR, sau đó giữ cố định các giá trị này cho toàn bộ các ablation runs phía sau.
* **Gating Function của Router:** Paper SAGE định nghĩa rõ việc sử dụng **Top-K Sigmoid Gating** (không phải Softmax). Tức là sau khi chọn Top-K, các expert được chọn sẽ nhận trọng số dựa trên sigmoid độc lập của logit, thay vì bị ép chuẩn hóa tổng bằng 1 qua softmax. Việc này đã được xác nhận ở phần supplement của paper gốc. Do đó, cấm việc đưa softmax/sigmoid thành biến ablation, mà phải khóa chết `gating_type = "sigmoid"`.

### 13.8. Kích thước Lớp ẩn Router (`router_hidden_dim`)
**`router_hidden_dim: 256 → 64 (baseline)`**
- `router_hidden_dim` là số neuron của lớp ẩn bên trong Router, quyết định độ lớn và khả năng biểu diễn của mạng dùng để tính routing scores cho các experts.
- SAGE-Lite sử dụng backbone nhỏ hơn SAGE gốc, nên giữ nguyên `router_hidden_dim=256` có thể làm Router lớn hơn mức cần thiết so với tổng thể kiến trúc Lite. Chọn `64` làm baseline nhằm giảm số tham số và chi phí tính toán của Router, đồng thời tạo một cấu hình gọn phù hợp với mục tiêu SAGE-Lite.
- 64 là lựa chọn thiết kế baseline, không phải giá trị bắt buộc suy ra từ kích thước backbone. Hiệu quả thực tế vẫn cần được xác nhận bằng thực nghiệm.
- **Các giá trị ablation tiềm năng (32 / 128 / 256)** có thể được dùng để kiểm tra ảnh hưởng của capacity Router:
  - `32`: Router rất nhỏ, ưu tiên regularization và hiệu suất.
  - `64`: Baseline cân bằng giữa capacity và độ gọn.
  - `128`: Tăng capacity để kiểm tra trường hợp 64 chưa đủ khả năng biểu diễn routing.
  - `256`: Cấu hình lớn, dùng làm mốc đối chứng với SAGE gốc.

### 13.9. Trạng thái Đóng băng Backbone (`freeze_encoder` / `freeze_transformer`)
**`freeze_encoder=False`, `freeze_transformer=False` → Giữ nguyên mặc định**
- `False/False` là baseline phù hợp cho SAGE-Lite: backbone pretrained vẫn được fine-tune thay vì freeze hoàn toàn.
- Thay vì đóng băng cứng, hệ thống sẽ sử dụng mức Learning Rate thấp hơn cho backbone thông qua chiến lược differential LR.
- Thiết kế này cho phép backbone thích nghi nhẹ với domain crack (vết nứt) mà vẫn hạn chế làm mất các pretrained features giá trị.

### 13.10. Dropout của Chuyên gia (`expert_dropout`)
**`expert_dropout = 0.1` → Giữ nguyên, hợp lý**
- Đây là hyperparameter regularization độc lập với kích thước backbone, do đó không cần scale theo backbone size (như `router_hidden_dim`).
- Với dataset tương đối nhỏ, `0.1` được giữ làm baseline hợp lý để hạn chế overfitting bên trong các SA-Hub Adapter.
- Chưa cần thay đổi hay tạo ablation trước khi có kết quả thực nghiệm.

### 13.11. Hệ số Phân bằng Tải (`load_balance_factor`)
**`load_balance_factor = 0.01` → Giữ nguyên baseline, cần theo dõi**
- **Vị trí:** Tham số nội bộ của `SageRouter`, được sử dụng trong `compute_load_balance_loss()`.
- **Source code:** Loss được tính theo dạng Switch-style:
  ```python
  load_balance_loss = self.expert_pool_size * (f * P).sum()
  return self.load_balance_factor * load_balance_loss
  ```
- Đây là hệ số scale nội bộ của `L_balance`, không phụ thuộc trực tiếp vào kích thước backbone.
- Source code mặc định `0.01` và áp dụng sau khi tính `M * Σ(f_i P_i)`.
- **SAGE-Lite:** Do Pool của mô hình Lite nhỏ hơn nhiều so với SAGE gốc (28 experts), giá trị của hàm Loss `M * (f * P).sum()` sẽ bị giảm đi theo tỷ lệ thuận. Điều này khiến mạng dễ bề gặp expert collapse hơn ở cùng một mức load-balancing pressure. Tuy nhiên, vẫn giữ `0.01` làm baseline nguyên bản, không tự ý tăng trước khi có dữ liệu thật. Nếu qua theo dõi (`expert_usage_ratio`) cho thấy dấu hiệu collapse sớm (routing dồn cục bộ vào 1-2 experts), lúc đó mới cân nhắc tăng hệ số `load_balance_factor` trong router config lên mức `0.03-0.05`.
- (Nhắc lại: `gs_tracker` chỉ theo dõi `g_s`; expert collapse phải xem qua **`expert_usage_ratio` / routing statistics**).

### 13.12. Phân bổ Epoch (Stage 1 vs Stage 2)
**Không kế thừa 100/300 của SAGE gốc → Quyết định dựa trên Benchmark throughput**
- Không kế thừa tỷ lệ epoch `100/300` của SAGE-GlaS.
- Tổng số epoch mục tiêu hiện tại dự kiến khoảng 15-30 (phù hợp với compute budget ngắn hạn), nhưng **chưa chốt chính thức**.
- Cần thực hiện benchmark Batch Size và Throughput (tốc độ huấn luyện) thực tế trên GPU T4 trước. Dựa vào tốc độ thực tế đó, chúng ta mới tính toán và quyết định tổng số Epoch tối ưu cũng như tỷ lệ chia cho Stage 1 / Stage 2.

## 15. Giao thức Đánh giá (Evaluation Protocol) - Lý do & Quyết định

- **Chốt Per-sample Foreground Dice làm kim chỉ nam:** Bài toán Crack Segmentation chỉ quan tâm duy nhất đến pixel nứt. Nếu dùng `mean_dice` (tính trung bình cả class background như SAGE gốc), điểm số sẽ luôn cao giả tạo (hơn 90%) do background chiếm đa số tuyệt đối, che khuất hoàn toàn sự cải thiện/suy giảm trên nét nứt thật sự. Chọn Checkpoint bắt buộc phải dùng per-sample Foreground Dice.
- **Tách bạch Global IoU và Crack-Present IoU (Secondary):** 
  - *Global IoU* gom tổng Intersection và Union trên toàn Dataset trước khi chia. Nó phản ánh chính xác bức tranh tổng thể vì không bị thiên lệch bởi các patch có kích thước vết nứt bé.
  - *Crack-Present IoU* ép hệ thống chỉ đánh giá trên ảnh có nứt thật sự (`GT > 0`). Nó chống lại việc các ảnh thuần Background 100% "ăn gian" đẩy mean IoU lên cao, giúp đo lường sức mạnh thuần túy của bộ giải mã (decoder) khi đối diện với vật thể.
- **Đóng băng Validation Data (Deterministic):** Validate trên Crack500 dùng Non-overlap 448x448 cố định và cấm tuyệt đối RandomCrop. Bất kỳ sự xê dịch ngẫu nhiên nào của Seed cũng làm nhiễu điểm số giữa các ablation. 
- **Baseline Ladder (Thang đo B0->B2):** Tách bạch rõ ràng sự đóng góp của từng thành phần kiến trúc:
  - **B0**: ConvNeXtV2-Femto + U-Net (Pure CNN baseline).
  - **B1**: B0 + 6 ViT-Tiny (ConvNeXtV2-Femto + 6 ViT-Tiny + U-Net thuần, không có cơ chế SAGE). Đo đóng góp thuần túy của Attention / Global Context.
  - **B2**: SAGE-Lite hoàn chỉnh (B1 + SA-Hub + Router + SAGE Injection). Đo đóng góp trọn vẹn của cơ chế Shape-Adapting Gated Experts.
- **Tại sao bỏ qua g_s Tracker:** Lấy thống kê trực tiếp từ **Expert Usage Ratio** phản ánh sự thật về routing behavior (Expert nào nằm trong Top-K), thay vì phụ thuộc vào tham số nội bộ `g_s` (chỉ theo dõi gap logit).
- **Vì sao 1 metric để chọn Checkpoint là đủ, không phải "ít"?** 
  - *Quan hệ toán học trực tiếp*: Dice và IoU không độc lập. `IoU = Dice / (2 - Dice)`, nghĩa là chúng là hàm đơn điệu tăng của nhau. Việc chọn checkpoint theo Dice cao nhất gần như chắc chắn sẽ có IoU cao nhất (hoặc rất gần). Do đó, nhồi nhét cả 2 để chọn checkpoint không mang lại thêm thông tin độc lập mà chỉ gây trùng lặp.
  - *Sự nhất quán với Loss*: Loss training đã có `1.5 * Soft Dice` làm thành phần chính. Việc dùng Dice để chọn checkpoint là nhất quán 100% với chính mục tiêu tối ưu, chứ không phải chọn ngẫu nhiên theo một tiêu chí rời rạc mà mô hình không được huấn luyện để nhắm tới.
- **Rủi ro thực sự (Tại sao lại cần Boundary IoU để Diagnostic):** Rủi ro không nằm ở "số lượng metric" mà ở "loại metric". Cả Dice và IoU đều đo *area overlap* (diện tích chồng lấn), chúng mù lòa trước *connectivity/topology* (tính liên kết/cấu trúc mạng lưới). Với đối tượng cấu trúc mỏng, dài như vết nứt, hoàn toàn có khả năng: 2 checkpoint có Dice bằng nhau (vì tổng diện tích đoán trúng như nhau), nhưng một cái cho ra nét đứt đoạn, một cái cho ra nét liền mạch. Dice và IoU thuần túy không thể phân biệt được lỗi đứt đoạn này. Đây chính là lý do ta phải sử dụng **Boundary IoU** ở khâu đánh giá cuối cùng — nó nhạy bén với hình thái và độ cong của biên hơn rất nhiều so với Dice/IoU thuần, đóng vai trò "khám bệnh" những lỗi mà area-metrics bỏ sót.

## 16. Khóa cứng Preprocessing Canonical (Ngày 2026-09-22)
> ⚠️ **QUYẾT ĐỊNH BẤT BIẾN:** Từ ngày 22-09-2026, toàn bộ pipeline tiền xử lý và augmentations đã được **CHỐT CỨNG (FROZEN)** làm chuẩn mực duy nhất xuyên suốt cho đến khi hoàn thành SAGE-Lite. Cấm tuyệt đối việc tự ý thêm/bớt/sửa các bước tiền xử lý trong mã nguồn và config, trừ khi có yêu cầu làm ablation hoặc sửa protocol cụ thể từ người dùng.
- **Crack500 Train:** Reflect Pad → RandomCrop 448 → smart filter `fg_pixels >= 20` (max 20 crop attempts × 10 source resamples) → Augmentation: `HorizontalFlip`, `VerticalFlip`, `RandomRotate90`, `RandomBrightnessContrast`, `GaussianBlur` → ImageNet Normalize → `ToTensorV2`.
- **Crack500 Eval:** Tiling Setting A (448×448 non-overlap) & Setting B (448×448 overlap 50%, stride 224, average probabilities → threshold 0.5).
- **DeepCrack Train & Eval:** Mask nhị phân `{0, 255} → {0, 1}` → Dynamic Pad-to-Square (`target = max(H, W, 448)`, BORDER_CONSTANT=0) → `Resize(448, 448)` với `cv2.INTER_NEAREST` mask → cùng bộ Augmentation trên (chỉ train) → ImageNet Normalize → `ToTensorV2`. Eval chạy direct 1-pass full-image.
- **Loại bỏ vĩnh viễn:** `CLAHE`, `ElasticTransform`, `GridDistortion`, `ShiftScaleRotate`, `HueSaturationValue`.

## 17. Kết quả B1 và Quyết định ViT-Depth Ablation trước B2 (Ngày 2026-09-22)
1. **Kết quả B1 Crack500 (6 blocks ViT-Tiny):**
   - Huấn luyện: Best Val Dice = **0.7420** (@ epoch 21, Val Loss = 1.1453), EarlyStopping @ 27.
   - Official Eval Test: Setting A Dice = **0.6857**, Setting B Dice = **0.6895** (HD95 giảm mạnh từ 94.37 px xuống **77.43 px** so với B0).
   - Kết luận: **B1 hoàn toàn không underperform B0**. Mức cải thiện là nhất quán trên 100% metrics nhưng ở mức vừa phải (~0.009 Dice).
2. **Quyết định Chiến lược: Chưa chuyển sang B2 vội:**
   - Số lượng ViT blocks quyết định trực tiếp số lượng router / SAGE injection points trong B2 (`N_injection = 4 + num_transformer_layers`).
   - Nếu sweep ở B2 sẽ thay đổi đồng thời cả backbone depth lẫn routing points, làm mất khả năng cô lập biến số và tốn kém tài nguyên.
   - Tiến hành **ViT-Depth Ablation trên B1** (0, 3, 6, 12 blocks) để chọn depth tối ưu dựa trên tập **VALIDATION** (tuyệt đối không dùng TEST để chọn).
   - Sau khi chốt depth ở B1, khóa cấu hình B1 rồi mới wire sang **B2 (Full SAGE-Lite)**.

