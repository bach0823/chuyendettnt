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

## 18. Cập nhật Định hướng Kiến trúc SAGE-Lite B2: Nút Thắt High-Resolution CNN→ViT (Ngày 2026-09-24)

Sau khi thẩm định và đánh giá độc lập điểm nghẽn tính toán của các cuộc gọi CNN Stage 0/1 $\to$ ViT expert ($N=12,544$ và $N=3,136$ tokens, tiêu tốn 34 ms/call và chiếm 62.6% forward pass), quyết định định hướng kiến trúc chính thức được xác lập:

### Thứ Tự Ưu Tiên Chiến Lược
1. **PRIMARY BASELINE $\to$ Proposal 3: Feature/Detail Enhancement $\to$ Spatial Compression**
   - *Quy trình*: CNN Stage 0/1 feature $\to$ lightweight learnable detail refinement $\to$ spatial compression $\to$ pretrained ViT global attention $\to$ SA-Hub adapt về CNN shape $\to$ residual fusion với main path.
   - *Nguyên tắc*: Chỉ áp dụng trên nhánh expert. Main CNN path ($112 \times 112$) và UNet skip connections giữ nguyên vẹn 100%. ViT block giữ nguyên dạng black-box pretrained chuẩn từ `timm`, không can thiệp attention internals. Định tuyến đa phương thức được bảo toàn trọn vẹn.
2. **SECOND $\to$ Proposal 1: Spatial Reduction Attention (SRA)**
   - $Q$ giữ full spatial resolution ($112 \times 112 = 12,544$), $K, V$ được nén không gian (ví dụ $28 \times 28 = 784$). Attention toàn cục trên tập K/V nén.
   - Cần sửa attention bên trong ViT block, tái sử dụng pretrained Q/K/V projections và xử lý an toàn cơ chế shared expert giữa Bottleneck ($N=196$) và high-res calls ($N=12,544$).
3. **THIRD $\to$ Proposal 2: Restricted High-Resolution Routing**
   - Stage 0/1 chỉ được route tới CNN experts. Stage 2/3 và ViT layers giữ full 16 experts.
   - Bắt buộc phải giải quyết bài toán $top\_k=4$ trên tập chỉ có 4 CNN experts trước khi code (nguy cơ forced-selection và sụp đổ entropy của router).

### Các Nguyên Tắc Kiến Trúc Bắt Buộc
- **Decoupled Roles**: Main CNN path là nguồn cung cấp chi tiết nứt độ phân giải cao; expert branch đóng vai trò context booster cộng dồn, không thay thế main path.
- **Không quy chụp**: Không được mặc định attention trên $N=12,544$ sẽ tự động biến thành uniform distribution.
- **Tách bạch 4 loại tổn thất**: Phân biệt rành mạch giữa mất resolution, mất global context, mất cross-modal interaction, và nhòe do interpolation.
- **Kích thước không gian chuẩn xác**: Stage 0 = $112 \times 112$ ($12,544$), Stage 1 = $56 \times 56$ ($3,136$), Stage 2 = $28 \times 28$ ($784$), Stage 3 / ViT = $14 \times 14$ ($196$).
- **Microbenchmark != Guaranteed Speedup**: Các phép ngoại suy thời gian epoch chỉ là ước lượng tham chiếu, không được xem là hiệu năng end-to-end bảo đảm.

### ⚠️ Bất Biến Hướng Định Tuyến Cốt Lõi (Core Routing Direction Invariant)
Bất kỳ xử lý tối ưu hóa đặc biệt nào cho nhánh CNN high-resolution $\to$ ViT **CHỈ ĐƯỢC PHÉP ÁP DỤNG KHI THỎA MÃN ĐỒNG THỜI CẢ 2 ĐIỀU KIỆN**:
1. `source ∈ {CNN Stage 0, CNN Stage 1}`
2. `target ∈ {ViT / Transformer experts}`

**Mọi hướng định tuyến khác BẮT BUỘC giữ nguyên hành vi SAGE chuẩn (Normal SAGE)**:
- `CNN S0/S1 → CNN expert` $\implies$ Normal SAGE.
- `CNN S2/S3 → CNN expert` $\implies$ Normal SAGE.
- `CNN S2/S3 → ViT expert` $\implies$ Normal SAGE.
- `ViT → CNN S0/S1` $\implies$ **Normal SAGE 100%** (ViT xuất phát từ bottleneck $N=196$, không tạo ra nút thắt attention lớn).
- `ViT → CNN S2/S3` $\implies$ Normal SAGE.
- `ViT → ViT` $\implies$ Normal SAGE.

**Yêu cầu kiểm tra khi code**:
Bắt buộc kiểm tra đồng thời:
`if source_is_cnn_stage_0_or_1 and target_is_transformer_expert:`
Tuyệt đối không kiểm tra đơn lẻ số chiều tensor, số channels, hoặc chỉ số expert.

### Quy Trình Tiếp Theo
- **TUYỆT ĐỐI KHÔNG CODE VÀO THỜI ĐIỂM NÀY**.
- Báo cáo đầy đủ đề xuất cụ thể cho P3 (module enhancement, target compression, bảo toàn interface SAGE, minimal ablation) để phê duyệt trước khi hiện thực hóa.

---

## Nghiệm Thu Thực Nghiệm & Đính Chính Giả Thuyết: GAP Routing-Signal Bottleneck (Crack500 Val)

*Ngày ghi nhận: 2026-09-28*  
*Mô hình: Canonical D4 P3-C (ASDW) Standalone Model (8 experts, top-k=4)*  
*Dữ liệu thực nghiệm: 348 mẫu Validation Crack500*  
*Tệp kết quả gốc: `results/diagnostics/gap_study/GAP_ROUTING_STUDY_REPORT.md`*

Kết quả này chỉnh lại hai điều: câu chuyện "GAP làm mất tín hiệu" của report trước, và cả nhận định "gần như ensemble tĩnh" mà tôi đã nói.

Trước hết, các con số khớp chéo với những gì tôi đã thấy: 61/348 = 17.53%, `g_s` trung bình của từng router trùng `gs_statistics.csv`, và Δ tính lại từ `g_s` cho ra -0.151 ở S1 và +0.103 ở S2. Tôi không mở được file trên máy bạn nên chưa tự kiểm được phần còn lại.

### Những gì bị bác bỏ hoặc phải rút lại

- **"Vết nứt <1% diện tích":** sai. Trung vị là 4.2%, chỉ 17.5% ảnh dưới 1%, 80% dưới 10%. Vết nứt nhỏ nhưng không nhỏ đến mức GAP xóa sạch.
- **"Tín hiệu gần như bằng 0 sau GAP":** vector đã pool dự đoán được `gt_area` với R² 0.79 đến 0.85 từ S3 đến B3. Thông tin về diện tích vẫn còn nguyên.
- **"Routing gần ensemble tĩnh" (của tôi):** con số 35.4% so với 36.0% là trung bình trên cả 8 router, nên các dịch chuyển ngược chiều triệt tiêu nhau. Theo diện tích vết nứt, B3 chuyển từ 57.2% CNN (Q1) xuống 7.2% (Q4), còn B1 chuyển ngược từ 31.0% lên 59.5%. Các router ViT sâu có thích nghi theo đầu vào, chỉ là không theo hướng tôi tưởng.
- **"SAGE gốc định tuyến theo token":** router gốc dùng cùng `AdaptiveAvgPool2d` và `mean(dim=1)`, cùng vị trí dòng. GAP là thứ kế thừa, không phải sự khác biệt thiết kế.

### Những gì còn đứng vững

- **Router theo dõi diện tích, không theo dõi độ mảnh.** R² của `thinness` chỉ tối đa 0.20 và âm ở B2, B3 (-0.18, -0.16), tức là dự đoán kém hơn dùng giá trị trung bình. Điều này chỉ nói về đọc tuyến tính, nhưng độ mảnh mới là biến liên quan chặt hơn đến ca khó (bảng quartile của bạn cho thấy Dice tụt từ 0.85 xuống 0.66). Diện tích thì cũng liên quan đến độ khó nhưng yếu hơn.
- **Cổng `g_s` gần như bất động:** độ lệch chuẩn của Δ chỉ 0.004 đến 0.012. Cơ chế điều biến phân cấp không tạo ra khác biệt. Phần thích nghi thực tế đến từ SAR logits.
- **Chưa biết lựa chọn thích nghi có giúp Dice hay không.** Hoán đổi routing tĩnh khi suy luận vẫn là phép kiểm tra quyết định.

### Hai lưu ý khi đọc bảng

1. Mỗi quartile khoảng 87 ảnh, và JSD giữa hai nhóm nhỏ có sàn nhiễu dương. Các giá trị dưới khoảng 0.02 (S0, S1) có thể chỉ là nhiễu. Nên chạy hoán vị nhãn quartile 200 lần để có ngưỡng. Các giá trị như 0.22 (B1) và 0.38 (B3) thì rõ ràng vượt nhiễu.
2. Base logits dao động rất ít (độ lệch chuẩn 0.014 đến 0.087), tức các expert gần như hòa nhau. Khi đó một thay đổi nhỏ của đầu vào cũng đảo được top-4, nên việc lựa chọn thay đổi theo diện tích chưa chắc là "định tuyến có nghĩa". Cũng cần nhớ khi huấn luyện router cộng nhiễu khám phá `randn × softplus(noise_projection(x))`. Nếu `noise_projection` giữ gần khởi tạo thì `softplus ≈ 0.69`, lớn hơn nhiều so với tín hiệu 0.05, nghĩa là lúc train việc chọn expert gần như ngẫu nhiên, còn lúc đánh giá thì tất định. Đây là giả thuyết, kiểm tra được trong vài phút bằng cách in giá trị trung bình của `softplus(noise_projection(x))` trên checkpoint.

   *Kết quả kiểm chứng thực nghiệm trực tiếp trên checkpoint D4 Canonical (`results/checkpoints/P3_C_D4_best_model_b2_global.pth`) với toàn bộ 348 mẫu validation:*
   - **S0** (ConvNeXt Stage 0): `softplus` mean = **0.5423** (std = 0.0622, min = 0.4060, max = 0.8068) — Gần mức khởi tạo, nhiễu lớn.
   - **S1** (ConvNeXt Stage 1): `softplus` mean = **0.4030** (std = 0.1700, min = 0.1174, max = 0.8836).
   - **S2** (ConvNeXt Stage 2): `softplus` mean = **0.2907** (std = 0.1514, min = 0.1235, max = 0.7699).
   - **S3** (ConvNeXt Stage 3): `softplus` mean = **0.0980** (std = 0.0480, min = 0.0332, max = 0.2588) — Bắt đầu triệt tiêu nhiễu mạnh (< 0.10).
   - **B0** (ViT Block 0): `softplus` mean = **0.0796** (std = 0.0173, min = 0.0440, max = 0.1591).
   - **B1** (ViT Block 1): `softplus` mean = **0.0441** (std = 0.0104, min = 0.0272, max = 0.0769) — **Nhiễu cực bé (~0.04)**, nhỏ hơn độ biến thiên base logits (0.053).
   - **B2** (ViT Block 2): `softplus` mean = **0.0550** (std = 0.0171, min = 0.0216, max = 0.1276).
   - **B3** (ViT Block 3): `softplus` mean = **0.0388** (std = 0.0123, min = 0.0162, max = 0.1024) — **Nhiễu cực bé (~0.038)**, nhỏ hơn độ biến thiên base logits (0.087).

   *Ý nghĩa*: Đúng như giả thuyết, ở các tầng CNN nông (S0–S2), router duy trì nhiễu cao lúc train. Nhưng ở các tầng sâu (S3, B0–B3) — đặc biệt là B1 và B3 nơi có sự chuyển dịch routing mạnh nhất theo diện tích (B1 JSD = 0.22, B3 JSD = 0.38) — mạng đã **tự động học cách ức chế nhiễu khám phá (noise suppression)** xuống chỉ còn ~0.038 – 0.044, nhỏ hơn cả biên độ tín hiệu base logits. Điều này cho thấy sự thích nghi ở các tầng sâu lúc train không hoàn toàn bị nhiễu xóa nhòa.

### Quyết định

Chưa có cơ sở để thay GAP. Bằng chứng nghiêng về việc GAP giữ được thông tin về diện tích, còn độ mảnh thì khó đọc tuyến tính nhưng chưa biết có giúp ích không.

---

## Đối Chiếu Hệ Thống với SAGE Gốc (Mã Nguồn & Figure 9 SAGE Paper)

*Ngày bổ sung: 2026-09-28*

Dựa trên việc kiểm tra trực tiếp mã nguồn SAGE gốc (`SAGE/sage/components/router.py`) và phân tích biểu đồ tiến hóa gating $g_s$ qua huấn luyện (Figure 9 của paper SAGE gốc):

### 1. Thẩm định Mã Nguồn Router SAGE Gốc vs SAGE-Lite
- **Cơ chế Pooling**: SAGE gốc và SAGE-Lite sử dụng **chính xác cùng một cơ chế** ở cấp mã nguồn:
  - CNN layers: `self.feature_aggregator = nn.AdaptiveAvgPool2d((1, 1))` (dòng 106), sau đó trích xuất `aggregated = self.feature_aggregator(x).squeeze(-1).squeeze(-1)` (dòng 189).
  - ViT layers: `aggregated = x.mean(dim=1)` (dòng 191).
  - Cổng $g_s$ và query SAR đều tính từ vector đã pool này (`g_s = torch.sigmoid(self.shared_expert_gate(aggregated))` và `query = self.query_projection(aggregated)`).
- **Kết luận kiến trúc**: GAP không phải là sự sai lệch hay thiếu sót khi chuyển thể sang SAGE-Lite, mà là sự kế thừa nguyên vẹn từ thiết kế gốc của tác giả.

### 2. So sánh Hành vi Động của Cổng $g_s$ (SAGE Gốc Fig. 9 vs SAGE-Lite Test 5)
- **SAGE gốc (Hình 9a, 9b, 9c)**:
  - *Phân phối ban đầu vs Cuối cùng (Fig 9a)*: $g_s$ trải rộng (spread) rõ rệt từ dạng hẹp ban đầu (~0.5) sang phân phối 2 thung lũng rõ ràng ở cuối quá trình train, mở rộng mạnh về cả hai phía (<0.4 và >0.6).
  - *Độ lệch theo tầng (Fig 9c)*: Tầng CNN duy trì $g_s$ cao ổn định (~0.60–0.70, thiên về shared), trong khi tầng Transformer dao động rất mạnh qua các epoch (0.35 đến 0.65), thể hiện tính biến thiên lớn.
- **SAGE-Lite (D4 Crack500 - Test 5)**:
  - Biên độ điều biến $\Delta_{\text{family}} = \ln(g_s / (1 - g_s))$ có độ lệch chuẩn per-sample cực kỳ hẹp (**$\text{std} = 0.004 \to 0.012$**).
  - Giá trị $g_s$ thực tế bị co cụm (ví dụ S1 dao động quanh 0.462, S2 quanh 0.526, B1-B3 quanh 0.501–0.504), không học được độ mở rộng/phân tách biên độ lớn như SAGE gốc.
- **Nguyên nhân**: Sự khác biệt hành vi không nằm ở phương trình hay hàm pooling (vì code y hệt nhau), mà nằm ở:
  1. *Đặc trưng miền bài toán*: WSI y tế có sự đa dạng mô học (cellular heterogeneity) lớn hơn nhiều so với binary crack segmentation.
  2. *Training regime*: SAGE gốc train 60 epochs (có thời gian cho gate mở rộng trọng số) so với ngân sách epoch ngắn (~15–30 epochs) của SAGE-Lite.

### 3. Tóm Lược Tình Trạng Định Tuyến SAGE-Lite Hiện Tại
- **Không phải static ensemble toàn phần**: SAGE-Lite ở các tầng ViT sâu (đặc biệt B1 và B3) có sự thích nghi định tuyến rất rõ nét theo quy mô diện tích vết nứt ($gt\_area$, JSD đạt tới 0.38 bits; B3 chuyển từ 57% CNN ở vết nứt nhỏ xuống 7% CNN ở vết nứt lớn).
- **Điểm mù thực sự (Blind Spot)**: Router hoàn toàn không đọc được độ mảnh ($thinness$, probe $R^2 < 0$, JSD $\approx 0$ trên mọi router), dù đây là biến tương quan âm mạnh nhất với các ca sụp đổ hiệu năng (Dice tụt).
- **Cơ chế gating $g_s$ bị tê liệt biên độ**: Tín hiệu thích nghi hiện tại được dẫn dắt chủ yếu bởi SAR Base Logits ở các tầng sâu (khi nhiễu khám phá tự triệt tiêu xuống ~0.04), trong khi hệ số $g_s$ chưa phát huy được vai trò điều biến phân cấp do biên độ quá nhỏ ($\Delta \le 0.04$).

---

## Thực Nghiệm Chẩn Đoán Đa Chiều: Biểu Diễn Thinness trong Router

*Ngày: 2026-09-28 | Mô hình: D4 P3-C Canonical | N=348 val Crack500 | Eval mode, seed=42*  
*Tệp gốc: `diagnostics/thinness_representation/THINNESS_REPRESENTATION_REPORT.md`*

### Bối cảnh & Câu hỏi

Linear probe từ pooled feature `h` cho `thinness` có R² gần-0 hoặc âm (đặc biệt B2=-0.178, B3=-0.165), trong khi `gt_area` đạt R²=0.83–0.85. Hai giả thuyết được kiểm tra:
- **H1**: Thông tin thinness tồn tại nhưng theo quan hệ phi tuyến; linear query không đọc được.
- **H2**: GAP làm mất thông tin local geometry (biên cục bộ).

### Kết quả Đa Chiều (5-fold CV, R² trên thinness)

| Router | (1) Linear Ridge | (2) Poly Deg-2 | (3) MLP D→64→1 | (4) Pre-GAP Std | (5) Pre-GAP GMP |
|--------|:---:|:---:|:---:|:---:|:---:|
| S0 (112×112) | -0.145 | +0.002 | -0.085 | **+0.231** | -0.055 |
| S1 (56×56) | +0.088 | -0.096 | +0.111 | **+0.228** | -0.143 |
| S2 | +0.122 | +0.110 | +0.104 | +0.026 | -0.192 |
| S3 | +0.049 | +0.115 | **+0.182** | -0.235 | -0.922 |
| B0 | +0.200 | +0.039 | **+0.201** | -0.699 | -1.009 |
| B1 | +0.117 | +0.049 | **+0.305** | -0.132 | -0.806 |
| B2 | -0.179 | +0.103 | **+0.286** | -0.385 | -0.556 |
| B3 | -0.165 | +0.105 | **+0.240** | -0.288 | -0.847 |

### Kết Luận Đã Được Chứng Thực

**Cả hai giả thuyết đều đúng — phân hóa rõ ràng theo độ sâu:**

**Tầng nông S0/S1 — H2 đúng (GAP xóa sổ local geometry):**
- Feature map 112×112 và 56×56 có spatial resolution cao, nhưng sau GAP, R² âm (-0.145).
- Spatial Std (σ_c = std_{u,v}(X_{c,u,v})) trên pre-GAP feature map tăng vọt lên +0.231 (S0) và +0.228 (S1).
- Nguyên nhân: Vết nứt mảnh tạo sharp localized peaks; phép tích phân phẳng của GAP (1/HW Σ X) triệt tiêu tỷ lệ biên-trên-diện tích. Spatial Std phản ánh được độ sắc nhọn cục bộ này.
- GMP thất bại âm ở cả hai tầng → max pooling không capture được geometry của thin crack theo cách spatial std làm được.

**Tầng sâu B1/B2/B3 — H1 đúng (thông tin phi tuyến trong h):**
- Linear probe: B2=-0.179, B3=-0.165 (âm).
- MLP D→64→1 (ReLU, Dropout=0.4, WD=0.01) đảo chiều hoàn toàn: B2=+0.286 (Δ+0.465), B3=+0.240 (Δ+0.405), B1=+0.305 (Δ+0.188).
- Pre-GAP Spatial Std sụp đổ ở các tầng sâu (R² âm nặng đến -1.0) → ở độ phân giải 14×14, mỗi token đại diện cho vùng 32×32 px, chi tiết hình học mảnh đã được nén vào channel representation, không còn đọc được qua spatial statistics đơn giản.
- **Cơ chế**: Thông tin thinness ở tầng sâu bị entangle phi tuyến giữa các kênh (dạng tỷ số h_a/h_b). Router hiện tại dùng `q = W_q h` và `logit = h^T w_m` — hoàn toàn tuyến tính — không giải mã được.

**Trần tuyệt đối còn khiêm tốn:**
- MLP tốt nhất chỉ đạt R²≈0.30 (B1), Spatial Std tốt nhất ≈0.23 (S0).
- So với gt_area: R²>0.82–0.85 → thinness vốn dĩ là tín hiệu yếu hơn nhiều trong backbone này.

### Chuỗi Suy Luận & Giới Hạn

```
Thinness info tồn tại (MLP decode được)     ✅ Confirmed (B1-B3)
GAP xóa sổ local geometry ở tầng nông      ✅ Confirmed (S0-S1 via Spatial Std)
          ↓
Linear query_projection không đọc được      ✅ Confirmed (linear probe âm)
          ↓
Router mù với thinness khi routing          ✅ Confirmed (JSD≈0 tất cả routers)
          ↓
Thêm nonlinear vào router → fix?            ❓ CHƯA XÁC NHẬN — cần ablation thực sự
          ↓
Routing theo thinness → tăng Dice?          ❓ CHƯA XÁC NHẬN — cần expert chuyên hóa thin crack
```

### Quyết Định Kiến Trúc

**Không thay đổi GAP hay router architecture ở giai đoạn này.** Diagnostic xác nhận bottleneck nhưng chưa chứng minh fix sẽ tăng Dice downstream. Post-hoc probe R² cao ≠ routing utility. Cần ablation thực sự (train lại với intervention) để validate benefit.

Nếu muốn investigate tiếp: bước rẻ nhất tiếp theo là **static routing swap inference test** (fixed routing không adaptive vs routing hiện tại, so Dice) để tách contribution của routing khỏi ensemble quality.

---

## Nghiệm Thu Kết Quả P3-C D4 K2 (Top-k = 2 Capacity Screening)

*Ngày: 2026-09-28 | Mô hình: B2 P3-C D4 K2 Standalone (top_k=2, 8 experts pool: 4 CNN + 4 ViT) | Ngân sách: 35 Epochs (17 S1 + 18 S2) | Crack500 Val 348 mẫu, Setting A*  
*Tệp lưu trữ: `results/P3_C_Canonical_Base_D4_K2_Full.zip`, Checkpoints: `results/checkpoints/P3_C_D4_K2_*`, Log: `results/logs/P3_C_Canonical_Base_D4_K2.log`*

### 1. Diễn Biến Huấn Luyện 2 Giai Đoạn
- **Stage 1 (Frozen Backbone, 17 eps)**:
  - Đạt đỉnh tại **Epoch 13** với Val Dice = **0.7304**, Val Loss = `1.2559`.
  - So với D4 K4 (Stage 1 peak = 0.7295 ở Ep 14): D4 K2 đạt hiệu năng Stage 1 tương đương, thậm chí nhỉnh hơn nhẹ (+0.0009).
- **Stage 2 (Joint Training, 18 eps)**:
  - Đạt đỉnh toàn cục tại **Epoch 16** với Val Dice = **0.7618**, Val Loss = `0.9518`.
  - Epoch 17: Val Dice = 0.7571. Epoch 18 (sàn LR 1e-6): Val Dice = 0.7594, Val Loss = 0.9504.

### 2. So Sánh Phân Phối Lỗi & Dung Lượng Kích Hoạt (D4 K4 vs D4 K2)

| Chỉ số / Đặc trưng | P3-C D4 K4 (Baseline) | P3-C D4 K2 ($k=2$) | Chênh lệch ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Số expert chọn mỗi router ($k/M$)** | $4 / 8$ (50% pool) | $2 / 8$ (25% pool) | Giảm 50% expert kích hoạt |
| **Peak Stage 1 Val Dice** | 0.7295 (Ep 14) | **0.7304** (Ep 13) | +0.0009 |
| **Peak Stage 2 Val Dice (Global)** | **0.7639** (Ep 14) | **0.7618** (Ep 16) | **-0.0021** (-0.21%) |
| **Val Loss @ Global Peak** | 0.9533 | **0.9518** | -0.0015 |
| **Mean IoU** | **0.6412** | 0.6386 | -0.0026 (-0.26%) |
| **Median Dice** | **0.8066** | 0.8058 | -0.0008 |
| **Mean Precision** | 0.7298 | 0.7216 | -0.0082 |
| **Mean Recall** | 0.8498 | **0.8573** | +0.0075 |

*Nhận định*: Giảm $top\_k$ từ 4 xuống 2 giúp giảm một nửa số lượng chuyên gia cần tính toán trên mỗi tầng router, trong khi hiệu năng phân đoạn hầu như được bảo toàn trọn vẹn (chỉ giảm khiêm tốn **0.21% Dice** và **0.26% IoU**, với Loss hội tụ tương đương).

---

## Nghiên Cứu Can Thiệp Định Tuyến (Routing Intervention Diagnostic)

*Ngày: 2026-09-28 | Mô hình: Canonical D4-P3-C Standalone Checkpoint (`best_model_b2_global.pth`, Ep 14, Val Dice: 0.7639)*  
*Giao thức: Inference Only (Zero Training, Zero Arch Modification), Setting A, 348 mẫu Crack500 Val split*  
*Tệp kết quả: `results/diagnostics/routing_intervention/` (Commit `63cfbd7` trên `SpecialSubjectTTNT`, `b0331a3` trên `SAGE_LITE`)*

### 1. Câu Hỏi Nghiên Cứu Cốt Tử
$$\text{Adaptive Routing (Định tuyến Thích nghi Mẫu-theo-Mẫu)} \longrightarrow \text{Downstream Segmentation Utility?}$$

Để tách bạch triệt để giữa **chất lượng của tập hợp chuyên gia (Ensemble Quality)** và **lợi ích của chính sách chọn động (Dynamic Routing Utility)**, nghiên cứu can thiệp không xâm lấn thông qua forward hook trên toàn bộ 8 router, so sánh 3 chế độ:
1. **Adaptive Baseline**: Router gốc, deterministic inference (`eval()` mode, noise OFF).
2. **Static Top-4**: Cố định 4 expert phổ biến nhất tại mỗi router dựa trên tần suất chọn tích lũy từ checkpoint (`static_expert_policy.json`).
3. **Random Top-4 (10 seeds: 42..51)**: Mỗi mẫu / mỗi router chọn ngẫu nhiên đồng đều không lặp 4/8 expert.
*Bất biến trọng số gating*: Trọng số kích hoạt của các expert được chọn luôn được tính từ chính `modulated_logits` của mô hình thông qua $g = \sigma(\operatorname{gather}(\text{modulated\_logits}, \text{selected\_indices}))$, đảm bảo khác biệt chỉ đến từ chính sách chọn tập expert.

### 2. Kết Quả Tổng Thể (Overall Metrics)

| Chế độ | Mean Dice ± Std | Median Dice | Mean IoU | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Adaptive Baseline** | **0.7639** ± 0.1640 | 0.8049 | **0.6412** | 0.7298 | 0.8498 |
| **Static Top-4** | **0.7636** ± 0.1648 | **0.8074** | **0.6409** | 0.7291 | 0.8506 |
| **Random Top-4 (10 seeds)** | **0.7633** ± 0.0004 | 0.8057 | **0.6404** | 0.7277 | 0.8517 |

### 3. Kiểm Định Thống Kê Theo Cặp (Paired Statistical Tests, N = 348)

| Phép so sánh | Mean $\Delta$ | Median $\Delta$ | 95% Confidence Interval | Paired $t$-test ($p$-value) | Wilcoxon signed-rank ($p$-value) | Win Rate (Wins / Ties / Losses) | Ý nghĩa ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Adaptive vs. Static** | **+0.00037** | +0.00021 | `[-0.00051, +0.00124]` | $t = 0.8192$ ($p = \mathbf{0.4133}$) | $W = 27942.0$ ($p = \mathbf{0.3053}$) | **54.31%** (189 / 4 / 155) | ❌ **KHÔNG CÓ Ý NGHĨA** |
| **Adaptive vs. Random** | **+0.00063** | +0.00036 | `[+0.00005, +0.00121]` | $t = 2.1341$ ($p = \mathbf{0.0335}$) | $W = 22814.0$ ($p = \mathbf{0.00011}$) | **59.77%** (208 / 2 / 138) | ✅ **CÓ Ý NGHĨA** (Effect size siêu bé) |
| **Static vs. Random** | **+0.00026** | +0.00017 | `[-0.00057, +0.00110]` | $t = 0.6179$ ($p = \mathbf{0.5370}$) | $W = 25539.0$ ($p = \mathbf{0.0162}$) | **54.60%** (190 / 2 / 156) | ⚠️ Chỉ có ý nghĩa trên Wilcoxon |

### 4. Phân Tích Phân Vị Độ Mảnh ($Q1..Q4$)

Khóa cứng phân vị độ mảnh ground truth: $Q1 \le 0.083 < Q2 \le 0.124 < Q3 \le 0.176 < Q4$.

| Chế độ | Q1 (Vết nứt thô nhất) | Q2 | Q3 | Q4 (Vết nứt mảnh nhất) |
| :--- | :---: | :---: | :---: | :---: |
| **Adaptive Baseline** | 0.8473 | 0.8005 | **0.7443** | **0.6637** |
| **Static Top-4** | **0.8474** | **0.8006** | 0.7427 | 0.6636 |
| **Random Top-4 (10 seeds)** | 0.8472 ± 0.0004 | 0.8000 ± 0.0006 | 0.7437 ± 0.0010 | 0.6623 ± 0.0005 |
| **Chênh lệch (Adaptive - Static)** | *-0.0001* | *-0.0001* | *+0.0016* | *+0.0001* |

### 5. Can Thiệp Theo Tầng Router (Router-wise Depth Ablation)

| Cấu hình | Router bị ngẫu nhiên hóa | Router giữ Adaptive | Mean Dice ± Std | $\Delta$ vs Adaptive | Mean IoU |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **All_Random** | Toàn bộ 8 router (S0..B3) | Không có | 0.7634 ± 0.0004 | -0.00051 | 0.6405 |
| **Shallow_CNN_Only** | S0, S1, S2 (3 tầng CNN nông) | S3, B0, B1, B2, B3 | 0.7632 ± 0.0004 | -0.00070 | 0.6403 |
| **All_CNN_Only** | S0, S1, S2, S3 (Toàn bộ 4 CNN) | B0, B1, B2, B3 (4 ViT) | 0.7632 ± 0.0005 | -0.00072 | 0.6402 |
| **Deep_ViT_Only** | B0, B1, B2, B3 (Toàn bộ 4 ViT) | S0, S1, S2, S3 (4 CNN) | **0.7643** ± 0.0001 | **+0.00036** | **0.6417** |

---

### 6. Tổng Kết Khoa Học & Quyết Định Chiến Lược

1. **Thực tế về Adaptive Routing**:
   - Adaptive routing của SAGE-Lite **hoàn toàn không mang lại lợi thế vượt trội so với một ensemble tĩnh tối ưu (Static Top-4)** trên tập dữ liệu vết nứt Crack500 ($p = 0.4133 > 0.05$, chênh lệch $\Delta = +0.00037$ nằm trọn trong khoảng tin cậy chứa 0).
   - Lợi ích của mạng chủ yếu đến từ **sự hiện diện của đa dạng chuyên gia (Heterogeneous Expert Pool)** kết hợp với việc **tính toán trọng số cổng liên tục $\sigma(\text{modulated\_logits})$** thay vì cơ chế lựa chọn rời rạc từng mẫu.
2. **Không có sự chuyên hóa vết nứt mảnh**:
   - Ở phân vị vết nứt mảnh nhất ($Q4$), Adaptive Dice (0.6637) và Static Dice (0.6636) chỉ lệch nhau đúng 0.0001 (0.01% Dice). Điều này xác nhận kết luận từ nghiên cứu thinness representation trước đó: router hoàn toàn "mù" với độ mảnh và không hề điều hướng chuyên gia để giải cứu các ca nứt mảnh.
3. **Độ dư thừa ở tầng sâu (Deep ViT Redundancy)**:
   - Khi chọn ngẫu nhiên expert ở 4 tầng ViT sâu (`Deep_ViT_Only`), Dice đạt **0.7643** (cao hơn cả Adaptive Baseline 0.7639). Các ViT experts ở tầng sâu có tính bù trừ rất lớn cho nhau.
4. **Hướng đi kiến trúc cho Phase 2**:
   - Việc chỉ tinh chỉnh router hay hy vọng GAP/Adaptive routing tự phát huy tác dụng trên vết nứt là không khả thi nếu không có cơ chế đưa thông tin hình học cục bộ (Local Geometry / High-frequency edge) trực tiếp vào router hoặc tái cấu trúc router với non-linear projection.







