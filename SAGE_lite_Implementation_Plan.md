# Kế hoạch Triển khai SAGE-lite cho Phân vùng Vết nứt (Crack Segmentation)
*(Kế hoạch migrate từ SAGE gốc → SAGE-lite)*

### 1. Backbone (Mạng xương sống)
- **Mô hình**: Chuyển sang kiến trúc lai (Hybrid) kết hợp **ConvNeXtV2-Femto** (rút trích local features) và nửa đầu của **ViT-Tiny (patch16)** (6 blocks đầu tiên).
- **ViT-Depth Sweep trên B1 (Chốt số lượng ViT Blocks)**: Ban đầu khởi điểm tại 6 blocks (điểm giữa của 12 blocks ViT-Tiny). Kết quả B1-6blocks đạt Val Dice = 0.7420 (vượt B0 0.7318, không underperform).
  - **Quy định chiến lược**: Chưa chuyển sang B2 vội mà tiến hành **ViT-depth ablation trên B1** để chốt số blocks tối ưu dựa trên tập **VALIDATION** (không dùng TEST để chọn).
  - **Lý do**: Số lượng ViT blocks quyết định trực tiếp số lượng SAGE injection points / routers của B2 (`N_injection = 4 + num_transformer_layers`). Nếu sweep ở B2 sẽ bị confound giữa backbone depth và routing capacity. Sweep ở B1 giúp cô lập hoàn toàn đóng góp của Transformer depth với chi phí tính toán rẻ hơn rất nhiều.
  - Sau khi chốt depth ở B1, cấu hình B1 sẽ được khóa cứng trước khi wire sang B2.
- **Cấu hình Task**: Đặt `num_classes=1` vì phân vùng vết nứt là bài toán phân loại nhị phân (vết nứt vs. nền).
- **Các bước triển khai**: Tải trọng số pre-trained cho backbone, gỡ bỏ classification head, cắt ViT xuống 6 blocks, và trích xuất các feature map phân cấp.

## 2. Positional Embedding (Mã hóa Vị trí) & CLS Token
- **Cơ chế Interpolation (Không bắt buộc ở Baseline 448)**: Ở baseline `img_size = 448`, ConvNeXt downsample 32 lần tạo ra bottleneck feature map `14x14 = 196` spatial tokens. Pretrained `vit_tiny_patch16_224` cũng có patch grid `14x14 = 196`. Do đó, target grid và pretrained grid đã khớp nhau hoàn toàn. Phép nội suy positional embedding ở runtime lúc này là **identity/no-op**. Tuy nhiên, vẫn giữ cơ chế interpolation trong code architecture để hỗ trợ các experiment đổi resolution sau này. 
- **Xử lý CLS Token**: 
  - Nếu pretrained ViT có chứa CLS token (kiểm chứng qua Shape Audit), BẮT BUỘC phải tách CLS positional embedding khỏi patch positional embeddings trước khi reshape/interpolate:
    ```python
    pos_embed = vit.pos_embed
    cls_pos = pos_embed[:, :1]
    patch_pos = pos_embed[:, 1:]
    ```
  - Chỉ `patch_pos` mới được phép reshape thành spatial grid và đưa vào hàm interpolate. Không được reshape trực tiếp toàn bộ `pos_embed` chứa `CLS + patch tokens` (197) thành HxW vì sẽ gây mismatch.
- **Quyết định giữ hay bỏ CLS Token tại Bottleneck**: KHÔNG ĐƯỢC giả định. Phải chờ xác nhận qua Shape Audit xem ViT có CLS token hay không. Trong implementation, phải có code explicitly để xử lý việc giữ hoặc bỏ. Nếu bỏ, decoder chỉ nhận patch/spatial tokens. Nếu giữ, CLS phải được xử lý riêng, tuyệt đối không được reshape lẫn lộn vào không gian `H x W` spatial grid.

## 3. Decoder (Mạng giải mã)
- **Tích chập tiêu chuẩn (Standard Convolutions)**: Mặc định, sử dụng **Tích chập 3x3 Tiêu chuẩn** cho các lớp decoder để đảm bảo năng lực biểu diễn đặc trưng không gian trong quá trình upsample.
- **Chạy thử nghiệm DWSC (Smoke-Test)**: Chỉ triển khai **Depthwise Separable Convolutions (DWSC)** dưới dạng một tùy chọn có thể bật/tắt (ví dụ: cờ use_dwsc=False) để dùng cho các bài test khói (smoke-testing) nhằm đánh giá hoặc khi cần thu nhỏ mô hình cực độ.

## 4. Router (Bộ định tuyến)
- **Kích thước Lớp ẩn (`router_hidden_dim`)**: Đặt `64` làm baseline khởi điểm nhằm giảm tham số và chi phí tính toán, cho ra một bộ định tuyến gọn nhẹ phù hợp với kiến trúc SAGE-Lite. (Có thể mở rộng ablation với các mức 32/128/256 về sau).
- **Router Noise & Fusion**: Giữ nguyên Exploration Noise làm baseline (sẽ làm ablation sau) và triển khai Residual Fusion.
- **Tham số Top-K (`top_k`)**: Được xác định là một biến số thực nghiệm (hyperparameter). Bắt đầu huấn luyện với `top_k = 4` làm cơ sở (baseline) trước, sau đó mới tiến hành thử nghiệm và tinh chỉnh tiếp dựa trên hiệu năng.

## 5. Fusion (Dung hợp)
- **Dung hợp phần dư (Residual Fusion)**: Triển khai chiến lược dung hợp residual đơn giản thay vì dùng tỷ lệ alpha phức tạp.
- **Các bước triển khai**: Đầu ra từ các SAGE/adapter modules phải được cộng trực tiếp vào đặc trưng của luồng backbone chính (`fused_output = main_features + adapter_output`), đảm bảo dòng gradient mượt mà và giữ nguyên vẹn luồng biểu diễn gốc.

## 6. SA-Hub (Hub Thích ứng Hình dáng)
- **Giữ nguyên O(D²) Eager Pre-population gốc**: Bắt buộc phải giữ nguyên hàm `pre_populate_sa_hubs` của tác giả gốc để sinh ra ma trận adapter O(D²) (D là số unique channel dims).
- **Lý do kỹ thuật 1 (Routing Động)**: Do SAGE sử dụng bộ định tuyến động toàn cục (Global Dynamic Routing). Ở thời gian chạy (runtime), một Tensor ở Stage 1 hoàn toàn có thể bị điều hướng vọt lên Expert ở Stage 4. Vì đường đi thay đổi theo từng ảnh đầu vào, không có cách nào "quét tĩnh" (auto-scan) trước được. Nếu thiếu bất kỳ tổ hợp adapter nào, hệ thống sẽ văng lỗi `RuntimeError` ngay lập tức.
- **Lý do kỹ thuật 2 (Bảo toàn Vết nứt)**: Bắt buộc giữ nguyên cơ chế ánh xạ 1-1 (Pairwise) thay vì dùng không gian ẩn chung (Shared Latent Dimension). Mặc dù không gian ẩn chung giúp giảm tham số xuống O(N), nhưng nó làm tăng số lượng phép tính (FLOPs) lúc inference và tạo ra nút thắt thông tin (low-rank bottleneck). Nút thắt này sẽ đóng vai trò như một bộ lọc thông thấp (low-pass filter), nghiền nát hoàn toàn các dải tần số không gian cao (vết nứt 1-2 pixel). Việc giữ lại thiết kế Pairwise O(D²) là sự hy sinh cần thiết để đảm bảo ánh xạ Full-rank, bảo toàn 100% hình thái không gian của vết nứt.

## 7. Injection (Bơm Cơ chế SAGE)
- **Số lượng Router (N_injection)**: Bắt đầu với **Full Injection** để giữ đúng nguyên tắc Minimal Migration và thiết lập một reference baseline ổn định. Đối với SAGE-lite, số lượng Router cho Full Injection phụ thuộc trực tiếp vào số block của backbone: **`4 (ConvNeXt stages) + num_transformer_layers`**. (Ví dụ ở baseline 6 blocks thì Full Injection = 10 Routers). Mức này sẽ linh động thay đổi nếu ta điều chỉnh số block của ViT. Về sau mới cân nhắc thử nghiệm các cấu hình Sparse (vd: N=2, N=3) để đánh giá marginal value của từng injection point.
- **Kiểm định cấp Module (Module-level Auditing)**: Triển khai các quy tắc màng lọc (dùng `isinstance`) để cấm tuyệt đối việc bơm SAGE vào các lớp downsamplers (thu nhỏ ảnh), patch embeddings, và LayerNorms. SAGE chỉ được phép bọc các khối tính toán thuần túy.
- **Xử lý Tuple (Tuple Wrappers)**: Tạm hoãn việc can thiệp sâu vào file `wrappers.py` để bóc tách Tuple (đối với các block trả về nhiều tensor). Kế hoạch là giữ nguyên wrapper gốc và tiến hành chạy thử nghiệm (smoke-test). Chỉ viết thêm logic bóc tách Tuple nếu hệ thống thực sự báo lỗi `Cannot unpack Tuple` ở thời gian chạy, nhằm tránh phức tạp hóa mã nguồn từ giai đoạn đầu.
- **Tối ưu hóa Self-selection (Zero-cost Bypass):** SAGE gốc cho phép Router chọn lại chính layer hiện tại nhưng lại gọi `forward` thêm 1 lần (lãng phí FLOPs). SAGE-lite khắc phục bằng cách cấp `my_index` cho `SageLayer`. Nếu `expert_idx == my_index`, layer sẽ cắt trực tiếp `main_output` để xài, bypass hoàn toàn SA-Hub và module tính toán, giữ đúng nguyên bản semantics nhưng khử triệt để overhead tính toán trùng.

## 8. Dữ liệu & Tiền xử lý (Data & Preprocessing) — ⚠️ FROZEN CANONICAL
> **QUY TẮC BẤT BIẾN:** Pipeline tiền xử lý dưới đây đã được nghiệm thu và **KHÓA CỨNG (FROZEN)** làm chuẩn mực duy nhất cho toàn bộ SAGE-Lite. Tuyệt đối không tự ý thay đổi trong quá trình code/train/audit. Chỉ thay đổi khi có yêu cầu ablation hoặc sửa protocol rõ ràng từ người dùng.

- **Cấu hình YAML Độc lập:** Bắt buộc tách thành 2 file cấu hình riêng biệt (`b0_deepcrack.yaml` và `b0_crack500.yaml` / tương ứng cho SAGE-Lite) do bản chất dataset và pipeline tiền xử lý hoàn toàn khác nhau.
- **Kích thước chuẩn:** Thiết lập `img_size = 448x448` cho mọi luồng dữ liệu (để bảo toàn chi tiết mảnh của vết nứt, tạo ra feature map 14x14 = 196 token ở cuối ConvNeXt).
- **Crack500 Dataset:**
  - **Lúc Train:** `Reflect Pad` nếu cần → `RandomCrop(448, 448)` → smart filter kiểm tra `fg_pixels >= 20` (nếu `< 20` thì reject và crop lại, tối đa 20 crop attempts × 10 source resamples) → Augmentation → ImageNet Normalize → `ToTensorV2`.
  - **Lúc Test/Eval:** Đánh giá chính thức bằng Tiling trên ảnh gốc:
    - **Setting A:** Non-overlapping tiling 448×448 (stride 448).
    - **Setting B:** Overlapping tiling 448×448 với 50% overlap (stride 224), trung bình xác suất sigmoid (Average Probabilities) → threshold 0.5.
- **DeepCrack Dataset:**
  - **Train & Eval:** Mask nhị phân `{0, 255} → {0, 1}` → **Dynamic Pad-to-Square** (`target = max(H, W, 448)`, đệm đối xứng BORDER_CONSTANT=0 giữ nguyên aspect ratio) → `Resize(448, 448)` với `cv2.INTER_NEAREST` cho mask → Augmentation (chỉ train) → ImageNet Normalize → `ToTensorV2`. Eval chạy direct 1-pass full-image.
- **Augmentation Protocol (Khóa cứng):**
  - **Giữ:** `HorizontalFlip(p=0.5)`, `VerticalFlip(p=0.5)`, `RandomRotate90(p=0.5)`, `RandomBrightnessContrast(p=0.5)`, `GaussianBlur(p=0.3)`.
  - **Loại bỏ hoàn toàn:** `CLAHE`, `ElasticTransform`, `GridDistortion`, `ShiftScaleRotate`, `HueSaturationValue`.

## 9. Chiến lược Huấn luyện & Tối ưu hóa (Training Strategy & Optimizations)
- **Ngân sách Huấn luyện (Epoch Budget & EarlyStopping)**: Cấu hình `Epoch budget = 30` (ngân sách tối đa trần, không phải con số cố định bắt buộc chạy đủ 30 epoch) kết hợp `EarlyStopping patience = 6`. B0 thực tế dừng sớm quanh epoch 14-17 (best checkpoint @ epoch 8 và 11). Single-stage training cho baseline B0, B1.
- **Load Balance Factor (`load_balance_factor=0.01`)**: Giữ tham số nội bộ của Router ở mức `0.01` làm baseline. Chỉ tăng lên (vd: `0.03-0.05`) nếu quan sát thấy hiện tượng sụp đổ định tuyến (expert collapse) thông qua biến `expert_usage_ratio`.
- **Expert Dropout (`expert_dropout=0.1`)**: Khóa giữ nguyên giá trị `0.1` làm baseline để hạn chế overfitting cho SA-Hub trên dataset nhỏ. (Đây là tham số độc lập, không cần scale theo kích thước backbone).
- **Fine-tune Backbone (`freeze_encoder=False, freeze_transformer=False`)**: Mặc định không đóng băng backbone ở baseline để mạng thích nghi với domain crack. Sẽ dùng differential LR thấp hơn cho backbone để bảo tồn pretrained features.
- **Hàm Tổn thất (Loss Function):** Sử dụng công thức `1.0 * BCE + 1.5 * Soft Dice + 1.0 * L_balance`. Trọng số Soft Dice (1.5x) là cơ chế chính để chống lại class imbalance. Không sử dụng `pos_weight` cho BCE.
- **Fix Lỗi cấu trúc gốc (Double Forward Pass):** Bản gốc gọi forward pass 2 lần làm lãng phí thời gian tính toán và sai lệch thống kê BatchNorm. SAGE-lite bọc nhánh if-else cẩn thận, chỉ gọi duy nhất 1 lần forward pass cho mỗi batch:
```python
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
- **Thêm AMP (Automatic Mixed Precision)**: Bắt buộc áp dụng để tiết kiệm VRAM. Fix bug "NaN ngầm" trong `router.py` bằng cách thay `eps = 1e-9` thành `eps = 1e-5`, thay phép nhân mask bằng `torch.where`, và xử lý `g_s` trong FP32 qua `torch.autocast(enabled=False)`.
- **Tỷ lệ Learning Rate (Stage 2)**: Không có tỷ lệ chuẩn duy nhất — paper gốc báo cáo 1:5 (shared_lr cao hơn base_lr 5×, dùng 1e-5/5e-5), nhưng code thực tế cho EBHI/GlaS lại dùng tỷ lệ ngược 2:1 (base_lr cao hơn shared_lr). Hai nguồn không khớp nhau, nên không có cơ sở để bám theo một con số "chuẩn" từ SAGE gốc.
  Với SAGE-Lite: khởi đầu `1:1` (`stage2_base_lr = stage2_shared_lr`) làm baseline trung lập. Theo dõi qua `gs_tracker` trong quá trình train:
  - Nếu thấy fine-grained experts học chậm/không đặc hóa được → thử nghiêng về phía paper (shared cao hơn, ví dụ 1:3 hoặc 1:5)
  - Nếu thấy shared experts bị lấn át, không ổn định → thử nghiêng về phía code thực tế (base cao hơn, ví dụ 2:1)
  Không giả định trước hướng nào đúng — để dữ liệu routing thực tế quyết định.
- **Tránh lỗi `AttributeError` khi trích xuất Shared Experts**: Code gốc hard-code theo tên biến. Cần thêm property `@property def num_sage_experts(self)` vào model `ConvNeXtV2ViTTinyUNet` và ưu tiên gọi property này trong hàm trích xuất thay vì `len(model.convnext.stages)`.
- **Warmup Epochs (2-3 epoch)**: Rút ngắn warmup xuống 2-3 epoch. Hoàn toàn an toàn do `warmup_epochs` chỉ ảnh hưởng đến lịch trình tăng của `LinearLR`.
- **EarlyStopping Patience (5-7 epoch)**: Áp dụng để "fail-fast" trong chu kỳ thực nghiệm ngắn. Việc cắt ngắn Stage 1 hoàn toàn an toàn, không chặn Stage 2 và code luôn tự động nạp lại checkpoint tốt nhất của Stage 1 trước khi khởi động Stage 2.
- **Tracking (`gs_tracker.py`)**: Giữ nguyên. Sự lầm tưởng về "4 CNN + 12 ViT" đến từ bug nhãn trục tọa độ trong `visualization.py`. Việc đổi số lượng experts trong Pool sẽ vô tình vẫn được hiển thị đúng nhãn.

- **Main LR Scheduler**: Giữ `LinearLR` warmup trong 2-3 epoch đầu. Sau warmup, sử dụng `CosineAnnealingLR` (`eta_min=1e-6`). Khóa cố định scheduler cho toàn bộ các ablation.
- **DropPath (Stochastic Depth)**: Không cấu hình thêm ở baseline SAGE-lite; giữ hành vi mặc định của backbone. Chỉ xem xét DropPath như một regularization ablation khi thực nghiệm cho thấy dấu hiệu overfitting (nếu có ablation, sẽ bắt đầu với `drop_path_rate = 0.1`).
- **Gating Function**: Khóa cứng `gating_type = "sigmoid"` để tuân thủ thiết kế SAGE gốc (các expert được chọn nhận gating weight bằng sigmoid độc lập, không chuẩn hóa bằng softmax). Không đưa vào làm biến ablation.
- **Gradient Checkpointing**: Sẽ được bật hoặc tắt hoàn toàn dựa trên kết quả của quá trình OOM Probe (Nếu tắt mà vẫn đủ VRAM cho batch size mục tiêu thì ưu tiên tắt để train nhanh, ngược lại thì bật). Giữ nguyên trạng thái cho toàn bộ ablations.
- **Reproducibility (Cố định Seed Toàn cục)**: Phải thiết lập chính sách seed cố định (`seed=42` hoặc một số nguyên quy định trước) cho toàn bộ pipeline để đảm bảo khả năng tái lập. Cụ thể: 
  - **Data split**: Chia tập train/val/test phải deterministic.
  - **Augmentation & Dataloader**: Khởi tạo Dataloader worker generators và các biến đổi ngẫu nhiên phải cùng một trạng thái ban đầu. (Validation/Test phải cấm tuyệt đối sự ngẫu nhiên, tắt random crops).
  - **Weight initialization**: Phải gọi hàm `set_seed` ngay trước khi khởi tạo model để đảm bảo các lớp Convolution/Linear mới thêm vào luôn có cùng một bộ trọng số ngẫu nhiên ban đầu ở mọi Baseline (từ B0 đến B2).
  Điều này là yếu tố cốt lõi để Baseline Ladder so sánh công bằng.
- **Mixed Precision (AMP) - Bắt buộc dùng FP16:** Thiết lập AMP dtype là `torch.float16` (FP16), tuyệt đối không dùng `bfloat16` (BF16) như cấu hình gốc của SAGE trên A100. Lý do: Tesla T4 thuộc kiến trúc Turing, không hỗ trợ BF16 Tensor Core, chỉ chạy AMP tốt ở FP16. (Đây cũng là lý do bắt buộc phải vá lỗi `eps = 1e-9` ở Load Balance Loss để chống underflow).

## 10. Quy trình Diagnostic Khởi động (Pre-training HPs Tuning)

Trước khi chạy hàng loạt Ablation 15 epoch, bắt buộc chạy chuỗi script diagnostic sau để chốt Hyperparameter (thay vì đoán mò):

- **OOM Probe (Test Batch Size):** Chạy RIÊNG cho hai cấu hình cực đoan: nhẹ nhất (chỉ Backbone + Decoder) và nặng nhất (Full SAGE-lite) - chi tiết các bậc xem mục 11.4. Sử dụng **Gradient Accumulation** để giữ Effective Batch Size CỐ ĐỊNH xuyên suốt các bậc nhằm đảm bảo fairness khi so sánh accuracy. Ghi nhận lại RAW batch size và Peak VRAM của từng mô hình để trả lời Câu hỏi 5 về độ phức tạp.
- **Đo lường `pos_weight` (Diagnostic)**: Quét trực tiếp trên các mask gốc của DeepCrack và Crack500 (không qua Dataset/augmentation/filter) để tính tỷ lệ pixel nền/nứt. Chỉ lưu số liệu này làm chẩn đoán (không dùng trong lần chạy R1). Chỉ thử dùng Weighted BCE nếu validation thực tế cho thấy mô hình under-predict vết nứt.
- **LR Range Test (Leslie Smith):** Chạy RIÊNG cho cấu hình nhẹ nhất và nặng nhất (Full SAGE-lite) vì loss landscape hoàn toàn khác nhau (đặc biệt khi bản Full có thêm L_balance). Các bậc trung gian dùng nội suy giữa 2 kết quả này hoặc test nhanh nếu nghi ngờ. Plot biểu đồ và xác định vùng LR phù hợp. Không dùng chung 1 LR cho tất cả nếu landscape lệch hẳn.
- **Weight Decay Tách biệt:** Cố định base_wd = 0.05 cho AdamW làm baseline của các run chính. Bắt buộc triển khai hàm get_param_groups để loại trừ các tham số LayerNorm và bias khỏi Weight Decay (weight_decay=0.0 cho nhóm này), trong khi các parameter còn lại dùng base_wd. Không thực hiện WD sweep trước R1; chỉ mở ablation 0.01 nếu kết quả validation sau đó cho thấy cần investigate regularization/overfitting.

### 10.2 Generic Routing Diagnostics

**Mục tiêu:**
- Theo dõi hành vi của Router sau khi SK4/SK5 được triển khai.
- Hỗ trợ phân tích các ablation về ViT depth, `top_k`, `N_injection`, `router_hidden_dim`, `load_balance_factor`, Exploration Noise...
- Phát hiện sớm routing collapse / dead expert / routing quá tập trung.
- Cho phép so sánh routing behavior giữa các baseline/ablation mà không phải sửa visualization code.

**Các diagnostic cần hỗ trợ:**
1. **Top-K Activation Map:** Heatmap 2D theo `layer × expert`. Thể hiện expert nào thực sự xuất hiện trong Top-K. Số layer và số expert phải dynamic, không hard-code số row/column.
2. **Expert Usage Ratio:** Tính tỷ lệ mỗi expert được chọn trong Top-K. Mẫu số (denominator) là tổng số lượt Top-K selections trong phạm vi aggregate tương ứng. Cần hỗ trợ aggregate theo layer/batch và đặc biệt là time-series theo epoch. Dùng làm tín hiệu cảnh báo expert collapse (nhưng KHÔNG tự động thay đổi `load_balance_factor`).
3. **Routing Entropy / Concentration:** Tính từ một phân phối được chuẩn hóa/derive từ `gating_scores`. Tuyệt đối không thay đổi semantics của Router (vẫn dùng `sigmoid` cho gating thực tế, không tự ý đổi sang softmax). Ưu tiên theo dõi theo epoch để phát hiện xu hướng routing ngày càng tập trung hoặc phân tán. Đây là diagnostic metric, không phải checkpoint selection metric.
4. **Affinity Heatmap (Optional):** Có thể derive từ cùng `gating_scores` dùng cho routing diagnostics. Không cần xây một pipeline thu thập dữ liệu riêng.

**Nguyên tắc generic (Tuyệt đối không hard-code):**
- Không hard-code các thông số: `num_transformer_layers = 6`, `num_experts = 20`, `N_injection = 10`, `top_k = 4`.
- Routing diagnostics phải đọc trực tiếp từ metadata runtime. (Ví dụ Full Injection có `N_injection = 4 + num_transformer_layers`, nên 6 ViT blocks → 10 injection points, 8 → 12).
- Visualization phải tự scale theo số lượng routing entries.

**Routing metadata schema:**
Dữ liệu `routing_infos` cần có schema tự mô tả, bao gồm runtime expert count/namespace, và `topk_indices` phải dùng cùng expert index space với `gating_scores` (tuyệt đối không hard-code 20 experts). Ví dụ:
```python
{
    "layer_name": "CNN_stage_2",
    "layer_type": "cnn",
    "num_experts": 20, # Dynamic runtime count
    "gating_scores": ..., # Cùng index space với topk
    "topk_indices": ...
}
```
`routing_infos` là dạng list có độ dài thay đổi. Không dựa vào index cố định kiểu `CNN 0..3`. Cả 4 visualizations đều lấy nguồn dữ liệu chung từ schema này, không xây thành các pipeline độc lập.

**Cross-reference với Router Analysis:**
Các phân tích router trong phần Failure Analysis (§11.5) bắt buộc phải tái sử dụng schema và metric definitions của §10.2 này, tuyệt đối không tạo hệ metric độc lập.

**Collapse thresholds & Khả năng tương thích:**
- KHÔNG chốt cứng các ngưỡng heuristic (ví dụ: usage < 0.05 = dead expert hay > 0.60 = dominant expert). Diagnostic chỉ tính metric, vẽ, và cảnh báo.
- Thiết kế phải dùng lại được cho: ViT depth 6/8/10/12, các `top_k` khác nhau, Full/Sparse Injection, router hidden dim khác nhau, load-balance khác nhau, Exploration Noise ON/OFF. Không tạo implementation riêng cho B0/B1/B2 hay R1/2/3.
- **Lưu ý Implementation:** Khi training dài, ưu tiên aggregate routing statistics theo batch/epoch và chỉ lưu summary cần thiết; KHÔNG mặc định persist toàn bộ token-level gating tensors (tránh gây bùng nổ dung lượng lưu trữ).
## 11. Evaluation Protocol (Giao thức Đánh giá)

### 11.1 Primary Metric & Checkpoint Selection
- **Metric chính**: Foreground Crack Dice (per-sample mean).
- Tính Dice **chỉ trên foreground crack**, không tính background. Phân ngưỡng `sigmoid(logits) > 0.5`.
- Công thức mỗi sample: `Dice = (2TP + eps) / (Pred + GT + eps)`
- **Chọn Checkpoint**: Dựa **duy nhất** vào **Best Val Foreground Crack Dice**. 
  - *Lý do*: Dice và IoU có quan hệ đơn điệu trực tiếp, dùng Dice làm tiêu chí chính là đủ và hoàn toàn nhất quán với Loss (thành phần `1.5 * Soft Dice`). Không dùng bất kỳ metric nào khác để tham gia selection.
- **Tie-breaking Rule (Đảm bảo Reproducibility)**:
  - Nếu `Val Foreground Crack Dice > best_dice + 1e-4` → Lưu checkpoint mới.
  - Nếu chênh lệch Dice `<= 1e-4` → Ưu tiên epoch có `val_loss` thấp hơn. (Lưu ý: `val_loss` chỉ là tie-break, tuyệt đối không phải secondary metric).
- **Validation Crack500**: Bắt buộc deterministic. Dùng non-overlapping tiling 448x448 cố định. Seed không được làm thay đổi các patch validation.

### 11.2 Secondary Metrics / Diagnostics (Không dùng để chọn checkpoint)
- **Global Pixel-Level Foreground IoU**: Gom toàn bộ intersection và union của tất cả sample rồi mới chia. Có guard `total_union = 0`.
- **Crack-Present Foreground IoU**: Tính mean IoU riêng biệt trên tập sample có `GT_pixels > 0`.
- **Boundary IoU**: Diagnostic đánh giá độ chính xác của hình thái/đường viền vết nứt.
- **HD95 (Hausdorff Distance 95%)**: Diagnostic đo lường khoảng cách sai lệch biên lớn nhất (95th percentile).
*(Lưu ý: Các metric phụ và Boundary IoU/HD95 chỉ dùng để phân tích và sanity-check tại best checkpoint, tuyệt đối không can thiệp vào Checkpoint Selection).*

### 11.3 Dataset & Inference Protocol
- **Crack500**:
  - **Setting A (Non-overlap)**: Patch 448x448. Tiling deterministic. Áp dụng cho mọi ablation để so sánh công bằng.
  - **Setting B (Final overlap)**: Patch 448x448, stride 224 (overlap 50%). Tái cấu trúc bằng **Average blending**. Dùng để báo cáo kết quả cuối cùng.
- **DeepCrack**: Không tiling. Đánh giá trực tiếp.
- **Nguyên tắc chung**: Không dùng random spatial sampling khi Val/Test. Mọi baseline dùng chung một config data.

### 11.4 Baseline Ladder
Nhằm cô lập đóng góp của từng thành phần, hệ thống sẽ trải qua 3 mốc:
- **B0**: ConvNeXtV2-Femto + U-Net (Pure CNN baseline, không ViT, không SAGE) — *[Đã hoàn thành trên Crack500 & DeepCrack]*.
- **B1**: B0 + 6 ViT-Tiny (ConvNeXtV2-Femto + 6 ViT-Tiny + U-Net thuần, Late Fusion, không SAGE). Cô lập đóng góp của Attention / Global Context.
- **B2 (Full SAGE-Lite)**: B1 + full SAGE mechanism (SAGE Router + heterogeneous Expert Pool + SA-Hub + Load-Balance Loss). Đo lường toàn bộ năng lực thích ứng hình thái vết nứt của SAGE-Lite.

  **QUY TẮC HUẤN LUYỆN (QUAN TRỌNG)**:
  Mỗi bậc B0, B1, B2 phải train **ĐỘC LẬP** từ pretrained ImageNet backbone gốc. Tuyệt đối **KHÔNG warm-start** (không load checkpoint trọng số đã học từ bậc liền trước). Các component mới (ViT ở B1; SA-Hub, Router ở B2) ở mỗi bậc luôn khởi tạo random (seed=42). Điều này đảm bảo kết quả đo lường được đóng góp thật của component, không bị confound bởi số epoch tích lũy.


### 11.5 Failure Analysis
Thực hiện trên checkpoint tốt nhất của từng model:
- **Phân phối hiệu năng**: Percentile P10, P25, P50, P75, P90 theo per-sample Dice/IoU.
- **Hard Samples**: Bắt 30 sample khó nhất (Lưu Input, GT, Pred, Dice, IoU, ID).
- **Router Analysis**: Phân tích tỷ lệ sử dụng (expert usage ratio) và phân phối router theo độ khó. Bỏ qua `g_s` từ GS Tracker.

### 11.6 Statistical / Reporting
- Báo cáo tách biệt theo 2 dataset. Crack500 phải tách riêng bảng của Setting A và Setting B.
- Tracking đầy đủ: Metric chính, metric phụ, Loss curve, Hardest samples, Router stats.


## 12. Precondition: Shape Audit & Preflight Verification
**Mục tiêu**: Tuyệt đối không code mù (blind coding) dựa trên tài liệu lý thuyết. Trước khi bắt tay vào implement `SageLayer` hay `SAGE_injection`, bắt buộc phải viết một script `shape_audit.py` để khởi tạo mô hình thực tế từ thư viện `timm` (`convnextv2_femto` và `vit_tiny_patch16_224`) với input `448x448` và chạy một forward pass thực tế (dummy pass).

**Các thông số cần in ra log và đối chiếu với Plan:**
1. **ConvNeXtV2-Femto channels**: Kiểm chứng xem 4 stages có đúng là sinh ra channels `[48, 96, 192, 384]` hay không.
2. **Feature/token shapes**: In shape của các tensor khi đi qua từng layer (từ 2D sang 1D flattened).
3. **Token count**: Kiểm tra số lượng token sau quá trình projection/patchification có đúng là `196` hay không.
4. **Positional Embedding & CLS Token**: 
   - Kiểm tra `vit.pos_embed.shape`.
   - Xác nhận có CLS token hay không.
   - Xác nhận số lượng patch positional embeddings và Patch Grid của pretrained ViT (vd: 14x14).
   - Xác nhận Target Bottleneck Grid của SAGE-lite.
   - Kiểm tra xem thuật toán có tách riêng CLS token trước mọi phép reshape/interpolation hay không.
5. **ViT Blocks**: Xác nhận số lượng Transformer blocks thực tế đang được wrap.
6. **N_injection**: Tính toán cấu hình `4 + num_transformer_layers` (ví dụ `10` nếu 6 blocks) từ số block thực tế lấy ra được.

**Hành động:** 
- Nếu có bất kỳ sự sai lệch (mismatch) nào giữa thực tế forward pass và các hằng số lý thuyết đang ghi trong Plan này, **phải cập nhật lại Plan** ngay lập tức.
- Script Audit này là **bước tiên quyết (Precondition)**, không phải ablation study.
## 12.5 B2 Experimental Roadmap
Lộ trình thực nghiệm chi tiết cho cấu hình B2 (Full SAGE-Lite) từ Phase 0 (Runtime preflight) đến Phase 6 (Regularization) được định nghĩa tại `docs/B2_Experimental_Roadmap.md` (bao gồm cả Phase A: Residual Scale Sweep và Phase B: Adaptive Fusion Comparison).

## 13. Research & Ablation Categorization (Scope)

Để phân định rõ ràng các mục tiêu nghiên cứu và tránh nhầm lẫn giữa các kiến trúc, toàn bộ các thử nghiệm được chia thành 3 nhóm độc lập:

### Nhóm 1: Core SAGE-lite Baseline Ladder (Ablation Chính)
Mục tiêu: Đo lường hiệu quả tăng thêm (incremental effect) của từng thành phần cấu thành nên hệ thống SAGE-lite.
* **B0**: ConvNeXtV2-Femto + U-Net (Pure CNN, no ViT, no SAGE) — **[ĐÃ HOÀN THÀNH trên Crack500 & DeepCrack]**.
* **B1**: B0 + 6 ViT-Tiny (ConvNeXtV2-Femto + 6 ViT-Tiny + U-Net thuần, Late Fusion, không SAGE). Đo lường tác động của Attention / Global Context.
* **B2 (Full SAGE-Lite)**: B1 + full SAGE mechanism (SAGE Router + heterogeneous Expert Pool + SA-Hub + Load-Balance Loss). Đo lường sức mạnh trọn vẹn của cơ chế Shape-Adapting Gated Experts.


**Quy tắc (PLAN LOCK)**: Mỗi baseline (B0, B1, B2) phải được train độc lập hoàn toàn từ weights pretrained của ImageNet. Tuyệt đối không warm-start (ví dụ: không lấy weights của B0 để train tiếp lên B1).

### Nhóm 2: Supplementary Backbone Ablation (Thử nghiệm Phụ trợ)
Mục tiêu: Trả lời câu hỏi độc lập *"Trong cùng điều kiện Decoder, backbone CNN và Transformer khác nhau thế nào về hiệu năng?"*
* **CNN Backbone**: ConvNeXtV2-Femto + U-Net (Thực chất chính là B0).
* **Transformer Backbone**: ViT-Tiny + U-Net.

**Quy tắc (PLAN LOCK)**: 
* Đây là nhóm bổ sung (supplementary).
* Không bao giờ được gọi model ViT-only là "B0" hay "B1".
* Không dùng kết quả hiệu năng (B1 trừ B0) để suy ra gián tiếp hiệu năng của ViT-only. ViT-only phải được train và đánh giá độc lập để so sánh trực tiếp với B0.

### Nhóm 3: Native / Reference Architecture Baselines (Nhóm Tham chiếu)
Mục tiêu: Đối chiếu hiệu năng của SAGE-lite với các kiến trúc chuẩn mực (native) của từng loại backbone.
* **Ví dụ**: MiT-B0 + SegFormer decoder.
* **Ví dụ**: ConvNeXtV2 + UperNet.

**Quy tắc (PLAN LOCK)**: Nhóm này chỉ mang tính tham chiếu (reference). Không được dùng để làm controlled backbone ablation, vì cả backbone và decoder đều bị thay đổi đồng thời, làm mất tính đối chứng độc lập (controlled variables).

---
**📍 TÌNH TRẠNG HIỆN TẠI (CURRENT SCOPE)**
* **B0 đã hoàn thành 100%**:
  - Crack500: Best Val Dice 0.7318, Test Setting A Dice 0.6771, Setting B Dice 0.6801.
  - DeepCrack: Best Val Dice 0.6240, Test Direct Dice 0.6953.
* **B1-6blocks đã hoàn thành trên Crack500**:
  - Best Val Dice: 0.7420 (@ epoch 21, Val Loss 1.1453).
  - Test Dice: Setting A 0.6857, Setting B 0.6895 (HD95: 77.43 px, giảm 16.93 px so với B0).
  - Kết luận: B1 hoàn toàn không underperform B0.
* **QUYẾT ĐỊNH MỚI**: Chưa chuyển sang B2. Tiến hành **ViT-Depth Ablation trên B1** (chọn depth tối ưu dựa trên VALIDATION, giữ nguyên toàn bộ pipeline khác). Sau khi chốt depth mới khóa B1 và wire sang B2.
* Preprocessing đã khóa cứng (Frozen Canonical).

