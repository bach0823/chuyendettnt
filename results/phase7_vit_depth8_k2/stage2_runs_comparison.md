# So sánh Đối đầu Chi tiết 2 Run Stage 2: Phase 7 (ViT Depth = 8, K=2, 12 Experts)

## 1. Tổng quan Cấu hình Thực nghiệm
- **Dataset:** Crack500 (1896 train samples, image_size = 448)
- **Kiến trúc:** B2 UNet + ConvNeXt-V2 (femto) + ViT-Tiny (8 blocks, embed_dim=192)
- **Routing:** SAGE Hierarchical Router, Top-k=2, 12 Experts (4 CNN + 8 ViT blocks), Stage 2 Gating (s2_gate = True, k=3)
- **Stage 2 Config:** `configs/p3_ablation/phase6_full_s1/a1_s2g_depth8.yaml`, epochs = 18
- **Run 1:** Khởi tạo từ `last_model_b2_stage1_weights.pth` (Stage 1 Epoch 16, Val Dice: 0.7400)
- **Run 2:** Khởi tạo từ `best_model_b2_stage1.pth` (Stage 1 Epoch 8, Val Dice: 0.7495)

## 2. Bảng Tóm tắt Chỉ số Chính (Summary Comparison)

| Chỉ số (Metric) | Run 1 (From Last Stage 1) | Run 2 (From Best Stage 1) | Chênh lệch (Best - Last) |
| :--- | :---: | :---: | :---: |
| **Stage 1 Checkpoint Nguồn** | Epoch 16 (Dice 0.7400) | Epoch 8 (Dice 0.7495) | +0.0095 |
| **Best Val Dice (Stage 2)** | **0.7624** (Epoch 17) | **0.7658** (Epoch 18) | **+0.0034 (+0.34%)** |
| **Val Loss tại Best Dice** | **1.3017** | 1.3769 | +0.0752 (Run 1 tốt hơn về Loss) |
| **Final Epoch 18 Val Dice** | 0.7615 | **0.7658** | +0.0043 |
| **Final Epoch 18 Val Loss** | **1.3037** | 1.3769 | +0.0732 |
| **Final Train Loss (LB)** | 1.3115 (0.1203) | 1.4007 (0.1204) | -0.0892 |
| **Final Train Dice** | 0.7525 | 0.7490 | -0.0035 |

## 3. Bảng Chi tiết Từng Epoch (Full 18 Epochs Trajectory)

| Epoch | Run 1 Train Loss (LB) | Run 1 Val Loss | Run 1 Val Dice | Run 2 Train Loss (LB) | Run 2 Val Loss | Run 2 Val Dice | Δ Val Dice (Run 2 - Run 1) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.6949 (0.1206) | 1.6199 | 0.7460 | 1.8233 (0.1208) | 1.7297 | 0.7470 | +0.0010 |
| 2 | 1.6972 (0.1206) | 1.6469 | 0.7418 | 1.8188 (0.1208) | 1.7233 | 0.7403 | -0.0015 |
| 3 | 1.6736 (0.1206) | 1.5611 | 0.7542 | 1.7915 (0.1208) | 1.6556 | 0.7500 | -0.0042 |
| 4 | 1.6421 (0.1205) | 1.5317 | 0.7460 | 1.7564 (0.1207) | 1.6401 | 0.7341 | -0.0119 |
| 5 | 1.5869 (0.1205) | 1.5092 | 0.7358 | 1.7032 (0.1206) | 1.5796 | 0.7371 | +0.0013 |
| 6 | 1.5456 (0.1204) | 1.4228 | 0.7306 | 1.6535 (0.1206) | 1.5289 | 0.7441 | +0.0135 |
| 7 | 1.5015 (0.1204) | 1.3946 | 0.7587 | 1.6130 (0.1205) | 1.4858 | 0.7524 | -0.0063 |
| 8 | 1.4736 (0.1204) | 1.4542 | 0.7154 | 1.5738 (0.1205) | 1.4991 | 0.7539 | +0.0385 |
| 9 | 1.4357 (0.1204) | 1.3415 | 0.7590 | 1.5368 (0.1205) | 1.4071 | 0.7561 | -0.0029 |
| 10 | 1.4066 (0.1203) | 1.3611 | 0.7233 | 1.5109 (0.1204) | 1.4507 | 0.7381 | +0.0148 |
| 11 | 1.3789 (0.1203) | 1.3378 | 0.7450 | 1.4789 (0.1204) | 1.4106 | 0.7633 | +0.0183 |
| 12 | 1.3619 (0.1203) | 1.3259 | 0.7509 | 1.4605 (0.1204) | 1.3999 | 0.7543 | +0.0034 |
| 13 | 1.3557 (0.1203) | 1.3118 | 0.7509 | 1.4424 (0.1204) | 1.4190 | 0.7494 | -0.0015 |
| 14 | 1.3352 (0.1203) | 1.2972 | 0.7578 | 1.4260 (0.1204) | 1.4132 | 0.7612 | +0.0034 |
| 15 | 1.3260 (0.1203) | 1.3014 | 0.7599 | 1.4238 (0.1204) | 1.4086 | 0.7636 | +0.0037 |
| 16 | 1.3215 (0.1203) | 1.2946 | 0.7596 | 1.4119 (0.1204) | 1.3730 | 0.7594 | -0.0002 |
| 17 | 1.3096 (0.1203) | 1.3017 | 0.7624 | 1.4040 (0.1204) | 1.3861 | 0.7646 | +0.0022 |
| 18 | 1.3115 (0.1203) | 1.3037 | 0.7615 | 1.4007 (0.1204) | 1.3769 | 0.7658 | +0.0043 |

## 4. Phân tích Chi tiết Quỹ đạo Học tập (Learning Trajectory Analysis)
1. **Xuất phát điểm và Khả năng Tổng quát hóa:**
   - **Run 2 (từ Best Stage 1):** Bắt đầu với trọng số Stage 1 tại điểm cực trị tối ưu nhất (Epoch 8, Dice 0.7495). Ngay từ Epoch 11, mô hình đã vượt mốc 0.763 (`0.7633`), và tiếp tục tinh chỉnh ổn định dần dần lên đỉnh **0.7658** ở Epoch 18. Train loss ở Run 2 dừng ở mức 1.4007 và Train Dice là 0.7490, cho thấy mô hình không bị overfit, giữ được khả năng tổng quát hóa xuất sắc trên tập validation.
   - **Run 1 (từ Last Stage 1):** Bắt đầu từ Epoch 16 của Stage 1 (Dice 0.7400, sau khi Stage 1 đã có dấu hiệu chững lại và giảm nhẹ từ peak 0.7495). Mô hình fit sâu hơn vào dữ liệu huấn luyện (Train loss giảm mạnh xuống 1.3115, Train Dice lên 0.7525), kéo theo Val loss thấp hơn đáng kể (1.3017 vs 1.3769). Tuy nhiên đỉnh Val Dice bị chặn ở **0.7624** (Epoch 17), thấp hơn 0.34% so với Run 2.
2. **Sự Ổn định của Stage 2 Gating:**
   - Cả hai run đều tăng trưởng vượt bậc so với Baseline Stage 1 (+0.0129 Dice ở Run 1 và +0.0163 Dice ở Run 2), khẳng định cơ chế S2 Gate (Top-k=2) hoạt động rất hiệu quả trên backbone 8 ViT blocks.