# Phase 7D: Top-K Capacity Sweep Results

Thư mục này lưu trữ kết quả và trọng số checkpoints của đợt huấn luyện **Phase 7D — Top-K Capacity Sweep** (`K=3` vs `K=4`, `seed=42`, Full Two-Stage 35 Epochs, dataset Crack500).

## 1. Nguồn Gốc Dữ Liệu & Lưu Trữ ZIP Gốc
Các tệp ZIP kết quả chạy trên Google Colab đã được lưu trữ vĩnh viễn trong thư mục `zips/`:
* `zips/Phase7D_TopK_Sweep_seed42 (1).zip` (336.29 MB) $\to$ Trích xuất vào `batch1_topk3_peak07658/`
* `zips/Phase7D_TopK_Sweep_seed42.zip` (335.92 MB) $\to$ Trích xuất vào `batch1_topk4_peak07696/`
* `zips/Phase7D_TopK_Sweep_seed42 (2).zip` (336.23 MB) $\to$ Trích xuất vào `batch2_topk3_peak07670/`
* `zips/Phase7D_TopK_Sweep_seed42 (3).zip` (335.75 MB) $\to$ Trích xuất vào `batch2_topk4_peak07630/`

Mỗi thư mục chứa đầy đủ 5 file weights `.pth` (`best_model_b2_global.pth`, `best_model_b2_stage1.pth`, `best_model_b2_stage2.pth`, `last_model_b2_stage1.pth`, `last_model_b2_stage2.pth`), `train.log`, cùng toàn bộ báo cáo chẩn đoán routing và error taxonomy `P3_C_Routing_Diagnostics/`.

---

## 2. Bảng Tổng Hợp Kết Quả Thực Nghiệm (Validation Set N=348)

| Cấu hình | Best Ep | Val Dice | Val IoU | Precision | Recall | Thin Crack Dice | Routing Entropy | CNN % | ViT % | HHI Concentration | High Quality (>75%) | Thin Failure | Breaks | False Bridges |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (K=2)** | Ep 14 | $0.7618 \pm 0.1644$ | $0.6386$ | $0.7216$ | $0.8573$ | $0.6996$ | $1.0000$ | $36.94\%$ | $63.06\%$ | $0.1811$ | 124 | 28 | 68 | 121 |
| **K=3 (Batch 1)** | Ep 17 | $0.7659 \pm 0.1609$ | $0.6432$ | $0.7348$ | $0.8444$ | $0.7093$ | $1.5849$ | $45.34\%$ | $54.66\%$ | $0.1522$ | 115 | 26 | 81 | 112 |
| **K=3 (Batch 2)** | Ep 16 | $0.7669 \pm 0.1578$ | $0.6440$ | $0.7428$ | $0.8360$ | $0.7143$ | $1.5849$ | $44.77\%$ | $55.23\%$ | $0.1462$ | 131 | 24 | 84 | 108 |
| **K=4 (Batch 1)** | Ep 14 | $\mathbf{0.7696} \pm 0.1615$ | $\mathbf{0.6480}$ | $\mathbf{0.7628}$ | $0.8165$ | $\mathbf{0.7193}$ | $2.0000$ | $40.07\%$ | $59.93\%$ | $0.1394$ | 126 | 21 | 82 | 121 |
| **K=4 (Batch 2)** | Ep 14 | $0.7630 \pm 0.1701$ | $0.6413$ | $0.7572$ | $0.8104$ | $0.7155$ | $2.0000$ | $52.41\%$ | $47.59\%$ | $0.1390$ | 117 | 22 | 75 | 114 |

---

## 3. Nhận Xét & Phân Tích Kỹ Thuật

1. **Hiệu quả của việc mở rộng Top-K Capacity ($K=2 \to K=3 \to K=4$):**
   * Cả $K=3$ và $K=4$ đều cải thiện vượt trội so với Baseline $K=2$ ($0.7618$ Dice):
     - **Top-K = 3** đạt độ ổn định rất cao qua 2 lần lặp: Dice trung bình $\approx 0.7664$, cải thiện Recall ($83.6\% - 84.4\%$) và giảm số ca False Bridges xuống mức thấp nhất ($108 - 112$ ca).
     - **Top-K = 4 (Batch 1)** thiết lập mốc kỷ lục mới của toàn bộ dự án với Val Dice **$0.7696$** (+0.78% so với baseline), Precision vọt lên **$76.28\%$**, Thin Crack Dice đạt **$0.7193$**, và số ca Thin Crack Failure giảm từ 28 xuống chỉ còn 21 ca.
2. **Hành vi Routing:**
   * Mức Entropy định tuyến tuân thủ tuyệt đối lý thuyết thông tin:
     - Với $K=2$: $H = \log_2(2) = 1.0000$ bit.
     - Với $K=3$: $H = \log_2(3) \approx 1.5849$ bit.
     - Với $K=4$: $H = \log_2(4) = 2.0000$ bit.
   * Khi $K$ tăng, sự phân tán tải (HHI Concentration) giảm từ $0.1811 \to 0.149 \to 0.139$, cho phép mạng kích hoạt đồng thời nhiều góc nhìn đa tầng từ cả nhánh CNN (cục bộ, cạnh) lẫn ViT (toàn cục, ngữ cảnh).
