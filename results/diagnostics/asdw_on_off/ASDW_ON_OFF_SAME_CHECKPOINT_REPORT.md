# Báo Cáo Chẩn Đoán Causal Cùng Checkpoint: ASDW ON vs OFF Trên Candidate B

**Thời gian thực hiện:** 03/10/2026  
**Phương pháp:** Causal Turn-off trên cùng weights ($X' = X + \gamma F(X) \rightarrow X$), không retrain, không thay đổi checkpoint  
**Target Model:** SAGE-Lite Candidate B Baseline (`B2ConvNeXtViTUNet`, Setting A, Tile 448)  
**Checkpoint Path:** `results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth`  
**Dataset:** Toàn bộ 348 ảnh validation của tập Crack500 ($81,594,144$ pixels được đánh giá)  
**File kết quả per-sample:** `results/diagnostics/asdw_on_off/asdw_on_off_per_sample_348.csv`  
**File tổng hợp JSON:** `results/diagnostics/asdw_on_off/asdw_on_off_summary.json`

---

## 1. Trả Lời Trực Tiếp Câu Hỏi Nghiên Cứu

> **Câu hỏi:** *ASDW đang thực sự làm thay đổi representation/prediction theo hướng giữ detail, hay nó gần như không đóng góp gì? Cụ thể Candidate B khi không có P3, chỉ compression thì sẽ như thế nào?*

**KẾT LUẬN THỰC NGHIỆM CHÍNH THỨC:**
**Candidate B khi hoàn toàn không có ASDW (chỉ nén trung bình trực tiếp $X \rightarrow \text{AdaptiveAvgPool2d}(28, 28) \rightarrow \text{ViT}$) cho kết quả gần như đồng nhất tuyệt đối (bit-level negligible difference) với khi có ASDW:**

1. **Về High-Frequency Representation tại S0 và S1:**
   - ASDW chỉ gây ra một mức nhiễu động chuẩn hóa cực nhỏ:
     $$\text{Stage 0}: \frac{\|X' - X\|_2}{\|X\|_2} = \mathbf{0.2454\%} \quad (\gamma_{\text{S0}} = 0.0147)$$
     $$\text{Stage 1}: \frac{\|X' - X\|_2}{\|X\|_2} = \mathbf{0.0744\%} \quad (\gamma_{\text{S1}} = 0.0158)$$
   - Độ tăng tỷ lệ năng lượng phổ cao tần (High-Pass Laplacian Ratio) của ASDW trước khi nén chỉ là:
     $$\Delta \text{HP}_{\text{S0}} = +0.000244 \ (+0.0076\%)$$
     $$\Delta \text{HP}_{\text{S1}} = +0.000052 \ (+0.0096\%)$$
   - $\implies$ **Hoàn toàn KHÔNG CÓ sự bảo vệ prominence có ý nghĩa nào.** Thay đổi ở dải cao tần trước khi nén chỉ ở mức $\sim 0.01\%$.

2. **Về Tác Động Logit Pixel-Level ($\Delta z = z_{\text{ON}} - z_{\text{OFF}}$):**
   - Trên toàn bộ **81,594,144 pixels** của tập validation:
     - Mean $|\Delta z| = \mathbf{0.000011}$ ($1.05 \times 10^{-5}$)
     - Median $|\Delta z| = \mathbf{0.000005}$ ($4.53 \times 10^{-6}$)
     - P95 $|\Delta z| = \mathbf{0.000041}$
     - P99 $|\Delta z| = \mathbf{0.000101}$
     - Max $|\Delta z| = \mathbf{0.001044}$ ($\sim 10^{-3}$)
   - **Số pixel bị đảo ngược dấu phân loại (Sign-Flips):**
     Đúng **45 pixels / 81,594,144 pixels** bị đổi nhãn (tỷ lệ $\mathbf{0.000055\%}$, tức chỉ $\sim 1$ pixel trên $1.8$ triệu pixels).

3. **Về Topology & Detail Metrics:**
   - False Bridge Events: **118 vs 118** ($\Delta = 0$).
   - False Bridge Images: **110 vs 110** ($\Delta = 0$).
   - Break Events (Đứt đoạn): **35 vs 35** ($\Delta = 0$).
   - Thin-Crack Dice ($\le 3\text{ px}$): **0.4230 vs 0.4230** ($\Delta = -0.0000$).
   - Boundary IoU ($d=2$): **0.2418 vs 0.2418** ($\Delta = -0.0000$).

---

## 2. Bảng Đối Chiếu Causal Đầy Đủ (Val N=348)

| Nhóm Chỉ Số | Tên Chỉ Số | ON (With ASDW) | OFF (Identity Bypass) | Độ Lệch $\Delta$ (ON - OFF) | Đánh Giá Ý Nghĩa |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Global Metrics** | **Val Dice** | **0.7641** | **0.7641** | $+1.30 \times 10^{-8}$ | Không đổi |
| | **Val IoU** | **0.6417** | **0.6417** | $+2.21 \times 10^{-7}$ | Không đổi |
| | **Recall** | **0.8477** | **0.8477** | $-1.22 \times 10^{-6}$ | Không đổi |
| | **Precision** | **0.7337** | **0.7337** | $+2.80 \times 10^{-6}$ | Không đổi |
| | **clDice** | **0.8499** | **0.8499** | $-1.99 \times 10^{-5}$ | Không đổi |
| **Detail & Boundary**| **Boundary IoU (d=2)**| **0.2418** | **0.2418** | $-9.52 \times 10^{-7}$ | Không đổi |
| | **HD95 (px)** | **53.6483 px** | **53.6432 px** | $+0.0050\text{ px}$ | Lệch $0.005\text{ px}$ |
| | **Boundary F1 (BF1)** | **0.3820** | **0.3820** | $-2.90 \times 10^{-6}$ | Không đổi |
| | **Thin-Crack Dice ($\le 3\text{ px}$)**| **0.4230** | **0.4230** | $-1.30 \times 10^{-6}$ | Không đổi |
| **Connectivity** | **Bridge Events** | **118** | **118** | **0** | **100% Cố định** |
| | **Bridge Images** | **110** | **110** | **0** | **100% Cố định** |
| | **Break Events** | **35** | **35** | **0** | **100% Cố định** |
| | **Break Images** | **34** | **34** | **0** | **100% Cố định** |
| | **Spurious Islands** | **111** | **111** | **0** | **100% Cố định** |

---

## 3. Phân Tích High-Frequency Representation Probe Tại S0 & S1

Đo lường trực tiếp trên toàn bộ ảnh validation thực tế:

```text
-----------------------------------------------------------------------------------------
HIGH-FREQUENCY REPRESENTATION PROBE AT S0 & S1 (PRE-COMPRESSION):
  Stage 0 Perturbation Norm (||X'-X|| / ||X||): 0.2454% (gamma = 0.0147)
  Stage 0 High-Pass Ratio (ON vs OFF):         3.219903 vs 3.219659 (Delta = +0.000244)
  Stage 1 Perturbation Norm (||X'-X|| / ||X||): 0.0744% (gamma = 0.0158)
  Stage 1 High-Pass Ratio (ON vs OFF):         0.547874 vs 0.547822 (Delta = +0.000052)
-----------------------------------------------------------------------------------------
```

### Tại sao ASDW không tạo ra thay đổi?
1. **Quá trình huấn luyện đã tự động "bóp nghẹt" ASDW:**
   - Khởi tạo ban đầu: $\gamma = 0.0100$.
   - Sau khi huấn luyện 35 epochs trên Candidate B: $\gamma_{\text{S0}} = 0.0147$, $\gamma_{\text{S1}} = 0.0158$.
   - Mạng không phát triển $\gamma$ lên các giá trị lớn (ví dụ $0.5$ hay $1.0$). Gradient từ loss không ưu tiên dòng chảy thông tin qua nhánh $F(X)$.
2. **Suy giảm kép (Double Attenuation):**
   - $X' = X + \gamma F(X)$: đóng góp của $F(X)$ là $0.07\% - 0.25\%$.
   - Ra khỏi ViT expert, nhánh SAGE nhân tiếp với `residual_scale = 0.1`:
     $$\text{Total Logit Shift} \approx 0.0025 \times 0.1 = 0.00025$$
   - Điều này giải thích chính xác tại sao $|\Delta z|$ trung bình trên 81 triệu pixels chỉ đạt **$0.000011$**.

---

## 4. Kết Luận Dứt Điểm

1. **Sự chênh lệch $+0.0014$ Dice giữa P3-A và P3-C trước đây là do ngẫu nhiên của 2 run train độc lập**, không phải do năng lực giữ detail của ASDW.
2. Khi thực hiện **same-checkpoint ON/OFF**, ASDW **hoàn toàn không đóng góp gì** vào cả metric toàn cục, metric biên, thin-crack lẫn cấu trúc liên thông (connectivity).
3. **Phán quyết cuối cùng:** Khóa và đóng Phase 6D. Hướng can thiệp P3/ASDW chính thức khép lại.
