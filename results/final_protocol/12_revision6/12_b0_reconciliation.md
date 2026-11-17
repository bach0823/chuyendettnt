# 12_b0_reconciliation.md — Đối Chiếu Toàn Diện Lịch Sử Baseline B0 Với Các Tệp Nguồn Gốc (Revision 6)

**Ngày lập:** 2026-11-17  
**Nguồn dữ liệu đối chứng:**
- `results/B0_Crack500_Run3_Results.md`
- `results/baselines/50ep/B0_Crack500/training_completion.json`
- `results/baselines/50ep/B0_Crack500/official_eval_test.json`
- `results/baselines/100ep/B0_Crack500/training_completion.json`
- `results/baselines/100ep/B0_Crack500/official_eval_all_splits.json`

---

## 1. Bảng Tổng Hợp Tham Số Huấn Luyện Và Kết Quả Của Baseline B0

Ghép cặp Val và Test **CHỈ KHI CÙNG GIAO THỨC ĐO LƯỜNG**; tuyệt đối không ghép chéo giao thức:

| Lần chạy | Checkpoint | Ngân sách Epoch tối đa (Patience) | Epoch tốt nhất (Best Epoch) | Giao thức đo Val | Giao thức đo Test | Val Dice | Test Dice | Khoảng cách Val $\to$ Test | Đường dẫn tệp nguồn đối chứng |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **B0 Run 3** (Setting A) | `best_model_b0.pth` | 30 epochs (Patience = 6) | Epoch 8 (dừng ở ep 14) | **Setting A** (tiling stride 448) | **Setting A** (tiling stride 448) | **0.7318** | **0.6771** | **+5.47 điểm** | `results/B0_Crack500_Run3_Results.md` |
| **B0 Run 3** (Setting B) | `best_model_b0.pth` | 30 epochs (Patience = 6) | Epoch 8 (dừng ở ep 14) | **Setting B** (stride 224 overlap) | **Setting B** (stride 224 overlap) | **0.7360** | **0.6801** | **+5.59 điểm** | `results/B0_Crack500_Run3_Results.md` |
| **B0 50ep** | `best_model_b0.pth` | 50 epochs | Epoch 41 (chạy đủ 50 ep) | **direct** (1-pass resize 448) | **direct** (1-pass resize 448) | **0.7789** | **0.7042** | **+7.47 điểm** | `results/baselines/50ep/B0_Crack500/training_completion.json` và `official_eval_test.json` |
| **B0 100ep** | `best_model_b0.pth` | 100 epochs | Epoch 45 (dừng ở ep 60) | **direct** (1-pass resize 448) | **direct** (1-pass resize 448) | **0.7806** | **0.7037** | **+7.69 điểm** | `results/baselines/100ep/B0_Crack500/training_completion.json` và `official_eval_all_splits.json` |

---

## 2. Giải Mã Sự Không Khớp Số Liệu B0 50ep: 0.7042 (Rev 3/4) vs 0.6974 (Rev 5)

### A. Truy vết tệp nguồn
1. Kiểm tra trực tiếp tệp `results/baselines/50ep/B0_Crack500/official_eval_test.json`:
   ```json
   {
     "precision": 0.6823106871526812,
     "recall": 0.7765935073057353,
     "dice": 0.7041702268606304,
     "iou": 0.563296435855112,
     "loss": 0.670171183875341
   }
   ```
   Giá trị Dice chính thức của tập Test cho B0 50ep được ghi nhận chính xác tuyệt đối là **0.704170** (làm tròn: **0.7042**).
2. Kiểm tra nguồn gốc con số `0.6974`:
   Trong kho lưu trữ, số `0.697452` chỉ xuất hiện tại dòng 192 của tệp `results/final_protocol/05_test_final_report.json` — đây là điểm Dice của một mẫu crop đơn lẻ (crop index 190) của mô hình LOCKED, hoàn toàn không liên quan đến B0 50ep.
3. **Kết luận:**
   - Số **0.7042** ở Revision 3 và 4 là **con số chuẩn xác duy nhất** được lưu trong tệp kết quả gốc của B0 50ep.
   - Số `0.6974` ở Revision 5 là lỗi gõ nhầm (clerical typo) và bị bãi bỏ hoàn toàn trong Revision 6.

---

## 3. Nhận Định So Sánh Đúng Giao Thức

1. **Hiệu năng Baseline B0 tăng rõ rệt theo ngân sách epoch:**
   - Huấn luyện 30 epochs (B0 Run 3): Val Setting A = 0.7318, Test Setting A = 0.6771.
   - Huấn luyện 50 epochs (B0 50ep): Val direct = 0.7789, Test direct = **0.7042**.
   - Huấn luyện 100 epochs (B0 100ep): Val direct = **0.7806**, Test direct = **0.7037**.
2. **Khoảng cách Val $\to$ Test của B0 (cùng direct 1-pass):**
   - B0 50ep: Gap = $0.7789 - 0.7042 = \mathbf{+7.47}$ điểm.
   - B0 100ep: Gap = $0.7806 - 0.7037 = \mathbf{+7.69}$ điểm.
   - B2 Canonical run2: Gap = $0.7793 - 0.7149 = \mathbf{+6.44}$ điểm.
   - B2 LOCKED: Gap = $0.7853 - 0.7158 = \mathbf{+6.96}$ điểm.
3. Khi so sánh trên cùng giao thức direct 1-pass:
   - Điểm Val của B0 100ep (**0.7806**) cao hơn B2 Canonical (**0.7793**) là $+0.13$ điểm.
   - Điểm Test của B2 Canonical (**0.7149**) cao hơn B0 100ep (**0.7037**) là $+1.12$ điểm (chưa kiểm định ghép cặp).
