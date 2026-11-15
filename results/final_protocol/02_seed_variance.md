# Phase 2 — Phương sai Seed thật (Multi-Seed Variance)

**Trạng thái:** ✅ **HOÀN THÀNH**  
**Cập nhật:** 2026-11-15 00:52  
**Nguồn dữ liệu:** Huấn luyện độc lập 3 seed trên Tesla T4 (Google Colab) thông qua notebook [`phase2_seed_template.ipynb`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/phase2_seed_template.ipynb) và giải nén vào thư mục chính của dự án.

---

## 1. Lưu trữ Checkpoints & Artifacts theo Protocol

Toàn bộ artifacts của 3 run huấn luyện độc lập đã được đưa về thư mục chính của project:
- Thư mục Seed 42: [`results/final_protocol/phase2_seed42/`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/phase2_seed42/)
- Thư mục Seed 43: [`results/final_protocol/phase2_seed43/`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/phase2_seed43/)
- Thư mục Seed 44: [`results/final_protocol/phase2_seed44/`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/phase2_seed44/)

Mỗi thư mục chứa đầy đủ:
1. `config_snapshot.yaml`: Cấu hình canonical đóng băng.
2. `git_hash.txt`: Mã commit Git nguồn (`e8598ea`).
3. `sha256_checkpoint.txt`: Mã băm SHA-256 của trọng số tốt nhất (`best_model_b2_global.pth`).
4. `run_metadata.json`: Metadata kiểm toán và metrics nghiệm thu.
5. `train.log`: Nhật ký huấn luyện chi tiết 35 epochs (Stage 1: 17 ep, Stage 2: 18 ep).
6. Checkpoint cặp đôi: `best_model_b2_global.pth`, `best_model_b2_stage1.pth`, `best_model_b2_stage2.pth`, `last_model_b2_stage1.pth`, `last_model_b2_stage2.pth`.
7. `routing_diagnostics/`: Ma trận định tuyến router và thống kê Gumbel-Softmax (`routing_statistics.json`, `expert_usage.csv`, `family_routing.csv`).

### Bảng định danh Checkpoint (SHA-256)

| Seed | File Checkpoint (`best_model_b2_global.pth`) | SHA-256 Checksum |
|:---:|:---|:---|
| **42** | `results/final_protocol/phase2_seed42/best_model_b2_global.pth` | `4B409B60A7DF788AE48DECE727751E1B8D904CD95EE1102E53D78563F2556CB7` |
| **43** | `results/final_protocol/phase2_seed43/best_model_b2_global.pth` | `7379C7346616ACE3569AE2D9D640D838098638AD70F17443A34166AC89D4A271` |
| **44** | `results/final_protocol/phase2_seed44/best_model_b2_global.pth` | `F5F7DC55C9D0CBEB606D5EEA396E6B8983ADC64391F23FE96395A19C5EB75F09` |

---

## 2. Kết quả Đánh giá Validation Set (348 ảnh) từng Seed

*(Dữ liệu từ `val_result.json`, đánh giá trực tiếp 1-pass `predict_full_image_direct` tại ngưỡng $\tau = 0.5$ chuẩn)*

| Chỉ số | Seed 42 | Seed 43 | Seed 44 | Trung bình ($\mu$) | Độ lệch chuẩn mẫu ($s$) | Min | Max |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Val Dice** | **0.780409** | **0.782932** | **0.772572** | **0.778638** | **0.005402** | 0.772572 | 0.782932 |
| **Val Macro IoU** | 0.659005 | 0.662575 | 0.648723 | 0.656768 | 0.007205 | 0.648723 | 0.662575 |
| **Global Pixel IoU**| 0.701305 | 0.704006 | 0.698232 | 0.701181 | 0.002889 | 0.698232 | 0.704006 |
| **Precision** | 0.752963 | 0.754663 | 0.736989 | 0.748205 | 0.009748 | 0.736989 | 0.754663 |
| **Recall** | 0.845736 | 0.848121 | 0.848417 | 0.847425 | 0.001471 | 0.845736 | 0.848417 |
| **Boundary IoU** | 0.134821 | 0.136103 | 0.135830 | 0.135585 | 0.000676 | 0.134821 | 0.136103 |
| **HD95 (px)** | 38.08 px | 37.83 px | 38.74 px | 38.22 px | 0.47 px | 37.83 px | 38.74 px |
| **Stage 1 Best Epoch** | Ep 17 (0.7630)| Ep 14 (0.7609)| Ep 14 (0.7477)| — | — | — | — |
| **Stage 2 Best Epoch** | Ep 15 (0.7804)| Ep 18 (0.7829)| Ep 14 (0.7726)| — | — | — | — |

---

## 3. Xác định Ngưỡng Phát hiện Tối thiểu (MDE)

Độ lệch chuẩn thực tế giữa 3 seed huấn luyện lại từ đầu:
$$s_{\text{seed}} = 0.005402$$

Ngưỡng phát hiện tối thiểu (Minimum Detectable Effect) theo quy tắc protocol:
$$\text{MDE} = 2 \times s_{\text{seed}} = 2 \times 0.005402 = 0.010804 \quad (\approx 1.08\% \text{ Dice})$$

### Kiểm định lại các giả định và kết quả trước đó theo MDE:
1. **Lợi ích TTA Flip (`flip_hv` ở Phase 1):**
   - Mức tăng ghi nhận: $\Delta = +0.003158$ (95% CI: $[+0.001021, +0.005301]$).
   - So sánh với MDE: $\Delta = 0.003158 < \text{MDE} = 0.010804$.
   - **Kết luận theo Rule 3:** Dù CI bootstrap ghép cặp trên cùng một checkpoint loại trừ 0, mức tăng $+0.0032$ nằm trong phạm vi phương sai ngẫu nhiên giữa các lần khởi tạo seed ($s = 0.0054$). Chưa đủ bằng chứng để khẳng định vượt trội hơn phương sai của seed.
2. **Khác biệt Run 1 vs Run 2 trong Phase 7 cũ:**
   - Chênh lệch cũ: $|\Delta| = 0.0010$.
   - So sánh: $0.0010 \ll \text{MDE} = 0.0108$. Hoàn toàn là nhiễu nội tại trong quá trình tối ưu hóa.

---

## 4. Đánh giá Ensemble 3 Seed (Trung bình Xác suất)

Script đánh giá thực nghiệm: [`run_ensemble_phase2.py`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/run_ensemble_phase2.py)  
File kết quả lưu trữ: [`02_ensemble_val_result.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/02_ensemble_val_result.json)

| Phương pháp | Val Dice | Precision | Recall | Macro IoU | Global Pixel IoU | Boundary IoU | HD95 (px) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Seed 42** | 0.780409 | 0.7530 | 0.8457 | 0.6590 | 0.7013 | 0.1348 | 38.08 px |
| **Seed 43** (Best đơn lẻ) | 0.782932 | 0.7547 | 0.8481 | 0.6626 | 0.7040 | 0.1361 | 37.83 px |
| **Seed 44** | 0.772572 | 0.7370 | 0.8484 | 0.6487 | 0.6982 | 0.1358 | 38.74 px |
| **Trung bình Đơn lẻ ($\mu$)** | **0.778638** | 0.7482 | 0.8474 | 0.6568 | 0.7012 | 0.1356 | 38.22 px |
| **Ensemble (3 Seeds)** | **0.785003** | **0.7540** | **0.8532** | **0.6647** | **0.7061** | 0.1340 | **36.05 px** |
| **$\Delta$ (Ensemble vs $\mu$)** | **+0.006365** | +0.0058 | +0.0058 | +0.0079 | +0.0049 | -0.0016 | **-2.17 px** |
| **$\Delta$ (Ensemble vs Best 43)**| **+0.002071** | -0.0007 | +0.0051 | +0.0021 | +0.0021 | -0.0021 | **-1.78 px** |

---

## 5. Danh mục File Đầu ra

- [`results/final_protocol/02_phase2_summary.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/02_phase2_summary.json)
- [`results/final_protocol/02_ensemble_val_result.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/02_ensemble_val_result.json)
- [`results/final_protocol/phase2_seed42/`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/phase2_seed42/)
- [`results/final_protocol/phase2_seed43/`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/phase2_seed43/)
- [`results/final_protocol/phase2_seed44/`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/phase2_seed44/)
