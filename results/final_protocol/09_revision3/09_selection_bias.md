# 09_selection_bias.md — Độ Lạc Quan Do Chọn Epoch Tốt Nhất Trên Val (Revision 3)

**Ngày:** 2026-11-17  
**Nguồn:** `results/final_protocol/phase2_results/phase2_seed{42,43,44}/train.log` (đã parse); `results/B0_Crack500_Run3_Results.md`

---

## 1. Phase 2 — Epoch-by-epoch Val Dice (3 seed)

Từ `train.log` của từng seed — INFO lines `Epoch X/18 ... Val Dice: Y`:

### Seed 42 (best epoch = 15, val = 0.7804)

| Epoch | Val Dice | Epoch | Val Dice |
|:---:|:---:|:---:|:---:|
| 1 | 0.7625 | 10 | 0.7719 |
| 2 | 0.7497 | 11 | 0.7697 |
| 3 | 0.7472 | 12 | 0.7625 |
| 4 | 0.7602 | 13 | 0.7765 |
| 5 | 0.7474 | 14 | 0.7785 |
| 6 | 0.7608 | **15** | **0.7804 ← BEST** |
| 7 | 0.7689 | 16 | 0.7778 |
| 8 | 0.7647 | 17 | 0.7768 |
| 9 | 0.7677 | 18 | 0.7785 |

> **Ghi chú:** `stage2_completion.json` ghi `best_dice = 0.7793` (run2 epoch 16) nhưng đây là checkpoint *run2* (phase7_combination). Seed 42 Phase 2 (`phase2_results/phase2_seed42`) có `best_epoch=15, best_dice_from_log=0.7804`. Hai số khác nhau vì đây là hai training run khác nhau (Phase 2 vs Phase 7 run2).

### Seed 43 (best epoch = 18, val = 0.7829)

| Epoch | Val Dice | Epoch | Val Dice |
|:---:|:---:|:---:|:---:|
| 1 | 0.7598 | 10 | 0.7662 |
| 2 | 0.7443 | 11 | 0.7750 |
| 3 | 0.7597 | 12 | 0.7769 |
| 4 | 0.7511 | 13 | 0.7793 |
| 5 | 0.7617 | 14 | 0.7778 |
| 6 | 0.7597 | 15 | 0.7794 |
| 7 | 0.7644 | 16 | 0.7786 |
| 8 | 0.7705 | 17 | 0.7822 |
| 9 | 0.7664 | **18** | **0.7829 ← BEST** |

### Seed 44 (best epoch = 14, val = 0.7726)

| Epoch | Val Dice | Epoch | Val Dice |
|:---:|:---:|:---:|:---:|
| 1 | 0.7406 | 10 | 0.7465 |
| 2 | 0.7369 | 11 | 0.7622 |
| 3 | 0.7481 | 12 | 0.7685 |
| 4 | 0.7408 | 13 | 0.7627 |
| 5 | 0.7215 | **14** | **0.7726 ← BEST** |
| 6 | 0.7264 | 15 | 0.7675 |
| 7 | 0.7314 | 16 | 0.7700 |
| 8 | 0.7560 | 17 | 0.7719 |
| 9 | 0.7502 | 18 | 0.7717 |

---

## 2. Tính Selection Bias

**Phương pháp 1 — Best trừ trung bình tất cả epoch:**

| Seed | Best epoch | Best val | Mean tất cả (18 epoch) | Bias (điểm) |
|:---:|:---:|:---:|:---:|:---:|
| 42 | 15 | 0.7804 | 0.7668 | **1.36 điểm** |
| 43 | 18 | 0.7829 | 0.7687 | **1.42 điểm** |
| 44 | 14 | 0.7726 | 0.7525 | **2.01 điểm** |
| **Trung bình** | — | — | — | **1.60 điểm** |

**Phương pháp 2 — Best trừ trung bình 5 epoch cuối (epoch 14–18):**

| Seed | Best epoch | Best val | Mean 5 epoch cuối | Bias (điểm) |
|:---:|:---:|:---:|:---:|:---:|
| 42 | 15 | 0.7804 | 0.7784 | **0.20 điểm** |
| 43 | 18 | 0.7829 | 0.7802 | **0.27 điểm** |
| 44 | 14 | 0.7726 | 0.7707 | **0.19 điểm** |
| **Trung bình** | — | — | — | **0.22 điểm** |

---

## 3. So sánh với B0 Run3 (tham chiếu)

Từ `results/B0_Crack500_Run3_Results.md` (epoch table, epochs 1–14):

- Best epoch 8: val = 0.7318
- Mean tất cả 14 epoch: (0.2398+0.6090+0.6403+0.6945+0.6735+0.6842+0.7026+0.7318+0.7288+0.7221+0.7170+0.7220+0.7293+0.7134)/14 = **0.6935**
- Bias B0 phương pháp 1: 0.7318 − 0.6935 = **3.83 điểm**

> B0 có bias lớn hơn vì học từ epoch 1 tăng mạnh (epoch 1 val=0.2398 — warmup), nên trung bình bị kéo xuống thấp. Phase 2 đã fine-tune từ checkpoint tốt nên biến động nhỏ hơn.

---

## 4. Giải thích

**Selection bias ~1.60 điểm (phương pháp 1)** là ước lượng thô: val được đo trên cùng 50 ảnh gốc để chọn epoch, nên best val có thể lạc quan hơn val thực trên phân phối rộng hơn. Mức bias nhỏ hơn khi tính theo 5 epoch cuối (0.22 điểm) vì model đã hội tụ, biến động giữa epoch nhỏ.

**Giới hạn:**
- Đây là bias do chọn epoch trên val, không phải bias test. Val và test khác phân phối nguồn ảnh (50 vs 148 ảnh gốc), nhưng không đo được ảnh hưởng riêng của từng yếu tố.
- Seed 44 có bias cao nhất (2.01 điểm) do val dice epoch 5-6 thấp bất thường (0.7215, 0.7264), kéo trung bình xuống.

**Nguồn:** `results/final_protocol/phase2_results/phase2_seed{42,43,44}/train.log` — INFO lines pattern `Epoch X/18 ... Val Dice: Y`
