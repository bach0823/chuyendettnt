# Phase 0 — Kiểm toán và Tái lập

**Thực hiện:** 2026-11-14 | **Script:** `results/final_protocol/00_audit/run_phase0.py`

---

## Giả định / Lựa chọn hợp lý đầu báo cáo

- Checkpoint canonical: `best_model_b2_stage2.pth` trong `phase7_s2g_new_pipeline_run2/`  
  Căn cứ: `diagnostics/final_test_summary.md` ghi rõ `Evaluated Checkpoint: best_model_b2_stage2.pth`.  
  `best_model_b2_global.pth` là file riêng (SHA256 khác), không phải alias.
- Config dùng: `configs/p3_ablation/phase7/phase7_b2_s2g_b1_b2_end_to_end.yaml`.
- Dataset root override sang đường dẫn tuyệt đối `datasets/Crack500_ready` (config dùng relative path).

---

## 1. Checkpoint

| Thuộc tính | Giá trị |
|---|---|
| **Đường dẫn** | `results/phase7_combination/phase7_s2g_new_pipeline_run2/best_model_b2_stage2.pth` |
| **SHA256** | `68BF82F6B6F4F16E4032F1B58A2C3D5F3A3C2E9911F1020F5C5356713DCC98DB` |
| **Kích thước** | 42,124,300 bytes |
| **Epoch lưu** | Stage 2 Epoch 16 |
| **best_dice trong meta** | 0.7793468928939803 |

Không có file `.partNN` → không cần ghép.

---

## 2. Config Canonical

| Tham số | Giá trị |
|---|---|
| `seed` | 42 |
| `img_size` | 448 |
| `epochs` | 35 (`stage1_epochs=17`) |
| `model` | B2 |
| `use_s2_gate` | true (k=3) |
| `use_s2_gate_block2` | true (k=3, d=2) |
| `boundary_iou_weight` | 0.50 (SoftBIoU, d=2) |
| `sage_config.top_k` | 2 |

---

## 3. Tái lập Val Dice

| | Giá trị |
|---|---|
| Claimed (`stage2_completion.json`) | 0.779347 |
| Run 1 (tái lập) | 0.779294 |
| Delta | −0.000053 |
| Tolerance | 0.0005 |
| Kết quả | ✅ PASS |

Chỉ số đầy đủ tại tau=0.5:

| Metric | Val |
|---|---|
| Dice | 0.779294 |
| Precision | 0.752526 |
| Recall | 0.844582 |
| Boundary IoU | 0.131849 |
| HD95 | 38.69 px |
| Params | 10,205,755 (10.206M) |

---

## 4. Tính tất định

- Run 2 Val Dice: 0.779294 — giống Run 1 bit-for-bit (diff = 0.00000000)
- **Kết luận: suy luận tất định.** (cudnn disabled cho GTX 16xx tại `train_crack.py:636-638`)

---

## 5. Run1 vs Run2 — Khác seed không?

**Kết luận: KHÔNG có bằng chứng khác seed.**

Bằng chứng:
- Cả hai dùng cùng config với `seed: 42`.
- `train_crack.py:627` gọi `set_seed(config.get('seed', 42))` — không có CLI override seed trong log.
- Chênh lệch Dice Stage 2 (`0.7793 − 0.7783 = 0.0010`) là noise từ optimizer + GPU non-determinism, không phải seed khác nhau có chủ đích.

**Hậu quả:** Run1 và Run2 KHÔNG được tính là 2 seed độc lập. Cả hai thực chất là seed=42. MDE chưa xác lập → Phase 2 bắt buộc train lại với seed 42, 43, 44.

---

## 6. File đầu ra

- `results/final_protocol/00_audit/audit_result.json`
- `results/final_protocol/00_audit/run_phase0.py`
