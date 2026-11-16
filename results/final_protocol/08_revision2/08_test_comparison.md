# 08_test_comparison.md — So sánh Test (Revision 2)

**Nguồn:** `results/phase7_combination/`, `results/final_protocol/`  
**Ngày viết:** 2026-11-16

---

## 1. Xác định đối tượng so sánh

### 0.7149 — Canonical Phase 7 Run 2, direct 1-pass
- **Checkpoint:** `results/phase7_combination/phase7_s2g_new_pipeline_run2/diagnostics/official_direct_evaluation.json`
  - Dice = `0.7148690704104006`
- **Checkpoint metadata:** `stage2_completion.json` → epoch=16, best_dice=0.7793468928939803
- **Seed:** 42 (xác nhận từ PROGRESS.md: "Canonical checkpoint: phase7_s2g_new_pipeline_run2/best_model_b2_stage2.pth")
- **Val Dice (tái lập Phase 0):** 0.779294 (source: `results/final_protocol/00_audit/audit_result.json`)
- **SHA256:** `68BF82F6B6F4F16E4032F1B58A2C3D5F3A3C2E9911F1020F5C5356713DCC98DB` (source: `PROGRESS.md`)
- **Giao thức:** Direct full-image 1-pass, τ=0.5, **không TTA**, **không ensemble**
- **Ngày:** Không rõ ngày chạy inference, checkpoint train xong 2026-10-17

### 0.7158 — Ensemble Phase 2, 3 seed, TTA flip_hv (Phase 5 LOCKED)
- **File nguồn:** `results/final_protocol/05_test_final_report.json`
  - Dice = 0.7158 (mean crop-level, 1,124 crops)
- **Checkpoint:** Ensemble 3 model (seed 42, 43, 44 — Phase 2 canonical)
  - SHA256: seed42=`4B409B...`, seed43=`7379C7...`, seed44=`F5F7DC...`
- **Giao thức:** Direct 1-pass + TTA flip_hv, τ=0.5, ensemble trung bình 3 model
- **Ngày LOCK:** Trước Phase 5 (cấu hình trong `05_frozen_config.json`)

---

## 2. Khác biệt giao thức — không so sánh trực tiếp được

| Thuộc tính | 0.7149 (run2 canonical) | 0.7158 (ensemble LOCKED) |
|---|---|---|
| Số checkpoint | 1 (seed=42) | 3 (seed 42, 43, 44) |
| TTA | Không | flip_hv |
| Giao thức | Direct 1-pass | Direct 1-pass + TTA + ensemble |
| Val Dice tương ứng | 0.7793 (seed=42, no TTA) | 0.7850 (ensemble + TTA) |

**Kết luận:** Δ = 0.7158 − 0.7149 = **+0.0009 (0.09 điểm)** phản ánh đồng thời tác động của TTA VÀ ensemble VÀ thay đổi checkpoint set. Không thể phân tách. Không nên kết luận "TTA/ensemble cải thiện 0.09 điểm ở test".

---

## 3. So sánh hợp lệ — cùng checkpoint, cùng giao thức

| Metric | Val | Test | Khoảng cách |
|---|:---:|:---:|:---:|
| Dice (seed=42, no TTA, direct 1-pass) | **0.7793** | **0.7149** | **0.0644 Dice = 6.44 điểm** |

- Val Dice 0.7793 = run2 canonical audit (Phase 0): `results/final_protocol/00_audit/audit_result.json`, val_dice=0.779294 (tái lập), claimed 0.779347.
- Test Dice 0.7149 = `official_direct_evaluation.json`: dice=0.7148690704104006, cùng checkpoint, cùng giao thức direct 1-pass.
- **Đây là so sánh đúng nhất hiện có: cùng 1 checkpoint, cùng giao thức.**

---

## 4. Lỗi trong Revision 1 cần sửa

| Lỗi | Sửa |
|---|---|
| "Canonical đơn-seed val tốt nhất (seed 43) = 0.7149" | 0.7149 là seed=**42**, run2, direct 1-pass. Seed 43 không có con số test nào. |
| "Δ = +0.0009" diễn giải như hiệu quả TTA/ensemble | Δ = +0.09 điểm giữa hai đối tượng **hoàn toàn khác nhau** (1 ckpt no TTA vs 3-ckpt+TTA). Không rút ra kết luận về TTA/ensemble. |
| "Val = 0.7786 (mean 3 seed) vs Test = 0.7158 → 6.3 điểm" | So sánh đúng hơn: seed=42 val=0.7793 vs test=0.7149 → **6.44 điểm** (cùng checkpoint, cùng giao thức). Dòng ensemble (val=0.7850 vs test=0.7158 → 6.92 điểm) giữ nhưng ghi rõ hai đối tượng khác nhau. |

---

## 5. Test của seed 43 (Phase 2)

**Chưa bao giờ đo.** Không có file JSON test nào tương ứng với Phase 2 seed=43 checkpoint riêng lẻ.  
Mọi con số test đã có đều là: canonical run2 seed=42 (0.7149), hoặc ensemble 3-seed Phase 2 (0.7158), hoặc các checkpoint Phase 7 trước đó.
