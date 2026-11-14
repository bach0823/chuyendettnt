# Báo cáo tiến độ — Final Protocol

**Cập nhật:** 2026-11-14 15:07  
**Canonical checkpoint:** `phase7_s2g_new_pipeline_run2/best_model_b2_stage2.pth`  
SHA256: `68BF82F6B6F4F16E4032F1B58A2C3D5F3A3C2E9911F1020F5C5356713DCC98DB`

---

## Tổng quan theo phase

| Phase | Mô tả | Trạng thái | Kết quả chính |
|---|---|---|---|
| **0** | Kiểm toán & tái lập | ✅ XONG | Val Dice tái lập = 0.779294, Δ=−0.000053 (**PASS**). Tất định. Run1≡Run2 (cùng seed=42). |
| **1** | Tau sweep + TTA + bootstrap | ✅ XONG | tau thay đổi KHÔNG có nghĩa. TTA flip_hv **CÓ nghĩa** (CI=[+0.001,+0.005]). |
| **2** | Phương sai seed thật | ⏸ **CHỜ COLAB** | Notebook sẵn sàng. Cần train 3 seed. |
| **5** | Test đúng 1 lần | 🔒 Khóa đến sau Phase 2 | — |
| **3** | Đối chứng router/expert | 🔒 Chờ MDE từ Phase 2 | — |
| **6** | Dataset tường riêng | 🔒 Chờ nhãn | — |
| **4** | Lịch huấn luyện | 🔒 Tùy chọn | — |

---

## Phase 0 — Kết quả chi tiết

| Mục | Kết quả |
|---|---|
| SHA256 | `68BF82F6...CC98DB` |
| Params | 10,205,755 (10.206M) |
| Val Dice tái lập @ tau=0.5 | **0.779294** (claimed 0.779347, Δ=−0.000053 ✅) |
| Precision / Recall | 0.7525 / 0.8446 |
| Boundary IoU | 0.1318 |
| HD95 | 38.69 px |
| Tính tất định | ✅ PASS (run1 = run2 bit-for-bit) |
| Run1 ≠ Run2 về seed | ❌ Cả hai là seed=42 — không phải multi-seed độc lập |

---

## Phase 1 — Kết quả chi tiết

### Tau sweep

| tau | Val Dice | Δ vs 0.5 | CI bootstrap | Quyết định |
|---|---|---|---|---|
| 0.5 | 0.779294 | — | — | Baseline |
| 0.6 | 0.780141 | +0.000847 | [−0.000234, +0.001985] | ❌ Không nhận (CI chứa 0) |

Plateau thực tế: tau=0.550→0.650 biên độ Dice < 0.0002.

### TTA

| TTA | Val Dice @ tau=0.5 | Δ vs no-TTA | CI | Quyết định |
|---|---|---|---|---|
| none | 0.779294 | — | — | Baseline |
| flip_h | 0.781055 | +0.001761 | — | — |
| flip_v | 0.781611 | +0.002317 | — | — |
| **flip_hv** | **0.782452** | **+0.003158** | **[+0.001021, +0.005301]** | ✅ **Nhận** |

TTA flip_hv tại tau=0.6: Δ=+0.003263, CI=[+0.001073,+0.005360] ✅

### Stability (20 lần 2 nửa val)

| Metric | Giá trị |
|---|---|
| Mean cross-eval Dice | 0.779867 |
| Std cross-eval Dice | 0.006081 |

Tau chọn ổn định trong dải 0.550–0.650. Nhưng CI tau change vẫn chứa 0 → giữ tau=0.5.

### Thin Crack Dice

Tại tau=0.5 baseline: **0.3532** — thấp. Tăng tau hoặc TTA chỉ cải thiện lên ~0.361.  
Đây là giới hạn của kiến trúc, không giải quyết được ở inference.

### Quyết định Phase 1

- **Giữ: tau=0.5**
- **Nhận: TTA flip_hv** (CI dương rõ ở cả tau=0.5 và 0.6)
- Val Dice với flip_hv @ tau=0.5: **~0.782** (+0.003 vs baseline)

> [!IMPORTANT]
> TTA flip_hv đã vượt CI bootstrap nhưng **chưa được so với MDE**. MDE = 2×std_seed chỉ có sau Phase 2. Nếu std_seed > 0.0016, TTA không vượt MDE và cần đánh giá lại.

---

## Phase 2 — Việc cần làm trên Colab

**File notebook:** [`results/final_protocol/phase2_seed_variance.ipynb`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/phase2_seed_variance.ipynb)

### Các bước:

1. Upload notebook lên Colab (hoặc mở từ Drive).
2. Sửa 3 đường dẫn trong Cell 1:
   - `DRIVE_ROOT` / `SAGE_LITE_CODE`: thư mục chứa code SAGE_LITE trên Drive.
   - `DATASET_ROOT`: thư mục chứa `Crack500_ready/train`, `val`, `test`.
   - `PHASE2_OUT`: thư mục lưu kết quả.
3. Chọn runtime GPU (T4 hoặc A100).
4. Chạy Cell 1→5 một lần.
5. **Seed 42:** Đặt `SEED=42` trong Cell 6, chạy Cell 6 → Cell 7. Lưu kết quả.
6. **Seed 43:** Đặt `SEED=43`, chạy lại Cell 6 → Cell 7.
7. **Seed 44:** Đặt `SEED=44`, chạy lại Cell 6 → Cell 7.
8. Chạy Cell 8 (tổng hợp + tính MDE).
9. Tùy chọn: Cell 9 (ensemble 3 seed).
10. Download/sync `PHASE2_OUT/phase2_summary.json` về local.

### Đầu ra cần lấy về

```
phase2_results/
├── phase2_seed42/
│   ├── best_model_b2_global.pth
│   ├── stage2_completion.json
│   ├── run_metadata.json        ← Val Dice, SHA256, git hash
│   ├── config_snapshot.yaml
│   └── train.log
├── phase2_seed43/   (tương tự)
├── phase2_seed44/   (tương tự)
└── phase2_summary.json          ← mean, std, MDE
```

### Thời gian ước tính (T4)

- 1 run ≈ 35 epoch × ~3.5 min/epoch ≈ 2 giờ.
- 3 seed = ~6 giờ (chạy tuần tự).

---

## Điều CHƯA làm được

| Hạng mục | Lý do |
|---|---|
| Phase 2 (seed variance) | Cần GPU Colab để train |
| MDE thực tế | Chờ Phase 2 |
| TTA stability qua 2 nửa val | Script Phase 1 chỉ kiểm tra stability của tau selection, không phải TTA |
| Test (1,124 ảnh) | Khóa đến sau Phase 2 + chọn cấu hình cuối |
| Phase 3 (router ablation) | Chờ MDE từ Phase 2 |
| Phase 6 (dataset tường) | Chờ nhãn + split.csv |

---

## File đầu ra (local)

| File | Mô tả |
|---|---|
| [`results/final_protocol/00_audit.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/00_audit.md) | Báo cáo Phase 0 |
| [`results/final_protocol/01_inference_only.md`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/01_inference_only.md) | Báo cáo Phase 1 |
| [`results/final_protocol/00_audit/audit_result.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/00_audit/audit_result.json) | Phase 0 raw data |
| [`results/final_protocol/01_inference_only/tau_sweep.csv`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/01_inference_only/tau_sweep.csv) | Tau sweep 17 điểm |
| [`results/final_protocol/01_inference_only/tta_sweep.csv`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/01_inference_only/tta_sweep.csv) | TTA 4 cấu hình |
| [`results/final_protocol/01_inference_only/tta_bootstrap.json`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/01_inference_only/tta_bootstrap.json) | Bootstrap CI flip_hv |
| [`results/final_protocol/01_inference_only/stability.csv`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/01_inference_only/stability.csv) | 20-rep stability |
| [`results/final_protocol/phase2_seed_template.ipynb`](file:///d:/truong/SpecialSubjectTTNT/results/final_protocol/phase2_seed_template.ipynb) | Notebook Phase 2 (3 tab song song) |
