# 08_phase3_description.md — Mô tả Phase 4 và Phase 6 (Revision 2)

**Nguồn:** `docs/Crack500_Experimental_Protocol_Roadmap.md`, `results/final_protocol/PROGRESS.md`  
**Ngày viết:** 2026-11-16

---

## 1. Lỗi trong Revision 1

Revision 1 (§7) ghi:
> "Phase 4 (domain adaptation/augmentation) và Phase 6 (inference optimization/quantization) chưa thực hiện."

Cả hai mô tả đều sai. Dưới đây là mô tả đúng từ nguồn gốc.

---

## 2. Phase 4 — Lịch huấn luyện (Learning Rate Screening)

**Nguồn:** `docs/Crack500_Experimental_Protocol_Roadmap.md` §4.4:
> "PHASE 4 — Sàng Lọc Tốc Độ Học (Learning Rate Screening)"  
> Screening 3 LR: 5e-5, 1e-4, 2e-4; chạy 8–10 epoch; chọn LR tốt nhất.

**Nguồn:** `results/final_protocol/PROGRESS.md`:
> "Phase 4 — Lịch huấn luyện | 🔒 Tùy chọn"

**Mô tả đúng:** Phase 4 là **sàng lọc tốc độ học (LR screening)**, tùy chọn.  
Không phải domain adaptation. Không phải augmentation.  
**Trạng thái:** Chưa thực hiện (đánh dấu "Tùy chọn").

---

## 3. Phase 6 — Dataset tường riêng (Cross-dataset Evaluation)

**Nguồn:** `docs/Crack500_Experimental_Protocol_Roadmap.md` §4.6:
> "PHASE 6 — Xác Nhận Toàn Diện (Full Confirmation)"  
> Huấn luyện xác nhận toàn diện cho candidate tốt nhất (1 config, 30 epoch).

**Nguồn:** `results/final_protocol/PROGRESS.md`:
> "Phase 6 — Dataset tường riêng | 🔒 Chờ nhãn"

**Lưu ý về hai tài liệu:** `Crack500_Experimental_Protocol_Roadmap.md` là roadmap ban đầu (Phase 6 = Full Confirmation trong Crack500). `PROGRESS.md` phản ánh kế hoạch thực tế hiện tại của final_protocol, trong đó Phase 6 đã được tái định nghĩa thành **dataset tường riêng (cross-dataset evaluation, chờ nhãn)**.

**Mô tả đúng (theo PROGRESS.md — kế hoạch thực tế):** Phase 6 là **đánh giá trên dataset tường riêng (cross-dataset)**, chờ nhãn.  
Không phải inference optimization. Không phải quantization.  
**Trạng thái:** Chưa thực hiện (chờ nhãn từ dataset tường).

---

## 4. Sửa trong báo cáo

Thay dòng trong §7 Revision 1:
> ~~"Phase 4 (domain adaptation/augmentation) và Phase 6 (inference optimization/quantization) chưa thực hiện."~~

Bằng:
> "Phase 4 (sàng lọc tốc độ học — LR screening, tùy chọn) và Phase 6 (đánh giá cross-dataset — chờ nhãn) chưa thực hiện."
