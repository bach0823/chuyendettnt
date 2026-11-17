# 10_phase3_reconcile.md — Đối Chiếu Số Liệu Từng Seed Phase 3 (Revision 4)

**Ngày:** 2026-11-17  
**Nguồn dữ liệu:** `results/final_protocol/phase3_variant*/run_metadata.json`, `phase2_results/phase2_seed*/run_metadata.json`.  
**Script trích xuất:** `results/final_protocol/10_revision4/scratch/dump_phase3_raw.py` và `compute_sha256.py`.

---

## 1. Đối chiếu Số liệu Từng Seed Giữa Các Phiên bản Báo cáo

Bảng đối chiếu giá trị Val Dice từng seed xuất hiện trong Revision 1, Revision 3 và file gốc `run_metadata.json`:

| Cấu hình | Seed | Giá trị trong Rev 1 | Giá trị trong Rev 3 | Giá trị GỐC (`run_metadata.json`) | Đánh giá sai lệch | SHA-256 Checkpoint (`best_model_b2_global.pth`) |
|---|:---:|:---:|:---:|:---:|---|---|
| **Variant D** | 42 | 0.773668 | 0.7775 | **0.773668** | Rev 1 đúng gốc; Rev 3 bị làm tròn/lệch (+0.38 điểm) | `38383C9EBF8E5BAFF3553BEA1F7524D61E109214446184C02A2CB58A30C71C3B` |
| **Variant D** | 43 | 0.782093 | 0.7783 | **0.782093** | Rev 1 đúng gốc; Rev 3 bị làm tròn/lệch (−0.38 điểm) | `A24A905D709AA8B44ECA25842942230B432A366F42A37AA6CC1F1D2FCDBBF68B` |
| **Variant R** | 42 | 0.779086 | 0.7788 | **0.779086** | Rev 1 đúng gốc; Rev 3 bị điều chỉnh (−0.03 điểm) | `F3043B6EA57804648FD49CA00E10914A66F794E2AFBC4B25CC321BFB8C5BC76C` |
| **Variant R** | 43 | 0.777913 | 0.7782 | **0.777913** | Rev 1 đúng gốc; Rev 3 bị điều chỉnh (+0.03 điểm) | `CB9CF335DF663D8EDB24CDF49B9A0FE1F214A95E64CD920DA66C27B2266D311F` |
| **Variant N** | 42 | 0.778212 | 0.778212 | **0.778212** | Cả Rev 1 và Rev 3 đều đúng gốc | `161C9EF30C031602466E86E9412DF3E316BC63ECC16EF93DEB169BD3B9CC6FD7` |
| **Variant N** | 43 | 0.777812 | 0.777812 | **0.777812** | Cả Rev 1 và Rev 3 đều đúng gốc | `756CC7A00378554A73D898ACEFAC2574F960C64B0D42957077ED5CBE4FC5CE9A` |
| **Canonical** | 42 | 0.780409 | 0.780409 | **0.780409** | Khớp file nguồn | `4B409B60A7DF788AE48DECE727751E1B8D904CD95EE1102E53D78563F2556CB7` |
| **Canonical** | 43 | 0.782932 | 0.782932 | **0.782932** | Khớp file nguồn | `7379C7346616ACE3569AE2D9D640D838098638AD70F17443A34166AC89D4A271` |
| **Canonical** | 44 | 0.772572 | 0.772572 | **0.772572** | Khớp file nguồn | `F5F7DC55C9D0CBEB606D5EEA396E6B8983ADC64391F23FE96395A19C5EB75F09` |

> [!NOTE]
> **Kết luận đối chiếu:** Giá trị từng seed trong Revision 1 là số đo thực tế lấy trực tiếp từ `run_metadata.json`. Bản Revision 3 đã đưa vào các số làm tròn nhân tạo cho Variant D và R (0.7775 / 0.7783 và 0.7788 / 0.7782) mà không kiểm tra lại file gốc. Báo cáo Revision 4 khôi phục 100% số liệu gốc từ file JSON.

---

## 2. Độ Phân Tán Đa Seed và Hệ Quả đối với Ngưỡng MDE

| Cấu hình | Seed 42 | Seed 43 | Seed 44 | Trung bình ($\mu$) | Độ phân tán giữa các seed (Max − Min) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Variant D** | 0.773668 | 0.782093 | — | **0.777881** | **0.008425 (0.84 điểm)** |
| **Variant R** | 0.779086 | 0.777913 | — | **0.778500** | **0.001173 (0.12 điểm)** |
| **Variant N** | 0.778212 | 0.777812 | — | **0.778012** | **0.000400 (0.04 điểm)** |
| **Canonical (2 seed)** | 0.780409 | 0.782932 | — | **0.781670** | **0.002523 (0.25 điểm)** |
| **Canonical (3 seed)** | 0.780409 | 0.782932 | 0.772572 | **0.778638** | **0.010360 (1.04 điểm)** |

### Hệ quả đối với việc kiểm định MDE:
1. **Độ phân tán của Variant D rất lớn:**  
   Khoảng chênh lệch giữa hai seed của Variant D lên tới **0.84 điểm** (Seed 42 là 0.7737 nhưng Seed 43 là 0.7821), gấp 11 lần mức chênh lệch trung bình giữa Variant D và Canonical (−0.08 điểm). Mức độ biến thiên này gần bằng toàn bộ độ rộng của $\text{MDE} = 1.08$ điểm.
2. **Cỡ mẫu 2 seed không đủ sức mạnh phân biệt các mức chênh lệch nhỏ:**  
   Với phương sai lớn giữa các seed, việc các biến thể chênh lệch nhau từ 0.01 đến 0.38 điểm hoàn toàn nằm trong phạm vi nhiễu ngẫu nhiên của việc khởi tạo trọng số và xáo trộn batch dữ liệu.
3. **Ý nghĩa khoa học:**  
   Không thể đưa ra kết luận rằng Variant D "kém hơn" hay "tốt hơn" Canonical ở mức 0.08 điểm vì sự biến động giữa các seed nội tại của chính nó đã là 0.84 điểm. Khẳng định chuẩn xác duy nhất theo nguyên tắc thống kê là: **"Không phát hiện sự khác biệt vượt quá ngưỡng MDE = 1.08 điểm"**.
