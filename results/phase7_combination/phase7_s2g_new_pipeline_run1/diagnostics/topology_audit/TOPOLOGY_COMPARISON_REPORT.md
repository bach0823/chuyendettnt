# Báo Cáo Đối Soát Tô-Pô Hình Thái: Phase 6 vs Phase 7

| Chỉ số Tô-pô & Hình thái | Phase 6 (Chỉ có S2G Block 1) | Phase 7 (Dual S2G: Block 1 + Block 2) | Chênh lệch ($\Delta$ P7 - P6) | Nhận định Giả thuyết |
| :--- | :---: | :---: | :---: | :--- |
| **Validation Dice** | **0.7676** | **0.7535** | `-0.0141` | Giảm |
| **Centerline Dice (clDice)** | **0.8440** | **0.8286** | `-0.0154` | Suy giảm độ liền trục |
| **Boundary IoU** | **0.1352** | **0.1269** | `-0.0083` | - |
| **HD95 (px, thấp hơn là tốt)**| **70.15 px** | **102.86 px** | `+32.71 px` | Biên sai lệch lớn hơn |
| **Số sự kiện Gãy đứt (Break events)** | **40** | **52** | `+12` | KHÔNG ĐẠT: Tăng đứt đoạn |
| **Ảnh bị gãy đứt vết nứt** | **38 (10.9%)** | **49 (14.1%)** | `+11` | - |
| **Số sự kiện Nối dính giả (Bridge events)** | **120** | **123** | `+3` | KHÔNG ĐẠT: Tăng cầu nối giả |
| **Ảnh bị dính cầu nối giả** | **112 (32.2%)** | **117 (33.6%)** | `+5` | - |
| **Đảo nhiễu giả (Spurious islands)** | **189** | **291** | `+102` | - |
| **Precision** | **0.7407** | **0.7110** | `-0.0296` | - |
| **Recall** | **0.8384** | **0.8499** | `+0.0115` | - |

