# Báo Cáo Đối Soát Tô-Pô Hình Thái: Phase 6 vs Phase 7

| Chỉ số Tô-pô & Hình thái | Phase 6 (Chỉ có S2G Block 1) | Phase 7 (Dual S2G: Block 1 + Block 2) | Chênh lệch ($\Delta$ P7 - P6) | Nhận định Giả thuyết |
| :--- | :---: | :---: | :---: | :--- |
| **Validation Dice** | **0.7676** | **0.7586** | `-0.0090` | Giảm |
| **Centerline Dice (clDice)** | **0.8440** | **0.8336** | `-0.0103` | Suy giảm độ liền trục |
| **Boundary IoU** | **0.1352** | **0.1317** | `-0.0036` | - |
| **HD95 (px, thấp hơn là tốt)**| **70.15 px** | **93.07 px** | `+22.93 px` | Biên sai lệch lớn hơn |
| **Số sự kiện Gãy đứt (Break events)** | **40** | **46** | `+6` | KHÔNG ĐẠT: Tăng đứt đoạn |
| **Ảnh bị gãy đứt vết nứt** | **38 (10.9%)** | **44 (12.6%)** | `+6` | - |
| **Số sự kiện Nối dính giả (Bridge events)** | **120** | **117** | `-3` | ĐẠT: Giảm cầu nối giả |
| **Ảnh bị dính cầu nối giả** | **112 (32.2%)** | **110 (31.6%)** | `-2` | - |
| **Đảo nhiễu giả (Spurious islands)** | **189** | **242** | `+53` | - |
| **Precision** | **0.7407** | **0.7258** | `-0.0148` | - |
| **Recall** | **0.8384** | **0.8400** | `+0.0016` | - |

