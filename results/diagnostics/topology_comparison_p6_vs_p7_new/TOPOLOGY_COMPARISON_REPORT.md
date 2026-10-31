# Báo Cáo Đối Soát Tô-Pô Hình Thái: Phase 6 vs Phase 7

| Chỉ số Tô-pô & Hình thái | Phase 6 (Chỉ có S2G Block 1) | Phase 7 (Dual S2G: Block 1 + Block 2) | Chênh lệch ($\Delta$ P7 - P6) | Nhận định Giả thuyết |
| :--- | :---: | :---: | :---: | :--- |
| **Validation Dice** | **0.7676** | **0.7689** | `+0.0013` | Tăng |
| **Centerline Dice (clDice)** | **0.8440** | **0.8460** | `+0.0020` | Cải thiện độ liền trục |
| **Boundary IoU** | **0.1353** | **0.1356** | `+0.0003` | - |
| **HD95 (px, thấp hơn là tốt)**| **70.12 px** | **70.42 px** | `+0.29 px` | Biên sai lệch lớn hơn |
| **Số sự kiện Gãy đứt (Break events)** | **38** | **43** | `+5` | KHÔNG ĐẠT: Tăng đứt đoạn |
| **Ảnh bị gãy đứt vết nứt** | **37 (10.6%)** | **42 (12.1%)** | `+5` | - |
| **Số sự kiện Nối dính giả (Bridge events)** | **120** | **119** | `-1` | ĐẠT: Giảm cầu nối giả |
| **Ảnh bị dính cầu nối giả** | **112 (32.2%)** | **113 (32.5%)** | `+1` | - |
| **Đảo nhiễu giả (Spurious islands)** | **185** | **168** | `-17` | - |
| **Precision** | **0.7408** | **0.7409** | `+0.0001` | - |
| **Recall** | **0.8384** | **0.8396** | `+0.0011` | - |

