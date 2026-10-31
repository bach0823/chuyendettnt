# Báo Cáo Đối Soát Tô-Pô Hình Thái: Phase 6 vs Phase 7

| Chỉ số Tô-pô & Hình thái | Phase 6 (Chỉ có S2G Block 1) | Phase 7 (Dual S2G: Block 1 + Block 2) | Chênh lệch ($\Delta$ P7 - P6) | Nhận định Giả thuyết |
| :--- | :---: | :---: | :---: | :--- |
| **Validation Dice** | **0.7676** | **0.7623** | `-0.0054` | Giảm |
| **Centerline Dice (clDice)** | **0.8440** | **0.8400** | `-0.0040` | Suy giảm độ liền trục |
| **Boundary IoU** | **0.1353** | **0.1335** | `-0.0018` | - |
| **HD95 (px, thấp hơn là tốt)**| **70.12 px** | **67.56 px** | `-2.56 px` | Rút ngắn khoảng cách biên |
| **Số sự kiện Gãy đứt (Break events)** | **38** | **44** | `+6` | KHÔNG ĐẠT: Tăng đứt đoạn |
| **Ảnh bị gãy đứt vết nứt** | **37 (10.6%)** | **42 (12.1%)** | `+5` | - |
| **Số sự kiện Nối dính giả (Bridge events)** | **120** | **116** | `-4` | ĐẠT: Giảm cầu nối giả |
| **Ảnh bị dính cầu nối giả** | **112 (32.2%)** | **110 (31.6%)** | `-2` | - |
| **Đảo nhiễu giả (Spurious islands)** | **185** | **157** | `-28` | - |
| **Precision** | **0.7408** | **0.7343** | `-0.0064` | - |
| **Recall** | **0.8384** | **0.8371** | `-0.0013` | - |

