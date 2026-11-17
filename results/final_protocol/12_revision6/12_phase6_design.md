# 12_phase6_design.md — Thiết Kế Thử Nghiệm Chuẩn Hóa Cho Giai Đoạn Đánh Giá Trên Dataset Tường Riêng (Revision 6)

**Ngày lập:** 2026-11-17  
**Mục tiêu:** Thiết lập khung thực nghiệm chuẩn tắc nhằm đánh giá thực chất năng lực tổng quát hóa của kiến trúc ngoài tập dữ liệu Crack500, tuân thủ các nguyên tắc thống kê nghiêm ngặt và tối ưu hóa chi phí tính toán.

---

## 1. Dữ Liệu Mục Tiêu Và Các Mô Hình Đưa Vào So Sánh Đối Đầu

### A. Tập dữ liệu thực nghiệm
- Đánh giá trên **dataset tường riêng của người dùng** (ảnh chụp nứt bề mặt tường bê tông thực tế thu thập từ công trình), phản ánh trung thực bài toán ứng dụng thực địa.
- Dữ liệu phải được gắn mã định danh cụm theo từng bức tường độc lập (`wall_id`) để phục vụ kiểm soát tương quan cụm (clustering effect).

### B. Ba mô hình cốt lõi đưa vào so sánh
1. **Baseline B0 (100 epochs):** Mô hình mạng cơ sở nhẹ (ConvNeXt-Femto + U-Net tiêu chuẩn, không có SAGE router hay chuyên gia).
2. **Canonical B2:** Mô hình đầy đủ gồm Backbone Femto + 4 khối ViT + SAGE router + S2-Gate.
3. **Variant N (B2 không SAGE):** Mô hình giữ nguyên Backbone Femto + 4 khối ViT + S2-Gate nhưng triệt tiêu hoàn toàn nhánh SAGE (`residual_scale = 0.0`), đóng vai trò đối chứng sạch cho S2-Gate và ViT mà không có cơ chế định tuyến chuyên gia.

---

## 2. Ba Kịch Bản Đánh Giá Bắt Buộc

Mỗi mô hình phải được đo lường độc lập qua 3 kịch bản chuyển giao:
1. **Kịch bản 1: Zero-shot Transfer:**
   - Nạp trực tiếp trọng số đã được huấn luyện trên Crack500 và thực hiện suy luận trực tiếp trên dataset tường riêng của người dùng mà không cập nhật bất kỳ trọng số nào.
   - Đo lường khả năng tổng quát hóa miền chéo (cross-domain generalization) mà không có dữ liệu thích ứng.
2. **Kịch bản 2: Fine-tuning:**
   - Tinh chỉnh mô hình đã tiền huấn luyện từ Crack500 trên một tập huấn luyện nhỏ thuộc dataset tường riêng.
   - Cố định cùng số lượng epoch tinh chỉnh, cùng learning rate scheduler và cùng bộ siêu tham số cho cả 3 mô hình.
3. **Kịch bản 3: Huấn luyện từ ImageNet Pretrain (Train from ImageNet):**
   - Khởi tạo trọng số Backbone từ ImageNet pretrain chuẩn, huấn luyện từ đầu trực tiếp trên dataset tường riêng của người dùng với cùng ngân sách epoch.
   - Đo lường khả năng học biểu diễn đặc trưng trực tiếp từ miền dữ liệu đích.

---

## 3. Tiêu Chuẩn Thống Kê Bắt Buộc Và Nguyên Tắc Quyết Định Pareto

### A. Tiêu chuẩn thống kê đối với cỡ mẫu nhỏ
Do tập dữ liệu tường thực tế thường có kích thước mẫu hạn chế và cấu trúc ảnh chụp phân cụm theo từng công trình/bức tường:
1. **Quy tắc đa seed:** Mọi phép so sánh hiệu năng giữa các mô hình bắt buộc phải chạy trên **ít nhất 2 seed độc lập** (khuyến nghị $\ge 3$ seed) để ước lượng phương sai giữa các lần chạy (seed variance).
2. **Kiểm định Bootstrap cụm theo tường (`wall_id`):**
   - Toàn bộ khoảng tin cậy 95% và giá trị $p$-value ghép cặp phải được tính bằng phương pháp **Cluster Bootstrap resample theo từng bức tường (`wall_id`)**.
   - Tuyệt đối không tính toán thống kê hay tương quan trên từng ảnh cắt lẻ (crops) để tránh hiện tượng thổi phồng bậc tự do và sai lệch chọn mẫu (pseudoreplication).

### B. Nguyên Tắc Quyết Định Pareto (Pareto Optimization Rule)
- Mô hình Baseline B0 có cấu trúc đơn giản, số lượng tham số và chi phí tính toán (FLOPs, bộ nhớ VRAM, thời gian suy luận trên mỗi khung hình) thấp hơn đáng kể so với kiến trúc B2.
- **Quy tắc quyết định kỹ thuật:**
  - Nếu mô hình Baseline B0 đạt hiệu năng tương đương hoặc chỉ chênh lệch không đáng kể (nằm trong khoảng sai số thống kê của kiểm định cụm) so với Canonical B2 trên dataset tường riêng của người dùng:
  - **Mô hình Baseline B0 bắt buộc phải được ưu tiên lựa chọn làm kiến trúc tối ưu cho triển khai thực tế.**
  - Việc chấp nhận kiến trúc B2 phức tạp hơn chỉ được xem xét khi và chỉ khi Canonical B2 hoặc Variant N chứng minh được sự vượt trội có ý nghĩa thống kê vượt qua Minimum Detectable Effect (MDE) trên cả kiểm định cụm và đa seed.
