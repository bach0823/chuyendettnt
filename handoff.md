# HANDOFF – Đề tài AI: Nhận dạng ảnh tường nhà bị nứt
# Cập nhật: 2026-09-21 (Hoàn thành Phase 1 Migration & B0 Preflight PASS)

> File này được tạo để agent (hoặc session mới) đọc vào và tiếp tục hỗ trợ người dùng mà không cần giải thích lại từ đầu.
> **ĐỌC TOÀN BỘ FILE NÀY TRƯỚC KHI LÀM BẤT CỨ ĐIỀU GÌ.**
> ⚠️ **BẮT BUỘC ĐỌC VÀ TUÂN THỦ SKILL:** Agent mới PHẢI đọc kỹ file skill [d:/truong/SpecialSubjectTTNT/.agents/skills/project-ai-assistant/SKILL.md](file:///d:/truong/SpecialSubjectTTNT/.agents/skills/project-ai-assistant/SKILL.md) và file [d:/truong/SpecialSubjectTTNT/.agents/skills/sage-lite-colab/SKILL.md](file:///d:/truong/SpecialSubjectTTNT/.agents/skills/sage-lite-colab/SKILL.md) trước khi bắt đầu. Tuyệt đối tuân thủ nguyên tắc **Problem-First** (không vội nhảy vào code hay model khi chưa rõ bài toán), duy trì đồng bộ 2 file state (milestone_and_progress.md và user_capability.md), và bám sát quy trình feedback loop.

---

## 1. TÌNH HUỐNG CỦA NGƯỜI DÙNG

- Sinh viên năm 4, ngành **Trí tuệ Nhân tạo & Khoa học Dữ liệu**.
- Môn: **Chuyên đề Trí tuệ Nhân tạo**.
- Thầy giao đề tài dựa trên bài báo SAGE (CVPR 2026):
  > "Em đọc bài này và làm đề tài nhận dạng ảnh tường nhà bị nứt nhé"
- Hình thức: Bài tập lớn (slide + code + demo).
- Trình độ: Python cơ bản. Đã hoàn thành huấn luyện 5 mô hình thực tế trên Colab T4.

---

## 2. YÊU CẦU CỦA THẦY (NGUYÊN VĂN)

`
1, So sánh giữa các phương pháp cơ sở như thế nào
2, So sánh giữa các mô hình nền tảng như thế nào
3, Loại trừ từng module thì độ chính xác sẽ ảnh hưởng như thế nào
4, Tăng tham số ảnh hưởng như thế nào đến mô hình của dự án trong thực tế
5, Độ phức tạp tính toán của mô hình ảnh hưởng như thế nào
yc: Chạy 5 method trở lên | có link và minh họa trực tiếp | có slide, code, chú ý kĩ phần loss function, backpoint (layer, cơ sở)
`

**Yêu cầu dataset:** 1 bộ tự thu thập (tự chụp + label) + 1 bộ public.

---

## 3. PROBLEM STATEMENT (ĐÃ XÁC LẬP)

**Mục tiêu học thuật (quan trọng nhất):**
> Kiểm chứng xem ý tưởng Dual-Path + SA-Hub + MoE từ bài báo SAGE có thể chuyển giao sang bài toán vết nứt tường không.

**Hypothesis cốt lõi:**
> Vết nứt tường và tế bào ung thư đều có đặc điểm đa dạng hình thái (thin cracks, branching, hairline) và cần ngữ cảnh toàn cục. Do đó kiến trúc Dual-Path + SA-Hub từ SAGE có thể mang lại lợi ích tương tự cho crack segmentation.

---

## 4. BÀI BÁO SAGE – TÓM TẮT KỸ THUẬT

- **Tên:** SAGE: Shape-Adapting Gated Experts for Adaptive Histopathology Image Segmentation
- **Hội nghị:** CVPR Findings 2026
- **Link arxiv:** https://arxiv.org/abs/2511.18493
- **Bản tóm tắt & phân tích chi tiết trong dự án:** Đọc ngay file [SAGE_Paper_Summary.md](file:///d:/truong/SpecialSubjectTTNT/SAGE_Paper_Summary.md) (Đã phân tích sâu bối cảnh, 5 nhóm kiến trúc đối chiếu, chi tiết toán học của Dual-path, Hierarchical Router và SA-Hub).

---

## 5. TRẠNG THÁI DỰ ÁN HIỆN TẠI (Cập nhật 2026-10-07)

Dự án đã trải qua tiến trình nghiên cứu chuyên sâu có kiểm soát từ B0 đến hoàn thành toàn bộ Phase 6 A1 + S2-Gate Combination.

### Kết quả các mốc nền tảng chính:
- **Baseline B0 (Pure CNN):** Val Dice **0.7318** (Crack500), **0.6240** (DeepCrack).
- **Baseline B1 (Hybrid no SAGE):** Val Dice **0.7428** (Depth 4), **0.7420** (Depth 6).
- **B2 Phase 1 (Depth Sweep):** Khóa chính thức ViT Depth $D=4$ (Sweet spot cân bằng biểu diễn đa tỷ lệ).
- **B2 Phase 2 (Routing Capacity):** Tạm khóa $k=2$ (provisional top_k=2).
- **B2 Phase 3 (Router Hidden Dim Lock):** Khóa chính thức $H = 64$ (Val Dice **0.7618**).
- **B2 Phase 4 (Load Balance Factor Lock):** Khóa chính thức $LB = 0.010$ (Val Dice **0.7641**).
- **B2 Phase 5 (Optimization Stability & Candidate B Lock):** Khóa cấu hình chuẩn tắc **Candidate B** ($D=4, K=2, H=64$, Stage 2 base $r=1.00$) với Val Dice **0.7641**, Mean IoU **0.6417**.
- **B2 Phase 6-A.1 (Soft Boundary IoU Loss):** Val Dice **0.7684**, cải thiện vượt bậc trên vết nứt siêu mảnh.
- **B2 Phase 6-A.2 (Pure PLU-Head):** Val Dice **0.7664**, Precision tăng mạnh lên **0.7491**.
- **B2 Phase 6-B.1 (AB-BPL Loss):** Val Dice **0.7685**, BM Precision **0.7511**.
- **B2 Phase 6-D.3 (Causal ASDW Probe):** Chứng minh đóng góp của P3-C ASDW là $0\%$ ($\Delta\text{Dice} < 10^{-6}$). Chính thức LOẠI BỎ P3-C khỏi kiến trúc chuẩn SAGE-Lite.
- **B2 Phase 6-U1-S2G-v2 (Spatial Gate Conv3x3 tại Decoder Block 1):** Phục hồi 81.8% ca worsened, giữ vững 27.91% tỷ lệ chữa lành cầu nứt giả, giảm HD95 ngoạn mục 11.34 px ($53.65 \to 42.31\text{ px}$).
- **B2 Phase 6 A1 + S2-Gate Combination (ĐỈNH CAO DỰ ÁN MỚI 🏆):**
  * **Run 1 (`a1_s2g_end_to_end`)**: Val Dice **0.7676** (Epoch 16).
  * **Run 2 (`a1_s2g_stage2`)**: Val Dice 🏆 **0.7702** (Epoch 14), Precision **0.7510**, Recall **0.8301**, Thin Crack Dice **0.4260**, Loss đáy **1.3471**.
  * **Static Top-2 Routing**: Đạt Val Dice **0.7687**, IoU **0.6467**, Precision **0.7486** (vượt Dynamic Adaptive 0.7676).
  * **Static Top-$K$ Capacity Sweep**: Bão hòa dung lượng tại $K=2$ ($25\%$), tăng $K \ge 3$ gây suy thoái biểu diễn.
  * **Expert Diversity**: Cosine Similarity trung bình $0.2694$ (High Diversity); Intra-ViT $0.6869$ khẳng định trần biểu diễn đã bão hòa ở $D=4$.

---

## 6. TRẠNG THÁI THỰC THI & NEXT STEPS (Cập nhật 2026-10-07)

1. **Thực nghiệm đã nghiệm thu đầy đủ**:
   - Tài liệu tổng hợp toàn diện: [`results/PHASE6_S2G_ROUTING_AND_CAPACITY_SYNTHESIS.md`](file:///d:/truong/SpecialSubjectTTNT/SAGE_LITE/results/PHASE6_S2G_ROUTING_AND_CAPACITY_SYNTHESIS.md).
   - Scripts đánh giá chuẩn hóa:
     * `scripts/diagnostics/evaluate_routing_and_asdw_ablation.py`
     * `scripts/diagnostics/evaluate_static_topk_sweep.py`
     * `scripts/diagnostics/run_expert_diversity_diagnostic.py`
   - Dữ liệu định lượng: Lưu trữ tại `results/diagnostics/` (`routing_and_asdw_ablation/`, `static_topk_sweep/`, `expert_diversity/`, `error_analysis/`, `routing_analysis/`).

2. **Next Steps cho Agent kế thừa**:
   - Triển khai **Candidate C Synthesis (Final Model)**: Tích hợp đầy đủ bộ 3 thành phần chiến thắng:
     1. Objective regularization: Soft Boundary IoU Loss ($\lambda=0.5, d=2$).
     2. Skip spatial regulation: S2-Gate (Conv3x3 tại Decoder Block 1).
     3. Tùy chọn mở rộng: Khảo sát khả năng bổ sung cổng không gian tương tự tại Decoder Block 2 (112x112) để lọc nhiễu cho Skip S0, hoặc đưa PLU-Head vào đánh giá cuối cùng.
   - Thử nghiệm trên tập Test chính thức (Setting A & Setting B) để đối chiếu số liệu tổng kết toàn bộ đề tài.
   - Chuẩn bị slide báo cáo học thuật, đối chiếu 5 câu hỏi gốc của Giảng viên hướng dẫn.

3. **Cảnh báo nghiêm ngặt về dữ liệu (Non-negotiable Invariant)**:
   - Tuyệt đối KHÔNG tự ý xóa bất kỳ thư mục kết quả nào trong `results/`.
   - Các run chạy từ cùng checkpoint Stage 1 hiển nhiên có hash `.pth` đầu vào giống nhau, nhưng hành trình Stage 2 và kết quả cuối cùng là độc lập (ví dụ Run 1 đạt 0.7676, Run 2 đạt 0.7702). Bắt buộc phải đọc `train.log` trước khi kết luận.



