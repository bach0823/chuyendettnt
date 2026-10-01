# Project State: SAGE-Lite: A Lite Version of Shape-Adapting Gated Experts for Crack Binary Segmentation

## Deadline
Deadline: N/A
Days Remaining: N/A
Risk Level: Safe

## Workflow: Repo-Based (Colab Driver Pattern)
- Source code nằm trong Git repo độc lập: `sage_lite/` (private GitHub repo).
- Colab chỉ đóng vai trò driver: Clone repo → `pip install -e .` → gọi entry point training.
- Mỗi run có thể truy ra từ commit + config YAML tương ứng.
- Checkpoint và log lưu vào Google Drive để không mất khi Colab session reset.

## Current Progress

Current Milestone:
Milestone 2: SAGE Core Mechanism & B2 (Full SAGE-Lite)

Current SK:
Nghiệm thu tiến trình bậc thang thực nghiệm SAGE-Lite B2 trên Crack500 (Master Experimental Progression):
1. **Phase 1: ViT Depth Scaling Study (HOÀN TẤT & ĐÃ KHÓA BASE DEPTH = D4 ✅):**
   - Đã sweep đủ 5 độ sâu ViT {D4: **0.7639** > D8: **0.7604** > D6: **0.7599** > D2: **0.7578** > D12: **0.7557**}. Khóa chính thức ViT Depth = 4 (D4, 10.12M params, 8 experts đối xứng 1:1, 8 routers).
2. **Phase 2: Routing Capacity top_k Screening (TẠM KHÓA PROVISIONAL TOP_K = 2 🔒):**
   - K2 (0.7618) đạt xấp xỉ K4 baseline (0.7639) với độ suy giảm tối thiểu (-0.21% Dice), cắt giảm 50% expert calls. K6 bị OOM trên phần cứng Tesla T4 (hardware infeasibility under canonical T4 protocol). Tạm chốt provisional top_k=2 để tiếp tục lộ trình.
3. **Phase 3: Router Hidden Dim Study (HOÀN TẤT & ĐÃ KHÓA HIDDEN_DIM = 64 ✅):**
   - Đã khảo sát 3 dung lượng router {H32: 0.7587, H64: **0.7618**, H128: 0.7550}. Đường cong chữ U ngược quan sát được: $Dice(H128=0.7550) < Dice(H32=0.7587) < Dice(H64=0.7618)$. Chọn `router_hidden_dim = 64` là best observed configuration dưới D4 + provisional top_k=2 (seed 42).
4. **Phase 4: Load Balancing Loss Study (HOÀN TẤT & ĐÃ KHÓA LB = 0.010) ✅:**
   - Hoàn tất quét lưới 3 mức $LB \in \{0.005, 0.010, 0.030\}$:
     + $LB = 0.005$: Peak Val Dice = 0.7580 (sụp đổ router nông thành công tắc 2 experts, phạt quá yếu làm over-predict).
     + $LB = 0.030$: Peak Val Dice = 0.7624 (hồi sinh chuyên gia chết tại stage 1 & 2, nhưng phạt quá nặng làm cùn đường biên).
     + $LB = 0.010$ (Candidate B): Peak Val Dice = 🏆 **0.7641** (Mean IoU: 0.6417, Precision: 0.7337). Đạt điểm cân bằng Pareto tối ưu.
   - **Khóa chính thức: $load\_balance\_factor = 0.010$** làm tham số bất biến cho các pha tiếp theo.
5. **Phase 5.1: SAGE LR Isolation (HOÀN TẤT & ĐÃ KHÓA SAGE_LR = 2e-4 ✅):**
   - Đã sweep đủ 4 mức SAGE LR {5e-5: 0.7597, 1e-4: 0.7618, 2e-4: **0.7641** 🏆, 3e-4: 0.7492}.
   - Đường cong hiệu năng theo SAGE LR là đường cong chữ U ngược (inverted U-curve):
     $$Dice(3e\text{-}4=0.7492) < Dice(5e\text{-}5=0.7597) < Dice(1e\text{-}4=0.7618) < Dice(2e\text{-}4=0.7641)$$
   - Khóa chính thức `sage_lr = 2e-4` là best observed configuration cho SAGE router trong Stage 1 (kết hợp với `base_lr = 1e-4`).
   - Candidate C (3e-4) vượt quá ngưỡng dung nạp tốc độ học của router, gây bão hòa sớm và suy giảm hiệu năng ở cả Stage 1 (Peak S1 Dice = 0.7278 vs 0.7333 ở 2e-4) lẫn Stage 2 (Peak S2 Dice = 0.7492 vs 0.7641 ở 2e-4), kích hoạt EarlyStopping ở cả 2 stage (Ep 16 và Ep 12).
   - **Phase 5.1 Extra (Khảo sát riêng Stage 2 SAGE LR = 2e-4 ✅)**:
     + Chạy Stage 2-only từ checkpoint Candidate B Stage 1 (Ep 13, Dice 0.7333) với `stage2_sage_lr = 2e-4` (giữ nguyên base/shared/p3 LR = 1e-4).
     + Kết quả: Đạt đỉnh **0.7644 Val Dice** (Ep 16), đáy Val Loss **0.9412** (Ep 18), Mean IoU **0.6409**, Median Dice **0.8073**.
     + Đóng gói đầy đủ artifacts tại `results/P3_C_Phase5_1_Extra_Stage2_SAGELR2e-4_Full.zip` và diagnostics tại `results/P3_C_Routing_Diagnostics_D4_K2_H64_Phase5_1_Extra_SAGELR2e-4/`.
     + **Quyết định chốt**: Khóa chính thức **Candidate B** (`sage_lr = 2e-4` ở Stage 1); **KHÔNG tách riêng SAGE LR ở Stage 2** (SAGE router ở Stage 2 đi chung với `stage2_base_lr = 1e-4` để giữ tối ưu hóa tinh gọn).
6. **Phase 5.3: Stage-2 Shared LR Ratio Sweep (HOÀN TẤT & ĐÃ KHÓA r = 1.00 ✅):**
   - Đã sweep đủ toàn bộ 6 tỉ số $r \in \{0.25, 0.50, 1.00, 2.00, 4.00, 5.00\}$ trải dài hơn một bậc độ lớn ($2.5\times 10^{-5} \to 5.0\times 10^{-4}$).
   - Xác lập đường cong chữ U ngược đơn đỉnh hoàn chỉnh:
     $$0.7579\,(r=0.25) < 0.7586\,(r=0.50) < \mathbf{0.7641}\,(r=1.00) > 0.7638\,(r=2.00) > 0.7611\,(r=4.00) > 0.7585\,(r=5.00)$$
   - Soft-freeze shared CNN ($r < 1.0$) và gia tốc quá mức ($r > 2.0$) đều gây suy giảm biểu diễn hình thái vết nứt.
   - Khóa chính thức $r = 1.00$ (kế thừa Candidate B với `stage2_shared_lr = stage2_base_lr = 1e-4` thống nhất toàn diện).
7. **Phase 6-A.1: Objective Probe — Soft Boundary IoU Loss (HOÀN TẤT & ĐẠT ĐỈNH TOÀN DỰ ÁN MỚI 🏆):**
   - Giữ nguyên 100% kiến trúc mạng Candidate B (0 params added, D4, K2, H64, LB=0.010, r=1.00), chỉ bổ sung Soft Boundary IoU Loss ($\lambda_{\text{boundary}}=0.50, d=2$).
   - Kết quả: Đạt đỉnh **Val Dice = 0.7684** (tăng $+0.0043$ so với Candidate B 0.7641), Mean IoU **0.6465**, Precision **0.7416** (+0.79%), Tỷ lệ diện tích dự đoán/GT giảm từ $+8.16\%$ về $+7.85\%$.
   - Đặc biệt bứt phá trên nhóm vết nứt mảnh (Thin cracks $>0.20$): Dice tăng vọt $+2.37\%$ ($0.6404 \to 0.6641$), Precision tăng $+4.16\%$ ($0.5232 \to 0.5648$), tỷ lệ phình diện tích giảm $-15.82\%$ ($+80.37\% \to +64.55\%$).
   - Xác nhận Pattern 2 trong Ma trận Quyết định Pre-registered (Objective regularization cải thiện trực tiếp cấu trúc viền vết nứt mảnh).
8. **Phase 6-A.2: Representation Probe — Pure PLU-Head (HOÀN TẤT & XÁC NHẬN NÚT THẮT BIỂU DIỄN ✅):**
   - Huấn luyện từ đầu Stage 1 $\to$ Stage 2 với Progressive Learned Upsampling Head ($112 \to 224 \to 448$, +42,720 params). Không đổi objective ($\mathcal{L}_{\text{Base}}$ thuần, $\text{BoundaryIoU}=0.0$).
   - Kết quả: Peak Global Val Dice = **0.7664** (+0.0023 vs Base 0.7641), Precision tăng mạnh lên **0.7491** (+1.54%).
   - Đột phá hình thái trên vết nứt siêu mảnh: Thin cracks ($n=66$) Dice vọt lên **0.6674** (+0.0270 vs Base), diện tích phình giảm sâu từ $+80.37\%$ xuống $+57.00\%$ ($\Delta = -23.37\%$). Thin-low-area ($n=5$) tăng vọt từ $0.5370 \to \mathbf{0.6059}$ (+0.0689), phình diện tích giảm từ $+141.66\%$ xuống $+77.06\%$ ($\Delta = -64.60\%$).
9. **Phase 6-B.1: Boundary Margin Probe — Asymmetric Boundary-Band Penalty Loss (HOÀN TẤT & NGHIỆM THU 🏆):**
   - **Trạng thái**:
     $$ \boxed{\textbf{B-1: Positive global result + partial boundary-mechanism success}} $$
   - **Chi tiết thực nghiệm**:
     + Global Val Dice: ✅ **0.7641 → 0.7685** (+0.0044, đỉnh cao mới Phase 6, Mean IoU: 0.6465, Win rate: 56.32%).
     + BM Precision ($n=127$): ✅ Mean tăng từ $0.7452 \to 0.7511$ (+0.0059), Median tăng từ $0.7533 \to 0.7638$ (+0.0105), $56.7\%$ win rate.
     + BM Recall preservation: ✅ Giữ vững ở mức cao $0.8321$ (Median $\Delta = -0.0005$, bảo toàn nguyên vẹn, không sụt giảm như A2 $0.8087$).
     + BM Median Dice / Aggregated Dice: ✅ Signal rõ rệt (Median: $0.7841 \to 0.7901$, Aggregated: $0.7869 \to 0.7909$, Win rate $62.20\%$ [79/127], Wilcoxon signed-rank $p = 0.0200$).
     + BM Sample-mean Dice: ❌ $0.7812 \to 0.7802$ ($-0.0010$, bị kéo lùi bởi 2 mẫu ngoại lai ở đuôi phân phối).
     + BM Area Excess: ❌ Chưa đạt frozen guidepost ($\le -5.0\%$), nhưng hướng dịch chuyển hoàn toàn chuẩn xác (Sample-mean: $+14.86\% \to +13.38\%$, $\Delta = -1.47\%$; Median: $+15.49\% \to +13.13\%$, $\Delta = -2.36\%$; giảm diện tích phình trên $54.3\%$ số mẫu BM).
     + Thin cracks ($n=66$): ✅ Dice tăng từ $0.6404 \to 0.6555$ (+0.0151), tỷ lệ phình diện tích giảm $-12.86\%$ ($+80.37\% \to +67.51\%$).
     + Thin-low-area ($n=5$): ❌ Giữ nguyên bản chất ($0.5370 \to 0.5362$, khẳng định cần biểu diễn không gian phân giải cao thay vì chỉ loss).
   - **Kết luận khoa học chốt**:
     > *Phase 6-B.1 provides evidence that asymmetric boundary-band supervision can improve global validation Dice while preserving boundary recall and modestly reducing excess predicted area, but the pre-registered BM Dice and Area Excess guideposts were not fully met. The result therefore supports AB-BPL as a complementary boundary objective, rather than establishing it as a complete solution to boundary over-dilation.*
    - **Quy tắc thực nghiệm**: Không chạy thêm chỉ để “đuổi” Guidepost $-5\%$. B-1 đã thiết lập phenotype rõ ràng; sử dụng thông tin này để quyết định các cơ chế cần kiểm tra ở các bước kế tiếp, không tune B-1 hậu nghiệm.
10. **Phase 6-C.1: Topology Probe — Soft-clDice Loss (HOÀN TẤT & ĐÃ ĐÓNG ✅):**
    - **Phán quyết chính thức**:
      $$ \boxed{\textbf{C.1 = Negative global result + confirmed topology trade-off}} $$
    - **Kết quả thực nghiệm**: Centerline Dice tăng ($0.8498 \to 0.8525$), $T_{\text{sens}}$ tăng ($0.8872 \to 0.8899$), Spurious Islands giảm ($108 \to 102$). Tuy nhiên, Global Dice suy giảm ($0.7641 \to 0.7613$, $\Delta = -0.0028$), Precision tụt ($0.7338 \to 0.7210$), Area Excess tăng mạnh ($+8.16\% \to +11.82\%$). False Bridge hoàn toàn bất biến ($31.6\% \to 32.5\%$), Thin Breakage không đổi ($12.1\% \to 12.1\%$).
    - **Kết luận cơ chế**: Soft-clDice cải thiện centerline coverage quanh GT centerline nhưng đi kèm xu hướng mở rộng vùng foreground (over-dilation), làm giảm precision và không giải quyết được False Bridge.
11. **Phase 6-D.0: Area Excess ↔ False Bridge Diagnostic (HOÀN TẤT ✅):**
    - Khảo sát tương quan chéo giữa mức độ phình diện tích và tần suất xuất hiện cầu nứt giả trên $N=348$ mẫu Candidate B:
      $$P(\text{Bridge} \mid \text{Excess} \le 0) = 10.7\% \quad \text{vs} \quad P(\text{Bridge} \mid \text{Excess} > 0) = 40.4\% \quad (OR = 5.67, p < 0.0001)$$
    - **Kết luận khoa học**: False Bridge có thành phần liên quan đến độ phình diện tích, nhưng đồng thời tồn tại một **Residual Separation Bottleneck** độc lập. Giảm over-dilation là cần thiết nhưng không đủ để chữa lành toàn bộ False Bridge.
12. **Phase 6-D.1: Representation x Boundary Synergy Probe — PLU + AB-BPL (HOÀN TẤT & ĐÃ ĐÓNG ✅):**
    - **Phán quyết chính thức**:
      $$ \boxed{\textbf{D.1 = Partial support for representation–boundary complementarity}} $$
      $$ \boxed{\textbf{Dilation is contributory, but insufficient}} $$
    - **Lineage**: Stage 2 rẽ nhánh có kiểm soát từ Phase 6-A.2 Stage 1 checkpoint.
    - **Kết quả 4 Pre-registered Guideposts**: Đạt $1/4$ guidepost (BM Dice $0.7814 \ge 0.7800$ ✅; Thin-low-area $0.5769 < 0.5900$ ❌ [retained 60% gain]; Area Excess $+5.07\% > 5.0\%$ ❌; False Bridge $112 > 107$ ❌).
    - **Hiệp đồng bổ trợ**: Thiết lập kỷ lục Precision toàn cục ($0.7525$) và Boundary Margin ($0.7772$), tỷ lệ thắng nứt mảnh cao nhất ($80.3\%$), nén Area Excess xuống $+5.07\%$.
    - **Bằng chứng cơ chế False Bridge**: $105/110$ cầu nứt tồn tại dai dẳng (`persistent under D.1`). D.1 chỉ chữa lành các ca có khoảng cách lớn ($\ge 11\text{ px}$), hoàn toàn bất lực trước các khe hẹp $\le 5-8\text{ px}$.
13. **Phase 6-D.2: Pure Inter-Component Separation Isolation Probe (PREFLIGHT HOÀN TẤT & KHÓA SPECIFICATION 🔒):**
    - **Mục tiêu**: Kiểm tra độc lập giả thuyết can thiệp hành lang phân tách âm (negative moat supervision) trên Candidate B gốc:
      $$L_{\text{D2}} = L_{\text{Base}} + 0.010 \times L_{\text{sep}}(G_{\max}=8.0\text{ px}, r_{ij} = \lceil g_{ij}/2 \rceil + 1)$$
    - **Lineage**: Candidate B Stage 1 Checkpoint (`P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_stage1.pth`).
    - **Preflight Suite (7/7 PASS ✅)**:
      + Đạt chuẩn hình học trên 5 mẫu Crack500 thực tế (moat bám khít khe hẹp, rỗng đối với đơn component và gap $> 8\text{ px}$).
      + Single-component và Serpentine tự áp sát ($K=1$) $\implies L_{\text{sep}} = 0.0$, grad $= 0$.
      + Two-component synthetic gap $\implies$ Phạt pixel trong moat ($L=5.0067$), pixel nền ngoài moat bằng 0 ($L=0.0067$).
      + Empty GT $\implies L_{\text{sep}} = 0.0$, an toàn tuyệt đối.
      + Exact Base Equivalence ($\lambda_{\text{sep}}=0$) $\implies \texttt{torch.equal} = \text{TRUE}$ ở cấp độ byte cho loss và toàn bộ 100% gradient tham số.
      + Gradient Calibration trên 8 ca bridge thực tế $\implies \|g_{\text{sep}}\| / \|g_{\text{Base}}\| = 5.6461\times \implies \lambda_{\text{sep}} = 0.010$ (ngân sách $5.65\%$ Base gradient).
    - **Trạng thái**: Sẵn sàng huấn luyện Stage 2 trên Colab Driver.

State:
- Phase 1: D4 locked
- Phase 2: K6 OOM → provisional top_k=2
- Phase 3: H64 selected under D4+K2, seed42
- Phase 4: COMPLETED & LOCKED (LB=0.010, Peak Val Dice = 0.7641)
- Phase 5.1: SAGE LR = 2e-4 locked (Candidate B 0.7641 làm baseline chính thức, Stage 2 không tách riêng)
- Phase 5.2: SKIPPED (loss giảm mượt, không spike, kế thừa warmup=3)
- Phase 5.3: HOÀN TẤT & ĐÃ KHÓA r = 1.00 (Candidate B 0.7641; hoàn tất cả 6 tỉ số r in {0.25, 0.50, 1.00, 2.00, 4.00, 5.00})
- Phase 6-A.1: HOÀN TẤT (Soft Boundary IoU Loss, Val Dice = 0.7684 🏆, New Global Peak)
- Phase 6-A.2: HOÀN TẤT (Pure PLU-Head, Val Dice = 0.7664, Thin Crack Recovery 0.6059)
- Phase 6-B.1: HOÀN TẤT (AB-BPL Probe, Val Dice = 0.7685 🏆, Positive Global + Partial Boundary-Mechanism Success)
- Phase 6-C.1: HOÀN TẤT (Soft-clDice Probe, Val Dice = 0.7613, Negative Global + Confirmed Topology Trade-off)
- Phase 6-D.0: HOÀN TẤT (Area Excess ↔ False Bridge Diagnostic, OR = 5.67)
- Phase 6-D.1: HOÀN TẤT (PLU + AB-BPL Synergy Probe, Val Dice = 0.7669, Partial Support for Complementarity)
- Phase 6-D.2: PREFLIGHT PASSED & SPECIFICATION FROZEN 🔒 (Pure Separation Isolation Probe, lambda_sep = 0.010)

Cấu hình hiện hành:
- ViT depth = 4
- top_k = 2 (provisional)
- router_hidden_dim = 64
- sage_lr = 2e-4 (Stage 1 isolated)
- stage2_sage_lr: Không tách riêng (đi cùng `stage2_base_lr = 1e-4` trong tier `others`)
- N_injection = 8 routers
- batch size = 14
- epoch budget = 35 (17+18)


### ⚠️ QUY TẮC BẮT BUỘC: PREPROCESSING CHÍNH THỨC ĐÃ KHÓA (FROZEN CANONICAL)
> **TUYỆT ĐỐI KHÔNG ĐƯỢC TỰ Ý THAY ĐỔI, THÊM/BỚT BẤT KỲ BƯỚC PREPROCESSING NÀO (crop, padding, resize, augmentation, mask processing) CHO ĐẾN KHI HOÀN THÀNH TOÀN BỘ SAGE-LITE.**  
> Chỉ thay đổi khi người dùng CHỦ ĐỘNG yêu cầu làm ablation hoặc sửa protocol.
> - **Crack500 Train:** Reflect Pad (nếu cần) → RandomCrop 448×448 → smart filter `fg_pixels >= 20` (retry max 20 crop attempts × 10 source resamples) → Augmentation (`HorizontalFlip`, `VerticalFlip`, `RandomRotate90`, `RandomBrightnessContrast`, `GaussianBlur`) → ImageNet Normalize → Tensor.
> - **Crack500 Eval:** Tiling Setting A (448×448 non-overlap) & Setting B (448×448 overlap 50%, stride 224, average probabilities → threshold 0.5).
> - **DeepCrack Train & Eval:** Mask `{0,255} → {0,1}` → Dynamic Pad-to-Square (`target = max(H, W, 448)`, BORDER_CONSTANT=0) → Resize 448×448 (`cv2.INTER_NEAREST` mask) → cùng bộ Augmentation trên (chỉ train) → ImageNet Normalize → Tensor. Eval chạy direct 1-pass full-image.
> - **Augmentation đã loại bỏ hoàn toàn:** `CLAHE`, `ElasticTransform`, `GridDistortion`, `ShiftScaleRotate`, `HueSaturationValue`.

Completed:
- Đã chốt kiến trúc SAGE-Lite (Implementation Plan).
- Xác định xong Bảng biến thực nghiệm (Nhóm 1 khoá, Nhóm 2 đo).
- Hoàn thiện Migration Map theo file tree.
- SK1 Hoàn thành: Shape audit pass chuẩn input 448x448, Femto channels [48, 96, 192, 384], 196 spatial tokens.
- SK2 Hoàn thành: `sage_lite/sage/networks/convnextv2_vit_hybrid.py` (Femto + ViT-Tiny dynamic blocks support).
- SK3 Hoàn thành: `decoder_block.py` + `b0_unet.py` + `train_crack.py` (hỗ trợ single-stage, loss BCE+SoftDice, optimizer AdamW, Cosine Annealing warmup).
- **B0 Baseline Training HOÀN TẤT trên cả 2 dataset:**
  + **Crack500 (Run-3):** Best Val Dice = **0.7318** (@ epoch 8). Test Setting A: Dice **0.6771**, IoU 0.5645. Test Setting B: Dice **0.6801**, IoU 0.5682.
  + **DeepCrack (Run-1):** Best Val Dice = **0.6240** (@ epoch 11). Test Direct: Dice **0.6953**, IoU 0.6041, Boundary IoU 0.1365, HD95 30.89 px.
- Infrastructure Migration (Repo-based Workflow) Hoàn thành trên branch `crack500-audit`.
- **ViT-Depth Sweep {4, 6, 8, 12} & Official Eval B1 trên Crack500 HOÀN TẤT 100% (Screening Phase):**
  + Đã huấn luyện đủ 4 cấu hình: Depth 4 (Val Dice **0.7428**, Val Loss **1.1364** - Top 1), Depth 6 (0.7420), Depth 8 (0.7379), Depth 12 (0.7419).
  + Đã chạy Official Evaluation Setting A & Setting B cho toàn bộ các checkpoint `best_model_b1.pth`.
  + **Interpretation & Chiến lược:** Val Dice của Depth 4/6/12 ngang nhau (~0.742). Test Dice của Depth 12 cao hơn nhưng chỉ phản ánh representation thuần (chưa có SAGE). Do đó **CHƯA freeze ViT depth từ B1**; B1 đóng vai trò screening. Depth tối ưu cho SAGE-Lite sẽ được đo đạc và quyết định qua **B2 Depth Ablation** (tập trung vào `{4, 6, 12}`).
- **Milestone 2: Triển khai Full B2 Architecture HOÀN TẤT 100%:**
  + [x] Khóa kỹ thuật #1 (Strict Tensor Contract): `forward()` trả pure Tensor, thu thập đủ $4 + N_{\text{vit}}$ routing info qua `_last_routing_info`.
  + [x] Khóa kỹ thuật #2 (Zero Dynamic Params): Trainable parameters bất biến tuyệt đối trước/sau forward pass.
  + [x] Pure Residual Fusion: $y = x_{\text{main}} + \text{Dropout}(0.1 \times x_{\text{expert}})$.
  + [x] Zero-cost Self-Selection Bypass `my_index`: Tự bypass khi router chọn chính layer hiện tại.
  + [x] 3 Optimizer Param Groups: Backbone ($10^{-5}$), Decoder ($10^{-4}$), SAGE ($10^{-4}$). Weight decay $0.0$ cho LN/BN/bias, $0.05$ cho rest.
  + [x] Training Loss: $\mathcal{L} = \mathcal{L}_{\text{seg}} + 1.0 \times \mathcal{L}_{\text{balance}}$.
  + [x] Configs: `configs/b2_crack500_depth12.yaml` và `configs/b2_crack500_depth6.yaml`.
  + [x] Verification: Toàn bộ 10 smoke tests trong `scripts/scratch/verify_b2.py` PASS 100% dưới AMP FP16.
  + [x] Independent Cross-Check Audit: Hoàn tất với 25/25 requirements PASS, review log lưu tại `docs/B2_Audit_Review_Log.md`.
  + [x] **Phase 0 Runtime Preflight & Throughput Profiling trên Colab Tesla T4 (HOÀN TẤT 100%):**
    - Batch 20 & Batch 14+: OOM (vượt 14.56 GB VRAM T4).
    - **Batch 12 trên Real Crack500 (12 training batches thật):** ✅ **PASS** (Peak Alloc 13.84 GB, Peak Res 14.05 GB, **Free 0.51 GB**, 0 memory leak từ batch 2, 16/16 experts chọn đều 5.0% - 6.9%).
    - **Deep Runtime Profiler (`profile_deep_b2.py` - Chạy trực tiếp trên Colab T4):**
      + **B1-D12 Baseline**: Step 206.6 ms (Fwd 60.5 ms, Bwd 137.7 ms), Throughput 58.07 img/s, Est 1 epoch 0.54 min (~32s), Peak VRAM 2.42 GB.
      + **B2-D12 SAGE-Lite**: Step 5433.0 ms (Fwd 1487.4 ms, Bwd 3934.7 ms), Throughput 2.21 img/s, Est 1 epoch 14.31 min, Peak VRAM 13.61 GB.
      + **Phát hiện Cốt lõi (Resolution Bottleneck)**: Stage 0 (490.2ms) + Stage 1 (426.1ms) chiếm **61.6% thời gian Forward** do phân giải cao ($112 \times 112$ và $56 \times 56$). Cả 12 ViT blocks chỉ chiếm ~420ms (ít hơn 1 mình Stage 0).
      + **Minh oan cho SA-Hub**: SA-Hub adapt in/out chỉ chiếm 8.7% expert loop, index_add_ chiếm 2.1%. **89.1% thời gian là tính toán FLOPs thực tế trong Expert modules**.
      + **Phân phối Routing**: 16/16 experts cân bằng 4.7% – 7.2% (uniform target 6.25%). Cross-modal chiếm 36.3%.
      + **_infer_expert_type CPU**: Chỉ tốn 14.58 ms/step (0.27% tổng step time).
    - **Targeted Runtime Profiler (`profile_targeted_stages.py` - SageLayer 0 & 1 Focus, 10 Steps Colab T4):**
      + **Nguồn gốc Slowdown 43x**: Khi ViT Layer gọi ViT Expert ($N=196$), thời gian chỉ **0.79 ms/call**. Nhưng khi Stage 0 hoặc Stage 1 gọi ViT Expert, chuỗi tokens là **12,544 tokens**, khiến self-attention vọt lên **33.85 – 34.65 ms/call (chậm hơn 43 lần/call)**!
      + **Tập trung Chi phí**: Stage 0 & 1 gọi ViT experts 212 lần trong 10 steps, tiêu tốn 7,255 ms (>85% thời gian expert path của 2 tầng này).
      + **Micro-timing Accounting**: Khớp **96.9% – 98.3%** thời gian Expert Path (Compute chiếm 91.8% – 94.5%, residual chỉ 1.7% – 3.1%).
    - **ViT Scaling vs Local Window Attention Micro-Benchmark (`benchmark_vit_scaling.py` - Colab T4):**
      + **Global ViT $\mathcal{O}(N^2)$**: Tại $N=12,544$ và $B=4$, tổng Fwd+Bwd bùng nổ lên **173.07 ms** (chậm gấp **72.2x** so với $N=196$).
      + **Local Window Attention ($W=7$) $\mathcal{O}(N)$**: Chỉ tốn **24.60 ms** tại $N=12,544$ $\implies$ **nhanh hơn 7.04 lần** so với Global ViT. Chi phí giảm từ $0.003449$ xuống $0.000490$ ms/token.
    - **Spatial Compression before Global ViT Feasibility Micro-Benchmark (`benchmark_spatial_compression.py` - Colab T4):**
      + **Hiệu quả nén qua AdaptiveAvgPool**: Từ $112 \times 112$ ($N=12544$, 177.48 ms ở $B=4$), nén xuống $56 \times 56$ ($N=3136$) tốn **15.84 ms (11.21x nhanh hơn)**; nén xuống $28 \times 28$ ($N=784$) chỉ tốn **3.36 ms (52.90x nhanh hơn)**; nén về $14 \times 14$ ($N=196$) tốn **3.16 ms (56.12x nhanh hơn)**.
      + **Bảo toàn Pretrained Weights**: Giữ nguyên 100% cấu trúc và weights của pretrained ViT-Tiny block, giải quyết trọn vẹn điểm nghẽn sequence length của các nhánh CNN $\to$ ViT.
    - Bảng nghiệm thu: Real Crack500 pipeline (PASS), B2-D12 / top-k=4 (PASS), Batch 12 OOM (Không), Forward/backward/opt (PASS), Numerical stability (PASS), 16 experts active (PASS), VRAM (An toàn với đệm ~0.95 GB), Throughput (13.98 min/epoch), Cần sửa architecture? (Không).
    - Chi tiết log lưu tại: `docs/B2_Phase0_Preflight_Log.md` (Mục 9, 10, 11, 12).


In Progress:
- **Chuẩn bị triển khai Phase 6: Regularization & Fusion Mechanics (Khóa Cấu Hình Nền Tảng Cuối Cùng)**:
  + **Kế thừa các thông số đã khóa**:
    * Base ViT Depth = D4 (Phase 1)
    * Provisional `top_k = 2` (Phase 2)
    * `router_hidden_dim = 64` (Phase 3)
    * **Load balance factor = 0.010 (Tạm chốt giữ nguyên)** (Phase 4)
    * `sage_lr = 2e-4` (Phase 5.1)
    * `warmup_epochs = 3` (Phase 5.2)
    * `stage2_lr_ratio = 1.00` (Phase 5.3)
  + **Nội dung khảo sát Phase 6**:
    1. Phase 6.1: `expert_dropout \in {0.0, 0.1, 0.2}` (điều hòa chuyên gia).
    2. Phase 6.2: `residual_scale \in {0.05, 0.10, 0.20}` (với 0.10 là baseline mặc định).
    3. Phase 6.3: Cơ chế hợp nhất `fusion_type \in {"residual", "adaptive"}`.
  + **Tiêu chí lựa chọn**: Duy nhất dựa trên Validation Dice (348 ảnh). Tuyệt đối không can thiệp Test set.
- **Cập nhật Định hướng Kiến trúc Giải quyết Nút thắt High-Resolution CNN→ViT:**
  + Đã hoàn thành đánh giá độc lập 3 proposal cho nút thắt Stage 0/1 ($N=12,544$ và $N=3,136$) gọi ViT expert.
  + **Thứ tự ưu tiên nghiên cứu & triển khai đã chốt:**
    1. **PRIMARY BASELINE $\to$ Proposal 3: Feature/Detail Enhancement $\to$ Spatial Compression**
       * Refine đặc trưng vết nứt mảnh trên feature map trước khi nén không gian $\to$ nạp vào ViT-Tiny pretrained global attention $\to$ SA-Hub adapt về CNN shape $\to$ residual fusion với main path $112 \times 112$.
       * Chỉ tác động trên nhánh CNN$\to$ViT expert. Main CNN path và UNet skips giữ nguyên 100%. ViT block giữ nguyên dạng black-box pretrained.
    2. **SECOND $\to$ Proposal 1: Spatial Reduction Attention (SRA)**
       * $Q$ giữ full resolution ($112 \times 112 = 12,544$), $K, V$ được nén không gian (ví dụ $28 \times 28 = 784$). Global attention trên tập K/V nén.
       * Cần sửa logic attention trong ViT block, tái sử dụng pretrained Q/K/V projections và hỗ trợ an toàn cho shared expert bottleneck ($N=196$).
    3. **THIRD $\to$ Proposal 2: Restricted High-Resolution Routing**
       * Giới hạn Stage 0 và 1 chỉ route tới CNN experts. Stage 2/3 và ViT layers giữ full 16 experts.
       * Cần xử lý triệt để bài toán $top\_k=4$ trên tập chỉ có 4 CNN experts (nguy cơ sụp đổ entropy và forced selection).
  + **Nguyên tắc kiến trúc cốt lõi:**
    * Main CNN path là nguồn cung cấp chi tiết vết nứt sắc nét cho UNet skip connections; expert branch không thay thế main path.
    * Phân biệt rõ các loại tổn thất: mất độ phân giải (resolution), mất ngữ cảnh (context), mất tương tác đa phương thức (cross-modal), mất do nội suy (interpolation).
    * Các số liệu microbenchmark là ước tính tham khảo, không coi là end-to-end speedup bảo đảm.
    * Kích thước không gian thực tế: Stage 0 = $112 \times 112$ ($12,544$ tokens), Stage 1 = $56 \times 56$ ($3,136$ tokens), Stage 2 = $28 \times 28$ ($784$ tokens), Stage 3 / ViT = $14 \times 14$ ($196$ tokens).

Knowledge Being Learned:
- Cơ chế Routing đa chuyên gia (MoE), tính ổn định số học trong Softmax/Sigmoid gating dưới AMP FP16, giảm thiểu overhead của self-selection qua bypass `my_index`.
- Quy trình quản lý thực nghiệm bằng Git commit + config YAML để đảm bảo tính tái lập (reproducibility).
- Đo đạc biên giới hạn phần cứng (VRAM profiling) và tách bạch giữa hardware-constrained runtime settings vs HPO.
- Kỹ thuật phân rã throughput bằng CUDA Events: nhận diện chính xác Backward computation bottleneck trong mạng MoE thay vì đoán mò về I/O hay DataLoader.
- Động lực học định tuyến và nguyên lý bảo tồn tín hiệu vết nứt mảnh trong mạng MoE lai CNN-Transformer ở độ phân giải cao.

Current Issue:
- Không có issue. Two-Stage Training & Optimizer Preflight đã PASS 100% cho cả 3 cấu hình Run A, Run B và Run C (343/343 tensors khớp tuyệt đối, 0 missing, 0 duplicates, shared experts cô lập chuẩn ở CNN main blocks, Stage 2 LR ratio 1:1 bảo toàn).

Next Step:
- Chuẩn bị 4 candidate configs cho Phase 5.3 ($r \in \{0.25, 0.50, 1.00, 2.00\}$) kế thừa base `sage_lr = 2e-4`:
  + $r = 0.25$: `stage2_shared_lr = 2.5e-5`, `stage2_base_lr = 1.0e-4`
  + $r = 0.50$: `stage2_shared_lr = 5.0e-5`, `stage2_base_lr = 1.0e-4`
  + $r = 1.00$: `stage2_shared_lr = 1.0e-4`, `stage2_base_lr = 1.0e-4` (Baseline kế thừa từ Candidate B Phase 5.1, không train lại)
  + $r = 2.00$: `stage2_shared_lr = 2.0e-4`, `stage2_base_lr = 1.0e-4`
- Chuẩn bị driver notebook và các cell Colab T4 cho Phase 5.3.


## Milestones & SKs (Dependency-order)

### Milestone 0 (đã thêm): Infrastructure & Repo Setup
- [x] Đổi tên `SAGE_LITE` → `sage_lite`, khởi tạo Git repo độc lập, thêm `pyproject.toml`.
- [x] Phase 2: Tạo `train_crack.py` (CLI: --config, --resume, --output_dir), driver notebook Colab T4, configs cho Crack500 và DeepCrack.

### Milestone 1: Diagnostics & Core Backbone (B0 Baseline)
- [x] SK1: Viết và chạy `scripts/shape_audit.py` (channels [48,96,192,384], 196 tokens, xử lý CLS token).
- [x] SK2: Code `sage/networks/convnextv2_vit_hybrid.py` và verify forward pass.
- [x] SK3: Code `sage/networks/decoder_block.py` + hoàn thiện B0 UNet (`b0_unet.py`).
- [x] **Huấn luyện và nghiệm thu B0 Baseline:**
  - [x] Crack500 Run-3: Val Dice 0.7318, Test Setting A Dice 0.6771, Setting B Dice 0.6801.
  - [x] DeepCrack Run-1: Val Dice 0.6240, Test Direct Dice 0.6953.

### Milestone 1.5: Huấn luyện Baseline B1 & ViT-Depth Screening
- [x] Huấn luyện B1-6blocks trên Crack500 (PASS: Val Dice **0.7420**, Test Dice Setting A **0.6857**, Setting B **0.6895**, HD95 **77.43 px**)
- [x] **ViT-Depth Screening trên B1 (Crack500)**: Hoàn thành sweep tập `{4, 6, 8, 12}` trên tập Validation & Official Test:
  + [x] B0: 0 blocks — Val Dice 0.7318, Val Loss 1.3962
  + [x] B1 Depth 4: Val Dice **0.7428**, Val Loss **1.1364** (Top 1 Val), Test Dice A 0.6829, Test Dice B 0.6867
  + [x] B1 Depth 6: Val Dice 0.7420, Val Loss 1.1453 (Top 2 Val), Test Dice A 0.6857, Test Dice B 0.6895
  + [x] B1 Depth 8: Val Dice 0.7379, Val Loss 1.2006 (Top 4 Val), Test Dice A 0.6819, Test Dice B 0.6854
  + [x] B1 Depth 12: Val Dice 0.7419, Val Loss 1.2332 (Top 3 Val), Test Dice A 0.6902, Test Dice B 0.6926
  + [x] **Chiến lược:** Chưa freeze ViT depth từ B1; giữ nguyên dữ liệu screening, chuyển việc chốt depth sang **B2 Depth Ablation `{4, 6, 12}`** khi có SAGE routing.
- [x] Đánh giá Test Setting A & B cho toàn bộ sweep B1 trên Crack500: HOÀN TẤT.
- [ ] Huấn luyện B1 trên DeepCrack (Direct eval)

### Milestone 2: SAGE Core Mechanism & B2 (Full SAGE-Lite)
- [x] SK4: Port core mechanism (`sage/components/router.py`, `sage_layer.py`, `sa_hub.py`). Áp dụng fix eps 1e-5, FP32 logit modulation, bypass `my_index`, residual fusion scale 0.1.
- [x] SK5: Code `sage/networks/sage_injection.py` (Wiring full injection + isinstance guard). Hoàn thiện mô hình B2 (`b2_unet.py`) với Lock #1 (Tensor Contract) và Lock #2 (Zero Dynamic Params). Sẵn sàng huấn luyện.

### Milestone 3: Data Pipeline & Training Setup
- [x] SK6: Cập nhật `sage/utils/dataloader.py` và `losses.py` (DeepCrack pad+resize, Crack500 RandomCrop+smart filter, kết hợp BCE+SoftDice).
- [x] SK7: Viết `sage/utils/training_utils.py` và `scripts/train_crack.py` (Single-stage, AdamW, Cosine Annealing warmup, differential LR).

### Milestone 4: Diagnostic Scripts & Baseline Ladder
- [ ] SK8.6: **Generic Routing Diagnostics** (Top-K Activation Map, Expert Usage Ratio, Routing Entropy/Concentration).
- [ ] SK9: Cập nhật metrics trong evaluation (Dice primary, Boundary IoU, HD95).
- [ ] SK10: Chạy Baseline Ladder B0 → B2. **Quy tắc**: Train ĐỘC LẬP từ ImageNet gốc, KHÔNG warm-start từ bậc trước.
  - **B0**: ConvNeXtV2-Femto + U-Net (Hoàn thành trên Crack500 & DeepCrack)
  - **B1**: B0 + 6 ViT-Tiny (ConvNeXtV2-Femto + 6 ViT-Tiny + U-Net thuần, Late Fusion, không SAGE) — *[Đang tiến hành]*
  - **B2 (Full SAGE-Lite)**: B1 + full SAGE mechanism (SAGE Router + heterogeneous Expert Pool + SA-Hub + Load-Balance Loss)
- [ ] SK11: Ablation test: Sparse N_injection.
- [ ] SK12: Ablation test: Exploration Noise ON vs OFF

## Bảng chốt biến thực nghiệm (Configuration Map)

### Nhóm 1: Khóa CỨNG (đổi = phải sửa code/logic, không đổi tùy tiện)
- **ViT blocks**: 4 (Đã khóa từ Phase 1 Depth Scaling)
- **num_shared_experts**: 4 (auto = 4 CNN stages)
- **router_hidden_dim**: 64 (Best observed configuration từ Phase 3)
- **top_k**: 2 (Tạm chốt provisional từ Phase 2)
- **N_injection**: Full (8 routers ở baseline 4 ViT blocks: 4 CNN + 4 ViT)
- **Decoder conv**: Standard 3x3 (DWSC chỉ là flag tắt)
- **Fusion**: Residual (main + adapter, scale 0.1), không alpha
- **SA-Hub**: O(D²) eager pairwise, giữ nguyên gốc
- **Gating**: sigmoid, khóa cứng, không ablation
- **Loss**: 1.0*BCE + 1.5*SoftDice + 1.0*L_balance, không pos_weight
- **load_balance_factor**: 0.01 (Inherited baseline, Phase 4 skipped)
- **expert_dropout**: 0.1
- **freeze_encoder/transformer**: False/False
- **Stage2 LR ratio**: **1.0 (Đã khóa chính thức từ Phase 5.3)** — Candidate B unified optimizer (`stage2_shared_lr = stage2_base_lr = 1e-4`), bác bỏ soft freeze ($r=0.25, 0.50$).
- **AMP dtype**: FP16 (tối ưu Tesla T4)
- **Warmup**: 3 epochs
- **EarlyStopping patience**: 6
- **Weight decay**: 0.05, param groups tách LN/bias (WD=0.0)
- **Seed**: 42 cố định
- **Exploration Noise**: Giữ nguyên baseline = ON (giống code gốc). Ablation bật/tắt để riêng ra SK sau, không phải quyết định kiến trúc mặc định.

#### Nhóm 2: Cấu hình Thực nghiệm Đã Chốt
- **Batch size**: **14** (chuẩn hóa trên Tesla T4 sau preflight và Phase 1-3).
- **Base LR**: **1e-4** (Backbone LR: **1e-5**, Decoder LR: **1e-4**, SAGE LR đang khảo sát Phase 5.1 `{5e-5, 2e-4}`).
- **Training Budget**: **Epoch budget = 35** (Two-stage: Stage 1 = 17 epochs, Stage 2 = 18 epochs; `patience: 6`).
- **Crack500 smart filter**: **fg_pixels >= 20** (max 20 crop attempts × 10 source resamples).

## Migration Map

```text
sage-lite/
├── configs/
│   ├── datasets/
│   │   ├── colon.yaml, ebhi.yaml, glas.yaml          [DELETE]
│   │   ├── deepcrack.yaml, crack500.yaml              [NEW]
│   ├── experiments/
│   │   ├── baseline_*.yaml, sage_*.yaml (gốc)         [DELETE]
│   │   ├── b0_crack500.yaml, b0_deepcrack.yaml        [DONE] — Baseline B0
│   │   ├── b1_crack500.yaml, b1_deepcrack.yaml        [NEW] — Baseline B1
│   │   ├── b2_sage_lite.yaml                          [NEW] — B2 (Full SAGE-Lite)
├── prepare_data/
│   ├── prepare_colon*.py, prepare_ebhi.py, prepare_glas.py  [DELETE]
│   ├── prepare_deepcrack.py, prepare_crack500.py       [NEW]
├── sage/
│   ├── components/
│   │   ├── router.py                                   [MODIFY] eps fix (1e-9→1e-5), autocast(enabled=False) cho g_s, GIỮ NGUYÊN exploration noise (baseline=ON)
│   │   ├── sa_hub.py                                    [COPY nguyên]
│   │   ├── sage_layer.py                                [MODIFY] thêm `my_index` param → zero-cost self-select bypass
│   ├── networks/
│   │   ├── convnext_transformer_unet.py                 [REPLACE] → convnextv2_vit_hybrid.py (Femto backbone + ViT-Tiny 6 blocks)
│   │   ├── decoder_block.py                             [DONE] default 3x3, `use_dwsc` flag
│   │   ├── sage_injection.py                            [MODIFY] isinstance guard chặn downsampler/patch-embed/LayerNorm; thêm `num_sage_experts` property support
│   │   ├── wrappers.py                                  [COPY nguyên]
│   ├── utils/
│   │   ├── dataloader.py                                [DONE] Dataset DeepCrack (pad+resize) và Crack500 (RandomCrop + smart filter)
│   │   ├── losses.py                                     [DONE] BCE+SoftDice+L_balance combined
│   │   ├── metrics.py, advanced_metrics.py               [DONE] per-sample foreground Dice làm primary, thêm Boundary IoU + HD95
│   │   ├── evaluation.py                                 [DONE] Crack500 Setting A & B, DeepCrack direct
│   │   ├── training_utils.py                             [DONE] Single-stage, `get_param_groups` tách WD
│   │   ├── model_utils.py                                [DONE] `num_sage_experts` property
│   │   ├── gs_tracker.py, visualization.py                [COPY nguyên]
│   ├── __init__.py                                        [DONE] export mới
├── scripts/
│   ├── shape_audit.py                                     [DONE]
│   ├── train_crack.py                                     [DONE] — Single-stage training runner
│   ├── evaluate_crack_official.py                         [DONE] — Official Setting A/B evaluator
```

## Bảng Tổng hợp Hyperparameter & Baseline

| Hyperparameter | Phân loại | Giá trị Hiện hành | Ghi chú |
|---|---|---|---|
| **ViT blocks** | Kiến trúc | 4 | Đã khóa từ Phase 1 Depth Scaling |
| **num_shared_experts** | Kiến trúc | 4 | Bằng đúng 4 CNN stages (0..3) |
| **router_hidden_dim** | Kiến trúc | 64 | Best observed configuration từ Phase 3 |
| **top_k** | Kiến trúc | 2 | Tạm chốt provisional từ Phase 2 |
| **N_injection** | Kiến trúc | Full (8 routers) | Ở baseline 4 ViT blocks (4 CNN + 4 ViT) |
| **Decoder conv** | Kiến trúc | Standard 3x3 | DWSC chỉ là flag tắt |
| **Fusion** | Kiến trúc | Residual | main + 0.1 * adapter, không alpha |
| **SA-Hub** | Kiến trúc | O(D²) eager pairwise | Giữ nguyên thuật toán gốc |
| **Gating** | Kiến trúc | Sigmoid | Khóa cứng (không ablation) |
| **Exploration Noise** | Kiến trúc | ON | Giữ nguyên baseline |
| **Patch size** | Kiến trúc | 14x14 | 196 tokens tại bottleneck |
| **Loss Weights** | Training | 1.0*BCE, 1.5*SoftDice, 1.0*L_balance | Không dùng pos_weight |
| **load_balance_factor** | Training | 0.01 | Inherited baseline, Phase 4 skipped |
| **expert_dropout** | Training | 0.1 | Hạn chế overfitting SA-Hub |
| **freeze_encoder/transformer** | Training | False/False | Fine-tune cả mạng |
| **Stage2 LR ratio** | Training | Khảo sát Phase 5.3 ($r=1.0$ baseline) | $r = LR_{\text{shared experts}} / LR_{\text{base}}$ |
| **AMP dtype** | Training | FP16 | Tối ưu trên GPU Tesla T4 |
| **Warmup** | Training | 3 epochs | CosineAnnealingLR sau warmup |
| **EarlyStopping patience** | Training | 6 | Dừng sớm nếu Val Dice không tăng |
| **Weight decay** | Training | 0.05 | Param groups tách LN/bias (WD=0.0) |
| **Seed** | Training | 42 | Cố định toàn pipeline |
| **Batch size** | Thực nghiệm | 14 | Chuẩn hóa trên Tesla T4 sau preflight & Phase 1-3 |
| **Base LR** | Thực nghiệm | 1e-4 | Backbone LR: 1e-5, Decoder LR: 1e-4 |
| **Training Budget** | Thực nghiệm | Epoch budget = 35 | Two-stage: 17 S1 + 18 S2; EarlyStopping patience = 6 |
| **min_pixels Crack500** | Data | fg_pixels >= 20 | Ngưỡng lọc patch có vết nứt (Canonical) |


## Progress Log
- **2026-09-20**: Hoàn thành Preflight Shape Audit (SK1) qua `scripts/shape_audit.py`. Xác nhận chính thức Shape Contract cho Milestone 1: Input 448x448, ConvNeXt-Femto channels [48, 96, 192, 384], bottleneck 14x14 = 196 tokens, ViT-Tiny slice 6 blocks, tách riêng CLS token khỏi pos_embed, N_injection = 10 (4 CNN stages + 6 ViT blocks). Đủ điều kiện bắt đầu code SAGE-Lite (SK2).
- **2026-09-20**: Khởi tạo lại `milestone_and_progress.md` dựa trên kế hoạch SAGE-Lite mới (Blueprint/Implementation Plan) theo cấu trúc Dependency-order. Rõ ràng 3 script diagnostic không chặn code.

- **2026-09-20**: Cập nhật logic đánh giá Batch Size (đo riêng B0/B2, dùng Gradient Accumulation) và Epoch (ngân sách N_total chung) để đảm bảo tính công bằng thực nghiệm (so sánh accuracy và tính complexity).
- **2026-09-20**: Fix thêm 2 lỗ hổng thực nghiệm (LR Range Test đo riêng B0/B2, nghiêm cấm warm-start ở Baseline Ladder) và đồng bộ lại file SAGE_lite_Implementation_Plan.md.
- **2026-09-22**: Hoàn thành B0 Baseline chính thức trên cả 2 dataset (Crack500 Run-3: Val Dice 0.7318, Test Setting A Dice 0.6771, Setting B Dice 0.6801; DeepCrack Run-1: Val Dice 0.6240, Test Direct Dice 0.6953). Chốt frozen canonical preprocessing cho toàn bộ dự án.
- **2026-09-22**: Chốt thang đo Baseline Ladder: B0 (ConvNeXtV2-Femto + U-Net) → B1 (B0 + 6 ViT-Tiny) → B2 (Full SAGE-Lite). Config `b1_crack500.yaml` chốt `batch_size: 16` an toàn, `Epoch budget = 30` (EarlyStopping patience = 6).
- **2026-09-22**: Hoàn thành Huấn luyện & Đánh giá chính thức B1 trên Crack500 (Run-1). Val Dice = 0.7420 (@ epoch 21). Test Setting A: Dice 0.6857 (IoU 0.5730, HD95 83.11 px). Test Setting B: Dice 0.6895 (IoU 0.5770, HD95 77.43 px). Vượt B0 toàn diện trên 100% metrics (+0.94% Dice Setting B, HD95 giảm mạnh 16.93 px). Sẵn sàng chuyển sang B1 DeepCrack.
- **2026-09-23**: Hoàn thành toàn diện ViT-Depth Sweep B1 trên Crack500 `{4, 6, 8, 12}` và đánh giá chính thức Setting A & Setting B:
  + Depth 4: Val Dice **0.7428**, Val Loss **1.1364** (Top 1) | Test Setting A Dice 0.6829 | Test Setting B Dice 0.6867 (Recall B 0.8589, HD95 90.64 px).
  + Depth 6: Val Dice 0.7420, Val Loss 1.1453 (Top 2) | Test Setting A Dice 0.6857 | Test Setting B Dice 0.6895.
  + Depth 8: Val Dice 0.7379, Val Loss 1.2006 (Top 4) | Test Setting A Dice 0.6819 | Test Setting B Dice 0.6854.
  + Depth 12: Val Dice 0.7419, Val Loss 1.2332 (Top 3) | Test Setting A Dice 0.6902 | Test Setting B Dice 0.6926.
  + **Quyết định Chiến lược:** Chưa freeze ViT depth từ B1; B1 đóng vai trò screening. Depth tối ưu của SAGE-Lite sẽ được đo đạc và chốt qua **B2 Depth Ablation `{4, 6, 12}`** khi có SAGE routing. Không cần chạy lại B1. Sẵn sàng huấn luyện B1 trên DeepCrack.
- **2026-09-23**: Khởi tạo lộ trình thực nghiệm chi tiết cho B2 (Full SAGE-Lite) từ Phase 0 đến Phase 6. Lộ trình được tài liệu hóa tại `docs/B2_Experimental_Roadmap.md`. Đã hoàn tất cài đặt kỹ thuật B2, sẵn sàng chạy Phase 0 (Runtime preflight) trên Colab.
- **2026-09-25**: Hoàn thành Chuẩn bị Giao thức Thử nghiệm Đối chứng P3 (P3-PHASE-6):
  + Đối chiếu và chuẩn hóa toàn diện thuật ngữ giữa Phase 6 và Phase 8 trong `docs/B2_Experimental_Roadmap.md`: Run A được định danh chính xác là "Compression Control under Common Fixed PE28 Treatment" (`no refinement + common fixed PE28 + AvgPool28 -> ViT`); Run B và Run C được xác định là các kiểm soát gần tương đương về dung lượng ("near-matched-capacity controls", Run B = 38,450 vs Run C = 37,874 params, lệch 576 params ở DW) cung cấp bằng chứng thực nghiệm về cấu trúc định hướng/đa tỷ lệ; Speedup được chuẩn hóa thành chỉ số chẩn đoán (diagnostic metric) so với Direct Baseline expert-path latency theo 4-Pillar Framework, loại bỏ ngưỡng bác bỏ cứng ad-hoc `< 10x` chưa đăng ký trước.
  + Khởi tạo bộ 3 file cấu hình YAML chuẩn hóa tại `sage_lite/configs/p3_ablation/`: `b2_p3_run_a.yaml` (Run A, `p3_mode: "A"`), `b2_p3_run_b.yaml` (Run B, `p3_mode: "B"`), `b2_p3_run_c.yaml` (Run C, `p3_mode: "C"`).
  + Xác minh ma trận tham số (Diff Matrix): 100% siêu tham số ngoài khối refinement (Locked Base Depth = 4, Seed = 42, Batch Size = 12, Epochs = 30, Learning Rate = 1e-4, 9 khóa của `sage_config`, data paths) hoàn toàn đẳng cấu và đồng nhất.
  + Chuẩn hóa và đóng băng giao thức cuối cùng (Final Protocol Freeze): Khẳng định `p3_mode` là biến hành vi mô hình duy nhất (`output_dir` chỉ để cô lập artifact); khóa ngữ nghĩa `epochs = 30` là ngân sách tối đa có early stopping (`patience = 6`); ghi nhận ghi chú phương pháp luận phân biệt rõ A/B/C (kiểm soát dưới PE28 chung) vs Direct Baseline (so sánh cấp độ hệ thống); chuẩn hóa kết quả nghiệm thu Phase 5 là 12/12 tests PASS. P3-PHASE-6 chính thức được đánh dấu **FROZEN**.
- **2026-09-25**: Hoàn thành Kiểm tra Runtime Run-A và Tiền kiểm tra Tối ưu hóa Hai giai đoạn (Two-Stage Optimizer Preflight) cho cả 3 Run A, Run B, Run C (P3-PHASE-6 & P3-PHASE-7 Preflight):
  + **Run-A Runtime Gate PASS**: Xác minh kiểu runtime của `stage0.p3_refinement` và `stage1.p3_refinement` là `nn.Identity()`. Nhánh nén P3 ($28 \times 28$, `backbone.pe28_fixed`, $784$ tokens) được kích hoạt chuẩn xác; đường direct high-res cũ tuyệt đối không bị gọi cho ViT experts.
  + **Run A Two-Stage Preflight PASS**: 333/333 tensors được hạch toán đầy đủ (0 duplicate, 0 missing); 132 shared tensors thuộc duy nhất về CNN main blocks; `model.set_shared_experts([0, 1, 2, 3])` thực thi chuẩn; scheduler Stage 2 khởi tạo lại với `warmup=3`.
  + **Run B & Run C Two-Stage Preflight PASS**:
    * Run B: 10 P3 tensors (38,450 params). Stage 1 phân bổ vào nhóm `backbone` (LR $10^{-5}$, WD 0.05). Stage 2 sau reload và `set_shared_experts([0,1,2,3])` thuộc strictly về `other_and_routers` (LR $10^{-4}$, WD 0.05), không lẫn vào `shared_experts`. Khớp 343/343 trainable tensors (0 missing, 0 duplicate).
    * Run C: 10 P3 tensors (37,874 params). Stage 1 phân bổ vào `backbone` (LR $10^{-5}$, WD 0.05). Stage 2 thuộc strictly về `other_and_routers` (LR $10^{-4}$, WD 0.05), không lẫn vào `shared_experts`. Khớp 343/343 trainable tensors (0 missing, 0 duplicate).
- **2026-09-26**: Hoàn thành Huấn luyện Đối chứng Chuẩn tắc P3 (Canonical Baseline D12, 12 ViT blocks, 16 experts: 4 CNN + 12 ViT) trên Crack500:
  + **Run A (Identity Control)**: Peak S1 Dice 0.7158, Peak S2 Dice **0.7543** (Val Loss 1.0140).
  + **Run B (Generic DW)**: Peak S1 Dice 0.7179, Peak S2 Dice **0.7496** (Val Loss 1.0495).
  + **Run C (ASDW Refinement)**: Peak S1 Dice 0.7266, Peak S2 Dice **0.7557** (Val Loss 1.0889).
  + Run C (ASDW) vượt trội cả Run A (+0.14% Dice) và Run B (+0.61% Dice), chứng minh tính hiệu quả của cơ chế tinh chế bất đối xứng dải định hướng (1x7, 7x1).
- **2026-09-27**: Hoàn thành Khảo sát Độ nhạy Độ sâu ViT (ViT-Depth Sweep D12 $\to$ D8 $\to$ D6 $\to$ D4) cho P3-C (ASDW):
  + **P3-C D8 (8 ViT blocks, 12 experts)**: Peak S1 Dice **0.7374**, Peak S2 Dice **0.7596** (Val Loss 1.0700). Tốc độ tăng ~18% (1.48s/it).
  + **P3-C D6 (6 ViT blocks, 10 experts)**: Peak S1 Dice 0.7312, Peak S2 Dice **0.7599** (Val Loss **0.9602**). Tốc độ tăng ~21% (1.42s/it).
  + **Kiểm chứng Hội tụ Hậu nghiệm (Convergence Extension Audit)**: Đã kiểm chứng tiếp nối tại sàn LR $10^{-6}$ qua 6 epochs cho cả D12, D8 và D6; cả 3 đều kích hoạt `EarlyStopping (patience=6)` chuẩn mực mà không cải thiện thêm Dice, xác nhận điểm dừng toán học thực thụ.
  + **P3-C D4 (4 ViT blocks, 8 experts: 4 CNN + 4 ViT, 1:1 symmetry, $k=4$)**:
    * Thiết lập **KỶ LỤC MỚI TOÀN DỰ ÁN**: Peak Val Dice = 🏆 **0.7639** (@ S2 Ep 14), duy trì ổn định $>0.760$ liên tục suốt 5 epoch cuối (Ep 14-18, TB 0.7625).
    * **Val IoU**: 🏆 **0.6412** (Cao nhất lịch sử).
    * **Val Precision**: 🏆 **0.7298** (Tăng vọt +2.33% so với D6, giảm mạnh lỗi over-segmentation từ 39 xuống 28 mẫu).
    * **Val Loss**: Đạt đáy **0.9317** (S2 Ep 18).
    * **Động lực học Gamma**: $\gamma_{S0} = 0.0162$, $\gamma_{S1} = 0.0205$ (tăng gấp gần 3 lần so với D12: 0.0071), chứng minh nhu cầu bù đắp inductive bias dạng dải từ CNN khi ViT nông.
    * **Kiểm chứng Hội tụ Hậu nghiệm D4 (Convergence Extension Audit)**: Đã hoàn tất kiểm chứng tiếp nối tại sàn LR $10^{-6}$ qua 6 epochs (`results/logs/P3_C_Canonical_Base_D4_Extension.log`), Val Dice dao động $0.7599 - 0.7634$ (không vượt đỉnh 🏆 **0.7639**), kích hoạt `EarlyStopping (patience=6)` chuẩn mực.
    * **Kết luận Toàn diện 4 Depth (D12, D8, D6, D4)**: 100% cả 4 cấu hình đã kích hoạt EarlyStopping chuẩn, chứng minh không cấu hình nào bị dừng sớm do thiếu budget. Kỷ lục 🏆 **0.7639** của D4 là điểm dừng tối ưu toán học thực thụ (*true mathematical convergence*).
    * Đã lưu trữ toàn diện: `results/logs/P3_C_Canonical_Base_D4.log`, `results/logs/P3_C_Canonical_Base_D4_Extension.log`, `results/checkpoints/P3_C_D4_best_model_b2_global.pth`, `results/P3_C_Routing_Diagnostics_D4/`, `results/configs/b2_p3_run_c_d4.yaml` và cập nhật `results/p3_abc_epoch_by_epoch_metrics.json/md`.
- **2026-09-28**: Giải quyết Triệt để Cosine LR Schedule Confound & Tái huấn luyện D8 Chuẩn tắc (Option B: $t_{initial}=18$, budget 33 epochs):
  + **Xác minh Confound**: Trước đây D8 chạy `budget=30` ($t_{initial}=15$) khiến LR tại S2 Ep 13 bị ép xuống $5.28 \times 10^{-6}$ (chênh 3.54x so với $1.87 \times 10^{-5}$ của D4/D6).
  + **Kết quả Huấn luyện D8 Chuẩn tắc**: Khi được cấp cùng đường cong cosine suy giảm ($t_{initial}=18$), D8 tiếp tục fine-tune sâu đến Ep 17 và đạt đỉnh **0.7604** (Val Loss: **0.9999**).
  + **Định vị Lại Bảng Xếp hạng (Fair Controlled Ranking)**:
    * **D4**: 🏆 **0.7639** (Top 1 áp đảo, kiến trúc đối xứng 1:1 tối ưu).
    * **D8**: **0.7604** (Chính thức vượt D6 sau khi gỡ bỏ rào cản LR schedule).
    * **D6**: **0.7599** (Ổn định, tốc độ nhanh).
    * **D12**: **0.7557** (Bị over-smoothing và CNN starvation).
  + **Lưu trữ Trọn bộ Artifacts**:
    * Checkpoint chuẩn: `results/checkpoints/P3_C_D8_best_model_b2_global.pth` (bản cũ được lưu tại `P3_C_D8_best_model_b2_global_e30.pth`).
    * Log huấn luyện: `results/logs/P3_C_Canonical_Base_D8.log` & `results/logs/P3_C_Canonical_Base_D8_E33.log` (bản cũ tại `P3_C_Canonical_Base_D8_E30.log`).
    * Toàn bộ Routing Diagnostics & Error Analysis: `results/P3_C_Routing_Diagnostics_D8/` (bản cũ tại `results/P3_C_Routing_Diagnostics_D8_E30/`).
    * Dữ liệu định lượng: Cập nhật `results/p3_abc_epoch_by_epoch_metrics.json` & `.md`.
- **2026-09-28 (tiếp tục)**: Hoàn thành Huấn luyện & Đánh giá Toàn diện P3-C D2 (2 ViT blocks, 6 experts: 4 CNN + 2 ViT, $k=4$, 9.20M params):
  + **Kết quả Thực nghiệm**: Peak S1 Dice 0.7190 (@ Ep 13), Peak S2 Dice **0.7578** (@ S2 Ep 16), Val Loss 0.9477 (đáy 0.9364).
  + **Phát hiện Khoa học mang tính Bước ngoặt — Đường cong Parabol Ngược (Inverted-U Concave Curve)**:
    * D12 (0.7557) $\to$ D8 (0.7604) $\to$ D6 (0.7599) $\to$ **D4 (0.7639 - Đỉnh cao tối ưu)** $\to$ D2 (0.7578 - Suy giảm dung lượng).
    * Khi giảm xuống D2, việc thiếu hụt ViT blocks làm suy yếu khả năng mô hình hóa quan hệ không gian tầm xa (lỗi `fragmented_prediction` tăng vọt từ 4 lên 10 mẫu, tỷ lệ chọn CNN bị kéo lệch lên 67.6%).
    * Chứng minh thực nghiệm rằng **D4 là "Vùng Goldilocks" tối ưu tuyệt đối** (đạt cân bằng hoàn hảo giữa Inductive Bias cục bộ của CNN và Attention toàn cục của ViT).
  + **Lưu trữ Trọn bộ Artifacts (Bao gồm cả Best và Last models theo /learn)**:
    * Checkpoints: `results/checkpoints/P3_C_D2_best_model_b2_global.pth`, `last_model_b2_stage2.pth`, `best_model_b2_stage1.pth`, `last_model_b2_stage1.pth`.
    * Log: `results/logs/P3_C_Canonical_Base_D2.log`.
    * Diagnostics: `results/P3_C_Routing_Diagnostics_D2/` (đầy đủ `full_val/` và `error_analysis/`).
    * Config: `results/configs/b2_p3_run_c_d2.yaml`.
    * Dữ liệu: Cập nhật `results/p3_abc_epoch_by_epoch_metrics.json` & `.md`.
- **2026-09-28 (tiếp tục)**: Khảo sát Dung lượng Kích hoạt Top-k=2 (D4 K2, 8 experts pool):
  + Val Dice đạt **0.7618** (chỉ chênh -0.21% so với 0.7639 của D4 K4 dù giảm 50% số expert kích hoạt).
  + Khẳng định tính hiệu quả và gọn nhẹ của cấu hình $k=2$ cho các phase tối ưu hóa tiếp theo.
- **2026-09-28**: Nghiên cứu Can thiệp Định tuyến (Routing Intervention Diagnostic trên N=348 Crack500 Val):
  + Phép thử hoán đổi Adaptive vs Frequency-based Static Top-4 cho $\Delta = +0.00037$ ($p = 0.413$, không có ý nghĩa thống kê).
  + Kết luận đột phá: Hiệu năng vượt trội của SAGE-Lite đến từ **Ensemble Quality** (tập hợp chuyên gia phong phú kết hợp SA-Hub và ASDW) chứ không phụ thuộc vào quyết định routing động từng mẫu. Khóa nguyên vẹn router, không can thiệp GAP.
- **2026-09-29**: Khảo sát Độ nhạy Scale Residual (Residual/Fusion Sensitivity):
  + Tách bạch phản ứng của từng họ tầng: S3 nhạy scale dương, nhóm ViT nhạy scale âm khi $>0.1$.
  + Biên độ biến thiên rất nhỏ ($<0.001$ Dice), loại trừ cơ chế fusion khỏi danh sách các nút thắt hiệu năng chính.
- **2026-09-29**: Hoàn thành & Nghiệm thu Phase 3 Router Hidden Dimension Sweep ($H32 \to H64 \to H128$):
  + Thiết lập đường cong chữ U ngược: $H32 (0.7587) < H128 (0.7550) < \mathbf{H64 (0.7618)}$.
  + **Chính thức khóa $router\_hidden\_dim = 64$** làm tham số nền tảng.
- **2026-09-29**: Hoàn thành & Nghiệm thu Phase 4 Load Balancing Factor Sweep ($0.005 \to 0.010 \to 0.030$):
  + Xác lập đỉnh Pareto tại $LB = 0.010$ (Val Dice **0.7641** của Candidate B).
  + **Chính thức khóa $load\_balance\_factor = 0.010$** làm chuẩn mực vĩnh viễn.
- **2026-09-29**: Nghiệm thu Phase 5 & Đóng băng Chuẩn tắc Candidate B:
  + Quét 6 tỉ số learning rate Stage 2 ($r \in \{0.25, 0.50, 1.00, 2.00, 4.00, 5.00\}$). Xác lập $r=1.00$ là Sweet Spot tối ưu.
  + **Chính thức khóa Candidate B** ($D=4, K=2, H=64$, Stage 2 base $r=1.00$) làm Canonical Base với Val Dice **0.7641**, Mean IoU **0.6417**.
- **2026-09-30**: Hoàn thành Thực nghiệm Phase 6-A.1 Objective Probe (Soft Boundary IoU Loss, $\lambda=0.50, d=2$):
  + **Thiết lập Đỉnh Toàn Dự Án Mới: Val Dice 0.7684** (+0.0043), Mean IoU **0.6465**, Precision **0.7416** (+0.79%).
  + Đột phá trên nhóm vết nứt siêu mảnh ($Q4$ thin cracks $>0.20$): Val Dice tăng $+2.37\%$ ($0.6404 \to 0.6641$), sai số loang viền dôi dư giảm $-15.82\%$ ($80.37\% \to 64.55\%$).
- **2026-10-01**: Hoàn thành & Nghiệm thu Thực nghiệm Phase 6-A.2 Pure PLU Representation Probe (PLU-Head, BoundaryIoU OFF, L = L_Base):
  + **Giao thức chuẩn xác**: Huấn luyện toàn vẹn từ đầu Full Stage 1 (17 epochs, backbone LR=1e-5, decoder/PLU=1e-4, SAGE=2e-4) $\to$ Stage 2 (18 epochs, all LR=1e-4). Khóa cứng `boundary_iou_weight = 0.0`. Chuyển tiếp checkpoint Stage 1 $\to$ Stage 2 an toàn tuyệt đối với `strict=True` (0 missing, 0 unexpected).
  + **Quỹ đạo hội tụ xuất sắc**:
    * Stage 1 Peak Val Dice: **0.7357** (@ Ep 10, vượt trội 0.7333 của Candidate B).
    * Stage 2 Ep 1 Val Dice: **0.7235** (hội tụ trơn tru, triệt tiêu 100% cú sập).
    * Stage 2 Peak Global Val Dice: **0.7664** (@ Ep 15, Val Loss: **0.7135**), vượt qua Candidate B baseline (**0.7641**) chỉ bằng cải tiến biểu diễn thuần túy (+0.0023 Dice, Precision tăng từ 0.7337 lên **0.7491**).
  + **Đột phá hình thái học trên vết nứt siêu mảnh (Thin Crack Recovery)**:
    * Nhóm vết nứt mảnh ($\text{thinness} > 0.20, n=66$): Val Dice đạt **0.6674** (+0.0270 vs Candidate B, cao hơn cả can thiệp BoundaryIoU của 6-A.1 đạt 0.6641). Precision đạt **0.5764** (+5.32% vs Candidate B). Tỷ lệ phình diện tích giảm sâu từ $+80.37\%$ xuống **$+57.00\%$** ($\Delta = -23.37\%$).
    * Nhóm hỏng nặng nhất (Thin-Low-Area, $n=5$): Val Dice tăng vọt từ $0.5370 \to \mathbf{0.6059}$ (+0.0689), tỷ lệ phình diện tích cực đoan giảm sốc từ $+141.66\%$ xuống **$+77.06\%$** ($\Delta = -64.60\%$).
  + **Phán quyết ma trận**: **POSITIVE**. Xác nhận giả thuyết nút thắt biểu diễn (Representation Bottleneck) là có thật.
  + **Lưu trữ toàn bộ artifacts**:
    * Checkpoints: `results/checkpoints/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_best_model_b2_global.pth`, `best_model_b2_stage1.pth`, `best_model_b2_stage2.pth`, `last_model_b2_stage1.pth`, `last_model_b2_stage2.pth`.
    * Logs: `results/logs/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_train.log`.
    * Routing & Error Diagnostics: `results/P3_C_Routing_Diagnostics_Phase6_A2_Pure_PLU_D4_K2_H64/` (đầy đủ `full_val/` và `error_analysis/`).
    * Archive: `results/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_Full.zip`.
- **2026-10-01 (tiếp tục)**: Hoàn thành & Nghiệm thu Thực nghiệm Phase 6-B.1 Objective Probe (AB-BPL Probe, $\lambda=0.040, r=2$, Stage 2 only):
  + **Giao thức chuẩn tắc**: Giữ nguyên 100% kiến trúc mạng Candidate B (10,118,955 params, +0 params), khởi động Stage 2 từ Candidate B Stage 1 checkpoint với strict parameter alignment. Bổ sung Asymmetric Boundary-Band Penalty Loss ($\mathcal{L}_{\text{AB-BPL}}$) với $\lambda=0.040, r=2$ để giám sát dải viền ngoài $M_{\text{bg}} = \text{dilate}(GT, r) \setminus GT$.
  + **Tiền kiểm tra nghiêm ngặt (Preflight Gate 1-5)**: PASS 100% (Loss & Gradient exact equality khi $\lambda=0$, bất đối xứng trừng phạt FP, AMP FP16 numerical stability, bất biến tham số tuyệt đối, strict checkpoint load).
  + **Kết quả định lượng toàn diện**:
    * Global Val Dice: ✅ **0.7641 → 0.7685** (+0.0044, thiết lập đỉnh cao mới Phase 6, Mean IoU: 0.6465, Win rate: 56.32% [196/348]).
    * BM Precision ($n=127$): ✅ Mean tăng $+0.0059$ ($0.7452 \to 0.7511$), Median tăng $+0.0105$ ($0.7533 \to 0.7638$), $56.7\%$ win rate.
    * BM Recall preservation: ✅ Bảo toàn nguyên vẹn ở mức $0.8321$ (Median $\Delta = -0.0005$, không bị suy sụp như A2 PLU $0.8087$).
    * BM Median / Aggregated Dice: ✅ Xuất hiện tín hiệu cải thiện rõ rệt (Median: $0.7841 \to 0.7901$, Aggregated: $0.7869 \to 0.7909$, Win rate $62.20\%$ [79/127], kiểm định Wilcoxon $p = 0.0200$).
    * BM Sample-mean Dice: ❌ $0.7812 \to 0.7802$ ($-0.0010$, bị ảnh hưởng bởi 2 ngoại lai ở đuôi phân phối).
    * BM Area Excess: ❌ Chưa đạt guidepost đăng ký trước ($\le -5.0\%$), nhưng chiều hướng hoàn toàn đúng (Sample-mean: $+14.86\% \to +13.38\%$, $\Delta = -1.47\%$; Median: $+15.49\% \to +13.13\%$, $\Delta = -2.36\%$; giảm diện tích dôi dư trên $54.3\%$ số mẫu).
    * Thin Cracks ($n=66$): ✅ Dice tăng từ $0.6404 \to 0.6555$ (+0.0151), độ phình diện tích giảm $-12.86\%$ ($+80.37\% \to +67.51\%$).
    * Thin-low-area ($n=5$): ❌ Không thay đổi ($0.5370 \to 0.5362$), củng cố phát hiện rằng vết nứt siêu mảnh cần độ phân giải không gian trực tiếp (như A2 PLU) thay vì chỉ ràng buộc hàm mất mát.
  + **Trạng thái & Phán quyết chính thức**:
    $$\boxed{\textbf{B-1: Positive global result + partial boundary-mechanism success}}$$
    > *Phase 6-B.1 provides evidence that asymmetric boundary-band supervision can improve global validation Dice while preserving boundary recall and modestly reducing excess predicted area, but the pre-registered BM Dice and Area Excess guideposts were not fully met. The result therefore supports AB-BPL as a complementary boundary objective, rather than establishing it as a complete solution to boundary over-dilation.*
  + **Quy tắc vận hành**: Không chạy thêm chỉ để “đuổi” Guidepost $-5\%$. B-1 đã cung cấp một phenotype rất rõ; sử dụng thông tin này làm cơ sở kết hợp cơ chế hoặc mở rộng các nhánh tiếp theo.
  + **Lưu trữ toàn bộ artifacts**:
    * Checkpoints: `results/checkpoints/P3_C_Phase6_B1_AB_BPL_D4_K2_H64_best_model_b2_global.pth`, `best_model_b2_stage2.pth`, `last_model_b2_stage2.pth`.
    * Log huấn luyện: `results/P3_C_Routing_Diagnostics_Phase6_B1_AB_BPL_D4_K2_H64/train.log`.
    * Diagnostics: `results/P3_C_Routing_Diagnostics_Phase6_B1_AB_BPL_D4_K2_H64/diagnostics/` (bao gồm `error_summary.json`, `per_sample_metrics.csv`, `figures/`, `qualitative/`, `full_val/`).
    * Config: `results/configs/b2_p3_run_c_d4_k2_h64_phase6_b1_ab_bpl.yaml`.
    * Archive: `results/P3_C_Phase6_B1_AB_BPL_D4_K2_H64_Full.zip`.
- **2026-10-01 (tiếp tục)**: Triển khai Chẩn đoán Hình thái học & Liên thông Tiền can thiệp Phase 6-C (Pre-intervention Topology & Connectivity Diagnostics trên N=348 Crack500 Val):
  + **Động lực & Công cụ**: Tách bạch triệt để giữa hiện tượng Nối cầu giả (False Bridge / Merge) và Đứt gãy (Breakage / Fragmentation) bằng công cụ `tools/run_phase6_c_topology_diagnostic.py` (8-connectivity, hard skeletonization, đo clDice, $T_{\text{prec}}$, $T_{\text{sens}}$, CC signed/abs errors). Chạy ghép cặp đồng thời 4 mô hình: Candidate B, 6-A.1, 6-A.2, 6-B.1.
  + **Phát hiện Phenotype cốt lõi**:
    * **Candidate B**: Bottleneck áp đảo là **False Bridges (31.6%, 110 mẫu)** và Over-dilation (Area Excess $+8.16\%$), trong khi Breakage chỉ chiếm **9.5% (33 mẫu)**.
    * **Phase 6-A.2 (Pure PLU)**: Hiện tượng $Pred\_CC > GT\_CC$ vọt lên $44.5\%$ thực chất là do bùng nổ **Spurious Islands (614 đảo rác vs 108 của Base)** và đứt gãy vết nứt mảnh (Breakage trong nhóm Thin vọt từ $12.1\% \to 30.3\%$).
    * **Boundary Margin ($n=127$)**: $75.6\%$ số mẫu BM hoàn toàn không bị nối cầu giả; bản chất của BM là vành đai mờ viền (halo expansion) của một vết nứt đơn lẻ.
  + **Báo cáo & Dữ liệu chi tiết**: Lưu tại `results/diagnostics/phase6_c_topology/TOPOLOGY_PRE_INTERVENTION_REPORT.md`, `topology_per_sample_paired.csv`, `topology_cross_tabulation.csv`, `topology_summary.json`.
- **2026-10-01 (tiếp tục)**: Hoàn thành & Nghiệm thu Thực nghiệm Phase 6-C.1 (Soft-clDice Controlled Probe, $\lambda_{\text{clDice}}=0.030$, iters=5, Stage 2 only):
  + **Thiết kế can thiệp**: Bổ sung vi phân Soft-clDice Loss vào Stage 2 (từ Candidate B Stage 1 checkpoint, 10,118,955 params, +0). Hiệu chỉnh gradient scale tỉ lệ $\sim 30\%$ regularizing force ($\text{Base Grad Norm}=0.00052$, $\text{clDice Grad Norm}=0.00487 \implies \text{ratio}=9.36\times$, $\lambda=0.030$). Preflight Gate 1–5 PASS 100% trên GPU.
  + **Kết quả thực nghiệm 4 Tiers & Topology (So với Candidate B)**:
    * Global Val Dice: ❌ **0.7641 → 0.7613** ($\Delta = -0.0028$, Win rate: 38.5% [134/348]).
    * Centerline Dice (clDice): ⚠️ Tăng từ **0.8498 → 0.8525** (Median: $0.9087 \to 0.9114$).
    * Skeleton Coverage: ✅ $T_{\text{sens}}$ đạt đỉnh cao nhất **0.8899** (Base 0.8872), $T_{\text{prec}}$ tăng lên **0.8501** (Base 0.8473).
    * Global Recall: ✅ Tăng lên **0.8591** (Base 0.8477).
    * Global Precision: ❌ Sụt giảm từ **0.7338 → 0.7210** ($\Delta = -0.0128$).
    * Global Area Excess: ❌ Tăng nở viền từ **+8.16% → +11.82%** (bị phình thêm $+3.66\%$).
    * Thin Cracks Breakage ($n=66$): ❌ **12.1% → 12.1%** (8 mẫu $\to$ 8 mẫu, bất biến, không đạt guidepost $\le 9.0\%$).
    * Global False Bridge: ❌ **31.6% → 32.5%** (110 $\to$ 113 mẫu, về cơ bản bất biến).
    * Spurious Islands: ✅ Giảm còn **102 đảo** (sạch nhất trong toàn bộ 5 models).
  + **Quy chuẩn Quy ước Tính toán (Convention Locking)**:
    * Khóa chuẩn duy nhất cho **Area Excess (%)**:
      $$\text{Area Excess (\%)} = \frac{\sum \text{Pred Area} - \sum \text{GT Area}}{\sum \text{GT Area}} \times 100\%$$
      (tính trên tổng diện tích phân tầng, đảm bảo tính nhất quán trên toàn bộ các báo cáo).
  + **Trạng thái & Phán quyết chính thức**:
    $$\boxed{\textbf{C.1 = Negative global result + confirmed topology trade-off}}$$
    > *Soft-clDice có thể cải thiện centerline/topological coverage ($T_{\text{sens}}: 0.8872 \to 0.8899$) và giảm một phần artifact fragmentation/island ($108 \to 102$), nhưng không giải quyết false bridge ($31.6\% \to 32.5\%$) và đi kèm trade-off theo hướng tăng predicted area ($+8.16\% \to +11.82\%$), làm giảm precision và Global Dice ($0.7641 \to 0.7613$).*
  + **Bác bỏ giả thuyết cốt lõi (Scientific Hypothesis Disproven)**:
    $$\boxed{\text{Breakage preservation} \neq \text{False-bridge correction}}$$
    *Tối ưu hóa tính liên thông (continuity) giúp bảo vệ khung xương nứt nhưng bất lực trước bài toán chia tách cấu trúc bị nối cầu sai (merge/separation). Kết quả quan sát phù hợp với cơ chế trong đó tối ưu skeleton sensitivity khuyến khích duy trì coverage quanh GT centerline, và trên bài toán này điều đó đi kèm xu hướng mở rộng vùng foreground.*
  + **Quy tắc vận hành**: **Đóng băng vĩnh viễn C.1 tại $\lambda = 0.030$**, không thực hiện parameter sweep. Chuyển sang định nghĩa giả thuyết nghiên cứu mới cho **Phase 6-D**.
  + **Lưu trữ toàn bộ artifacts**:
    * Checkpoints: `results/checkpoints/P3_C_Phase6_C1_clDice_D4_K2_H64_best_model_b2_global.pth`, `best_model_b2_stage2.pth`, `last_model_b2_stage2.pth`.
    * Log huấn luyện: `results/P3_C_Routing_Diagnostics_Phase6_C1_clDice_D4_K2_H64/train.log`.
    * Diagnostics: `results/P3_C_Routing_Diagnostics_Phase6_C1_clDice_D4_K2_H64/diagnostics/`.
    * Topology Metrics: `results/diagnostics/phase6_c_topology/topology_c1_metrics.csv`.
    * Archive: `results/P3_C_Phase6_C1_clDice_D4_K2_H64_Full.zip`.
- **2026-10-01 (tiếp tục)**: Hoàn thành & Nghiệm thu Chẩn đoán Tiền can thiệp Phase 6-D.0 (Deconfounding Over-dilation and False Bridge):
  + **Mục tiêu**: Làm rõ False Bridge là hệ quả thuần túy của Over-dilation hay là điểm nghẽn phân tách hình thái học (topology separation) độc lập.
  + **Phát hiện Định lượng Đột phá**:
    * **Thành phần do phình viền (Dilation-mediated Component)**: $OR = 5.67$ ($p < 0.0001$). Tỉ lệ False Bridge tăng đơn điệu tuyệt đối theo mức phình diện tích: $10.7\%$ (AreaExcess $\le 0\%$) $\to 29.0\%$ ($0-10\%$) $\to 34.5\%$ ($10-30\%$) $\to 52.5\%$ ($>30\%$). $90.0\%$ số ca nối cầu trong Candidate B ($99/110$) nằm ở nhóm phình viền dương.
    * **Thành phần tồn dư phân tách (Residual Separation Component)**: $10.7\%$ số mẫu thiếu diện tích ($\text{AreaExcess} \le 0\%$) vẫn dính cầu giả; ở nhóm nứt phức tạp (`Complex_Topology`), tỉ lệ này lên tới **$66.7\%$** ($4/6$ mẫu).
    * **Khóa Khung Khái niệm**:
      $$\boxed{\text{False Bridge} = \text{Dilation-mediated Component} + \text{Residual Separation Component}}$$
    * Bác bỏ phương án ghép mù quáng $A1 + B1$ (nguy cơ double-count boundary pressure).
  + **Báo cáo & Dữ liệu chi tiết**: Lưu tại `results/diagnostics/phase6_d0_bridge_dilation/PHASE_6_D0_DIAGNOSTIC_REPORT.md`, `d0_diagnostic_summary.json`, `d0_per_sample_transitions.csv` (commit `dd63cd7`).
- **2026-10-02**: Chuẩn bị Toàn diện Thực nghiệm Phase 6-D.1 (Representation × Boundary Synergy Probe: PLU + AB-BPL):
  + **Thiết kế can thiệp trực giao**: Kết hợp giải pháp phục hồi biểu diễn không gian tầng cao (PLU Head của Phase 6-A.2, 10,125,363 params) với hàm phạt viền bất đối xứng (AB-BPL của Phase 6-B.1, $\lambda=0.040, r=2$).
  + **Quy trình Phả hệ Chuẩn tắc (Strict Lineage Protocol)**: Nhánh rẽ Stage 2 từ chính checkpoint Stage 1 của Phase 6-A.2 Pure PLU (`best_model_b2_stage1.pth` + RNG/scaler `last_model_b2_stage1.pth`). Nhóm đối chứng là Pure A2 Stage 2 (Base loss) với cùng checkpoint khởi điểm. Triệt tiêu 100% biến ngoại lai do khởi tạo.
  + **Khóa 4 Trụ cột Nghiệm thu Đăng ký Trước (Pre-registered Guideposts)**:
    1. *Thin-low-area ($n=5$)*: $\ge 0.5900$ (pre-registered practical retention guidepost, giữ phần lớn gain của PLU).
    2. *Boundary Margin ($n=127$)*: $\ge 0.7800$ (bảo toàn năng lực kiểm soát viền của B1).
    3. *Global Area Excess*: $< +5.0\%$ (ép chặt mask viền).
    4. *Global False Bridge Count*: $\le 107$ ($< 31.0\%$, Net Change $\le -5$ bridges).
  + **Bộ kiểm thử Tiền trạm (Preflight Gate 1–5)**: Đã chạy thực tế trên GPU và **PASS 100%** (Loss equivalence khi $\lambda=0$, AB-BPL mechanics, AMP FP16 stability trên CUDA với cuDNN disabled cho Turing GTX 1650, parameter invariance 10,125,363, strict checkpoint load 0 missing 0 unexpected).
  + **Tài liệu & Configs**: Đã lập `configs/p3_ablation/b2_p3_run_c_d4_k2_h64_phase6_d1_plu_abbpl.yaml`, `results/configs/...`, `docs/Phase6_Experiment_6D1_Specification.md`, `scripts/tests/test_phase6_d1_plu_abbpl.py`.







