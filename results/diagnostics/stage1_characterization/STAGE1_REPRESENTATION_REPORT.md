# Báo Cáo Chẩn Đoán Biểu Diễn Stage-1 Main Path ($F_{\text{preSAGE}}^{56}$)

**Thời gian thực hiện:** 03/10/2026  
**Phương pháp:** Zero-training Representation Geometry & Semantic Axis Projection  
**Target Model:** SAGE-Lite Candidate B Baseline (`B2ConvNeXtViTUNet`, Setting A, Tile 448)  
**Cohort nghiên cứu:** Toàn bộ 43 wider-gap false bridge events ($D_{\text{gap}} > 5.0\text{ px}$) từ Phase 6 Diagnostic D  
**File đo đạc chi tiết:** `stage1_representation_measurements_43events.csv`  
**File tổng hợp thống kê:** `stage1_representation_summary.csv` & `stage1_representation_summary.json`

---

## 1. Dữ Liệu Thực Nghiệm (Observed Facts)

### 1.1. Bảng tổng hợp số liệu phân tầng (Stratified Metrics Summary)

Đại lượng đo lường:
- Vector dịch chuyển so với nền cục bộ: $\Delta v_{\text{corr}} = \bar{v}_{\text{corr}} - \bar{v}_{\text{bg}}$, $\Delta v_{\text{crack}} = \bar{v}_{\text{crack}} - \bar{v}_{\text{bg}}$.
- Tỉ số độ lớn dịch chuyển: $M = \frac{\|\Delta v_{\text{corr}}\|}{\|\Delta v_{\text{crack}}\|}$.
- Độ tương đồng hướng tâm (Centered Cosine / Directional Alignment): $\cos(\theta) = \frac{\langle \Delta v_{\text{corr}}, \Delta v_{\text{crack}} \rangle}{\|\Delta v_{\text{corr}}\| \|\Delta v_{\text{crack}}\|}$.
- Hệ số chiếu ngữ nghĩa chuẩn hóa (Normalized Semantic Crack Projection): $\alpha = \frac{\langle \Delta v_{\text{corr}}, \Delta v_{\text{crack}} \rangle}{\|\Delta v_{\text{crack}}\|^2} = M \cos(\theta)$.
- Tương quan kích hoạt theo kênh (Channel-wise Pearson Correlation): $r_{\text{channel}} = \text{Corr}(\Delta v_{\text{corr}}, \Delta v_{\text{crack}})$.

| Phân tầng (Cohort) | $N$ | $\bar{D}_{\text{gap}}$ (px) | $S_1\ \alpha$ (Full) | $S_1\ \cos(\theta)$ | $S_1\ M$ (Mag) | $S_1\ r_{\text{ch}}$ | $S_1\ \alpha$ (Interior) | $S_1\ \alpha$ (Endpoint) | $S_1\ \alpha$ (S0-off) | $S_0\ \alpha$ (preSAGE) | $S_0\ r_{\text{ch}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Group A ($5 < D \le 8\text{ px}$)** | 11 | 7.1 px | **0.2713** | 0.3033 | 0.9193 | 0.3049 | 0.2725 | 0.2764 | 0.2724 | 0.4641 | 0.4800 |
| **Group B ($D > 8\text{ px}$)** | 32 | 23.2 px | **0.2136** | 0.2853 | 0.7821 | 0.2865 | 0.2112 | 0.2218 | 0.2129 | 0.2230 | 0.3911 |
| **Overall (Toàn bộ cohort)** | **43** | **19.1 px** | **0.2284** | **0.2899** | **0.8172** | **0.2912** | **0.2269** | **0.2358** | **0.2281** | **0.2847** | **0.4139** |

*Ghi chú trung vị (Median):*
- Overall $S_1\ \alpha$: $0.2126$ (mean: $0.2284$).
- Group B $S_1\ \alpha$: $0.1985$ (mean: $0.2136$).
- Interior corridor $\alpha_{\text{int}}$: $0.2126$ (mean: $0.2269$).

---

### 1.2. Phân tích chi tiết từng khía cạnh đo đạc

1. **Phân tích hình học vector (Geometric Decomposition):**
   - Tỉ số độ lớn tương đối $M = \frac{\|\Delta v_{\text{corr}}\|}{\|\Delta v_{\text{crack}}\|}$ đạt mức trung bình **0.8172** (Group A: 0.9193, Group B: 0.7821). Điều này chứng minh corridor có năng lượng kích hoạt tương phản (contrast activation energy) rất đáng kể so với nền (bằng ~82% cường độ của vết nứt thật).
   - Tuy nhiên, độ thẳng hàng định hướng (Centered Cosine) $\cos(\theta)$ chỉ đạt **0.2899** (tương ứng với góc lệch không gian đặc trưng $\theta = \arccos(0.29) \approx 73.1^\circ$).
   - Do đó, phép chiếu chuẩn hóa lên trục nứt $\alpha = M \cos(\theta) = 0.2284$ bị chặn ở mức thấp (~0.23), không vượt quá 0.30.

2. **Tương quan kênh kích hoạt ($r_{\text{channel}}$):**
   - Tại Stage 1 (96 kênh): Hệ số tương quan Pearson giữa profile kích hoạt kênh của corridor và vết nứt thật đạt trung bình **$r = 0.2912$** (Group A: 0.3049, Group B: 0.2865).
   - Mức tương quan $r \approx 0.29$ khẳng định corridor **không kích hoạt chọn lọc các kênh đặc trưng vết nứt (crack-specific signature channels)**, mà kích hoạt một tập hợp kênh khác biệt, chỉ có độ giao thoa yếu đến trung bình.

3. **Phân tách không gian: Interior Gap vs Endpoint Transition:**
   - Tại Group B ($D_{\text{gap}} > 8\text{ px}$):
     - $\alpha_{\text{interior}} = 0.2112$ (loại trừ hoàn toàn vùng 2px quanh endpoints).
     - $\alpha_{\text{endpoint}} = 0.2218$.
   - Giá trị $\alpha$ tại lõi khoảng trống (interior) gần như bằng phẳng so với đầu mút vết nứt ($\Delta \alpha = 0.0106$). Hiện tượng biểu diễn tại Stage 1 là một gờ liên tục trải rộng suốt chiều dài cầu nối, không phải một đột biến cô lập tại biên.

4. **Kiểm định đóng góp của Stage-0 SAGE (Counterfactual Test):**
   - Khi tắt hoàn toàn Stage-0 SAGE trong main path của Stage 1 ($F_{\text{preSAGE, S0-off}}^1$):
     - $\alpha_{\text{S0-off}} = 0.2281$ so với $\alpha_{\text{baseline}} = 0.2284$ ($\Delta \alpha = -0.0003$).
     - $r_{\text{channel, S0-off}} = 0.2899$ so với $r_{\text{channel, baseline}} = 0.2912$ ($\Delta r = -0.0013$).
   - Tại Stage 0:
     - $\alpha_{\text{preSAGE}}^0 = 0.2847$ so với $\alpha_{\text{postSAGE}}^0 = 0.2848$ ($\Delta \alpha = +0.0001$).
   - Kết quả này chứng minh: Phần dư của SAGE tại Stage 0 ($0.1 E_0$) hoàn toàn không làm thay đổi biểu diễn của corridor ($\Delta \le 0.0003$).

---

## 2. Suy Luận Có Cơ Sở (Supported Inferences)

Dựa trên các bằng chứng thực nghiệm thu được:

1. **Bản chất biểu diễn của Stage-1 Main Path là "Generic Structural Continuity / Line Feature", KHÔNG PHẢI "Semantic True Crack":**
   - Nếu ConvNeXt Stage-1 main path thực sự nhận diện corridor là vết nứt ngữ nghĩa, ta phải quan sát thấy: $\cos(\theta) \to 1.0$, $r_{\text{channel}} \to 1.0$, và $\alpha \to 1.0$.
   - Thực tế đo được: $\cos(\theta) = 0.29$, $r_{\text{channel}} = 0.29$, và $\alpha = 0.23$.
   - Điều này chứng minh rằng: Vector đặc trưng của corridor không nằm trên đa tạp vết nứt (crack manifold) mà nằm trên một trục trực giao phần lớn ($\theta \approx 73^\circ$), đại diện cho tính liên tục cấu trúc chung (generic continuity, gờ cạnh mờ, ranh giới độ sáng giữa 2 đầu vết nứt).

2. **Tại sao Decoder Block 1 vẫn bị kích hoạt thành False Bridge?**
   - Kết hợp với thí nghiệm Causal Ablation ở bước trước ($\Delta_S = 0.6427 \gg \Delta_U = 0.2307$):
   - Decoder Block 1 chỉ cần một tín hiệu cấu trúc liên tục yếu ($\alpha \approx 0.23, M \approx 0.82$) từ Stage-1 skip branch kết hợp với độ mở rộng không gian sau Conv 3x3 để khuếch đại (amplify) và xác nhận thành logit dương ($z_{\text{med}} > 0$).
   - Nói cách khác: Decoder Block 1 thiếu cơ chế phản biện ngữ nghĩa (semantic rejection) đối với các đặc trưng liên tục mức thấp (low-level continuity) khi chúng có độ lớn năng lượng $M > 0.8$.

3. **Nguồn gốc phát sinh biểu diễn (Genesis Origin) nằm ở Upstream ConvNeXt (Stem & Stage 0):**
   - Cả Stage-1 SAGE residual ($0.1E_1$) và Stage-0 SAGE residual ($0.1E_0$) đều cho tác động bằng $0$ ($\Delta_{\text{SAGE}} \approx 0.000$).
   - Biểu diễn liên tục này được hình thành trực tiếp từ chuỗi tích chập của ConvNeXt: `Stem (4x4 stride 4)` $\to$ `Stage 0 (7x7 depthwise stride 1)` $\to$ `Downsample 2x2 stride 2` $\to$ `Stage 1`.

---

## 3. Các Giả Thuyết Bị Bác Bỏ (Unsupported Hypotheses)

Dưới ánh sáng của dữ liệu thực nghiệm, các giả thuyết sau chính thức bị bác bỏ:

1. ❌ **Giả thuyết "Corridor tại Stage 1 là một vết nứt giả mang đầy đủ ngữ nghĩa (Hallucinated Semantic Crack)":**
   - *Bác bỏ:* $r_{\text{channel}} = 0.29$ và $\alpha = 0.23$ chứng minh các kênh chuyên biệt cho vết nứt không được kích hoạt tương ứng trên corridor. Corridor không bị gán nhãn ngữ nghĩa là crack ở backbone, mà chỉ mang đặc trưng đường nét/tính liên tục.

2. ❌ **Giả thuyết "SAGE tại Stage 0 đóng vai trò điều hướng hoặc tạo ra bridge representation":**
   - *Bác bỏ:* Phép đo counterfactual Stage-0 SAGE-off cho thấy $\Delta \alpha = -0.0003$ và $\Delta r = -0.0013$. SAGE tại Stage 0 hoàn toàn không tham gia vào việc tạo ra hay củng cố vector biểu diễn cầu nối.

3. ❌ **Giả thuyết "Độ tương đồng Cosine thô ~0.99 chứng minh corridor giống hệt crack":**
   - *Bác bỏ:* Trong không gian đặc trưng chưa trừ nền (uncentered), cosine giữa background và crack thật đã là $0.9912$. Đây là hiện tượng "Common-Mode DC Offset" điển hình trong deep feature maps, không phản ánh khoảng cách ngữ nghĩa thực tế. Chỉ có centered cosine và semantic projection mới phản ánh đúng sự thật hình học.

---

## 4. Định Hướng Cho Bước Tiếp Theo (Implications for Next Step)

Từ chuỗi chẩn đoán hoàn chỉnh:
- Setting A vs Setting B: Bridge không phụ thuộc vào tile boundary (98.2% persistent).
- Decoder Localization: 100% committed tại Decoder Head 112, 70% bắt nguồn từ Decoder 28-56.
- 2x2 Factorial Pathway: Stage-1 Skip là dominant pathway ($\Delta_S = 0.6427 \gg \Delta_U = 0.2307$).
- SAGE Provenance: Cả Stage-0 và Stage-1 SAGE residuals đều vô can ($\Delta_{\text{SAGE}} \approx 0$).
- Representation Characterization: Corridor mang generic structural continuity ($\alpha \approx 0.23$, $\cos \theta \approx 0.29$), nhưng năng lượng tương đối cao ($M \approx 0.82$).

**Kết luận bản chất:**
Hiện tượng false bridge là hệ quả của việc ConvNeXt backbone (bắt đầu từ Stem non-overlapping 4x4 stride 4) làm mất thông tin biên sắc nét của các khoảng trống hẹp/vừa, tạo ra một vệt liên tục cấu trúc (structural continuity ridge). Tín hiệu này đi qua Stage-1 skip và bị Decoder Block 1 khuếch đại thành false bridge.

**Hướng đi có cơ sở nhất tiếp theo:**
Tập trung vào khâu **Upstream Genesis (Stem Information Preservation & Anti-Aliased Downsampling)** theo đề xuất chẩn đoán U0 (đã được Auditor thẩm định tài liệu độc lập) nhằm bảo toàn khoảng cách không gian (gap separation) ngay từ tầng thấp nhất của mạng, ngăn chặn sự hình thành của vệt liên tục tại cội nguồn.
