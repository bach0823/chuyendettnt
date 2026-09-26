---
name: sage-lite-colab
description: Invariants and best practices for creating and managing Colab Notebooks in the SAGE-Lite project (Driver Pattern, VRAM Sandbox).
---

# SAGE-Lite Colab Notebook Guidelines

Skill này định nghĩa các nguyên tắc bất biến (invariants) và best practices chung để viết Google Colab Notebooks cho dự án SAGE-Lite. Các nguyên tắc này đảm bảo tính tái lập (reproducibility), giữ code sạch và mô phỏng thực tế mà không bị ràng buộc cứng vào một experiment cụ thể.

## 1. Architectural Invariants
- **Notebook = Driver:** Notebook chỉ đóng vai trò "người lái" (mount drive, setup environment, chạy VRAM probe, gọi entry point training/evaluation). Tuyệt đối không viết logic model, kiến trúc layer hay hàm training trực tiếp vào notebook. Mọi implementation phải nằm trong source code của repo.
- **Experiment-Specific Constraints:** KHÔNG hard-code các giả định của một experiment (ví dụ: loss function, input shape, số blocks, batch size candidates, số iteration) làm template bắt buộc. Mọi thông số (candidate batch size, loss, model, input shape, etc.) phải được lấy theo **experiment hiện tại**. (Ví dụ: B0 là Baseline, nhưng R1, R2 sẽ có cấu trúc và parameter khác).
- **Target Hardware Invariant (Tesla T4 Primary):** Google Colab Tesla T4 (16GB VRAM) là môi trường phần cứng chuẩn duy nhất để đo đạc: OOM probe, batch-size search, throughput benchmark, và official training. Tuyệt đối **KHÔNG** chạy các tác vụ đo đạc tải CUDA nặng trên GPU local (GTX 1650 4GB). Máy local chỉ phục vụ code editing, static audit, git sync, hoặc test cú pháp/mock CPU.
- **Colab Cell Syntax Invariant:** Khi cung cấp lệnh để chạy trong Colab Notebook, **BẮT BUỘC** định dạng sẵn 100% cú pháp cell notebook: dùng `%cd` cho chuyển thư mục và `!` cho mọi lệnh shell (`!git`, `!python`, `!pip`). Tuyệt đối không đưa bash thô thiếu `!` và `%`.

## 2. Environment Setup & Preflight (Tập trung & Tái lập)
- **All-in-One Preflight Verification Cell Invariant:** Luôn chuẩn bị sẵn **MỘT CELL DUY NHẤT** tích hợp toàn bộ các bước tiền kiểm (Preflight) để User chỉ cần copy-paste bấm chạy 1 lần duy nhất trước khi chạy tác vụ chính:
  1. Đồng bộ Repo (`%cd /content/SAGE_LITE`, `!git fetch origin <branch>`, `!git checkout <branch>`, `!git pull origin <branch>`)
  2. Kiểm tra Git state (`!git rev-parse HEAD`, `!git status`)
  3. Kiểm tra Protocol / Unit Tests (`!python scripts/tests/test_two_stage_protocol.py` hoặc test suite tương ứng)
  4. Kiểm tra Dataset Sanity (`!python scripts/dataset_sanity_check.py ...`)
  *Tránh chia nhỏ lẻ tẻ thành nhiều cell gây tốn thao tác và dễ bỏ sót.*
- **Single Setup Cell:** Gom gọn quá trình Mount Google Drive, Clone Repo, Cài đặt Dependencies vào một Code Cell duy nhất.
- **Public Repository:** Vì repo SAGE_LITE là PUBLIC, clone trực tiếp repo. **TUYỆT ĐỐI KHÔNG** dùng `userdata`, Personal Access Tokens (PAT), hoặc URL dạng `oauth2` để clone (tránh gây lỗi `SecretNotFoundError`).
- **Standard Clone Command:**
  ```python
  !git clone https://github.com/bach0823/SAGE_LITE.git sage_lite
  ```

## 3. VRAM Probing Sandbox (Đo lường độc lập)
Khi experiment cần đo/chọn batch size bằng VRAM probe, cần phải Sandbox quá trình này để memory cache của PyTorch hoặc các momentum buffers của optimizer không cộng dồn chéo giữa các batch size candidates.
- **Mô phỏng thực tế:** Khi mục tiêu là đo training VRAM, vòng lặp test mỗi batch size phải bao gồm Forward $\to$ Backward $\to$ `optimizer.step()` (không áp đặt nếu chỉ đo inference/forward).
- **Sandbox Cleanup Rules:**
  1. Sao lưu trạng thái chuẩn của model lên CPU trước vòng lặp: `init_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}`.
  2. Tại mỗi vòng lặp `bs`, khôi phục lại weights của model và `model.zero_grad(set_to_none=True)`.
  3. Khởi tạo **riêng biệt** một Optimizer và Scaler mới bên trong vòng lặp.
  4. Sử dụng khối `finally` để xóa an toàn (`del` kèm check `locals()`) các tensors, optimizer, và scaler cục bộ.
  5. Gọi `gc.collect()` và `torch.cuda.empty_cache()` để trả lại VRAM trống hoàn toàn cho hệ thống.

<details>
<summary><b>Example Implementation (Reference Only - Lấy B0 làm ví dụ)</b></summary>
<em>Lưu ý: Đây CHỈ là ví dụ tham khảo. Các thông số input, loss, model phải được thay đổi dựa theo experiment thực tế.</em>

```python
import gc
import torch

def probe_batch_size(candidate_batches=[4, 8, 12, 16, 20]): # Phụ thuộc experiment
    results = []
    # Sao lưu trạng thái gốc
    init_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    for bs in candidate_batches:
        # Restore trạng thái nguyên bản
        model.load_state_dict(init_model_state)
        model.zero_grad(set_to_none=True)
        
        # Optimizer & Scaler ĐỘC LẬP
        probe_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        probe_scaler = torch.cuda.amp.GradScaler()
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            # Shape tùy thuộc experiment
            x_test = torch.randn(bs, 3, 448, 448, device=device)
            gt_test = (torch.rand(bs, 1, 448, 448, device=device) > 0.8).float()
            
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits_test = model(x_test)
                # Loss tùy thuộc experiment
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits_test, gt_test)
            
            probe_scaler.scale(loss).backward()
            probe_scaler.step(probe_optimizer)
            probe_scaler.update()
            
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
            results.append((bs, peak_vram, "SUCCESS"))
        except torch.cuda.OutOfMemoryError:
            results.append((bs, None, "OOM"))
            break
        finally:
            # Sandbox Cleanup bắt buộc (có check locals an toàn)
            if 'x_test' in locals(): del x_test
            if 'gt_test' in locals(): del gt_test
            if 'logits_test' in locals(): del logits_test
            if 'loss' in locals(): del loss
            if 'probe_optimizer' in locals(): del probe_optimizer
            if 'probe_scaler' in locals(): del probe_scaler
            
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()

    model.load_state_dict(init_model_state)
    return results
```
</details>

## 4. Automation & Workspace Hygiene
- **Sửa Notebook Programmatically:** Nếu cần viết script Python để tạo hoặc sửa `.ipynb` tự động, **BẮT BUỘC** dùng thư viện `json` (parse thành dict, sửa field `source`, rồi dump lại). Tuyệt đối **KHÔNG** dùng regex hay string replacement thuần túy trên toàn bộ văn bản file vì dễ sinh lỗi cú pháp JSON nghiêm trọng.
- **Vị trí lưu Scratch Scripts:** Mọi script phụ trợ, script test một lần, hay script update file phải được đặt trong thư mục `scripts/scratch/` hoặc `.gemini/scratch/`. TUYỆT ĐỐI không để rơi vãi file tạm ở thư mục root của dự án.

## 5. Consolidated Output & Reporting
- **Tổng hợp kết quả (Consolidated Output):** Khi notebook thực hiện nhiều bước kiểm tra, nghiệm thu (Shape, AMP, VRAM, Smoke Test) hoặc benchmark, nên lưu các kết quả quan trọng vào các biến toàn cục (ví dụ: eport_dict = {}). Ở cuối notebook, hãy thêm một cell chuyên dụng để in ra báo cáo tổng hợp. Việc này giúp người dùng dễ dàng nắm bắt bức tranh toàn cảnh và copy/paste kết quả cuối cùng một cách nhanh nhất.

## 6. Model Results Preservation & Repo Sync Protocol (Quy trình Lưu trữ & Đồng bộ Kết quả Mô hình)

Khi người dùng yêu cầu "lưu kết quả / lưu kq / lưu run X", Agent **BẮT BUỘC** tuân thủ quy trình chuẩn hóa sau:

1. **Lưu đầy đủ giá trị qua tất cả Epoch (Full Epoch Metrics Invariant):**
   - Không cần copy nguyên văn terminal log rác (tránh thanh tiến trình tqdm làm phình file).
   - **BẮT BUỘC phải lưu trọn vẹn toàn bộ các giá trị của MỌI epoch** (Train Loss, Train LB, Train Dice, Val Loss, Val Dice, LR từng nhóm tham số, Gamma $S0/S1$, thời gian/tốc độ $s/it$, cờ `is_stage_best` và `is_global_best`) vào `results/...metrics.json` và `results/...metrics.md`.
   - **Tuyệt đối KHÔNG chỉ lấy riêng giá trị đỉnh (peak) hay các mốc nổi bật**. Phải giữ toàn bộ chuỗi số liệu từ Epoch 1 đến Epoch cuối để phục vụ vẽ biểu đồ và phân tích đường cong học tập.
2. **Lưu Config tương ứng:** Lưu bản sao config vào `results/configs/<config_name>.yaml`.
3. **Bắt buộc Push lên Repo chính (`SpecialSubjectTTNT`):**
   - Toàn bộ kết quả phải nằm trong thư mục `results/` của repository chính (`SpecialSubjectTTNT/results/`), TUYỆT ĐỐI không lưu vào subrepo `SAGE_LITE`.
   - Ngay sau khi ghi nhận và cập nhật xong các file kết quả (`results/*.md`, `results/*.json`, `results/configs/*.yaml`, `results/figures/*.png`), Agent **bắt buộc phải `git add`, `git commit` và `git push` trực tiếp lên repository `SpecialSubjectTTNT`**.
4. **Tận dụng Công cụ Phân tích Sẵn có (Zero Reinventing the Wheel):**
   - Khi cần phân tích sâu hay trực quan hóa (routing diagnostics, error analysis, visual gallery), **Agent PHẢI ưu tiên sử dụng các công cụ chuẩn có sẵn trong repo** (`tools/analyze_routing.py`, `tools/run_p3_c_error_analysis.py`).
   - Cung cấp sẵn Cell Colab hoàn chỉnh 100% cú pháp notebook (`%cd`, `!python`) với đúng đường dẫn checkpoint, config, output dir trên Google Drive và lệnh nén zip kèm checkpoint `.pth` để người dùng chỉ việc copy-paste chạy 1 lần.

## 7. Experiment Results Log (BẮT BUỘC ĐỌC & CẬP NHẬT)

**[CRITICAL]** Tất cả kết quả thực nghiệm được lưu vĩnh viễn trong repository chính tại:
```
results/
```
Agent mới bắt đầu conversation: **ĐỌC CÁC FILE NÀY TRƯỚC** thay vì hỏi lại người dùng về kết quả cũ.
Sau khi hoàn thành một benchmark/training run mới: **APPEND kết quả vào file tương ứng, cập nhật JSON metrics và commit/push vào git repo SpecialSubjectTTNT**.

---

### B0 Baseline — Pure ConvNeXtV2-Femto + U-Net (Crack500) [COMPLETE]
*File đầy đủ:* `sage_lite/docs/experiments/b0_baseline_log.md`

**Architecture:**
- 7,376,593 params, 0 ViT blocks
- Input/Output: 448×448 → 448×448
- Branch: `crack500-audit`, Commit: `584ca21`

**T4 Throughput (Synthetic, BCE-only):**
| BS | Throughput | Peak VRAM |
|---:|---:|---:|
| **16 (chọn)** | 66.7 img/s | 2.68 GB |
| 28 (max tested) | 63.4 img/s | 4.60 GB |

**Full Training - Run-2 Official Protocol (Smart Crop 85/15 + Tiling Val):**
- Config: `BS=16`, `img_size=448`, `LR=1e-4`, Differential LR (backbone ×0.1)
- Loss: `BCE + 1.5×SoftDice`
- Stage 1 best: Val Dice `0.7207` @ epoch 12/15
- Stage 2 best (global): Val Dice **`0.7381`** @ epoch 6/15 (Early Stopped at epoch 12)
- Total Budget: 27/30 epochs (ES triggered)

**Official Metrics (Non-overlapping Tiling, Per-sample mean, FP32):**
| Split | Official Precision | Official Recall | Official Dice |
|:---|---:|---:|---:|
| Val | — | — | 0.7381 (from train loop) |
| Test | — | — | Đang chờ test |

*Kết luận:* Ngưỡng trần thực tế (Ceiling) của B0 với strict evaluation protocol là 73.81%. Mọi metrics của Run-1 (0.8082) đã bị loại bỏ vì inflated (dùng batch average và naive resize). 
**Candidate 2 (Weighted BCE): STANDBY** — Chờ kết quả precision/recall cụ thể từ tập Test mới quyết định.

---

### B2 Phase 0: Runtime & Hardware Characterization (Crack500) [COMPLETE]
*File đầy đủ:* `results/B2_Crack500_Phase0_Characterization.md`
- **Hardware:** Tesla T4 (14.56 GB usable VRAM, 2 vCPUs)
- **Max Safe BS:** Cả 3 depth D12, D6, D4 đều **OOM ở BS 16**. **BS 12 là trần vật lý** (D12 chiếm 14.73 GB, còn dư 178 MB buffer). Safe BS đề xuất = 12 (kèm `empty_cache()`) hoặc 8 (dư > 5 GB).
- **Worker Scaling:** DataLoader wait chỉ 0.3 - 0.8 ms ở workers=2/4. `num_workers = 2` là tối ưu nhất trên Colab (tránh IPC overhead).
- **ViT Depth Compute Scaling:** D4 (3.09 img/s, 613s/epoch) nhanh hơn D12 (2.26 img/s, 837s/epoch) ~26.8%.
- **Epoch Budget:** 30 epochs cho D12 mất ~7.0h (nguy cơ timeout cao). Đề xuất $N_{\text{total}} = 20\text{ epochs}$ (~4.7h D12, ~3.9h D6, ~3.4h D4) kèm Early Stopping `patience=5`.

---

### B2 Phase 7: Real-Data P3 Launch Preflight (Run A, B, C) [COMPLETE - GATED]
*File đầy đủ:* `results/B2_Crack500_P3_Launch_Preflight.md`
- **Config:** Depth = 12, BS = 12, Workers = 2, Real Crack500, Commit `bdb23f1`
- **Preflight Checks:** 24/24 invariant checks **PASS 100%** trên cả 3 Runs: Run A (Identity), Run B (Generic DW), Run C (ASDW).
- **Bitwise PE28 Invariance:** `Run A.pe28_fixed == Run B.pe28_fixed == Run C.pe28_fixed` (max diff = 0.0) -> PASS.
- **P3 Parameters:** Run A = 0 params; Run B & C = 10 params có active gradient hữu hạn.
- **Peak VRAM on T4:** Run A = 9.46 GB, Run B = 9.67 GB, Run C = 9.78 GB (an toàn dưới 10 GB).
- **Status:** **GATED** (Preflight passed 100%, chờ Locked Base checkpoint chính thức từ Phase 1).

