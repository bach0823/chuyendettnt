---
name: sage-lite-colab
description: Invariants and best practices for creating and managing Colab Notebooks in the SAGE-Lite project (Driver Pattern, VRAM Sandbox).
---

# SAGE-Lite Colab Notebook Guidelines

Skill này định nghĩa các nguyên tắc bất biến (invariants) và best practices chung để viết Google Colab Notebooks cho dự án SAGE-Lite. Các nguyên tắc này đảm bảo tính tái lập (reproducibility), giữ code sạch và mô phỏng thực tế mà không bị ràng buộc cứng vào một experiment cụ thể.

## 1. Architectural Invariants
- **Notebook = Driver (No-Drive Preference):** Notebook chỉ đóng vai trò "người lái" (setup environment, chạy VRAM probe, gọi entry point training/evaluation, nén và tải artifacts về máy tính cá nhân). Khuyến khích chạy thuần túy trên ổ đĩa ảo `/content/` của Colab và bỏ mount Google Drive để tránh phiền toái popup cấp quyền, độ trễ FUSE I/O và nghẽn dung lượng 15GB. Tuyệt đối không viết logic model, kiến trúc layer hay hàm training trực tiếp vào notebook. Mọi implementation phải nằm trong source code của repo.
- **Experiment-Specific Constraints:** KHÔNG hard-code các giả định của một experiment (ví dụ: loss function, input shape, số blocks, batch size candidates, số iteration) làm template bắt buộc. Mọi thông số (candidate batch size, loss, model, input shape, etc.) phải được lấy theo **experiment hiện tại**. (Ví dụ: B0 là Baseline, nhưng R1, R2 sẽ có cấu trúc và parameter khác).
- **Target Hardware Invariant (Tesla T4 Primary):** Google Colab Tesla T4 (16GB VRAM) là môi trường phần cứng chuẩn duy nhất để đo đạc: OOM probe, batch-size search, throughput benchmark, và official training. Tuyệt đối **KHÔNG** chạy các tác vụ đo đạc tải CUDA nặng trên GPU local (GTX 1650 4GB). Máy local chỉ phục vụ code editing, static audit, git sync, hoặc test cú pháp/mock CPU.
- **Colab Cell Syntax Invariant:** Khi cung cấp lệnh để chạy trong Colab Notebook, **BẮT BUỘC** định dạng sẵn 100% cú pháp cell notebook: dùng `%cd` cho chuyển thư mục và `!` cho mọi lệnh shell (`!git`, `!python`, `!pip`). Tuyệt đối không đưa bash thô thiếu `!` và `%`.
- **Checkpoint Archival Invariant (Best + Last Models):** Khi chuẩn bị cell export/download hoặc lưu trữ artifacts sau huấn luyện, **BẮT BUỘC** lưu trữ cả **Best Model** (`best_model_*.pth`) lẫn **Last Model** (`last_model_*.pth`). Tuyệt đối không chỉ lọc riêng `best_model`. Cả hai checkpoint này đều cần thiết: `best_model` để đánh giá benchmark đỉnh cao, và `last_model` để phục vụ audit quỹ đạo, kiểm tra over-fitting cuối lịch trình, hoặc resume tiếp nối.
- **All-in-One Post-Training Cell Invariant (Diagnose + Zip + Download):** Khi chuẩn bị thao tác hậu huấn luyện (post-training evaluation & archival), **BẮT BUỘC** gộp toàn bộ các bước:
  1. Routing Diagnostics (`tools/analyze_routing.py`)
  2. Error Analysis & Morphology (`tools/run_p3_c_error_analysis.py`)
  3. Đóng gói Zip toàn bộ thư mục chạy (bảo tồn Best + Last checkpoints + Logs + Diagnostics figures)
  4. Tự động kích hoạt tải file về máy qua `google.colab.files.download(...)`
  vào **MỘT CELL DUY NHẤT** (All-in-One Cell). Tuyệt đối **KHÔNG** tách nhỏ thành nhiều cell lẻ tẻ để người dùng chỉ cần copy-paste bấm chạy 1 lần duy nhất.
- **Zero-Drive & Minimal Assumptions Invariant:**
  - Tuyệt đối **KHÔNG** yêu cầu hoặc ép buộc người dùng mount Google Drive (`drive.mount('/content/drive')`), trừ khi người dùng chủ động yêu cầu.
  - Hiểu rõ cơ chế hệ thống tệp: `os.makedirs('/content/drive/MyDrive/...', exist_ok=True)` trong mã nguồn Python hoàn toàn tạo được cây thư mục trên ổ đĩa ảo cục bộ `/content/` mà không cần Google Drive FUSE mount. Checkpoint lưu vào đây đọc/ghi hoàn toàn bình thường; không được suy diễn rằng thiếu mount Drive sẽ gây crash.
  - Tuyệt đối **KHÔNG** tự ý chèn các lệnh `!pip install ...` hàng loạt vào đoạn code của người dùng nếu người dùng đang dùng notebook/script chuẩn đã chạy thành công trước đó (tôn trọng môi trường có sẵn).
- **Ephemeral Runtime Invariant (No-Reset Assumption Strictly Forbidden):**
  - Tuyệt đối **KHÔNG BAO GIỜ giả định Colab runtime chưa reset**, hoặc giả định các file/checkpoint từ các cell/phiên làm việc trước đó vẫn còn tồn tại sẵn trên máy ảo hoặc Google Drive (`/content/drive/MyDrive/...`).
  - Mọi workflow Colab khi cung cấp cho người dùng **BẮT BUỘC phải tự chứa (self-contained)**:
    1. Checkpoint tổ tiên (ancestor checkpoint, ví dụ Stage-1 checkpoint) phải luôn đi kèm lệnh tải trực tiếp vào thư mục cục bộ `/content/checkpoints/` từ raw GitHub/release URL.
    2. Cờ `--checkpoint` và `--rng-checkpoint` trong lệnh huấn luyện `train_crack.py` **PHẢI trỏ trực tiếp và khớp 100%** vào file cục bộ vừa tải tại `/content/checkpoints/` (ví dụ `/content/checkpoints/best_model_b2_stage1.pth`), tuyệt đối KHÔNG trỏ sang Google Drive giả định.
- **Ultra-Minimal Driver Cell Invariant (Zero-Boilerplate Standard):**
  - **Notebook là Driver thuần túy, KHÔNG phải Test Suite hay Script Runner**:
    - **Quy tắc vàng**: Code python kiểm tra tồn tại / tính toàn vẹn (file existence, assertions, metadata check) nếu đã test ở máy local rồi thì lên Colab **TUYỆT ĐỐI KHÔNG CẦN GHI THÊM**.
    - Tuyệt đối **KHÔNG** viết hàng chục dòng Python inline để assert file tồn tại (`assert os.path.isfile`), tính toán checksum SHA-256 (`hashlib.sha256`), in các banner ASCII phân cách (`print("=" * 80)`), hay đọc/parse JSON thủ công trong cell.
    - Mọi logic kiểm tra, fail-fast và xác thực tính toàn vẹn PHẢI nằm gọn trong Repository scripts/tools (`scripts/train_crack.py`, `tools/analyze_routing.py`, etc.).
  - **Chuẩn hóa 3 Cells tối giản mẫu (Chỉ chạy lệnh, không thừa một dòng):**
    1. **Cell 1 (Setup, Data & Ancestor Checkpoint):** Thuần các lệnh shell trực tiếp (`!nvidia-smi`, `!rm -rf ...`, `!git clone ...`, `%cd ...`, `!python prepare_data/...`, `!mkdir -p ...`, `!wget ...`).
    2. **Cell 2 (Huấn luyện Stage):** Chỉ gồm `%cd /content/SAGE_LITE` và duy nhất 1 lệnh CLI gọi `!python scripts/train_crack.py --config ... [flags]`.
    3. **Cell 3 (Chẩn đoán, Nén & Tải về):** Chỉ gồm các lệnh CLI chẩn đoán (`!python tools/analyze_routing.py ...`, `!python tools/run_p3_c_error_analysis.py ...`), nén zip (`shutil.make_archive(...)`) và kích hoạt tải về (`files.download(...)`).

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
   - Cung cấp sẵn Cell Colab hoàn chỉnh 100% cú pháp notebook (`%cd`, `!python`) với đường dẫn output dir cục bộ (hoặc Drive nếu người dùng yêu cầu) và lệnh nén zip kèm checkpoint `.pth` để người dùng chỉ việc copy-paste chạy 1 lần. Lưu ý rằng `tools/run_p3_c_error_analysis.py` đã tự động nội suy đường dẫn `routing_statistics.json` (từ thư mục con `full_val/`), `data_root` (từ file YAML config) và `checkpoint` (từ `output_dir`), giúp đơn giản hóa dòng lệnh tối đa.

## 7. Faithful Stateful Training Resumption & Dual Checkpoint Architecture Protocol

Để đảm bảo quá trình tiếp nối huấn luyện (resumption) sau khi ngắt kết nối hoặc chạy mở rộng hội tụ bảo toàn tính liên tục của quỹ đạo học tập (faithful training-state resumption), hệ thống áp dụng kiến trúc Checkpoint Kép:

### 1. Phân biệt rõ hai loại Checkpoint
- **`best_model_*.pth` (Evaluation & Diagnostics Artifact):**
  - Chỉ lưu `model_state_dict` + metrics (`best_dice`, `best_loss`, metadata kiến trúc).
  - File nhẹ, dùng cho inference, evaluation trên tập Test và chạy routing diagnostics.
- **`last_model_*.pth` (Full Stateful Continuity Anchor):**
  - Lưu tại mỗi epoch (cả single-stage và two-stage).
  - **BẮT BUỘC chứa Full Training State**:
    ```python
    {
        "epoch": epoch,
        "stage": stage,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),  # AdamW moments
        "scheduler_state_dict": scheduler.state_dict(),  # Scheduler state
        "scaler_state_dict": scaler.state_dict(),        # AMP scale factor
        "val_dice": val_dice,
        "val_loss": val_loss,
        "best_dice": best_stage_dice,
        "best_loss": best_stage_loss,
        "epochs_no_improve": epochs_no_improve,          # Early stopping counter
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "dataloader_generator_state": g.get_state(),     # DataLoader shuffle permutation state
        "model_type": model_type,
    }
    ```

### 2. Nguyên tắc Tiếp nối (Faithful Resumption Protocol)
- **Tiếp nối Trạng thái Chuẩn (Faithful Full-State Resumption - Từ D10 trở đi):**
  - Khôi phục đầy đủ: Model weights, AdamW 1st/2nd moments, Cosine scheduler phase, AMP scaler, Early stopping counter và toàn bộ các bộ sinh số ngẫu nhiên (RNG PyTorch, CUDA, NumPy, Python, cùng DataLoader Generator state).
  - Đảm bảo tính liên tục của quỹ đạo tối ưu (trajectory continuity) và thứ tự shuffle batch giữa các epoch mà không bị reset seed hay warm-restart về LR cao.
  - *Lưu ý về tính tái lập:* Tính tái lập là chuẩn mực toán học ở cấp độ trạng thái (stateful continuity); các khác biệt nhỏ ở mức bit-for-bit qua các môi trường GPU/CUDA driver khác nhau (non-deterministic floating-point kernels) là đặc tính cố hữu của phần cứng, nhưng quỹ đạo huấn luyện được bảo toàn trung thực.
- **Tiếp nối Hậu nghiệm cho Checkpoint Cũ (Post-hoc Low-LR Audit - D6):**
  - Với checkpoint lịch sử chỉ có weights: Dùng `--resume-stage2-low-lr` (`--low-lr 1e-6`) cùng các cờ tường minh `--best-dice`, `--best-loss`, `--initial-epochs-no-improve` để kiểm chứng hội tụ mà không làm sai lệch kết quả canonical.

## 8. Colab Archival & Export Pattern (No-Drive Standalone Download & Python shutil)

Nhằm tối ưu hóa tốc độ I/O, loại bỏ xung đột xác thực Google Drive FUSE mount, và tránh nguy cơ tràn dung lượng Google Drive 15GB, quy trình xuất xưởng dữ liệu (artifacts export) trên Google Colab được chuẩn hóa như sau:

1. **Ưu tiên chạy hoàn toàn trên Colab Local Disk (`/content/`):**
   - Lưu trữ checkpoint, tensorboard log, và diagnostic output trực tiếp trong thư mục `/content/...` (ví dụ `/content/runs/...` hoặc `/content/P3_C_Canonical_Base_D4`).
2. **Nén dữ liệu bằng Python `shutil.make_archive` (Tránh lệnh Shell `zip` lỗi path):**
   - Lệnh shell `!zip -r ...` thường xuyên gặp lỗi `name not matched: Nothing to do!` do đường dẫn thư mục nguồn không tồn tại, gõ nhầm version (ví dụ D8 thay vì D4), hoặc xử lý dấu gạch chéo không tương thích.
   - **BẮT BUỘC** sử dụng script Python với `shutil.make_archive` và kiểm tra `os.path.exists()` trước khi nén để fail-fast và hiển thị thông báo lỗi rõ ràng nếu đường dẫn sai.
3. **BẮT BUỘC lưu trữ cả Best Model (`best_model_*.pth`) lẫn Last Model (`last_model_*.pth`):**
   - Khi gom artifacts vào bundle hoặc đóng gói zip, **BẮT BUỘC** gom toàn bộ các file `.pth` sinh ra trong thư mục chạy (sử dụng `glob.glob(os.path.join(RUN_DIR, "*.pth"))`).
   - Tuyệt đối không chỉ lọc riêng `best_model_b2_global.pth`. Phải bảo tồn cả `last_model` (ví dụ `last_model_b2_stage2.pth`) để phục vụ audit quỹ đạo, kiểm tra over-fitting cuối lịch trình, hoặc resume tiếp nối trung thực.
4. **Tự động tải về máy cục bộ bằng `google.colab.files.download`:**
   - Kèm theo tính năng kiểm tra dung lượng file (in kích thước theo MB) trước khi kích hoạt download trình duyệt.

**Bộ 3 Cells Colab Chuẩn Mực Tối Giản (Copy-paste ready, Không rườm rà):**

```python
# ==============================================================================
# Cell 1: Environment Setup, Repo Clone, Dataset & Ancestor Checkpoint Download
# ==============================================================================
!nvidia-smi

!rm -rf /content/SAGE_LITE
!git clone -b crack500-audit https://github.com/bach0823/SAGE_LITE.git /content/SAGE_LITE
%cd /content/SAGE_LITE

!python prepare_data/prepare_crack500.py

!mkdir -p /content/checkpoints
!wget -q -O /content/checkpoints/best_model_b2_stage1.pth \
  "https://raw.githubusercontent.com/bach0823/chuyendettnt/main/results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_stage1.pth"
!wget -q -O /content/checkpoints/last_model_b2_stage1_rng.pth \
  "https://raw.githubusercontent.com/bach0823/chuyendettnt/main/results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_last_model_b2_stage1_rng.pth"
```

```python
# ==============================================================================
# Cell 2: Training Execution (Driver thuần túy - 1 lệnh thực thi duy nhất)
# ==============================================================================
%cd /content/SAGE_LITE

!python scripts/train_crack.py \
  --config configs/p3_ablation/<CONFIG_NAME>.yaml \
  --stage2-only \
  --checkpoint /content/checkpoints/best_model_b2_stage1.pth \
  --rng-checkpoint /content/checkpoints/last_model_b2_stage1_rng.pth \
  --data-root /content/dataset/Crack500
```

```python
# ==============================================================================
# Cell 3: Post-Training Diagnostics, Zip & Browser Download
# ==============================================================================
%cd /content/SAGE_LITE
from google.colab import files
import shutil

# 1. Routing Diagnostics (Full Val)
!python tools/analyze_routing.py \
    --config configs/p3_ablation/<CONFIG_NAME>.yaml \
    --checkpoint /content/runs/<RUN_FOLDER_NAME>/best_model_b2_global.pth \
    --output_dir /content/runs/<RUN_FOLDER_NAME>/P3_C_Routing_Diagnostics/full_val \
    --split val \
    --data_root /content/dataset/Crack500

# 2. Error Analysis (Setting A Tiling)
!python tools/run_p3_c_error_analysis.py \
    --config configs/p3_ablation/<CONFIG_NAME>.yaml \
    --checkpoint /content/runs/<RUN_FOLDER_NAME>/best_model_b2_global.pth \
    --routing-json /content/runs/<RUN_FOLDER_NAME>/P3_C_Routing_Diagnostics/full_val/routing_statistics.json \
    --output-dir /content/runs/<RUN_FOLDER_NAME>/P3_C_Routing_Diagnostics/error_analysis \
    --data-root /content/dataset/Crack500

# 3. Zip & Download
shutil.make_archive("/content/<RUN_FOLDER_NAME>_Full", "zip", "/content/runs", "<RUN_FOLDER_NAME>")
files.download("/content/<RUN_FOLDER_NAME>_Full.zip")
```


## 9. Experiment Results Log (BẮT BUỘC ĐỌC & CẬP NHẬT)

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

