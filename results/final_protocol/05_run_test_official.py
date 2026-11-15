"""
Phase 5 Official Test Evaluation Script — SAGE-Lite Final Protocol.
Đánh giá chính thức ĐÚNG 1 LẦN DUY NHẤT trên 1,124 ảnh test của Crack500.

Cấu hình khóa (theo 05_frozen_config.json):
- Strategy: ensemble_3seed (Phase 2 seed 42, 43, 44)
- Checkpoint seed42: results/final_protocol/phase2_seed42/best_model_b2_global.pth
- Checkpoint seed43: results/final_protocol/phase2_seed43/best_model_b2_global.pth
- Checkpoint seed44: results/final_protocol/phase2_seed44/best_model_b2_global.pth
- Tau: 0.5
- TTA: flip_hv (trung bình: ảnh gốc + lật ngang + lật dọc) — áp dụng mỗi model, rồi avg 3 model
- Inference Method: predict_full_image_direct (target_size=448)

LƯU Ý: Phải truyền cờ --confirm-unlock để thực thi.
"""

import sys
import os
import json
import hashlib
import argparse
import time
import numpy as np
import cv2
import torch
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "SAGE_LITE"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "SAGE_LITE", "scripts"))

from sage.networks import create_b2_unet
from evaluate_crack_official import get_image_mask_pairs, compute_boundary_iou, predict_full_image_direct
from sage.utils.advanced_metrics import calculate_hd95_bf1

CHECKPOINT_CONFIGS = [
    {
        "seed": 42,
        "path": os.path.join(PROJECT_ROOT, "results", "final_protocol", "phase2_seed42", "best_model_b2_global.pth"),
        "sha256": "4B409B60A7DF788AE48DECE727751E1B8D904CD95EE1102E53D78563F2556CB7"
    },
    {
        "seed": 43,
        "path": os.path.join(PROJECT_ROOT, "results", "final_protocol", "phase2_seed43", "best_model_b2_global.pth"),
        "sha256": "7379C7346616ACE3569AE2D9D640D838098638AD70F17443A34166AC89D4A271"
    },
    {
        "seed": 44,
        "path": os.path.join(PROJECT_ROOT, "results", "final_protocol", "phase2_seed44", "best_model_b2_global.pth"),
        "sha256": "F5F7DC55C9D0CBEB606D5EEA396E6B8983ADC64391F23FE96395A19C5EB75F09"
    },
]
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "p3_ablation", "phase7", "phase7_b2_s2g_b1_b2_end_to_end.yaml")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def build_model(config, device):
    sage_cfg = config.get("sage_config", {})
    model = create_b2_unet(
        num_transformer_layers=int(config.get("num_transformer_layers", 4)),
        pretrained=False,
        sage_config=sage_cfg,
        p3_mode=config.get("p3_mode"),
        use_plu_head=config.get("use_plu_head", False),
        use_s2_gate=config.get("use_s2_gate", False),
        s2_gate_kernel_size=config.get("s2_gate_kernel_size", 3),
        use_s2_gate_block2=config.get("use_s2_gate_block2", False),
        s2_gate_block2_kernel_size=config.get("s2_gate_block2_kernel_size", 3),
        s2_gate_block2_dilation=config.get("s2_gate_block2_dilation", 1),
    ).to(device)
    return model


def predict_with_tta_flip_hv(model, image, device, target_size=448):
    """
    TTA flip_hv: avg sigmoid prob của ảnh gốc, lật ngang, lật dọc.
    """
    logits_orig = predict_full_image_direct(model, image, device, target_size=target_size)
    prob_orig = 1.0 / (1.0 + np.exp(-logits_orig))

    img_h = cv2.flip(image, 1)
    logits_h = predict_full_image_direct(model, img_h, device, target_size=target_size)
    prob_h = 1.0 / (1.0 + np.exp(-logits_h))
    prob_h = cv2.flip(prob_h.astype(np.float32), 1)

    img_v = cv2.flip(image, 0)
    logits_v = predict_full_image_direct(model, img_v, device, target_size=target_size)
    prob_v = 1.0 / (1.0 + np.exp(-logits_v))
    prob_v = cv2.flip(prob_v.astype(np.float32), 0)

    return (prob_orig + prob_h + prob_v) / 3.0


def predict_ensemble_tta(models, image, device, target_size=448):
    """
    Ensemble 3 model × TTA 3 aug = 9 forward passes, avg prob.
    """
    all_probs = []
    for m in models:
        p = predict_with_tta_flip_hv(m, image, device, target_size)
        all_probs.append(p)
    return np.mean(all_probs, axis=0)


def evaluate_test_set(models, test_pairs, device, tau=0.5):
    all_dice, all_prec, all_rec, all_iou, all_biou, all_hd95, all_thin = [], [], [], [], [], [], []
    total_tp = total_fp = total_fn = 0
    kernel3 = np.ones((3, 3), np.uint8)

    print(f"\n[Test Evaluation] Bắt đầu đánh giá trên {len(test_pairs)} ảnh test...")
    start_time = time.time()

    for idx, (img_p, mask_p) in enumerate(test_pairs):
        img = cv2.imread(img_p)
        if img is None:
            print(f"  [WARN] Không đọc được ảnh: {img_p}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  [WARN] Không đọc được mask: {mask_p}")
            continue
        target = (mask > 127).astype(np.uint8)

        prob_map = predict_ensemble_tta(models, img_rgb, device, target_size=448)
        pred = (prob_map >= tau).astype(np.uint8)

        if pred.shape != target.shape:
            pred = cv2.resize(pred.astype(np.float32),
                              (target.shape[1], target.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        tp = int(np.sum((pred == 1) & (target == 1)))
        fp = int(np.sum((pred == 1) & (target == 0)))
        fn = int(np.sum((pred == 0) & (target == 1)))
        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = tp / (tp + fp + 1e-5)
        rec = tp / (tp + fn + 1e-5)
        dice = 2.0 * tp / (2 * tp + fp + fn + 1e-5)
        iou = tp / (tp + fp + fn + 1e-5)

        biou = compute_boundary_iou(pred, target, d=2)
        hd, _ = calculate_hd95_bf1(pred, target)

        # Thin Crack Dice (px <= 3, dùng erosion 3x3)
        thin_mask = target - cv2.erode(target, kernel3, iterations=1)
        thin_tp = int(np.sum((pred == 1) & (thin_mask == 1)))
        thin_fp = int(np.sum((pred == 1) & (thin_mask == 0) & (target == 0)))
        thin_fn = int(np.sum((pred == 0) & (thin_mask == 1)))
        thin_dice = 2.0 * thin_tp / (2 * thin_tp + thin_fp + thin_fn + 1e-5)

        all_dice.append(dice)
        all_prec.append(prec)
        all_rec.append(rec)
        all_iou.append(iou)
        all_biou.append(biou)
        all_hd95.append(hd)
        all_thin.append(thin_dice)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(test_pairs):
            elapsed = time.time() - start_time
            print(f"  Processed {idx + 1}/{len(test_pairs)} ảnh ({elapsed:.1f}s) "
                  f"- Current Dice: {np.mean(all_dice):.4f}")

    global_pixel_iou = float(total_tp / (total_tp + total_fp + total_fn + 1e-5))

    # Bootstrap CI 95% (10000 lần, theo ảnh)
    rng = np.random.default_rng(42)
    n = len(all_dice)
    boot_dice = np.array(all_dice)
    boot_means = [np.mean(rng.choice(boot_dice, size=n, replace=True)) for _ in range(10000)]
    ci_low, ci_high = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

    results = {
        "num_images": n,
        "mean_dice": float(np.mean(all_dice)),
        "dice_ci_95_low": ci_low,
        "dice_ci_95_high": ci_high,
        "mean_precision": float(np.mean(all_prec)),
        "mean_recall": float(np.mean(all_rec)),
        "mean_iou": float(np.mean(all_iou)),
        "global_pixel_iou": global_pixel_iou,
        "mean_boundary_iou": float(np.mean(all_biou)),
        "mean_hd95": float(np.mean(all_hd95)),
        "mean_thin_crack_dice": float(np.mean(all_thin)),
        "total_elapsed_seconds": float(time.time() - start_time),
        "per_image_dice": [float(x) for x in all_dice],
        "checkpoint_strategy": "ensemble_3seed_phase2",
        "tau": 0.5,
        "tta": "flip_hv",
        "seeds": [42, 43, 44]
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Official Test Evaluation — Ensemble 3 seed")
    parser.add_argument("--confirm-unlock", action="store_true",
                        help="Xác nhận mở khóa để chạy test chính thức 1 lần duy nhất sau Phase 2.")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 5 — OFFICIAL TEST EVALUATION (SAGE-Lite Ensemble 3 Seed)")
    print("=" * 70)

    if not args.confirm_unlock:
        print("[KHÓA BẢO VỆ DỮ LIỆU TEST - TEST INTEGRITY LOCK]")
        print("Tập test (1,124 ảnh) chỉ được mở khóa đánh giá đúng 1 lần sau khi:")
        print("1. Phase 2 (Multi-seed variance) đã hoàn thành và xác lập MDE.")
        print("2. 05_frozen_config.json đã được khóa (status=LOCKED).")
        print("\nĐể mở khóa và chạy đánh giá, chạy lại với cờ: --confirm-unlock")
        sys.exit(0)

    # 1. Kiểm tra frozen config
    frozen_cfg_path = os.path.join(SCRIPT_DIR, "05_frozen_config.json")
    with open(frozen_cfg_path, "r", encoding="utf-8") as f:
        frozen = json.load(f)
    assert frozen["status"] == "LOCKED", f"05_frozen_config.json chưa LOCKED: {frozen['status']}"
    print(f"[PASS] frozen_config status = LOCKED. locked_at = {frozen.get('locked_at', '?')}")

    # 2. Load config model
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["root_dir"] = os.path.join(PROJECT_ROOT, "datasets", "Crack500_ready")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị chạy: {device}")

    # 3. Load & verify 3 model
    models = []
    for ckpt_info in CHECKPOINT_CONFIGS:
        path = ckpt_info["path"]
        expected_sha = ckpt_info["sha256"]
        seed = ckpt_info["seed"]

        print(f"\n[Audit] Kiểm tra checkpoint seed={seed}: {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint không tồn tại: {path}")

        actual_sha = sha256_file(path)
        print(f"  Expected SHA256: {expected_sha}")
        print(f"  Actual   SHA256: {actual_sha}")
        if actual_sha != expected_sha:
            raise ValueError(f"SHA256 mismatch seed={seed}! Got {actual_sha}")
        print(f"  [PASS] SHA256 hợp lệ.")

        model = build_model(config, device)
        ckpt = torch.load(path, map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        model.eval()
        models.append(model)
        print(f"  [PASS] Model seed={seed} nạp thành công.")

    # 4. Lấy test pairs
    test_pairs = get_image_mask_pairs(config, "test")
    print(f"\nSố lượng ảnh test phát hiện: {len(test_pairs)} (Kỳ vọng: 1124)")
    assert len(test_pairs) == 1124, f"Test pair count {len(test_pairs)} != 1124"

    # 5. Đánh giá
    results = evaluate_test_set(models, test_pairs, device, tau=0.5)

    print("\n" + "=" * 70)
    print("KẾT QUẢ TEST CHÍNH THỨC (Crack500 — 1,124 images)")
    print("Checkpoint: Ensemble 3 seed Phase 2 (seed 42, 43, 44)")
    print("=" * 70)
    print(f"Mean Test Dice       : {results['mean_dice']:.6f}  "
          f"[95% CI: {results['dice_ci_95_low']:.6f}, {results['dice_ci_95_high']:.6f}]")
    print(f"Mean Test Precision  : {results['mean_precision']:.6f}")
    print(f"Mean Test Recall     : {results['mean_recall']:.6f}")
    print(f"Mean Test IoU        : {results['mean_iou']:.6f}")
    print(f"Global Pixel IoU     : {results['global_pixel_iou']:.6f}")
    print(f"Mean Boundary IoU    : {results['mean_boundary_iou']:.6f}")
    print(f"Mean HD95            : {results['mean_hd95']:.2f} px")
    print(f"Thin Crack Dice      : {results['mean_thin_crack_dice']:.6f}")
    print("=" * 70)

    # 6. Save JSON
    out_json = os.path.join(SCRIPT_DIR, "05_test_final_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Đã lưu kết quả chi tiết: {out_json}")

    # 7. Save Markdown
    out_md = os.path.join(SCRIPT_DIR, "05_test_report.md")
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Báo cáo Kết quả Test Chính thức (Crack500)\n\n")
        f.write(f"- **Ngày chạy:** {ts}\n")
        f.write(f"- **Checkpoint:** Ensemble 3 seed Phase 2 (seed 42, 43, 44) — `best_model_b2_global.pth`\n")
        f.write(f"- **SHA256 seed42:** `4B409B60A7DF788AE48DECE727751E1B8D904CD95EE1102E53D78563F2556CB7`\n")
        f.write(f"- **SHA256 seed43:** `7379C7346616ACE3569AE2D9D640D838098638AD70F17443A34166AC89D4A271`\n")
        f.write(f"- **SHA256 seed44:** `F5F7DC55C9D0CBEB606D5EEA396E6B8983ADC64391F23FE96395A19C5EB75F09`\n")
        f.write(f"- **Số lượng ảnh test:** {results['num_images']}\n")
        f.write(f"- **Cấu hình:** Tau = 0.5, TTA = flip_hv, Ensemble mean prob\n\n")
        f.write("## Bảng Chỉ số Chính thức\n\n")
        f.write("| Chỉ số | Giá trị |\n|---|---|\n")
        f.write(f"| **Test Dice** | **{results['mean_dice']:.4f}** |\n")
        f.write(f"| **Dice 95% CI** | [{results['dice_ci_95_low']:.4f}, {results['dice_ci_95_high']:.4f}] |\n")
        f.write(f"| **Precision** | {results['mean_precision']:.4f} |\n")
        f.write(f"| **Recall** | {results['mean_recall']:.4f} |\n")
        f.write(f"| **Mean IoU** | {results['mean_iou']:.4f} |\n")
        f.write(f"| **Global Pixel IoU** | {results['global_pixel_iou']:.4f} |\n")
        f.write(f"| **Boundary IoU (d=2)** | {results['mean_boundary_iou']:.4f} |\n")
        f.write(f"| **HD95** | {results['mean_hd95']:.2f} px |\n")
        f.write(f"| **Thin Crack Dice** | {results['mean_thin_crack_dice']:.4f} |\n")
    print(f"Đã tạo báo cáo markdown: {out_md}")


if __name__ == "__main__":
    main()
