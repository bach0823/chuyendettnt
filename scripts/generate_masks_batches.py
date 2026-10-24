#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Script sinh mask bán tự động bằng OpenCrack nnU-Net theo từng batch:
===================================================================
- Thư mục ảnh vào: C:\Users\ASUS\Desktop\Dataset\img (001.jpg -> 307.jpg)
- Thư mục mask ra: C:\Users\ASUS\Desktop\Dataset\mask (mask_001.png -> mask_307.png)
- Thư mục overlap ra: C:\Users\ASUS\Desktop\Dataset\overlap (overlap_001.jpg -> overlap_307.jpg)
- Quy tắc batch:
    + Batch 1: 5 ảnh (001 -> 005)
    + Batch 2 trở đi: 15 ảnh mỗi batch (006 -> 020, 021 -> 035, ...)
"""

import os
import sys
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(r"d:\truong\SpecialSubjectTTNT")
MODEL_DIR = PROJECT_ROOT / "models" / "opencrack_nnunet" / "Dataset501_OpenCrack" / "nnUNetTrainer__nnUNetPlans__2d"
CHECKPOINT_NAME = "checkpoint_ep0500.pth"

BASE_DATASET = Path(r"C:\Users\ASUS\Desktop\Dataset")
DIR_IMG = BASE_DATASET / "img"
DIR_MASK = BASE_DATASET / "mask"
DIR_OVERLAP = BASE_DATASET / "overlap"

OVERLAP = 0.20
MIN_AREA = 30.0


def filter_mask_min_area(mask: np.ndarray, min_area: float = 30.0) -> np.ndarray:
    """Loại bỏ các đốm nhiễu pixel nhỏ hơn min_area bằng Connected Components."""
    if min_area <= 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 255
    return clean_mask


def create_red_overlap(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tạo ảnh overlap 16MP với lớp mask đỏ bán trong suốt đè lên ảnh gốc."""
    overlay = img_bgr.copy()
    red_color = np.array([0, 0, 255], dtype=np.uint8)
    idx = mask == 255
    if np.any(idx):
        overlay[idx] = cv2.addWeighted(img_bgr[idx], 0.35, np.full_like(img_bgr[idx], red_color), 0.65, 0)
    return overlay


def register_custom_trainer():
    import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as trainer_mod
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    if not hasattr(trainer_mod, "nnUNetTrainerSaveEvery10"):
        class nnUNetTrainerSaveEvery10(nnUNetTrainer):
            pass
        setattr(trainer_mod, "nnUNetTrainerSaveEvery10", nnUNetTrainerSaveEvery10)


def get_batch_range(batch_idx: int):
    """Tính khoảng chỉ số bắt đầu và kết thúc cho từng batch."""
    if batch_idx == 1:
        return 1, 5
    start = 5 + (batch_idx - 2) * 15 + 1
    end = min(307, start + 14)
    return start, end


def parse_args():
    parser = argparse.ArgumentParser(description="Sinh mask OpenCrack nnU-Net theo từng batch.")
    parser.add_argument("--batch", type=int, default=1, help="Số thứ tự batch cần chạy (1: 5 ảnh, 2..22: 15 ảnh)")
    parser.add_argument("--all", action="store_true", help="Tự động chạy toàn bộ các batch từ 1 đến 22")
    parser.add_argument("--start", type=int, default=None, help="Chỉ định trực tiếp chỉ số bắt đầu")
    parser.add_argument("--end", type=int, default=None, help="Chỉ định trực tiếp chỉ số kết thúc")
    parser.add_argument("--min-area", type=float, default=30.0, help="Ngưỡng lọc diện tích tối thiểu pixel (default: 30.0)")
    parser.add_argument("--overwrite", action="store_true", help="Ghi đè toàn bộ mask nếu đã tồn tại")
    parser.add_argument("--overwrite-first", action="store_true", default=True, help="Ghi đè lại ảnh 001 với min_area cũ")
    return parser.parse_args()


def main():
    args = parse_args()

    DIR_MASK.mkdir(parents=True, exist_ok=True)
    DIR_OVERLAP.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"=== KHỞI ĐỘNG HỆ THỐNG SINH MASK OPENCRACK NNU-NET ===")
    print("=" * 70)
    print(f"Thư mục ảnh vào : {DIR_IMG}")
    print(f"Thư mục mask ra : {DIR_MASK}")
    print(f"Thư mục overlap : {DIR_OVERLAP}")
    print(f"Thiết bị tính toán: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    print(f"Cấu hình lọc: min_area = {args.min_area} px | overlap sliding window = {OVERLAP*100:.0f}%")
    print("=" * 70 + "\n")

    register_custom_trainer()
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO

    tile_step_size = max(0.1, min(1.0, 1.0 - OVERLAP))
    print(f"Đang nạp mô hình vào VRAM ({tile_step_size=:.2f})...")

    predictor = nnUNetPredictor(
        tile_step_size=tile_step_size,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=False,
        device=device
    )
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=str(MODEL_DIR),
        use_folds=(0,),
        checkpoint_name=CHECKPOINT_NAME
    )
    rw = NaturalImage2DIO()
    print("Khởi tạo mô hình thành công!\n")

    if args.all:
        batch_list = [(b, *get_batch_range(b)) for b in range(1, 23)]
    elif args.start is not None and args.end is not None:
        batch_list = [(0, args.start, args.end)]
    else:
        s, e = get_batch_range(args.batch)
        batch_list = [(args.batch, s, e)]

    total_processed = 0
    total_skipped = 0
    grand_start = time.time()

    for batch_num, start_idx, end_idx in batch_list:
        batch_label = f"Batch {batch_num} ({start_idx:03d} -> {end_idx:03d})" if batch_num > 0 else f"Tùy chỉnh ({start_idx:03d} -> {end_idx:03d})"
        num_images = end_idx - start_idx + 1
        print("=" * 70)
        print(f"=== BẮT ĐẦU CHẠY {batch_label.upper()} (TỔNG: {num_images} ẢNH) ===")
        print("=" * 70)
        sys.stdout.flush()

        batch_processed = 0
        batch_skipped = 0
        t_batch_start = time.time()

        for idx in range(start_idx, end_idx + 1):
            img_name = f"{idx:03d}.jpg"
            mask_name = f"mask_{idx:03d}.png"
            overlap_name = f"overlap_{idx:03d}.jpg"

            src_img_path = DIR_IMG / img_name
            dst_mask_path = DIR_MASK / mask_name
            dst_overlap_path = DIR_OVERLAP / overlap_name

            if not src_img_path.exists():
                print(f"[{idx:03d}] CẢNH BÁO: Không tìm thấy ảnh {src_img_path}")
                sys.stdout.flush()
                continue

            should_overwrite = args.overwrite or (idx == 1 and args.overwrite_first)
            if dst_mask_path.exists() and not should_overwrite:
                print(f"[{idx:03d}] {img_name} -> {mask_name} ĐÃ CÓ (Bỏ qua)")
                batch_skipped += 1
                total_skipped += 1
                sys.stdout.flush()
                continue

            t0 = time.time()
            print(f"[{idx:03d}/{end_idx:03d}] Đang xử lý: {img_name} -> {mask_name} ...", end="", flush=True)

            try:
                img_arr, props = rw.read_images([str(src_img_path)])
                pred = predictor.predict_single_npy_array(img_arr, props)
                binary_mask = (pred[0] == 1).astype(np.uint8) * 255
                clean_mask = filter_mask_min_area(binary_mask, min_area=args.min_area)

                # Lưu mask nhị phân [0, 255]
                cv2.imwrite(str(dst_mask_path), clean_mask)

                # Đọc ảnh gốc tạo overlap trực quan
                img_bgr = cv2.imread(str(src_img_path))
                if img_bgr is not None:
                    overlap = create_red_overlap(img_bgr, clean_mask)
                    cv2.imwrite(str(dst_overlap_path), overlap, [cv2.IMWRITE_JPEG_QUALITY, 93])

                crack_px = int(np.sum(clean_mask == 255))
                total_px = clean_mask.shape[0] * clean_mask.shape[1]
                elapsed = time.time() - t0
                batch_processed += 1
                total_processed += 1
                print(f" XONG ({elapsed:.1f}s | {crack_px:,} px ~ {crack_px/total_px*100:.2f}%)")
                sys.stdout.flush()

            except Exception as e:
                print(f" LỖI: {e}")
                sys.stdout.flush()

        batch_time = time.time() - t_batch_start
        print("\n" + "=" * 70)
        print(f">>> HOÀN TẤT {batch_label.upper()} <<<")
        print(f"- Tạo mới đợt này : {batch_processed} ảnh")
        print(f"- Đã có sẵn        : {batch_skipped} ảnh")
        print(f"- Thời gian batch   : {batch_time/60:.1f} phút ({batch_time:.1f}s)")
        print(f"- Lũy kế toàn bộ    : {total_processed + total_skipped}/307 ảnh")
        print("=" * 70 + "\n")
        sys.stdout.flush()

    grand_time = time.time() - grand_start
    print("=" * 70)
    print(f"=== TOÀN BỘ TIẾN TRÌNH ĐÃ HOÀN TẤT THÀNH CÔNG ===")
    print(f"Tổng số ảnh mới tạo : {total_processed}")
    print(f"Tổng số ảnh bỏ qua  : {total_skipped}")
    print(f"Tổng thời gian chạy : {grand_time/60:.1f} phút")
    print(f"Thư mục mask        : {DIR_MASK}")
    print(f"Thư mục overlap     : {DIR_OVERLAP}")
    print("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
