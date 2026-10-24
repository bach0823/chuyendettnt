#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script kiểm tra tính toàn vẹn và độ chuẩn xác của file mask gán nhãn:
1. File mở bằng PIL phải đạt Mode 'L' (1-channel Grayscale).
2. Đọc bằng cv2.imread(..., cv2.IMREAD_UNCHANGED) phải có ndim == 2 (không có kênh Alpha).
3. Kích thước (H, W) của mask phải trùng khớp 100% với ảnh gốc.
4. Tập giá trị pixel np.unique(mask) bắt buộc là tập con của {0, 255} (không có pixel xám lẻ 1..254).
"""

import sys
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def verify_single_mask(img_path: Path, mask_path: Path) -> tuple[bool, str]:
    if not img_path.exists():
        return False, f"Không tìm thấy ảnh gốc: {img_path}"
    if not mask_path.exists():
        return False, f"Không tìm thấy mask: {mask_path}"

    # 1. Kiểm tra PIL Mode
    try:
        with Image.open(mask_path) as im:
            pil_mode = im.mode
            pil_size = im.size # (W, H)
    except Exception as e:
        return False, f"Lỗi đọc file bằng PIL: {e}"

    if pil_mode != "L":
        return False, f"LỖI: Chế độ màu là '{pil_mode}', yêu cầu bắt buộc phải là 'L' (Grayscale 1 kênh, không alpha)."

    # 2. Kiểm tra OpenCV ndim
    mask_cv = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask_cv is None:
        return False, "OpenCV không đọc được file mask."

    if mask_cv.ndim != 2:
        return False, f"LỖI: ndim = {mask_cv.ndim} (phải là 2D ma trận, không được chứa kênh màu hoặc kênh alpha)."

    # 3. Kiểm tra kích thước tuyệt đối khớp ảnh gốc
    img_cv = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img_cv is None:
        return False, "OpenCV không đọc được ảnh gốc để đối chiếu kích thước."

    expected_shape = img_cv.shape[:2]
    if mask_cv.shape != expected_shape:
        return False, f"LỖI: Kích thước mask {mask_cv.shape} bị lệch so với ảnh gốc {expected_shape}."

    # 4. Kiểm tra giá trị pixel nhị phân thuần túy
    unique_vals = set(np.unique(mask_cv))
    invalid_vals = unique_vals - {0, 255}
    if invalid_vals:
        return False, f"LỖI: Mask bị 'bẩn', chứa các giá trị pixel không hợp lệ: {sorted(list(invalid_vals))} (chỉ chấp nhận 0 và 255)."

    crack_px = int(np.sum(mask_cv == 255))
    total_px = mask_cv.shape[0] * mask_cv.shape[1]
    pct = crack_px / total_px * 100

    return True, f"HỢP LỆ (Grayscale 1 kênh | {mask_cv.shape[1]}x{mask_cv.shape[0]} | {crack_px:,} px ~ {pct:.2f}% nứt)"


def main():
    base_dir = Path(r"C:\Users\ASUS\Desktop\Dataset")
    img_dir = base_dir / "img"
    mask_dir = base_dir / "mask"

    if len(sys.argv) > 1:
        indices = [int(x) for x in sys.argv[1:]]
    else:
        # Mặc định kiểm tra 5 ảnh mẫu
        indices = [1, 2, 5, 10, 20]

    print("=" * 70)
    print("=== KIỂM TRA ĐỘ CHUẨN XÁC MASK GÁN NHÃN (DATA INTEGRITY AUDIT) ===")
    print("=" * 70)

    all_passed = True
    for idx in indices:
        img_name = f"{idx:03d}.jpg"
        mask_name = f"mask_{idx:03d}.png"
        img_p = img_dir / img_name
        mask_p = mask_dir / mask_name

        ok, msg = verify_single_mask(img_p, mask_p)
        status_tag = "[PASS]" if ok else "[FAIL]"
        print(f"{status_tag} [{idx:03d}] {mask_name} -> {msg}")
        if not ok:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print(">>> TẤT CẢ FILE ĐÃ ĐẠT TIÊU CHUẨN NHỊ PHÂN TUYỆT ĐỐI! <<<")
    else:
        print(">>> CẢNH BÁO: PHÁT HIỆN FILE KHÔNG ĐẠT TIÊU CHUẨN HUẤN LUYỆN! <<<")
    print("=" * 70)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
