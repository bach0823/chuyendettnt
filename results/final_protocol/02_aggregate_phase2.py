"""
Phase 2 Aggregator: Tổng hợp kết quả huấn luyện 3 seed từ Google Colab.
Tự động tìm kiếm file zip hoặc thư mục kết quả, tính Mean, Std và MDE = 2 * std.
"""

import os
import sys
import glob
import json
import zipfile
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")
TARGET_DIR = os.path.join(BASE_DIR, "phase2_results")
SEEDS = [42, 43, 44]

def extract_zips_if_needed():
    os.makedirs(TARGET_DIR, exist_ok=True)
    # Check for zips in Downloads or TARGET_DIR
    for seed in SEEDS:
        seed_dir = os.path.join(TARGET_DIR, f"phase2_seed{seed}")
        if os.path.exists(seed_dir) and (os.path.exists(os.path.join(seed_dir, "val_result.json")) or os.path.exists(os.path.join(seed_dir, "stage2_completion.json"))):
            print(f"[OK] Found extracted directory for seed {seed}: {seed_dir}")
            continue

        # Look for zip file
        zip_candidates = glob.glob(os.path.join(DOWNLOADS_DIR, f"*seed{seed}*.zip")) + \
                         glob.glob(os.path.join(BASE_DIR, f"*seed{seed}*.zip")) + \
                         glob.glob(os.path.join(TARGET_DIR, f"*seed{seed}*.zip"))
        
        if zip_candidates:
            zip_path = zip_candidates[0]
            print(f"[Unzipping] {zip_path} -> {TARGET_DIR}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(TARGET_DIR)
        else:
            print(f"[Missing] No result or zip found for seed {seed}")

def parse_seed_metrics(seed):
    seed_dir = os.path.join(TARGET_DIR, f"phase2_seed{seed}")
    if not os.path.exists(seed_dir):
        # Maybe extracted without outer folder
        alt_dirs = glob.glob(os.path.join(TARGET_DIR, f"*seed{seed}*"))
        if alt_dirs:
            seed_dir = alt_dirs[0]
        else:
            return None

    # Priority 1: val_result.json
    val_json_path = os.path.join(seed_dir, "val_result.json")
    if os.path.exists(val_json_path):
        with open(val_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {
                "seed": seed,
                "val_dice": float(data.get("dice", data.get("val_dice", 0.0))),
                "val_iou": float(data.get("iou", data.get("val_iou", 0.0))),
                "source": "val_result.json"
            }

    # Priority 2: stage2_completion.json
    s2_json_path = os.path.join(seed_dir, "stage2_completion.json")
    if os.path.exists(s2_json_path):
        with open(s2_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {
                "seed": seed,
                "val_dice": float(data.get("best_dice", data.get("val_dice", 0.0))),
                "val_iou": float(data.get("best_iou", data.get("val_iou", 0.0))),
                "source": "stage2_completion.json"
            }

    # Priority 3: Parse train.log
    log_path = os.path.join(seed_dir, "train.log")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            best_dice = 0.0
            for line in f:
                if "Best Val Dice:" in line:
                    try:
                        val = float(line.split("Best Val Dice:")[1].split()[0])
                        if val > best_dice:
                            best_dice = val
                    except:
                        pass
            if best_dice > 0:
                return {"seed": seed, "val_dice": best_dice, "source": "train.log"}

    return None

def main():
    print("=" * 60)
    print("PHASE 2 — AGGREGATING MULTI-SEED VARIANCE METRICS")
    print("=" * 60)

    extract_zips_if_needed()

    results = []
    for seed in SEEDS:
        res = parse_seed_metrics(seed)
        if res:
            results.append(res)
            print(f"Seed {seed}: Val Dice = {res['val_dice']:.6f} (from {res['source']})")
        else:
            print(f"Seed {seed}: CHƯA CÓ KẾT QUẢ")

    if len(results) < len(SEEDS):
        print(f"\n[CẢNH BÁO] Mới có {len(results)}/{len(SEEDS)} seed. Cần đủ 3 seed để tính MDE chính xác.")
        if not results:
            print("Toàn bộ 3 seed đang chờ chạy xong trên Colab.")
            return

    dice_vals = [r["val_dice"] for r in results]
    mean_dice = float(np.mean(dice_vals))
    std_dice = float(np.std(dice_vals, ddof=1)) if len(dice_vals) > 1 else 0.0
    mde = 2.0 * std_dice

    print("\n" + "=" * 60)
    print(f"Mean Val Dice : {mean_dice:.6f}")
    print(f"Std Val Dice  : {std_dice:.6f}")
    print(f"MDE (2 * std) : {mde:.6f}")
    print("=" * 60)

    summary = {
        "seeds": results,
        "mean_val_dice": mean_dice,
        "std_val_dice": std_dice,
        "mde": mde,
        "num_seeds": len(results),
        "status": "COMPLETED" if len(results) == 3 else "PARTIAL"
    }

    out_json = os.path.join(BASE_DIR, "02_phase2_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Đã lưu kết quả tổng hợp vào: {out_json}")

if __name__ == "__main__":
    main()
