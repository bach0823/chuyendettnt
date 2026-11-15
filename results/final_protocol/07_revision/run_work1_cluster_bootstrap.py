"""
Việc 1 — Cluster Bootstrap theo ảnh gốc (không cần GPU).

Tên file dạng: YYYYMMDD_HHMMSS_X_Y.jpg
  -> mã ảnh gốc = YYYYMMDD_HHMMSS (8 ký tự đầu từ dấu gạch dưới đầu tiên là phần datetime)
     Thực ra: timestamp_X_Y → mã gốc = timestamp phần đầu (8 ký tự date + _ + 6 ký tự time)
     Pattern: YYYYMMDD_HHMMSS là 15 ký tự đầu.
     
  Ví dụ: 20160222_080850_1281_361.jpg -> gốc = "20160222_080850"
         X=1281, Y=361 là offset crop.
         
Dữ liệu có sẵn:
- val: 348 crop -> per_image_dice từ phase1 TTA sweep (tái chạy nhanh không cần GPU vì cần dữ liệu)
  NHƯNG per_image_dice không được lưu ra file trong phase1_summary.json hay tta_sweep.csv!
  -> TTA sweep chỉ lưu aggregate metrics.
  
Vì không có per_image_dice cho val (các biến thể TTA), ta sẽ:
  a) Cluster bootstrap cho Test Dice (đã có per_image_dice trong 05_test_final_report.json)
     Nhưng cần map 1124 indices -> filenames -> cluster IDs.
  b) Val dice: đọc 3 seeds (val_result.json) chỉ có aggregate -> cluster bootstrap không thực hiện được.
  
=> Ghi rõ hạn chế và chỉ làm những gì có dữ liệu.
"""
import os
import json
import numpy as np
import re

OUT_DIR = r"d:\truong\SpecialSubjectTTNT\results\final_protocol\07_revision"
os.makedirs(OUT_DIR, exist_ok=True)

REPO = r"d:\truong\SpecialSubjectTTNT"
VAL_IMG_DIR = os.path.join(REPO, "datasets", "Crack500_ready", "val", "images")
TEST_IMG_DIR = os.path.join(REPO, "datasets", "Crack500_ready", "test", "images")

# ============================================================
# 1. Trích mã ảnh gốc từ tên file
# ============================================================
def extract_original_id(filename):
    """
    Từ: 20160222_080850_1281_361.jpg
    Trả về: '20160222_080850'
    Pattern: YYYYMMDD_HHMMSS (8 chữ số + _ + 6 chữ số)
    """
    name = os.path.splitext(os.path.basename(filename))[0]  # strip .jpg
    # Lấy 2 phần đầu ngăn cách bởi '_': YYYYMMDD và HHMMSS
    parts = name.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return name  # fallback

val_files = sorted([f for f in os.listdir(VAL_IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
test_files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

val_orig_ids = [extract_original_id(f) for f in val_files]
test_orig_ids = [extract_original_id(f) for f in test_files]

val_unique = sorted(set(val_orig_ids))
test_unique = sorted(set(test_orig_ids))

print(f"Val:  {len(val_files)} crops | {len(val_unique)} ảnh gốc | avg {len(val_files)/len(val_unique):.2f} crop/ảnh")
print(f"Test: {len(test_files)} crops | {len(test_unique)} ảnh gốc | avg {len(test_files)/len(test_unique):.2f} crop/ảnh")

# Phân phối số crop mỗi ảnh gốc
from collections import Counter
val_crop_counts = Counter(val_orig_ids)
test_crop_counts = Counter(test_orig_ids)

print(f"\nVal  crop/ảnh: min={min(val_crop_counts.values())} max={max(val_crop_counts.values())} median={np.median(list(val_crop_counts.values())):.1f}")
print(f"Test crop/ảnh: min={min(test_crop_counts.values())} max={max(test_crop_counts.values())} median={np.median(list(test_crop_counts.values())):.1f}")

# ============================================================
# 2. Cluster Bootstrap cho TEST Dice (per_image_dice có sẵn)
# ============================================================
# Đọc per_image_dice từ phase 5 test report
test_report_path = os.path.join(REPO, "results", "final_protocol", "05_test_final_report.json")
with open(test_report_path) as f:
    test_report = json.load(f)

test_per_image_dice = np.array(test_report["per_image_dice"])
assert len(test_per_image_dice) == len(test_files), \
    f"Mismatch: {len(test_per_image_dice)} dice values vs {len(test_files)} test files"

# Map index -> cluster (ảnh gốc)
test_orig_arr = np.array(test_orig_ids)
test_unique_arr = np.array(sorted(test_unique))
n_test_clusters = len(test_unique_arr)

# Cluster bootstrap: resample ảnh gốc, 10000 lần
rng = np.random.default_rng(42)
N_BOOT = 10_000

def cluster_bootstrap_mean(per_image_dice, orig_ids, n_boot=10_000, rng=None):
    """
    Resample ảnh gốc (cluster), lấy tất cả crops của mỗi ảnh được chọn.
    Trả về array bootstrap means.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    orig_ids = np.array(orig_ids)
    uniq_ids = np.unique(orig_ids)
    n_clusters = len(uniq_ids)
    
    # Tạo dict cluster -> indices
    cluster_to_idx = {uid: np.where(orig_ids == uid)[0] for uid in uniq_ids}
    
    boot_means = np.zeros(n_boot)
    for b in range(n_boot):
        # Resample n_clusters ảnh gốc có thay thế
        sampled_ids = rng.choice(uniq_ids, size=n_clusters, replace=True)
        # Lấy tất cả crops của các ảnh gốc được chọn
        all_dice = np.concatenate([per_image_dice[cluster_to_idx[uid]] for uid in sampled_ids])
        boot_means[b] = np.mean(all_dice)
    return boot_means

print("\n--- Bootstrap cluster TEST Dice (n=10000, resample ảnh gốc) ---")
boot_test = cluster_bootstrap_mean(test_per_image_dice, test_orig_ids, N_BOOT, rng)
test_ci_lo = float(np.percentile(boot_test, 2.5))
test_ci_hi = float(np.percentile(boot_test, 97.5))
test_mean  = float(np.mean(test_per_image_dice))

print(f"Test Dice (mean): {test_mean:.6f}")
print(f"Test CI (crop bootstrap, cũ): [{test_report['dice_ci_95_low']:.6f}, {test_report['dice_ci_95_high']:.6f}]")
print(f"Test CI (cluster bootstrap): [{test_ci_lo:.6f}, {test_ci_hi:.6f}]")
print(f"Độ rộng CI cũ:    {test_report['dice_ci_95_high'] - test_report['dice_ci_95_low']:.6f}")
print(f"Độ rộng CI cluster: {test_ci_hi - test_ci_lo:.6f}")
print(f"CI rộng hơn: {(test_ci_hi - test_ci_lo) / (test_report['dice_ci_95_high'] - test_report['dice_ci_95_low']):.2f}x")

# ============================================================
# 3. CI cho Val Dice (3 seeds) - chỉ có aggregate, KHÔNG có per_image_dice
# ============================================================
# Phase 2 val_result.json chỉ lưu aggregate metrics (không có per_image_dice)
# => KHÔNG THỂ thực hiện cluster bootstrap cho val.
# Chú ý hạn chế này trong report.

phase2_base = os.path.join(REPO, "results", "final_protocol", "phase2_results")
seeds = [42, 43, 44]
val_dices = {}
for s in seeds:
    p = os.path.join(phase2_base, f"phase2_seed{s}", "val_result.json")
    with open(p) as f:
        d = json.load(f)
    val_dices[s] = float(d["dice"])
    print(f"Val Dice seed {s}: {val_dices[s]:.6f}")

val_dice_arr = np.array(list(val_dices.values()))
print(f"Val mean: {val_dice_arr.mean():.6f}, s={val_dice_arr.std(ddof=1):.6f}")

# MDE 95% CI using chi-squared (n=3)
# CI for sigma: [s*sqrt(n-1/chi2_hi), s*sqrt(n-1/chi2_lo)]
# chi2(0.025, df=2) = 0.0506, chi2(0.975, df=2) = 7.378
import math
n_seeds = 3
s_seed = val_dice_arr.std(ddof=1)
df = n_seeds - 1
# 95% CI for s (chi-squared)
chi2_lo = 0.0506  # chi2 crit 0.025 với df=2
chi2_hi = 7.378   # chi2 crit 0.975 với df=2
s_ci_lo = s_seed * math.sqrt(df / chi2_hi)
s_ci_hi = s_seed * math.sqrt(df / chi2_lo)
MDE_nominal = 2 * s_seed
MDE_ci_lo   = 2 * s_ci_lo
MDE_ci_hi   = 2 * s_ci_hi
print(f"\ns_seed = {s_seed:.6f}")
print(f"95% CI cho s: [{s_ci_lo:.6f}, {s_ci_hi:.6f}]")
print(f"MDE nominal: {MDE_nominal:.6f}")
print(f"95% CI cho MDE: [{MDE_ci_lo:.6f}, {MDE_ci_hi:.6f}]")

# ============================================================
# 4. Val TTA delta: chỉ có aggregate, không có per_image. 
# Cluster bootstrap KHÔNG KHẢ THI cho TTA delta trên val (không có per_image_dice).
# => Ghi hạn chế.
# ============================================================

# ============================================================
# 5. Xuất CSV tóm tắt cluster IDs
# ============================================================
import csv

val_cluster_summary = [(uid, val_crop_counts[uid]) for uid in val_unique]
test_cluster_summary = [(uid, test_crop_counts[uid]) for uid in test_unique]

with open(os.path.join(OUT_DIR, "07a_val_cluster_summary.csv"), "w", newline='') as f:
    w = csv.writer(f)
    w.writerow(["original_id", "num_crops"])
    w.writerows(val_cluster_summary)

with open(os.path.join(OUT_DIR, "07a_test_cluster_summary.csv"), "w", newline='') as f:
    w = csv.writer(f)
    w.writerow(["original_id", "num_crops"])
    w.writerows(test_cluster_summary)

# Save bootstrap result
result = {
    "val_cluster_stats": {
        "n_crops": len(val_files),
        "n_original": len(val_unique),
        "avg_crops_per_orig": round(len(val_files)/len(val_unique), 3),
        "min_crops": int(min(val_crop_counts.values())),
        "max_crops": int(max(val_crop_counts.values())),
        "median_crops": float(np.median(list(val_crop_counts.values()))),
    },
    "test_cluster_stats": {
        "n_crops": len(test_files),
        "n_original": len(test_unique),
        "avg_crops_per_orig": round(len(test_files)/len(test_unique), 3),
        "min_crops": int(min(test_crop_counts.values())),
        "max_crops": int(max(test_crop_counts.values())),
        "median_crops": float(np.median(list(test_crop_counts.values()))),
    },
    "test_dice_cluster_bootstrap": {
        "mean": test_mean,
        "ci_95_lo_cluster": test_ci_lo,
        "ci_95_hi_cluster": test_ci_hi,
        "ci_95_lo_crop_old": test_report["dice_ci_95_low"],
        "ci_95_hi_crop_old": test_report["dice_ci_95_high"],
        "ci_width_cluster": round(test_ci_hi - test_ci_lo, 6),
        "ci_width_crop": round(test_report["dice_ci_95_high"] - test_report["dice_ci_95_low"], 6),
        "ratio_cluster_over_crop": round((test_ci_hi - test_ci_lo) / (test_report["dice_ci_95_high"] - test_report["dice_ci_95_low"]), 3),
    },
    "mde_analysis": {
        "n_seeds": n_seeds,
        "s_seed": float(s_seed),
        "s_95ci_lo": float(s_ci_lo),
        "s_95ci_hi": float(s_ci_hi),
        "mde_nominal": float(MDE_nominal),
        "mde_95ci_lo": float(MDE_ci_lo),
        "mde_95ci_hi": float(MDE_ci_hi),
        "note": "MDE 95% CI dùng chi-squared (df=2)"
    },
    "val_cluster_bootstrap_possible": False,
    "val_cluster_bootstrap_reason": "val_result.json chỉ lưu aggregate dice, không có per_image_dice. Cluster bootstrap cho val không khả thi mà không chạy lại suy luận.",
    "tta_cluster_bootstrap_possible": False,
    "tta_cluster_bootstrap_reason": "phase1 TTA bootstrap dùng per_image_dice trong RAM, không lưu ra file. Không thể tái tính cluster bootstrap TTA mà không GPU.",
}

with open(os.path.join(OUT_DIR, "07a_cluster_bootstrap_results.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("\nXong. Kết quả lưu vào:", OUT_DIR)
