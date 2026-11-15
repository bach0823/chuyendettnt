"""
Việc 2 — Phân tích khoảng cách val→test qua nhãn (masks).
Chỉ dùng ground truth masks, không dùng dự đoán mô hình.

Chỉ số đo:
1. Tỉ lệ pixel nứt (crack pixel ratio)
2. Phân phối độ rộng vết (qua distance transform trên skeleton)
3. Tỉ lệ vết mảnh (<= 3px theo skeleton + dt)
4. Số thành phần liên thông mỗi crop
5. Tỉ lệ crop có rất ít crack pixel (< 1000 px)

Phương pháp đo độ rộng:
- Dùng cv2.distanceTransform trên mask crack.
- Với mỗi pixel crack, giá trị distance transform = khoảng cách đến biên nứt gần nhất (tức là ~nửa chiều rộng).
- Độ rộng trung bình vết ~ 2 * mean(dt[mask>0]).
- Tỉ lệ mảnh: pixel có dt <= 1.5 (tương đương chiều rộng <= 3px).
"""
import os, json, csv
import numpy as np
import cv2
from glob import glob
from scipy import ndimage

REPO = r"d:\truong\SpecialSubjectTTNT"
VAL_MASK_DIR  = os.path.join(REPO, "datasets", "Crack500_ready", "val",  "masks")
TEST_MASK_DIR = os.path.join(REPO, "datasets", "Crack500_ready", "test", "masks")
OUT_DIR = os.path.join(REPO, "results", "final_protocol", "07_revision")
os.makedirs(OUT_DIR, exist_ok=True)

def analyze_masks(mask_dir, split_name):
    mask_files = sorted(glob(os.path.join(mask_dir, "*.jpg")) +
                        glob(os.path.join(mask_dir, "*.png")))
    n = len(mask_files)
    print(f"\n=== {split_name}: {n} masks ===")
    
    crack_ratios = []
    mean_widths  = []
    thin_ratios  = []  # fraction of crack pixels with width <= 3px
    n_components = []
    has_little   = []  # crop có < 1000 crack pixels
    
    for mpath in mask_files:
        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  WARN: cannot read {mpath}")
            continue
        
        binary = (mask > 127).astype(np.uint8)
        total_px = binary.size
        crack_px = int(binary.sum())
        
        crack_ratios.append(crack_px / total_px)
        has_little.append(crack_px < 1000)
        
        if crack_px > 0:
            # Distance transform: distance from each crack pixel to nearest 0 pixel
            # (approximates half-width of crack at each point)
            dt = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            # Collect dt values only at crack pixels
            dt_crack = dt[binary > 0]
            mean_w = float(2.0 * np.mean(dt_crack))  # full width estimate
            mean_widths.append(mean_w)
            # Thin ratio: pixels where dt <= 1.5 (width <= 3px)
            thin_ratio = float(np.sum(dt_crack <= 1.5) / len(dt_crack))
            thin_ratios.append(thin_ratio)
        else:
            mean_widths.append(0.0)
            thin_ratios.append(0.0)
        
        # Number of connected components in crack
        n_comp, _ = cv2.connectedComponents(binary)
        n_components.append(n_comp - 1)  # subtract background
    
    crack_ratios  = np.array(crack_ratios)
    mean_widths   = np.array(mean_widths)
    thin_ratios   = np.array(thin_ratios)
    n_components  = np.array(n_components)
    has_little    = np.array(has_little, dtype=bool)
    
    stats = {
        "split": split_name,
        "n_crops": n,
        "crack_pixel_ratio": {
            "mean":   float(crack_ratios.mean()),
            "std":    float(crack_ratios.std()),
            "median": float(np.median(crack_ratios)),
            "p25":    float(np.percentile(crack_ratios, 25)),
            "p75":    float(np.percentile(crack_ratios, 75)),
        },
        "mean_crack_width_px": {
            "mean":   float(mean_widths.mean()),
            "std":    float(mean_widths.std()),
            "median": float(np.median(mean_widths)),
            "p25":    float(np.percentile(mean_widths, 25)),
            "p75":    float(np.percentile(mean_widths, 75)),
        },
        "thin_crack_pixel_fraction_width_le3px": {
            "mean":   float(thin_ratios.mean()),
            "std":    float(thin_ratios.std()),
            "median": float(np.median(thin_ratios)),
        },
        "n_connected_components": {
            "mean":   float(n_components.mean()),
            "std":    float(n_components.std()),
            "median": float(np.median(n_components)),
            "max":    int(n_components.max()),
        },
        "fraction_crops_lt1000_crack_px": float(has_little.mean()),
        "n_crops_lt1000_crack_px": int(has_little.sum()),
    }
    
    # Print
    print(f"  Crack ratio: mean={stats['crack_pixel_ratio']['mean']:.4f} ± {stats['crack_pixel_ratio']['std']:.4f}")
    print(f"  Mean crack width (px): mean={stats['mean_crack_width_px']['mean']:.2f} ± {stats['mean_crack_width_px']['std']:.2f}")
    print(f"  Thin crack (<=3px) fraction: mean={stats['thin_crack_pixel_fraction_width_le3px']['mean']:.4f}")
    print(f"  Connected components: mean={stats['n_connected_components']['mean']:.2f} ± {stats['n_connected_components']['std']:.2f}")
    print(f"  Crops with <1000 crack px: {stats['n_crops_lt1000_crack_px']}/{n} = {stats['fraction_crops_lt1000_crack_px']:.3f}")
    
    return stats, crack_ratios, mean_widths, thin_ratios, n_components

val_stats,  val_cr, val_mw, val_tr, val_nc  = analyze_masks(VAL_MASK_DIR,  "val")
test_stats, tst_cr, tst_mw, tst_tr, tst_nc = analyze_masks(TEST_MASK_DIR, "test")

# Mann-Whitney U test for distributional differences
from scipy import stats as scipy_stats

def compare_distributions(a, b, name):
    stat, pval = scipy_stats.mannwhitneyu(a, b, alternative='two-sided')
    diff_median = float(np.median(b) - np.median(a))
    print(f"  {name}: val_med={np.median(a):.4f}, test_med={np.median(b):.4f}, Δmedian={diff_median:+.4f}, MWU p={pval:.4f}")
    return {"val_median": float(np.median(a)), "test_median": float(np.median(b)),
            "delta_median": diff_median, "mannwhitney_p": float(pval)}

print("\n=== So sánh phân phối val vs test ===")
comparisons = {}
comparisons["crack_pixel_ratio"] = compare_distributions(val_cr, tst_cr, "crack_pixel_ratio")
comparisons["mean_crack_width"]  = compare_distributions(val_mw, tst_mw, "mean_crack_width_px")
comparisons["thin_crack_fraction"] = compare_distributions(val_tr, tst_tr, "thin_crack_fraction")
comparisons["n_components"]      = compare_distributions(val_nc, tst_nc, "n_components")

# Check val_dices vs test_dice để tính khoảng cách
# Val: aggregate từ phase2 seeds
# Test: mean = 0.715770
val_dice_mean  = 0.778638   # canonical phase2 mean
ensemble_val   = 0.785003   # ensemble val dice
test_dice_mean = 0.715770   # test locked

gap_single  = val_dice_mean - test_dice_mean
gap_ensemble = ensemble_val - test_dice_mean
print(f"\nVal→Test gap:")
print(f"  Single-seed mean val (0.7786) → test (0.7158): Δ = {gap_single:.4f} ({gap_single:.1%})")
print(f"  Ensemble val (0.7850) → test (0.7158): Δ = {gap_ensemble:.4f} ({gap_ensemble:.1%})")

# B0 reference gap (from report)
b0_val  = 0.7318
b0_test = 0.6771
b0_gap  = b0_val - b0_test
print(f"  B0 reference: val (0.7318) → test (0.6771): Δ = {b0_gap:.4f} ({b0_gap:.1%})")
print(f"  Nhận xét: B2 gap ({gap_single:.4f}) vs B0 gap ({b0_gap:.4f})")

# Save all results
all_results = {
    "val_stats":  val_stats,
    "test_stats": test_stats,
    "distribution_comparison": comparisons,
    "val_test_dice_gap": {
        "val_single_seed_mean":    val_dice_mean,
        "val_ensemble":            ensemble_val,
        "test_locked":             test_dice_mean,
        "gap_single_to_test_pt":   round(gap_single,  4),
        "gap_ensemble_to_test_pt": round(gap_ensemble, 4),
        "b0_reference_gap_pt":     round(b0_gap, 4),
    },
    "note_on_prediction_analysis": (
        "Dự đoán per-image trên val không được lưu ra file (phase1, phase2 chỉ lưu aggregate). "
        "Dự đoán per-image trên test có sẵn trong 05_test_final_report.json nhưng không có filenames đi kèm. "
        "Phân tích Dice theo nhóm (crack width, crack ratio) cho test không khả thi mà không map index->filename."
    )
}

with open(os.path.join(OUT_DIR, "07b_valtest_gap_results.json"), "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

# Xuất CSV so sánh hai split
with open(os.path.join(OUT_DIR, "07b_valtest_gap_comparison.csv"), "w", newline='', encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Chỉ số", "Val (mean)", "Test (mean)", "Δ (test-val)", "MWU p-value"])
    for key, c in comparisons.items():
        w.writerow([key, f"{c['val_median']:.4f}", f"{c['test_median']:.4f}",
                    f"{c['delta_median']:+.4f}", f"{c['mannwhitney_p']:.4f}"])

print("\nXong. Kết quả lưu vào:", OUT_DIR)
