import os
import sys
import json
import csv
import yaml
from glob import glob
from pathlib import Path
import cv2
import numpy as np
from scipy import stats

REPO = r"d:\truong\SpecialSubjectTTNT"
sys.path.insert(0, os.path.join(REPO, "SAGE_LITE", "scripts"))
from evaluate_crack_official import get_image_mask_pairs

CONFIG_PATH = os.path.join(REPO, "configs", "p3_ablation", "phase7", "phase7_b2_s2g_b1_b2_end_to_end.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg["root_dir"] = os.path.join(REPO, "datasets", "Crack500_ready")

# 1. Load test pairs and dice
test_pairs = get_image_mask_pairs(cfg, "test")
val_pairs = get_image_mask_pairs(cfg, "val")

with open(os.path.join(REPO, "results", "final_protocol", "05_test_final_report.json"), "r", encoding="utf-8") as f:
    report = json.load(f)
test_dice_list = report["per_image_dice"]

def extract_source_id(fname):
    parts = os.path.basename(fname).split("_")
    if len(parts) >= 2:
        return parts[0] + "_" + parts[1]
    return fname

def analyze_mask_single(mpath):
    mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read {mpath}")
    binary = (mask > 127).astype(np.uint8)
    total_px = binary.size
    crack_px = int(binary.sum())
    crack_ratio = crack_px / total_px
    
    if crack_px > 0:
        dt = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        dt_crack = dt[binary > 0]
        mean_width = float(2.0 * np.mean(dt_crack))
        thin_fraction = float(np.mean(dt_crack <= 1.5))
        num_labels, labels = cv2.connectedComponents(binary)
        n_comp = num_labels - 1
    else:
        mean_width = 0.0
        thin_fraction = 0.0
        n_comp = 0
        
    return {
        "crack_px": crack_px,
        "crack_ratio": crack_ratio,
        "mean_width": mean_width,
        "thin_fraction": thin_fraction,
        "n_components": n_comp,
        "is_sparse": crack_px < 1000
    }

print("Analyzing 1124 test masks...")
test_records = []
for idx, (img_p, mask_p) in enumerate(test_pairs):
    props = analyze_mask_single(mask_p)
    d = test_dice_list[idx]
    fname = os.path.basename(img_p)
    sid = extract_source_id(fname)
    rec = {
        "index": idx,
        "filename": fname,
        "source_id": sid,
        "dice": d,
        **props
    }
    test_records.append(rec)

print("Analyzing 348 val masks...")
val_records = []
for idx, (img_p, mask_p) in enumerate(val_pairs):
    props = analyze_mask_single(mask_p)
    fname = os.path.basename(img_p)
    sid = extract_source_id(fname)
    rec = {
        "index": idx,
        "filename": fname,
        "source_id": sid,
        **props
    }
    val_records.append(rec)

# Save per_image_dice_test.csv
csv_path = os.path.join(REPO, "results", "final_protocol", "08_revision2", "08_per_image_dice_test.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(test_records[0].keys()))
    writer.writeheader()
    writer.writerows(test_records)
print(f"Saved: {csv_path}")

# 2. Correlations with Dice
dices = np.array([r["dice"] for r in test_records])
ratios = np.array([r["crack_ratio"] for r in test_records])
widths = np.array([r["mean_width"] for r in test_records])
thins = np.array([r["thin_fraction"] for r in test_records])
comps = np.array([r["n_components"] for r in test_records])
crack_pxs = np.array([r["crack_px"] for r in test_records])

correlations = {}
for name, arr in [("crack_ratio", ratios), ("mean_width", widths), 
                  ("thin_fraction", thins), ("n_components", comps),
                  ("crack_px", crack_pxs)]:
    sp_corr, sp_p = stats.spearmanr(dices, arr)
    pr_corr, pr_p = stats.pearsonr(dices, arr)
    correlations[name] = {
        "spearman_rho": float(sp_corr),
        "spearman_p": float(sp_p),
        "pearson_r": float(pr_corr),
        "pearson_p": float(pr_p)
    }

# 3. Grouping Analysis
# a) Độ rộng vết: mảnh (<=3px), vừa (3-10px), rộng (>10px)
group_width = {
    "thin_le_3px": [r["dice"] for r in test_records if r["mean_width"] <= 3.0],
    "medium_3_to_10px": [r["dice"] for r in test_records if 3.0 < r["mean_width"] <= 10.0],
    "wide_gt_10px": [r["dice"] for r in test_records if r["mean_width"] > 10.0]
}

# b) Độ phức tạp: ít (<=2), vừa (3-5), nhiều (>5)
group_comp = {
    "simple_le_2": [r["dice"] for r in test_records if r["n_components"] <= 2],
    "moderate_3_to_5": [r["dice"] for r in test_records if 3 <= r["n_components"] <= 5],
    "complex_gt_5": [r["dice"] for r in test_records if r["n_components"] > 5]
}

# c) Sparse vs dense crack
group_sparse = {
    "sparse_lt_1000px": [r["dice"] for r in test_records if r["is_sparse"]],
    "dense_ge_1000px": [r["dice"] for r in test_records if not r["is_sparse"]]
}

def summarize_group(d_list):
    if len(d_list) == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0}
    return {
        "n": len(d_list),
        "mean_dice": float(np.mean(d_list)),
        "std_dice": float(np.std(d_list)),
        "median_dice": float(np.median(d_list)),
        "q25_dice": float(np.percentile(d_list, 25)),
        "q75_dice": float(np.percentile(d_list, 75))
    }

group_summary = {
    "width": {k: summarize_group(v) for k, v in group_width.items()},
    "complexity": {k: summarize_group(v) for k, v in group_comp.items()},
    "sparse": {k: summarize_group(v) for k, v in group_sparse.items()}
}

# 4. Cluster-level MWU test (Val vs Test by original image)
def aggregate_by_source(records):
    sources = {}
    for r in records:
        sid = r["source_id"]
        if sid not in sources:
            sources[sid] = []
        sources[sid].append(r)
    
    agg = {}
    for sid, recs in sources.items():
        agg[sid] = {
            "n_crops": len(recs),
            "crack_ratio": np.mean([x["crack_ratio"] for x in recs]),
            "mean_width": np.mean([x["mean_width"] for x in recs]),
            "thin_fraction": np.mean([x["thin_fraction"] for x in recs]),
            "n_components": np.mean([x["n_components"] for x in recs]),
            "sparse_fraction": np.mean([x["is_sparse"] for x in recs])
        }
    return agg

val_source_agg = aggregate_by_source(val_records)
test_source_agg = aggregate_by_source(test_records)

print(f"Val original images: {len(val_source_agg)}")
print(f"Test original images: {len(test_source_agg)}")

cluster_mwu = {}
for metric in ["crack_ratio", "mean_width", "thin_fraction", "n_components", "sparse_fraction"]:
    val_vals = [v[metric] for v in val_source_agg.values()]
    test_vals = [v[metric] for v in test_source_agg.values()]
    stat, pval = stats.mannwhitneyu(val_vals, test_vals, alternative="two-sided")
    cluster_mwu[metric] = {
        "val_mean": float(np.mean(val_vals)),
        "val_std": float(np.std(val_vals)),
        "test_mean": float(np.mean(test_vals)),
        "test_std": float(np.std(test_vals)),
        "u_stat": float(stat),
        "p_value": float(pval),
        "significant_005": bool(pval < 0.05)
    }

output_payload = {
    "correlations": correlations,
    "group_summary": group_summary,
    "cluster_mwu_val_vs_test": cluster_mwu
}

json_path = os.path.join(REPO, "results", "final_protocol", "08_revision2", "08_dice_by_group.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output_payload, f, indent=2)
print(f"Saved: {json_path}")
