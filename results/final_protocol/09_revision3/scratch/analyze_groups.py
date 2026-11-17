"""
Phân tích Dice theo nhóm val+test + Oaxaca-Blinder + Spearman mức ảnh gốc.
Chạy sau khi val_per_image_dice.csv đã có.
"""
import sys, os, json, csv, math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats
from scipy.ndimage import label as cc_label
from PIL import Image

RESULTS_DIR = r'd:\truong\SpecialSubjectTTNT\results\final_protocol\09_revision3'
VAL_CSV = os.path.join(RESULTS_DIR, 'val_per_image_dice.csv')
TEST_CSV = r'd:\truong\SpecialSubjectTTNT\results\final_protocol\08_revision2\08_per_image_dice_test.csv'
MASK_VAL_DIR = r'd:\truong\SpecialSubjectTTNT\SAGE_LITE\data\Crack500\val\masks'
MASK_TEST_DIR = r'd:\truong\SpecialSubjectTTNT\SAGE_LITE\data\Crack500\test\masks'

# ---- Load dice CSVs ----
print("Loading CSVs...")
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)
print(f"Val crops: {len(val_df)}, Test crops: {len(test_df)}")

# Source ID extraction
def get_source_id(fname):
    base = Path(fname).stem
    parts = base.split('_')
    return '_'.join(parts[:2]) if len(parts) >= 2 else base

if 'source_id' not in val_df.columns:
    val_df['source_id'] = val_df['filename'].apply(get_source_id)
if 'source_id' not in test_df.columns:
    test_df['source_id'] = test_df['filename'].apply(get_source_id)

print(f"Val source IDs: {val_df['source_id'].nunique()}, Test source IDs: {test_df['source_id'].nunique()}")

# ---- Compute morphological features from masks ----
def compute_features(fname, mask_dir):
    """Compute crack morphology features from ground truth mask."""
    mask_path = os.path.join(mask_dir, fname.replace('.jpg', '.png'))
    if not os.path.exists(mask_path):
        mask_path = os.path.join(mask_dir, fname)
    if not os.path.exists(mask_path):
        return None
    
    mask = np.array(Image.open(mask_path).convert('L'))
    gt = (mask > 127).astype(np.uint8)
    
    total_pixels = gt.size
    crack_pixels = gt.sum()
    crack_ratio = crack_pixels / total_pixels
    
    if crack_pixels == 0:
        return {'crack_ratio': 0, 'mean_width': 0, 'thin_crack_ratio': 0, 'n_components': 0}
    
    # Connected components
    labeled, n_comp = cc_label(gt)
    
    # Width estimation via vertical/horizontal run-length
    # Simple: mean distance transform ~ width/2
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(gt)
    mean_width = 2 * dist[gt > 0].mean()  # diameter ≈ 2 * mean radius
    
    # Thin crack ratio: components with mean width <= 3
    thin_comp = 0
    for c in range(1, n_comp+1):
        comp_mask = (labeled == c)
        comp_dist = dist[comp_mask]
        comp_width = 2 * comp_dist.mean()
        if comp_width <= 3:
            thin_comp += 1
    thin_ratio = thin_comp / n_comp if n_comp > 0 else 0
    
    return {
        'crack_ratio': crack_ratio,
        'mean_width': mean_width,
        'thin_crack_ratio': thin_ratio,
        'n_components': n_comp
    }

print("Computing val features...")
val_feats = []
for idx, row in val_df.iterrows():
    f = compute_features(row['filename'], MASK_VAL_DIR)
    if f:
        val_feats.append({**f, 'filename': row['filename'], 'dice': row['dice'], 'source_id': row['source_id']})
    if (idx+1) % 50 == 0:
        print(f"  Val {idx+1}/{len(val_df)}")

val_feat_df = pd.DataFrame(val_feats)
print(f"Val feats: {len(val_feat_df)}")

print("Computing test features...")
test_feats = []
for idx, row in test_df.iterrows():
    f = compute_features(row['filename'], MASK_TEST_DIR)
    if f:
        test_feats.append({**f, 'filename': row['filename'], 'dice': row['dice'], 'source_id': row['source_id']})
    if (idx+1) % 100 == 0:
        print(f"  Test {idx+1}/{len(test_df)}")

test_feat_df = pd.DataFrame(test_feats)
print(f"Test feats: {len(test_feat_df)}")

# ---- Width groups ----
def width_group(w):
    if w <= 3: return 'narrow (≤3px)'
    elif w <= 10: return 'medium (3-10px)'
    else: return 'wide (>10px)'

def ncomp_group(n):
    if n <= 2: return 'few (≤2)'
    elif n <= 5: return 'moderate (3-5)'
    else: return 'many (>5)'

val_feat_df['width_group'] = val_feat_df['mean_width'].apply(width_group)
test_feat_df['width_group'] = test_feat_df['mean_width'].apply(width_group)
val_feat_df['ncomp_group'] = val_feat_df['n_components'].apply(ncomp_group)
test_feat_df['ncomp_group'] = test_feat_df['n_components'].apply(ncomp_group)

# ---- Group analysis ----
groups = ['narrow (≤3px)', 'medium (3-10px)', 'wide (>10px)']
print("\n=== Width Group Analysis ===")
print(f"{'Group':<20} {'Val count':>10} {'Val %':>8} {'Val Dice':>10} {'Test count':>12} {'Test %':>8} {'Test Dice':>10}")
for g in groups:
    v = val_feat_df[val_feat_df['width_group']==g]
    t = test_feat_df[test_feat_df['width_group']==g]
    pct_v = len(v)/len(val_feat_df)*100
    pct_t = len(t)/len(test_feat_df)*100
    dv = v['dice'].mean() if len(v)>0 else float('nan')
    dt = t['dice'].mean() if len(t)>0 else float('nan')
    print(f"{g:<20} {len(v):>10} {pct_v:>7.1f}% {dv:>10.4f} {len(t):>12} {pct_t:>7.1f}% {dt:>10.4f}")

# ---- Oaxaca-Blinder decomposition on width groups ----
print("\n=== Oaxaca-Blinder Decomposition (Width Groups) ===")
overall_val = val_feat_df['dice'].mean()
overall_test = test_feat_df['dice'].mean()
print(f"Overall val dice: {overall_val:.4f}, test dice: {overall_test:.4f}")
print(f"Gap (val - test): {(overall_val - overall_test)*100:.2f} điểm")

mix_effect = 0.0
within_effect = 0.0
for g in groups:
    v = val_feat_df[val_feat_df['width_group']==g]
    t = test_feat_df[test_feat_df['width_group']==g]
    w_val = len(v)/len(val_feat_df)
    w_test = len(t)/len(test_feat_df)
    d_val = v['dice'].mean() if len(v)>0 else 0
    d_test = t['dice'].mean() if len(t)>0 else 0
    
    # Mix effect: (w_test - w_val) * d_val
    mix_g = (w_test - w_val) * d_val
    # Within effect: w_test * (d_test - d_val)
    within_g = w_test * (d_test - d_val) if len(t)>0 and len(v)>0 else 0
    mix_effect += mix_g
    within_effect += within_g
    print(f"  {g}: mix={mix_g*100:.3f}đ, within={within_g*100:.3f}đ")

total_oaxaca = mix_effect + within_effect
print(f"\nTotal decomposition: {total_oaxaca*100:.3f} điểm (val-test gap)")
print(f"  Mix effect (phân bố nhóm): {mix_effect*100:.3f} điểm ({mix_effect/total_oaxaca*100:.1f}% nếu khác dấu có thể xảy ra)")
print(f"  Within effect (Dice trong nhóm): {within_effect*100:.3f} điểm")
print(f"  Actual gap: {(overall_val - overall_test)*100:.3f} điểm")

# ---- Spearman correlation at source ID level ----
print("\n=== Spearman at source-ID level ===")
val_by_src = val_feat_df.groupby('source_id').agg({'dice':'mean','crack_ratio':'mean','mean_width':'mean','n_components':'mean'}).reset_index()
test_by_src = test_feat_df.groupby('source_id').agg({'dice':'mean','crack_ratio':'mean','mean_width':'mean','n_components':'mean'}).reset_index()

for feat in ['crack_ratio','mean_width','n_components']:
    for split, df in [('val', val_by_src), ('test', test_by_src)]:
        rho, pval = stats.spearmanr(df['dice'], df[feat])
        print(f"  {split} {feat}: rho={rho:.3f}, p={pval:.4f} (n={len(df)})")

# ---- Save summary ----
summary = {
    'val_n_crops': len(val_feat_df),
    'test_n_crops': len(test_feat_df),
    'val_mean_dice': float(overall_val),
    'test_mean_dice': float(overall_test),
    'gap_points': float((overall_val - overall_test)*100),
    'oaxaca_mix_points': float(mix_effect*100),
    'oaxaca_within_points': float(within_effect*100),
}
with open(os.path.join(RESULTS_DIR, 'val_analysis_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {os.path.join(RESULTS_DIR, 'val_analysis_summary.json')}")
print("DONE")
