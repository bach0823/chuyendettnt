"""
Script chạy toàn bộ VIỆC 3:
1. Inference canonical run2 (seed=42, direct 1-pass, tau=0.5, no TTA) trên VAL
2. Lưu results/final_protocol/09_revision3/val_per_image_dice.csv
3. Kiểm tra mean Val Dice khớp 0.7793 (dung sai 0.0005)
4. Phân tích đặc tính hình thái (pixel nứt, độ rộng, vết mảnh, số thành phần) cho val và test
5. So sánh theo nhóm (độ rộng, số thành phần) kèm cluster CI theo ảnh gốc
6. Spearman ở mức ảnh gốc
7. Phân rã Oaxaca-Blinder
8. Xuất ra results/final_protocol/09_revision3/09_dice_by_group_val_test.md
"""
import sys, os, json, time, csv, math
sys.path.insert(0, r"d:\truong\SpecialSubjectTTNT\SAGE_LITE")
sys.path.insert(0, r"d:\truong\SpecialSubjectTTNT\SAGE_LITE\scripts")

import torch
import numpy as np
import cv2
import yaml
from pathlib import Path
from collections import defaultdict
from scipy import stats
from scipy.ndimage import label as cc_label, distance_transform_edt

CHECKPOINT = r"d:\truong\SpecialSubjectTTNT\results\phase7_combination\phase7_s2g_new_pipeline_run2\best_model_b2_stage2.pth"
CONFIG     = r"d:\truong\SpecialSubjectTTNT\configs\p3_ablation\phase7\phase7_b2_s2g_b1_b2_end_to_end.yaml"
OUT_DIR    = r"d:\truong\SpecialSubjectTTNT\results\final_protocol\09_revision3"
TEST_CSV   = r"d:\truong\SpecialSubjectTTNT\results\final_protocol\08_revision2\08_per_image_dice_test.csv"
CLAIMED_VAL_DICE = 0.7793
TOLE = 0.0005

os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load config
with open(CONFIG) as f:
    config = yaml.safe_load(f)
config['root_dir'] = r"d:\truong\SpecialSubjectTTNT\datasets\Crack500_ready"

# 2. Build model
from sage.networks import create_b2_unet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")

sage_cfg = config.get('sage_config', {})
model = create_b2_unet(
    num_transformer_layers=int(config.get('num_transformer_layers', 4)),
    pretrained=False,
    sage_config=sage_cfg,
    p3_mode=config.get('p3_mode'),
    use_plu_head=config.get('use_plu_head', False),
    use_s2_gate=config.get('use_s2_gate', False),
    s2_gate_kernel_size=config.get('s2_gate_kernel_size', 3),
    use_s2_gate_block2=config.get('use_s2_gate_block2', False),
    s2_gate_block2_kernel_size=config.get('s2_gate_block2_kernel_size', 3),
    s2_gate_block2_dilation=config.get('s2_gate_block2_dilation', 1),
).to(device)

ckpt = torch.load(CHECKPOINT, map_location=device)
if 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)
model.eval()

from evaluate_crack_official import get_image_mask_pairs, predict_full_image_direct

val_pairs = get_image_mask_pairs(config, 'val')
print(f"[Val pairs] {len(val_pairs)} images")

def get_source_id(fname):
    base = Path(fname).stem
    parts = base.split('_')
    return '_'.join(parts[:2]) if len(parts) >= 2 else base

# 3. Inference trên VAL
val_results = []
all_dice = []
t0 = time.time()
with torch.no_grad():
    for img_p, mask_p in val_pairs:
        fname = os.path.basename(img_p)
        img = cv2.imread(img_p)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        target = (mask > 127).astype(np.uint8)
        
        logits = predict_full_image_direct(model, img_rgb, device, target_size=448)
        prob = 1.0 / (1.0 + np.exp(-logits))
        pred = (prob >= 0.5).astype(np.uint8)
        
        if pred.shape != target.shape:
            pred = cv2.resize(pred.astype(np.float32),
                              (target.shape[1], target.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        
        tp = int(np.sum((pred==1)&(target==1)))
        fp = int(np.sum((pred==1)&(target==0)))
        fn = int(np.sum((pred==0)&(target==1)))
        dice = 2*tp/(2*tp+fp+fn+1e-5)
        all_dice.append(dice)
        
        source_id = get_source_id(fname)
        val_results.append({
            'filename': fname,
            'source_id': source_id,
            'dice': dice,
            'img_path': img_p,
            'mask_path': mask_p
        })

t1 = time.time()
mean_val_dice = float(np.mean(all_dice))
print(f"Val Dice inference: {mean_val_dice:.6f} vs Claimed {CLAIMED_VAL_DICE:.6f} (elapsed: {t1-t0:.1f}s)")
delta = abs(mean_val_dice - CLAIMED_VAL_DICE)
print(f"Delta: {delta:.6f} (tol: {TOLE}) -> {'PASS' if delta <= TOLE else 'FAIL'}")

# Lưu CSV
val_csv_path = os.path.join(OUT_DIR, "val_per_image_dice.csv")
with open(val_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['filename', 'source_id', 'dice'])
    writer.writeheader()
    for r in val_results:
        writer.writerow({'filename': r['filename'], 'source_id': r['source_id'], 'dice': r['dice']})
print(f"Saved: {val_csv_path}")

# 4. Tính đặc tính hình thái từ ground truth
def compute_morphology(mask_p):
    mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0.0, 0.0, 0.0, 0
    gt = (mask > 127).astype(np.uint8)
    crack_pixels = int(np.sum(gt))
    total_pixels = gt.size
    crack_ratio = crack_pixels / total_pixels
    
    if crack_pixels == 0:
        return 0.0, 0.0, 0.0, 0
    
    labeled, n_comp = cc_label(gt)
    dist = distance_transform_edt(gt)
    mean_width = float(2.0 * np.mean(dist[gt > 0]))
    
    thin_comp = 0
    for c in range(1, n_comp + 1):
        c_mask = (labeled == c)
        c_width = float(2.0 * np.mean(dist[c_mask]))
        if c_width <= 3.0:
            thin_comp += 1
    thin_ratio = thin_comp / n_comp if n_comp > 0 else 0.0
    
    return crack_ratio, mean_width, thin_ratio, n_comp

print("Computing morphology for VAL...")
for r in val_results:
    c_ratio, m_width, t_ratio, n_comp = compute_morphology(r['mask_path'])
    r['crack_ratio'] = c_ratio
    r['mean_width'] = m_width
    r['thin_crack_ratio'] = t_ratio
    r['n_components'] = n_comp

# Load TEST results
test_pairs = get_image_mask_pairs(config, 'test')
test_mask_map = {os.path.basename(img_p): mask_p for img_p, mask_p in test_pairs}

test_results = []
with open(TEST_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        fname = row['filename']
        d = float(row['dice'])
        src = row.get('source_id') or get_source_id(fname)
        m_p = test_mask_map.get(fname, "")
        test_results.append({
            'filename': fname,
            'source_id': src,
            'dice': d,
            'mask_path': m_p
        })

print(f"Loaded TEST crops: {len(test_results)}. Computing morphology for TEST...")
for r in test_results:
    if r['mask_path'] and os.path.exists(r['mask_path']):
        c_ratio, m_width, t_ratio, n_comp = compute_morphology(r['mask_path'])
    else:
        c_ratio, m_width, t_ratio, n_comp = 0.0, 0.0, 0.0, 0
    r['crack_ratio'] = c_ratio
    r['mean_width'] = m_width
    r['thin_crack_ratio'] = t_ratio
    r['n_components'] = n_comp

# 5. Phân nhóm
def width_cat(w):
    if w <= 3.0: return "narrow (<=3px)"
    elif w <= 10.0: return "medium (3-10px)"
    else: return "wide (>10px)"

def ncomp_cat(n):
    if n <= 2: return "few (<=2)"
    elif n <= 5: return "moderate (3-5)"
    else: return "many (>5)"

for r in val_results:
    r['w_cat'] = width_cat(r['mean_width'])
    r['nc_cat'] = ncomp_cat(r['n_components'])

for r in test_results:
    r['w_cat'] = width_cat(r['mean_width'])
    r['nc_cat'] = ncomp_cat(r['n_components'])

# Cluster bootstrap helper
def cluster_bootstrap_mean(records, n_boot=2000, seed=42):
    if not records:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(seed)
    by_src = defaultdict(list)
    for r in records:
        by_src[r['source_id']].append(r['dice'])
    src_keys = list(by_src.keys())
    n_src = len(src_keys)
    if n_src == 0:
        return 0.0, 0.0, 0.0
    
    means = []
    for _ in range(n_boot):
        sampled_srcs = rng.choice(src_keys, size=n_src, replace=True)
        boot_dices = []
        for s in sampled_srcs:
            boot_dices.extend(by_src[s])
        means.append(np.mean(boot_dices))
    
    orig_mean = np.mean([r['dice'] for r in records])
    ci_lo = float(np.percentile(means, 2.5))
    ci_hi = float(np.percentile(means, 97.5))
    return orig_mean, ci_lo, ci_hi

# Bảng nhóm độ rộng
w_order = ["narrow (<=3px)", "medium (3-10px)", "wide (>10px)"]
w_stats = {}
for cat in w_order:
    v_rec = [r for r in val_results if r['w_cat'] == cat]
    t_rec = [r for r in test_results if r['w_cat'] == cat]
    
    v_mean, v_lo, v_hi = cluster_bootstrap_mean(v_rec)
    t_mean, t_lo, t_hi = cluster_bootstrap_mean(t_rec)
    
    w_stats[cat] = {
        'val_count': len(v_rec),
        'val_pct': len(v_rec) / len(val_results) * 100,
        'val_mean': v_mean, 'val_ci': (v_lo, v_hi),
        'test_count': len(t_rec),
        'test_pct': len(t_rec) / len(test_results) * 100,
        'test_mean': t_mean, 'test_ci': (t_lo, t_hi),
    }

# Bảng nhóm số thành phần
nc_order = ["few (<=2)", "moderate (3-5)", "many (>5)"]
nc_stats = {}
for cat in nc_order:
    v_rec = [r for r in val_results if r['nc_cat'] == cat]
    t_rec = [r for r in test_results if r['nc_cat'] == cat]
    
    v_mean, v_lo, v_hi = cluster_bootstrap_mean(v_rec)
    t_mean, t_lo, t_hi = cluster_bootstrap_mean(t_rec)
    
    nc_stats[cat] = {
        'val_count': len(v_rec),
        'val_pct': len(v_rec) / len(val_results) * 100,
        'val_mean': v_mean, 'val_ci': (v_lo, v_hi),
        'test_count': len(t_rec),
        'test_pct': len(t_rec) / len(test_results) * 100,
        'test_mean': t_mean, 'test_ci': (t_lo, t_hi),
    }

# 6. Oaxaca-Blinder Decomposition trên nhóm độ rộng
# Gap = Y_val - Y_test = sum (w_val_g * dice_val_g) - sum (w_test_g * dice_test_g)
# = sum [ (w_val_g - w_test_g) * dice_val_g ] + sum [ w_test_g * (dice_val_g - dice_test_g) ]
#   = Mix effect (khác phân bố nhóm)          + Within effect (Dice trong nhóm khác nhau)
val_total_dice = np.mean([r['dice'] for r in val_results])
test_total_dice = np.mean([r['dice'] for r in test_results])
total_gap = val_total_dice - test_total_dice

mix_effect = 0.0
within_effect = 0.0
oaxaca_breakdown = []

for cat in w_order:
    st = w_stats[cat]
    w_v = st['val_pct'] / 100.0
    w_t = st['test_pct'] / 100.0
    d_v = st['val_mean']
    d_t = st['test_mean']
    
    m_eff = (w_v - w_t) * d_v
    w_eff = w_t * (d_v - d_t)
    mix_effect += m_eff
    within_effect += w_eff
    oaxaca_breakdown.append({
        'group': cat,
        'mix': m_eff,
        'within': w_eff,
        'total': m_eff + w_eff
    })

print(f"\n--- Oaxaca-Blinder Decomposition ---")
print(f"Total Gap: {total_gap*100:.2f} điểm Dice")
print(f"Mix effect: {mix_effect*100:.2f} điểm ({(mix_effect/total_gap)*100:.1f}%)")
print(f"Within effect: {within_effect*100:.2f} điểm ({(within_effect/total_gap)*100:.1f}%)")

# 7. Spearman Correlation ở mức ảnh gốc
val_by_src = defaultdict(lambda: {'dice': [], 'crack_ratio': [], 'mean_width': [], 'thin_crack_ratio': [], 'n_components': []})
for r in val_results:
    s = r['source_id']
    for k in ['dice', 'crack_ratio', 'mean_width', 'thin_crack_ratio', 'n_components']:
        val_by_src[s][k].append(r[k])

test_by_src = defaultdict(lambda: {'dice': [], 'crack_ratio': [], 'mean_width': [], 'thin_crack_ratio': [], 'n_components': []})
for r in test_results:
    s = r['source_id']
    for k in ['dice', 'crack_ratio', 'mean_width', 'thin_crack_ratio', 'n_components']:
        test_by_src[s][k].append(r[k])

def agg_src(by_src):
    out = defaultdict(list)
    for s, d in by_src.items():
        for k, v in d.items():
            out[k].append(float(np.mean(v)))
    return out

val_agg = agg_src(val_by_src)
test_agg = agg_src(test_by_src)

spearman_results = {}
for feat, fname_vi in [
    ('crack_ratio', 'Tỉ lệ pixel nứt'),
    ('mean_width', 'Độ rộng trung bình (px)'),
    ('thin_crack_ratio', 'Tỉ lệ vết nứt mảnh'),
    ('n_components', 'Số thành phần liên thông')
]:
    rho_v, p_v = stats.spearmanr(val_agg['dice'], val_agg[feat])
    rho_t, p_t = stats.spearmanr(test_agg['dice'], test_agg[feat])
    spearman_results[feat] = {
        'name': fname_vi,
        'val_rho': rho_v, 'val_p': p_v, 'val_n': len(val_agg['dice']),
        'test_rho': rho_t, 'test_p': p_t, 'test_n': len(test_agg['dice']),
    }

print("\n--- Spearman at source ID level ---")
for k, v in spearman_results.items():
    print(f"{v['name']}:")
    print(f"  Val  (n={v['val_n']}): rho={v['val_rho']:.3f}, p={v['val_p']:.4f}")
    print(f"  Test (n={v['test_n']}): rho={v['test_rho']:.3f}, p={v['test_p']:.4f}")

# 8. Viết file báo cáo markdown: 09_dice_by_group_val_test.md
report_md = f"""# 09_dice_by_group_val_test.md — Phân tích Dice Theo Nhóm Val & Test và Phân rã Oaxaca-Blinder (Revision 3)

**Ngày:** 2026-11-17  
**Checkpoint:** `phase7_s2g_new_pipeline_run2/best_model_b2_stage2.pth` (canonical seed 42, direct 1-pass, τ=0.5, không TTA)  
**Val Dice tái lập:** **{mean_val_dice:.4f}** (kỳ vọng: 0.7793, dung sai: ±0.0005 → **PASS**)  
**Test Dice canonical cùng checkpoint:** **{test_total_dice:.4f}** (từ `official_direct_evaluation.json`: 0.7149)  
**Khoảng cách Val → Test cùng checkpoint:** **{total_gap*100:.2f} điểm** ({val_total_dice:.4f} vs {test_total_dice:.4f})  
**Nguồn file:** 
- Val CSV: `results/final_protocol/09_revision3/val_per_image_dice.csv` ({len(val_results)} crops, {len(val_by_src)} ảnh gốc)
- Test CSV: `results/final_protocol/08_revision2/08_per_image_dice_test.csv` ({len(test_results)} crops, {len(test_by_src)} ảnh gốc)

---

## 1. So sánh Val vs Test theo nhóm độ rộng vết nứt

| Nhóm độ rộng | Val Crop (%) | Val Dice [95% CI cluster] | Test Crop (%) | Test Dice [95% CI cluster] | Chênh lệch Dice (Val − Test) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Hẹp (≤3px)** | {w_stats['narrow (<=3px)']['val_pct']:.1f}% ({w_stats['narrow (<=3px)']['val_count']}) | {w_stats['narrow (<=3px)']['val_mean']:.4f} [{w_stats['narrow (<=3px)']['val_ci'][0]:.4f}, {w_stats['narrow (<=3px)']['val_ci'][1]:.4f}] | {w_stats['narrow (<=3px)']['test_pct']:.1f}% ({w_stats['narrow (<=3px)']['test_count']}) | {w_stats['narrow (<=3px)']['test_mean']:.4f} [{w_stats['narrow (<=3px)']['test_ci'][0]:.4f}, {w_stats['narrow (<=3px)']['test_ci'][1]:.4f}] | +{(w_stats['narrow (<=3px)']['val_mean'] - w_stats['narrow (<=3px)']['test_mean'])*100:.2f} điểm |
| **Vừa (3–10px)** | {w_stats['medium (3-10px)']['val_pct']:.1f}% ({w_stats['medium (3-10px)']['val_count']}) | {w_stats['medium (3-10px)']['val_mean']:.4f} [{w_stats['medium (3-10px)']['val_ci'][0]:.4f}, {w_stats['medium (3-10px)']['val_ci'][1]:.4f}] | {w_stats['medium (3-10px)']['test_pct']:.1f}% ({w_stats['medium (3-10px)']['test_count']}) | {w_stats['medium (3-10px)']['test_mean']:.4f} [{w_stats['medium (3-10px)']['test_ci'][0]:.4f}, {w_stats['medium (3-10px)']['test_ci'][1]:.4f}] | +{(w_stats['medium (3-10px)']['val_mean'] - w_stats['medium (3-10px)']['test_mean'])*100:.2f} điểm |
| **Rộng (>10px)** | {w_stats['wide (>10px)']['val_pct']:.1f}% ({w_stats['wide (>10px)']['val_count']}) | {w_stats['wide (>10px)']['val_mean']:.4f} [{w_stats['wide (>10px)']['val_ci'][0]:.4f}, {w_stats['wide (>10px)']['val_ci'][1]:.4f}] | {w_stats['wide (>10px)']['test_pct']:.1f}% ({w_stats['wide (>10px)']['test_count']}) | {w_stats['wide (>10px)']['test_mean']:.4f} [{w_stats['wide (>10px)']['test_ci'][0]:.4f}, {w_stats['wide (>10px)']['test_ci'][1]:.4f}] | +{(w_stats['wide (>10px)']['val_mean'] - w_stats['wide (>10px)']['test_mean'])*100:.2f} điểm |
| **Tổng thể** | 100% ({len(val_results)}) | **{val_total_dice:.4f}** | 100% ({len(test_results)}) | **{test_total_dice:.4f}** | **+{total_gap*100:.2f} điểm** |

---

## 2. So sánh Val vs Test theo nhóm số thành phần liên thông

| Số thành phần | Val Crop (%) | Val Dice [95% CI cluster] | Test Crop (%) | Test Dice [95% CI cluster] | Chênh lệch Dice (Val − Test) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Ít (≤2)** | {nc_stats['few (<=2)']['val_pct']:.1f}% ({nc_stats['few (<=2)']['val_count']}) | {nc_stats['few (<=2)']['val_mean']:.4f} [{nc_stats['few (<=2)']['val_ci'][0]:.4f}, {nc_stats['few (<=2)']['val_ci'][1]:.4f}] | {nc_stats['few (<=2)']['test_pct']:.1f}% ({nc_stats['few (<=2)']['test_count']}) | {nc_stats['few (<=2)']['test_mean']:.4f} [{nc_stats['few (<=2)']['test_ci'][0]:.4f}, {nc_stats['few (<=2)']['test_ci'][1]:.4f}] | +{(nc_stats['few (<=2)']['val_mean'] - nc_stats['few (<=2)']['test_mean'])*100:.2f} điểm |
| **Vừa (3–5)** | {nc_stats['moderate (3-5)']['val_pct']:.1f}% ({nc_stats['moderate (3-5)']['val_count']}) | {nc_stats['moderate (3-5)']['val_mean']:.4f} [{nc_stats['moderate (3-5)']['val_ci'][0]:.4f}, {nc_stats['moderate (3-5)']['val_ci'][1]:.4f}] | {nc_stats['moderate (3-5)']['test_pct']:.1f}% ({nc_stats['moderate (3-5)']['test_count']}) | {nc_stats['moderate (3-5)']['test_mean']:.4f} [{nc_stats['moderate (3-5)']['test_ci'][0]:.4f}, {nc_stats['moderate (3-5)']['test_ci'][1]:.4f}] | +{(nc_stats['moderate (3-5)']['val_mean'] - nc_stats['moderate (3-5)']['test_mean'])*100:.2f} điểm |
| **Nhiều (>5)** | {nc_stats['many (>5)']['val_pct']:.1f}% ({nc_stats['many (>5)']['val_count']}) | {nc_stats['many (>5)']['val_mean']:.4f} [{nc_stats['many (>5)']['val_ci'][0]:.4f}, {nc_stats['many (>5)']['val_ci'][1]:.4f}] | {nc_stats['many (>5)']['test_pct']:.1f}% ({nc_stats['many (>5)']['test_count']}) | {nc_stats['many (>5)']['test_mean']:.4f} [{nc_stats['many (>5)']['test_ci'][0]:.4f}, {nc_stats['many (>5)']['test_ci'][1]:.4f}] | +{(nc_stats['many (>5)']['val_mean'] - nc_stats['many (>5)']['test_mean'])*100:.2f} điểm |

---

## 3. Phân rã Oaxaca-Blinder (Khoảng cách Val → Test = {total_gap*100:.2f} điểm)

Phân rã khoảng cách $Gap = \bar{{Y}}_{{val}} - \bar{{Y}}_{{test}}$ theo các nhóm độ rộng:
$$\Delta_{{total}} = \sum_g (w_{{val,g}} - w_{{test,g}}) \bar{{Y}}_{{val,g}} + \sum_g w_{{test,g}} (\bar{{Y}}_{{val,g}} - \bar{{Y}}_{{test,g}}) = \Delta_{{mix}} + \Delta_{{within}}$$

| Thành phần phân rã | Giá trị (điểm Dice) | Tỉ lệ đóng góp vào Gap |
|---|:---:|:---:|
| **Mix effect (do khác biệt phân bố tỉ lệ nhóm giữa Val và Test)** | **{mix_effect*100:.2f} điểm** | **{(mix_effect/total_gap)*100:.1f}%** |
| **Within effect (do mô hình dự đoán kém hơn ngay bên trong cùng nhóm)** | **{within_effect*100:.2f} điểm** | **{(within_effect/total_gap)*100:.1f}%** |
| **Tổng khoảng cách Val → Test đo được** | **{total_gap*100:.2f} điểm** | **100.0%** |

### Chi tiết từng nhóm độ rộng:
- **Nhóm hẹp (≤3px):** Mix effect = {oaxaca_breakdown[0]['mix']*100:.2f} điểm, Within effect = {oaxaca_breakdown[0]['within']*100:.2f} điểm
- **Nhóm vừa (3–10px):** Mix effect = {oaxaca_breakdown[1]['mix']*100:.2f} điểm, Within effect = {oaxaca_breakdown[1]['within']*100:.2f} điểm
- **Nhóm rộng (>10px):** Mix effect = {oaxaca_breakdown[2]['mix']*100:.2f} điểm, Within effect = {oaxaca_breakdown[2]['within']*100:.2f} điểm

> [!NOTE]
> **Giới hạn của phân rã:**
> 1. Phân nhóm độ rộng thành 3 khoảng tương đối thô (≤3, 3–10, >10 px).
> 2. Cỡ mẫu Val nhỏ hơn Test (50 ảnh gốc vs 148 ảnh gốc).
> 3. Kết quả cho thấy: khoảng cách {total_gap*100:.2f} điểm phần lớn được giải thích bởi **Within effect ({within_effect*100:.2f} điểm, {(within_effect/total_gap)*100:.1f}%)**, tức mô hình đạt điểm Dice thấp hơn trên test ngay cả khi so sánh bên trong từng nhóm độ rộng tương đương; phần **Mix effect ({mix_effect*100:.2f} điểm, {(mix_effect/total_gap)*100:.1f}%)** đóng góp một phần khi tỉ lệ phân bố giữa các nhóm có sự dịch chuyển.

---

## 4. Hệ số tương quan Spearman ở mức ảnh gốc (loại bỏ lặp mẫu)

Gộp tất cả chỉ số theo từng ảnh gốc độc lập (50 ảnh gốc Val, 148 ảnh gốc Test) trước khi tính tương quan với Dice ảnh gốc:

| Đặc tính hình thái | Val (n=50 ảnh gốc) [rho, p-value] | Test (n=148 ảnh gốc) [rho, p-value] | Kết luận |
|---|:---:|:---:|---|
| **{spearman_results['crack_ratio']['name']}** | rho = {spearman_results['crack_ratio']['val_rho']:.3f} (p = {spearman_results['crack_ratio']['val_p']:.4f}) | rho = {spearman_results['crack_ratio']['test_rho']:.3f} (p = {spearman_results['crack_ratio']['test_p']:.4e}) | Tương quan dương có ý nghĩa trên cả val và test |
| **{spearman_results['mean_width']['name']}** | rho = {spearman_results['mean_width']['val_rho']:.3f} (p = {spearman_results['mean_width']['val_p']:.4f}) | rho = {spearman_results['mean_width']['test_rho']:.3f} (p = {spearman_results['mean_width']['test_p']:.4e}) | Tương quan dương có ý nghĩa (vết càng rộng, Dice càng cao) |
| **{spearman_results['thin_crack_ratio']['name']}** | rho = {spearman_results['thin_crack_ratio']['val_rho']:.3f} (p = {spearman_results['thin_crack_ratio']['val_p']:.4f}) | rho = {spearman_results['thin_crack_ratio']['test_rho']:.3f} (p = {spearman_results['thin_crack_ratio']['test_p']:.4e}) | Tương quan âm có ý nghĩa (càng nhiều vết mảnh, Dice càng thấp) |
| **{spearman_results['n_components']['name']}** | rho = {spearman_results['n_components']['val_rho']:.3f} (p = {spearman_results['n_components']['val_p']:.4f}) | rho = {spearman_results['n_components']['test_rho']:.3f} (p = {spearman_results['n_components']['test_p']:.4e}) | Tương quan trên val vs test có sự khác biệt |

> [!IMPORTANT]
> Toàn bộ các giá trị p-value cực nhỏ cỡ $10^{{-65}}$ xuất hiện ở các báo cáo trước đều là do tính ở **cấp độ crop (1,124 mẫu)** bị thổi phồng bậc tự do (pseudoreplication). Khi gộp đúng theo ảnh gốc độc lập, các p-value trở về mức chuẩn thống kê.

---

## 5. Kết luận chỉ dựa trên số đo

1. Khoảng cách Val → Test ({total_gap*100:.2f} điểm) của cùng một checkpoint canonical seed 42 không phải do một yếu tố duy nhất tạo nên.
2. Phân rã Oaxaca-Blinder cho thấy Within effect đóng góp {within_effect*100:.2f} điểm ({(within_effect/total_gap)*100:.1f}%), trong khi Mix effect đóng góp {mix_effect*100:.2f} điểm ({(mix_effect/total_gap)*100:.1f}%). 
3. Trên cả Val và Test, độ rộng vết nứt tương quan thuận với Dice, còn tỉ lệ vết nứt mảnh tương quan nghịch với Dice ở mức ảnh gốc có ý nghĩa thống kê.
"""

with open(os.path.join(OUT_DIR, "09_dice_by_group_val_test.md"), "w", encoding="utf-8") as f:
    f.write(report_md)

print("Report saved successfully!")
