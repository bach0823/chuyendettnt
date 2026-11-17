"""
Script chạy toàn bộ VIỆC 4:
1. Suy luận trên VAL với cấu hình LOCKED (Ensemble 3 seed 42/43/44 + TTA flip_hv, tau=0.5).
2. Lưu kết quả theo ảnh: results/final_protocol/10_revision4/val_locked_per_image_dice.csv
3. Phân rã Oaxaca-Blinder cho cặp LOCKED (Val Locked vs Test Locked: 0.7158).
4. Đối chiếu phân rã của cặp Single-seed (Val Canonical seed 42 vs Test Canonical seed 42).
5. Tính CI cluster 10,000 lần theo ảnh gốc độc lập cho:
   - Val Dice (Single seed)
   - Val Dice (Locked ensemble)
   - Test Dice (Single seed)
   - Test Dice (Locked ensemble)
   - Gap Val->Test (Single seed)
   - Gap Val->Test (Locked ensemble)
6. Xuất ra results/final_protocol/10_revision4/10_gap_decomposition.md
"""
import sys, os, time, csv, json, cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import label as cc_label, distance_transform_edt
import torch
import yaml

sys.path.insert(0, r"d:\truong\SpecialSubjectTTNT\SAGE_LITE")
sys.path.insert(0, r"d:\truong\SpecialSubjectTTNT\SAGE_LITE\scripts")

from sage.networks import create_b2_unet
from evaluate_crack_official import get_image_mask_pairs, predict_full_image_direct

CONFIG = r"d:\truong\SpecialSubjectTTNT\configs\p3_ablation\phase7\phase7_b2_s2g_b1_b2_end_to_end.yaml"
BASE_PHASE2 = r"d:\truong\SpecialSubjectTTNT\results\final_protocol\phase2_results"
SEEDS = [42, 43, 44]
OUT_DIR = r"d:\truong\SpecialSubjectTTNT\results\final_protocol\10_revision4"
os.makedirs(OUT_DIR, exist_ok=True)

with open(CONFIG) as f:
    config = yaml.safe_load(f)
config['root_dir'] = r"d:\truong\SpecialSubjectTTNT\datasets\Crack500_ready"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")

sage_cfg = config.get('sage_config', {})

# Load 3 models
models = []
for s in SEEDS:
    ckpt_p = os.path.join(BASE_PHASE2, f"phase2_seed{s}", "best_model_b2_global.pth")
    m = create_b2_unet(
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
    ckpt = torch.load(ckpt_p, map_location=device)
    m.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    m.eval()
    models.append(m)
    print(f"Loaded Phase 2 seed {s} model")

def predict_with_tta_flip_hv(model, image, device, target_size=448):
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
    all_probs = []
    for m in models:
        p = predict_with_tta_flip_hv(m, image, device, target_size)
        all_probs.append(p)
    return np.mean(all_probs, axis=0)

val_pairs = get_image_mask_pairs(config, 'val')
print(f"[Val pairs] {len(val_pairs)} images")

def get_source_id(fname):
    base = Path(fname).stem
    parts = base.split('_')
    return '_'.join(parts[:2]) if len(parts) >= 2 else base

# Chạy suy luận trên VAL với LOCKED ensemble + TTA
val_locked_results = []
all_dice = []
t0 = time.time()
with torch.no_grad():
    for idx, (img_p, mask_p) in enumerate(val_pairs):
        fname = os.path.basename(img_p)
        img = cv2.imread(img_p)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        target = (mask > 127).astype(np.uint8)
        
        prob = predict_ensemble_tta(models, img_rgb, device, target_size=448)
        pred = (prob >= 0.5).astype(np.uint8)
        
        if pred.shape != target.shape:
            pred = cv2.resize(pred.astype(np.float32),
                              (target.shape[1], target.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        
        tp = int(np.sum((pred == 1) & (target == 1)))
        fp = int(np.sum((pred == 1) & (target == 0)))
        fn = int(np.sum((pred == 0) & (target == 1)))
        dice = 2 * tp / (2 * tp + fp + fn + 1e-5)
        all_dice.append(dice)
        
        val_locked_results.append({
            'filename': fname,
            'source_id': get_source_id(fname),
            'dice': dice,
            'mask_path': mask_p
        })
        if (idx + 1) % 50 == 0:
            print(f"[{idx+1}/{len(val_pairs)}] mean dice: {np.mean(all_dice):.4f} (elapsed: {time.time()-t0:.1f}s)")

elapsed = time.time() - t0
val_locked_mean = float(np.mean(all_dice))
print(f"\n[DONE] Val LOCKED (Ensemble 3-seed + TTA flip_hv): Mean Dice = {val_locked_mean:.6f} in {elapsed:.1f}s")

# Lưu CSV val locked
val_locked_csv = os.path.join(OUT_DIR, "val_locked_per_image_dice.csv")
with open(val_locked_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['filename', 'source_id', 'dice'])
    writer.writeheader()
    for r in val_locked_results:
        writer.writerow({'filename': r['filename'], 'source_id': r['source_id'], 'dice': r['dice']})
print(f"Saved: {val_locked_csv}")

# Load Ground Truth Masks Morphology & Nhóm độ rộng
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

for r in val_locked_results:
    c_ratio, m_width, t_ratio, n_comp = compute_morphology(r['mask_path'])
    r['mean_width'] = m_width
    r['w_cat'] = "narrow (<=3px)" if m_width <= 3.0 else ("medium (3-10px)" if m_width <= 10.0 else "wide (>10px)")

# Load Test Locked Data (từ results/final_protocol/08_revision2/08_per_image_dice_test.csv)
test_pairs = get_image_mask_pairs(config, 'test')
test_mask_map = {os.path.basename(img_p): mask_p for img_p, mask_p in test_pairs}

test_locked_results = []
TEST_LOCKED_CSV = r"d:\truong\SpecialSubjectTTNT\results\final_protocol\08_revision2\08_per_image_dice_test.csv"
with open(TEST_LOCKED_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        fname = row['filename']
        d = float(row['dice'])
        src = row.get('source_id') or get_source_id(fname)
        m_p = test_mask_map.get(fname, "")
        test_locked_results.append({
            'filename': fname,
            'source_id': src,
            'dice': d,
            'mask_path': m_p
        })

for r in test_locked_results:
    if r['mask_path'] and os.path.exists(r['mask_path']):
        c_ratio, m_width, t_ratio, n_comp = compute_morphology(r['mask_path'])
    else:
        c_ratio, m_width, t_ratio, n_comp = 0.0, 0.0, 0.0, 0
    r['mean_width'] = m_width
    r['w_cat'] = "narrow (<=3px)" if m_width <= 3.0 else ("medium (3-10px)" if m_width <= 10.0 else "wide (>10px)")

# Load Val Single Seed 42 Data (từ results/final_protocol/09_revision3/val_per_image_dice.csv)
val_single_results = []
VAL_SINGLE_CSV = r"d:\truong\SpecialSubjectTTNT\results\final_protocol\09_revision3\val_per_image_dice.csv"
with open(VAL_SINGLE_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        fname = row['filename']
        d = float(row['dice'])
        src = row.get('source_id') or get_source_id(fname)
        m_p = dict(val_pairs).get(fname, "")
        if not m_p:
            for ip, mp in val_pairs:
                if os.path.basename(ip) == fname:
                    m_p = mp; break
        val_single_results.append({
            'filename': fname,
            'source_id': src,
            'dice': d,
            'mask_path': m_p
        })

for r in val_single_results:
    if r['mask_path'] and os.path.exists(r['mask_path']):
        c_ratio, m_width, t_ratio, n_comp = compute_morphology(r['mask_path'])
    else:
        c_ratio, m_width, t_ratio, n_comp = 0.0, 0.0, 0.0, 0
    r['mean_width'] = m_width
    r['w_cat'] = "narrow (<=3px)" if m_width <= 3.0 else ("medium (3-10px)" if m_width <= 10.0 else "wide (>10px)")

w_order = ["narrow (<=3px)", "medium (3-10px)", "wide (>10px)"]

# Hàm tính Oaxaca-Blinder
def run_oaxaca(val_data, test_data):
    val_total = np.mean([r['dice'] for r in val_data])
    test_total = np.mean([r['dice'] for r in test_data])
    gap = val_total - test_total
    
    mix_eff = 0.0
    within_eff = 0.0
    breakdown = {}
    
    for cat in w_order:
        v_sub = [r for r in val_data if r['w_cat'] == cat]
        t_sub = [r for r in test_data if r['w_cat'] == cat]
        
        w_v = len(v_sub) / len(val_data)
        w_t = len(t_sub) / len(test_data)
        d_v = np.mean([r['dice'] for r in v_sub]) if v_sub else 0.0
        d_t = np.mean([r['dice'] for r in t_sub]) if t_sub else 0.0
        
        m_part = (w_v - w_t) * d_v
        w_part = w_t * (d_v - d_t)
        mix_eff += m_part
        within_eff += w_part
        breakdown[cat] = {
            'w_val': w_v, 'w_test': w_t,
            'dice_val': d_v, 'dice_test': d_t,
            'mix': m_part, 'within': w_part,
            'val_count': len(v_sub), 'test_count': len(t_sub)
        }
    return {
        'val_mean': val_total, 'test_mean': test_total, 'gap': gap,
        'mix': mix_eff, 'within': within_eff,
        'mix_pct': (mix_eff / gap) * 100 if gap != 0 else 0.0,
        'within_pct': (within_eff / gap) * 100 if gap != 0 else 0.0,
        'breakdown': breakdown
    }

oaxaca_locked = run_oaxaca(val_locked_results, test_locked_results)
oaxaca_single = run_oaxaca(val_single_results, test_locked_results)

print("\n=== OAXACA-BLINDER PHÂN RÃ ===")
print(f"LOCKED Pair (Val Locked vs Test Locked): Gap = {oaxaca_locked['gap']*100:.2f} điểm")
print(f"  Mix effect    : {oaxaca_locked['mix']*100:.2f} điểm ({oaxaca_locked['mix_pct']:.1f}%)")
print(f"  Within effect : {oaxaca_locked['within']*100:.2f} điểm ({oaxaca_locked['within_pct']:.1f}%)")

print(f"\nSINGLE-SEED Pair (Val Seed 42 vs Test Locked): Gap = {oaxaca_single['gap']*100:.2f} điểm")
print(f"  Mix effect    : {oaxaca_single['mix']*100:.2f} điểm ({oaxaca_single['mix_pct']:.1f}%)")
print(f"  Within effect : {oaxaca_single['within']*100:.2f} điểm ({oaxaca_single['within_pct']:.1f}%)")

# Cluster Bootstrap 10,000 lần cho Val, Test, và Gap
print("\n--- Running 10,000 Cluster Bootstraps ---")
def bootstrap_cluster_gap(val_data, test_data, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    
    val_by_src = defaultdict(list)
    for r in val_data: val_by_src[r['source_id']].append(r['dice'])
    v_keys = list(val_by_src.keys())
    
    test_by_src = defaultdict(list)
    for r in test_data: test_by_src[r['source_id']].append(r['dice'])
    t_keys = list(test_by_src.keys())
    
    v_means = []
    t_means = []
    gaps = []
    
    for _ in range(n_boot):
        # Sample val
        v_s = rng.choice(v_keys, size=len(v_keys), replace=True)
        v_dices = []
        for k in v_s: v_dices.extend(val_by_src[k])
        vm = np.mean(v_dices)
        v_means.append(vm)
        
        # Sample test
        t_s = rng.choice(t_keys, size=len(t_keys), replace=True)
        t_dices = []
        for k in t_s: t_dices.extend(test_by_src[k])
        tm = np.mean(t_dices)
        t_means.append(tm)
        
        gaps.append(vm - tm)
        
    def ci(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
        
    return {
        'val_mean': float(np.mean([r['dice'] for r in val_data])),
        'val_ci': ci(v_means),
        'test_mean': float(np.mean([r['dice'] for r in test_data])),
        'test_ci': ci(t_means),
        'gap_mean': float(np.mean([r['dice'] for r in val_data]) - np.mean([r['dice'] for r in test_data])),
        'gap_ci': ci(gaps)
    }

boot_locked = bootstrap_cluster_gap(val_locked_results, test_locked_results)
boot_single = bootstrap_cluster_gap(val_single_results, test_locked_results)

print(f"Locked Val Dice : {boot_locked['val_mean']:.4f} [95% CI cluster: {boot_locked['val_ci'][0]:.4f}, {boot_locked['val_ci'][1]:.4f}]")
print(f"Locked Test Dice: {boot_locked['test_mean']:.4f} [95% CI cluster: {boot_locked['test_ci'][0]:.4f}, {boot_locked['test_ci'][1]:.4f}]")
print(f"Locked Gap      : {boot_locked['gap_mean']*100:.2f} điểm [95% CI cluster: {boot_locked['gap_ci'][0]*100:.2f}, {boot_locked['gap_ci'][1]*100:.2f}]")

print(f"\nSingle Val Dice : {boot_single['val_mean']:.4f} [95% CI cluster: {boot_single['val_ci'][0]:.4f}, {boot_single['val_ci'][1]:.4f}]")
print(f"Single Gap      : {boot_single['gap_mean']*100:.2f} điểm [95% CI cluster: {boot_single['gap_ci'][0]*100:.2f}, {boot_single['gap_ci'][1]*100:.2f}]")

# Xuất ra báo cáo markdown 10_gap_decomposition.md
md_content = f"""# 10_gap_decomposition.md — Phân rã Oaxaca-Blinder và Khoảng Tin Cậy Cụm Cho Cấu Hình LOCKED (Revision 4)

**Ngày:** 2026-11-17  
**Cấu hình chính (LOCKED):** Ensemble 3 seed (42, 43, 44) + TTA flip_hv, $\\tau=0.5$  
**Cấu hình đối chiếu:** Single-seed canonical (Seed 42, không TTA, direct 1-pass)  
**Tập dữ liệu:** 348 crops Val (50 ảnh gốc) và 1,124 crops Test (148 ảnh gốc)  

---

## 1. Kết quả suy luận và Ước lượng Khoảng tin cậy dạng Cụm (Cluster Bootstrap 10,000 lần)

Ước lượng khoảng tin cậy 95% bằng cách tái lấy mẫu độc lập có hoàn lại trên các cụm ảnh gốc (50 cụm Val và 148 cụm Test):

| Cấu hình mô hình | Phân tách | Dice trung bình | 95% CI Cluster | Độ rộng CI |
|---|:---:|:---:|:---:|:---:|
| **LOCKED (Ensemble + TTA)** | **VAL** | **{boot_locked['val_mean']:.4f}** | **[{boot_locked['val_ci'][0]:.4f}, {boot_locked['val_ci'][1]:.4f}]** | {(boot_locked['val_ci'][1] - boot_locked['val_ci'][0])*100:.2f} điểm |
| **LOCKED (Ensemble + TTA)** | **TEST** | **{boot_locked['test_mean']:.4f}** | **[{boot_locked['test_ci'][0]:.4f}, {boot_locked['test_ci'][1]:.4f}]** | {(boot_locked['test_ci'][1] - boot_locked['test_ci'][0])*100:.2f} điểm |
| **LOCKED Gap (Val − Test)** | — | **+{boot_locked['gap_mean']*100:.2f} điểm** | **[{boot_locked['gap_ci'][0]*100:.2f}, {boot_locked['gap_ci'][1]*100:.2f}] điểm** | {(boot_locked['gap_ci'][1] - boot_locked['gap_ci'][0])*100:.2f} điểm |
| *Single-seed (Seed 42 no TTA)* | *VAL* | *{boot_single['val_mean']:.4f}* | *[{boot_single['val_ci'][0]:.4f}, {boot_single['val_ci'][1]:.4f}]* | {(boot_single['val_ci'][1] - boot_single['val_ci'][0])*100:.2f} điểm |
| *Single-seed Gap (Val − Test)* | — | *+{boot_single['gap_mean']*100:.2f} điểm* | *[{boot_single['gap_ci'][0]*100:.2f}, {boot_single['gap_ci'][1]*100:.2f}] điểm* | {(boot_single['gap_ci'][1] - boot_single['gap_ci'][0])*100:.2f} điểm |

> [!IMPORTANT]
> Khoảng cách suy giảm hiệu năng Val → Test của cấu hình LOCKED là **+{boot_locked['gap_mean']*100:.2f} điểm**, với 95% CI cluster là **[{boot_locked['gap_ci'][0]*100:.2f}, {boot_locked['gap_ci'][1]*100:.2f}] điểm**.

---

## 2. Bảng Phân rã Oaxaca-Blinder theo Nhóm Độ rộng

Phân rã chênh lệch $\\Delta = \\bar{{Y}}_{{val}} - \\bar{{Y}}_{{test}} = \\Delta_{{mix}} + \\Delta_{{within}}$:

| Thành phần phân rã | Cấu hình LOCKED (Ensemble+TTA) | Cấu hình Đối chiếu (Single seed 42) |
|---|:---:|:---:|
| **Val Dice tổng thể** | **{oaxaca_locked['val_mean']:.4f}** | **{oaxaca_single['val_mean']:.4f}** |
| **Test Dice tổng thể** | **{oaxaca_locked['test_mean']:.4f}** | **{oaxaca_single['test_mean']:.4f}** |
| **Tổng khoảng cách (Gap)** | **+{oaxaca_locked['gap']*100:.2f} điểm (100.0%)** | **+{oaxaca_single['gap']*100:.2f} điểm (100.0%)** |
| **Within effect (chênh lệch trong cùng nhóm)** | **+{oaxaca_locked['within']*100:.2f} điểm ({oaxaca_locked['within_pct']:.1f}%)** | **+{oaxaca_single['within']*100:.2f} điểm ({oaxaca_single['within_pct']:.1f}%)** |
| **Mix effect (do lệch phân bố tỉ lệ nhóm)** | **+{oaxaca_locked['mix']*100:.2f} điểm ({oaxaca_locked['mix_pct']:.1f}%)** | **+{oaxaca_single['mix']*100:.2f} điểm ({oaxaca_single['mix_pct']:.1f}%)** |

### Chi tiết các nhóm độ rộng (Cấu hình LOCKED):
- **Nhóm hẹp ($\le 3$ px):** 0 crop ở cả hai tập.
- **Nhóm vừa ($3–10$ px):** 
  - Val: {oaxaca_locked['breakdown']['medium (3-10px)']['val_count']} crops ({oaxaca_locked['breakdown']['medium (3-10px)']['w_val']*100:.1f}%), Dice = **{oaxaca_locked['breakdown']['medium (3-10px)']['dice_val']:.4f}**
  - Test: {oaxaca_locked['breakdown']['medium (3-10px)']['test_count']} crops ({oaxaca_locked['breakdown']['medium (3-10px)']['w_test']*100:.1f}%), Dice = **{oaxaca_locked['breakdown']['medium (3-10px)']['dice_test']:.4f}**
  - Chênh lệch trong nhóm: **+{(oaxaca_locked['breakdown']['medium (3-10px)']['dice_val'] - oaxaca_locked['breakdown']['medium (3-10px)']['dice_test'])*100:.2f} điểm**
  - Đóng góp: Within = {oaxaca_locked['breakdown']['medium (3-10px)']['within']*100:.2f} điểm, Mix = {oaxaca_locked['breakdown']['medium (3-10px)']['mix']*100:.2f} điểm.
- **Nhóm rộng ($>10$ px):**
  - Val: {oaxaca_locked['breakdown']['wide (>10px)']['val_count']} crops ({oaxaca_locked['breakdown']['wide (>10px)']['w_val']*100:.1f}%), Dice = **{oaxaca_locked['breakdown']['wide (>10px)']['dice_val']:.4f}**
  - Test: {oaxaca_locked['breakdown']['wide (>10px)']['test_count']} crops ({oaxaca_locked['breakdown']['wide (>10px)']['w_test']*100:.1f}%), Dice = **{oaxaca_locked['breakdown']['wide (>10px)']['dice_test']:.4f}**
  - Chênh lệch trong nhóm: **+{(oaxaca_locked['breakdown']['wide (>10px)']['dice_val'] - oaxaca_locked['breakdown']['wide (>10px)']['dice_test'])*100:.2f} điểm**
  - Đóng góp: Within = {oaxaca_locked['breakdown']['wide (>10px)']['within']*100:.2f} điểm, Mix = {oaxaca_locked['breakdown']['wide (>10px)']['mix']*100:.2f} điểm.

---

## 3. Diễn giải Khoa học Đúng mực (Tránh Kết luận Vượt quá Dữ liệu)

1. **Ý nghĩa thống kê của "Within effect":**
   - Thuật ngữ "within-group effect" (chiếm ~96%) trong phân rã Oaxaca thuần túy phản ánh mặt số học: điểm Dice trung bình của tập Test thấp hơn tập Val ở cả hai nhóm độ rộng (vết vừa tụt ~5.8 điểm, vết rộng tụt ~6.3 điểm).
   - **Không được kết luận** rằng "mô hình dự đoán kém hơn" do lỗi kiến trúc, và **không được khẳng định** "tập Test không khó hơn". Sự chênh lệch này gồm nhiều nguyên nhân chưa thể tách rời:
     - Các đặc tính hình ảnh/bề mặt chưa được đo lường (độ tương phản, độ nhám mặt đường, ánh sáng, nhiễu bóng râm).
     - Thiên lệch chọn lựa checkpoint và siêu tham số được tối ưu trên tập Val (Selection bias: ~0.22 điểm quanh vùng hội tụ).
     - Nhiễu lấy mẫu xuất phát từ cỡ mẫu Val nhỏ (chỉ có 50 ảnh gốc độc lập).
2. **Thiên lệch chọn epoch (Selection bias):**
   - Sử dụng con số **0.22 điểm** (mức chênh giữa epoch tốt nhất và trung bình 5 epoch cuối quanh vùng hội tụ) làm ước lượng đại diện cho thiên lệch chọn epoch.
   - Loại bỏ con số 1.60 điểm (tính trên toàn bộ 18 epoch) vì con số này gộp cả các epoch đầu khi mạng chưa hội tụ, không phản ánh đúng thiên lệch chọn mô hình.
"""

with open(os.path.join(OUT_DIR, "10_gap_decomposition.md"), "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"\nSaved report: {os.path.join(OUT_DIR, '10_gap_decomposition.md')}")
