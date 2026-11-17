"""
Việc 3 — Val inference canonical run2 (seed=42, direct 1-pass, tau=0.5, no TTA)
Chạy trên GPU local GTX 1650.
"""
import sys, os, json, csv, math
sys.path.insert(0, r'd:\truong\SpecialSubjectTTNT\SAGE_LITE')

import torch
import numpy as np
import yaml
from pathlib import Path
from scipy.ndimage import label as cc_label

# ---- Config ----
CKPT = r'd:\truong\SpecialSubjectTTNT\results\phase7_combination\phase7_s2g_new_pipeline_run2\best_model_b2_stage2.pth'
CONFIG_YAML = r'd:\truong\SpecialSubjectTTNT\SAGE_LITE\configs\sage_b2_crack500.yaml'
TAU = 0.5
OUTPUT_DIR = r'd:\truong\SpecialSubjectTTNT\results\final_protocol\09_revision3'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'val_per_image_dice.csv')
EXPECTED_DICE = 0.7793  # tolerance 0.0005

# ---- Load config ----
with open(CONFIG_YAML, 'r') as f:
    config = yaml.safe_load(f)

config['root_dir'] = r'd:\truong\SpecialSubjectTTNT\SAGE_LITE'

# ---- Import evaluation script ----
eval_script_path = r'd:\truong\SpecialSubjectTTNT\SAGE_LITE\scripts\evaluate_crack_official.py'

# Try to import predict_full_image_direct
import importlib.util
spec = importlib.util.spec_from_file_location("evaluate_crack_official", eval_script_path)
eval_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_mod)

predict_fn = eval_mod.predict_full_image_direct
get_pairs_fn = eval_mod.get_image_mask_pairs

# ---- Build model ----
from sage.networks import create_b2_unet
model = create_b2_unet(config)
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = model.to(device)
model.eval()

# ---- Get val pairs ----
# Try to get val dataset
try:
    pairs = get_pairs_fn(config, 'val')
    print(f"Val pairs from config: {len(pairs)}")
except Exception as e:
    print(f"Error getting pairs: {e}")
    # Fallback: scan val directory
    val_img_dir = os.path.join(config.get('root_dir', ''), config.get('val_img_dir', ''))
    print(f"Trying val dir: {val_img_dir}")
    pairs = []

if not pairs:
    # Manual fallback
    import glob
    val_imgs = sorted(glob.glob(r'd:\truong\SpecialSubjectTTNT\SAGE_LITE\data\Crack500\val\images\*.jpg'))
    val_masks = [p.replace('images', 'masks').replace('.jpg', '.png') for p in val_imgs]
    pairs = list(zip(val_imgs, val_masks))
    print(f"Fallback val pairs: {len(pairs)}")

print(f"Total val pairs: {len(pairs)}")

# ---- Run inference ----
from PIL import Image
import torch.nn.functional as F

def dice_coeff(pred, target):
    p = pred.astype(float).flatten()
    t = target.astype(float).flatten()
    inter = (p * t).sum()
    total = p.sum() + t.sum()
    if total == 0:
        return 1.0
    return 2.0 * inter / total

results = []
all_dice = []

for i, (img_path, mask_path) in enumerate(pairs):
    fname = os.path.basename(img_path)
    try:
        # Load image
        img = Image.open(img_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('L')
        mask_arr = np.array(mask_img)
        gt = (mask_arr > 127).astype(np.uint8)

        # Run predict
        pred_prob = predict_fn(model, img, device=device, img_size=config.get('img_size', 448))
        pred_bin = (pred_prob >= TAU).astype(np.uint8)
        d = dice_coeff(pred_bin, gt)
        all_dice.append(d)
        results.append({'filename': fname, 'dice': d, 'img_path': img_path})

        if (i+1) % 50 == 0:
            print(f"[{i+1}/{len(pairs)}] mean dice so far: {np.mean(all_dice):.4f}")
    except Exception as e:
        print(f"Error on {fname}: {e}")
        continue

mean_dice = np.mean(all_dice)
print(f"\nFinal mean val dice: {mean_dice:.4f}")
print(f"Expected: {EXPECTED_DICE} (tolerance ±0.0005)")
print(f"Match: {abs(mean_dice - EXPECTED_DICE) <= 0.0005}")

# ---- Save CSV ----
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['filename', 'dice', 'source_id'])
    writer.writeheader()
    for r in results:
        fname = r['filename']
        parts = fname.replace('.jpg', '').split('_')
        source_id = '_'.join(parts[:2]) if len(parts) >= 2 else fname
        writer.writerow({'filename': fname, 'dice': r['dice'], 'source_id': source_id})

print(f"Saved: {OUTPUT_CSV}")
print(f"Total rows: {len(results)}")
