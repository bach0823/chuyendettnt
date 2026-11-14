"""
Phase 0 Audit Script: Tái lập và kiểm tra tính tất định của Val Dice canonical.
Canonical: phase7_s2g_new_pipeline_run2/best_model_b2_stage2.pth, Val Dice claim=0.7793
"""
import sys, os, json, hashlib, time
sys.path.insert(0, r"d:\truong\SpecialSubjectTTNT\SAGE_LITE")
sys.path.insert(0, r"d:\truong\SpecialSubjectTTNT\SAGE_LITE\scripts")

import torch
import numpy as np
import cv2
import yaml

CHECKPOINT = r"d:\truong\SpecialSubjectTTNT\results\phase7_combination\phase7_s2g_new_pipeline_run2\best_model_b2_stage2.pth"
CONFIG     = r"d:\truong\SpecialSubjectTTNT\configs\p3_ablation\phase7\phase7_b2_s2g_b1_b2_end_to_end.yaml"
OUT_DIR    = r"d:\truong\SpecialSubjectTTNT\results\final_protocol\00_audit"
CLAIMED_VAL_DICE = 0.7793
CLAIMED_TEST_DICE = 0.7149
TOLE = 0.0005

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()

print("=" * 60)
print("PHASE 0 — CHECKPOINT AUDIT")
print("=" * 60)

# 1. SHA256
sha = sha256_file(CHECKPOINT)
print(f"[SHA256] {sha}")
print(f"[Size]   {os.path.getsize(CHECKPOINT):,} bytes")

# 2. Load config
with open(CONFIG) as f:
    config = yaml.safe_load(f)

# Override root_dir to absolute path
config['root_dir'] = r"d:\truong\SpecialSubjectTTNT\datasets\Crack500_ready"
print(f"[Config] seed={config.get('seed')}, img_size={config.get('img_size')}, "
      f"epochs={config.get('epochs')}, s2_gate={config.get('use_s2_gate')}, "
      f"s2_gate_b2={config.get('use_s2_gate_block2')}")

# 3. Build model and load checkpoint
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
    ckpt_meta = {k: v for k, v in ckpt.items() if k != 'model_state_dict'}
else:
    model.load_state_dict(ckpt)
    ckpt_meta = {}
model.eval()
print(f"[Checkpoint meta] {ckpt_meta}")
num_params = sum(p.numel() for p in model.parameters())
print(f"[Params] {num_params:,} ({num_params/1e6:.3f}M)")

# 4. Tái lập Val Dice (run 1)
from evaluate_crack_official import get_image_mask_pairs, predict_full_image_direct, compute_boundary_iou
from sage.utils.advanced_metrics import calculate_hd95_bf1

def eval_val_direct(model, pairs, device, tau=0.5, diagnostic=False):
    """Custom eval loop using predict_full_image_direct, explicit tau."""
    all_dice, all_prec, all_rec, all_biou, all_hd95 = [], [], [], [], []
    total_tp = total_fp = total_fn = 0
    for img_p, mask_p in pairs:
        img = cv2.imread(img_p)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        target = (mask > 127).astype(np.uint8)
        
        logits = predict_full_image_direct(model, img_rgb, device, target_size=448)
        prob = 1.0 / (1.0 + np.exp(-logits))
        pred = (prob >= tau).astype(np.uint8)
        
        if pred.shape != target.shape:
            pred = cv2.resize(pred.astype(np.float32),
                              (target.shape[1], target.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        
        tp = int(np.sum((pred==1)&(target==1)))
        fp = int(np.sum((pred==1)&(target==0)))
        fn = int(np.sum((pred==0)&(target==1)))
        total_tp += tp; total_fp += fp; total_fn += fn
        prec = tp/(tp+fp+1e-5); rec = tp/(tp+fn+1e-5)
        dice = 2*tp/(2*tp+fp+fn+1e-5)
        all_dice.append(dice); all_prec.append(prec); all_rec.append(rec)
        if diagnostic:
            all_biou.append(compute_boundary_iou(pred, target, d=2))
            hd, _ = calculate_hd95_bf1(pred, target)
            all_hd95.append(hd)
    
    out = {
        "dice": float(np.mean(all_dice)),
        "precision": float(np.mean(all_prec)),
        "recall": float(np.mean(all_rec)),
        "global_pixel_iou": float(total_tp/(total_tp+total_fp+total_fn+1e-5)),
    }
    if diagnostic:
        out["boundary_iou"] = float(np.mean(all_biou)) if all_biou else 0.0
        out["hd95"] = float(np.mean(all_hd95)) if all_hd95 else 999.0
    return out
val_pairs = get_image_mask_pairs(config, 'val')
print(f"[Val pairs] {len(val_pairs)} images")

print("\n--- Run 1 (Val Dice re-run) ---")
t0 = time.time()
result_run1 = eval_val_direct(model, val_pairs, device, tau=0.5, diagnostic=True)
t1 = time.time()
dice_run1 = result_run1['dice']
print(f"Val Dice (run 1): {dice_run1:.6f}  (claimed: {CLAIMED_VAL_DICE})")
print(f"Precision: {result_run1['precision']:.6f}, Recall: {result_run1['recall']:.6f}")
print(f"Boundary IoU: {result_run1['boundary_iou']:.6f}, HD95: {result_run1['hd95']:.4f} px")
print(f"Time: {t1-t0:.1f}s")

delta = abs(dice_run1 - CLAIMED_VAL_DICE)
if delta > TOLE:
    print(f"[FAIL] Delta {delta:.6f} > tolerance {TOLE}. STOP — investigate.")
    sys.exit(1)
else:
    print(f"[PASS] Delta {delta:.6f} <= tolerance {TOLE}")

# 5. Kiểm tra tính tất định (run 2)
print("\n--- Run 2 (Determinism check) ---")
t0 = time.time()
result_run2 = eval_val_direct(model, val_pairs, device, tau=0.5, diagnostic=False)
t1 = time.time()
dice_run2 = result_run2['dice']
print(f"Val Dice (run 2): {dice_run2:.6f}")
print(f"Time: {t1-t0:.1f}s")

pixel_diff = abs(dice_run1 - dice_run2)
print(f"[Dice diff run1 vs run2] {pixel_diff:.8f}")
if pixel_diff < 1e-5:
    print("[PASS] Deterministic: Dice identical across 2 runs")
else:
    print(f"[WARN] Dice differs by {pixel_diff:.8f} (non-determinism from CUDA ops)")

# 6. Summary JSON
audit = {
    "checkpoint": CHECKPOINT,
    "sha256": sha,
    "num_params": num_params,
    "config_seed": config.get('seed'),
    "config_img_size": config.get('img_size'),
    "claimed_val_dice": CLAIMED_VAL_DICE,
    "run1_val_dice": dice_run1,
    "run1_precision": result_run1['precision'],
    "run1_recall": result_run1['recall'],
    "run1_boundary_iou": result_run1['boundary_iou'],
    "run1_hd95": result_run1['hd95'],
    "run2_val_dice": dice_run2,
    "delta_claimed_vs_run1": float(dice_run1 - CLAIMED_VAL_DICE),
    "delta_run1_vs_run2": float(pixel_diff),
    "pass_tolerance": bool(delta <= TOLE),
    "is_deterministic": bool(pixel_diff < 1e-5),
    "checkpoint_meta": {str(k): str(v) for k, v in ckpt_meta.items()},
}
out_path = os.path.join(OUT_DIR, "audit_result.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2, ensure_ascii=False)
print(f"\n[Saved] {out_path}")
print("PHASE 0 COMPLETE")
