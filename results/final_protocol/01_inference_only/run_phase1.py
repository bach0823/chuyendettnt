"""
Phase 1: Khai thác chỉ ở suy luận (không train, chỉ dùng val).
- Quét tau 0.30..0.70 bước 0.025
- Flip TTA: {none, h, v, hv}
- Kiểm tra ổn định: 2 nửa val, 20 lần lặp (seed 42..61)
- Bootstrap theo ảnh 10.000 lần, delta ghép cặp vs tau=0.5
"""
import sys, os, json, csv, time
import numpy as np
import torch
import cv2
import yaml

sys.path.insert(0, r"d:\truong\SpecialSubjectTTNT\SAGE_LITE")
sys.path.insert(0, r"d:\truong\SpecialSubjectTTNT\SAGE_LITE\scripts")

CHECKPOINT = r"d:\truong\SpecialSubjectTTNT\results\phase7_combination\phase7_s2g_new_pipeline_run2\best_model_b2_stage2.pth"
CONFIG     = r"d:\truong\SpecialSubjectTTNT\configs\p3_ablation\phase7\phase7_b2_s2g_b1_b2_end_to_end.yaml"
OUT_DIR    = r"d:\truong\SpecialSubjectTTNT\results\final_protocol\01_inference_only"
os.makedirs(OUT_DIR, exist_ok=True)

from sage.networks import create_b2_unet
from evaluate_crack_official import get_image_mask_pairs, compute_boundary_iou, predict_full_image_direct
from sage.utils.advanced_metrics import calculate_hd95_bf1

with open(CONFIG) as f:
    config = yaml.safe_load(f)
# Override root_dir to absolute path
config['root_dir'] = r"d:\truong\SpecialSubjectTTNT\datasets\Crack500_ready"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

sage_cfg = config.get('sage_config', {})
model = create_b2_unet(
    num_transformer_layers=int(config.get('num_transformer_layers', 4)),
    pretrained=False, sage_config=sage_cfg,
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
print("Checkpoint loaded.")

val_pairs = get_image_mask_pairs(config, 'val')
print(f"Val images: {len(val_pairs)}")

# ---- Helper: predict with optional TTA ----
def predict_probs(image, tta=None):
    """Return probability map (H,W) in [0,1]. tta: None/'h'/'v'/'hv'"""
    logits_orig = predict_full_image_direct(model, image, device, target_size=448)
    prob_orig = 1.0 / (1.0 + np.exp(-logits_orig))

    if tta is None:
        return prob_orig

    probs = [prob_orig]
    if 'h' in tta:
        img_h = cv2.flip(image, 1)
        logits_h = predict_full_image_direct(model, img_h, device, target_size=448)
        prob_h = 1.0 / (1.0 + np.exp(-logits_h))
        probs.append(cv2.flip(prob_h.astype(np.float32), 1))
    if 'v' in tta:
        img_v = cv2.flip(image, 0)
        logits_v = predict_full_image_direct(model, img_v, device, target_size=448)
        prob_v = 1.0 / (1.0 + np.exp(-logits_v))
        probs.append(cv2.flip(prob_v.astype(np.float32), 0))
    return np.mean(probs, axis=0)

# ---- Helper: compute metrics from prob maps ----
def compute_metrics_from_probs(prob_maps_list, pairs, tau, diagnostic=False):
    """pairs: list of (img_path, mask_path), prob_maps_list: list of np arrays"""
    all_dice, all_prec, all_rec, all_iou, all_biou, all_hd95, all_thin = [], [], [], [], [], [], []
    total_tp = total_fp = total_fn = 0
    for prob, (img_p, mask_p) in zip(prob_maps_list, pairs):
        pred = (prob >= tau).astype(np.uint8)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        target = (mask > 127).astype(np.uint8)
        
        # Resize pred to match target if needed
        if pred.shape != target.shape:
            pred = cv2.resize(pred.astype(np.float32), (target.shape[1], target.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        
        tp = int(np.sum((pred == 1) & (target == 1)))
        fp = int(np.sum((pred == 1) & (target == 0)))
        fn = int(np.sum((pred == 0) & (target == 1)))
        total_tp += tp; total_fp += fp; total_fn += fn
        
        prec = tp / (tp + fp + 1e-5)
        rec  = tp / (tp + fn + 1e-5)
        dice = 2.0 * tp / (2 * tp + fp + fn + 1e-5)
        iou  = tp / (tp + fp + fn + 1e-5)
        all_dice.append(dice); all_prec.append(prec); all_rec.append(rec); all_iou.append(iou)
        
        if diagnostic:
            biou = compute_boundary_iou(pred, target, d=2)
            all_biou.append(biou)
            hd, _ = calculate_hd95_bf1(pred, target)
            all_hd95.append(hd)
        
        # Thin crack Dice: mask pixels where local width <= 3px (approx by erosion)
        kernel3 = np.ones((3,3), np.uint8)
        thin_mask = target - cv2.erode(target, kernel3, iterations=1)
        thin_pred = pred.copy()
        thin_tp = int(np.sum((thin_pred == 1) & (thin_mask == 1)))
        thin_fp = int(np.sum((thin_pred == 1) & (thin_mask == 0) & (target == 0)))
        thin_fn = int(np.sum((thin_pred == 0) & (thin_mask == 1)))
        thin_dice = 2.0 * thin_tp / (2 * thin_tp + thin_fp + thin_fn + 1e-5)
        all_thin.append(thin_dice)
    
    gpi = float(total_tp / (total_tp + total_fp + total_fn + 1e-5))
    out = {
        "dice": float(np.mean(all_dice)),
        "precision": float(np.mean(all_prec)),
        "recall": float(np.mean(all_rec)),
        "iou": float(np.mean(all_iou)),
        "global_pixel_iou": gpi,
        "thin_dice": float(np.mean(all_thin)),
        "per_image_dice": all_dice,
    }
    if diagnostic:
        out["boundary_iou"] = float(np.mean(all_biou)) if all_biou else 0.0
        out["hd95"] = float(np.mean(all_hd95)) if all_hd95 else 999.0
    return out

# ========== Step 1: Pre-compute all prob maps (no TTA) ==========
print("\n=== Pre-computing probability maps (no TTA) ===")
prob_maps = []
images_cache = []
t0 = time.time()
for img_p, mask_p in val_pairs:
    img = cv2.imread(img_p)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images_cache.append(img)
    prob = predict_probs(img, tta=None)
    prob_maps.append(prob)
print(f"Done in {time.time()-t0:.1f}s")

# ========== Step 2: Tau sweep 0.30..0.70 step 0.025 ==========
print("\n=== Tau sweep ===")
taus = [round(0.30 + i*0.025, 3) for i in range(17)]  # 0.30 to 0.70
tau_results = []
for tau in taus:
    r = compute_metrics_from_probs(prob_maps, val_pairs, tau, diagnostic=True)
    tau_results.append({"tau": tau, **{k: v for k, v in r.items() if k != "per_image_dice"}})
    print(f"tau={tau:.3f} Dice={r['dice']:.6f} P={r['precision']:.4f} R={r['recall']:.4f} "
          f"ThinDice={r['thin_dice']:.4f} BIoU={r.get('boundary_iou',0):.4f} HD95={r.get('hd95',999):.2f}")

# Save CSV
csv_path = os.path.join(OUT_DIR, "tau_sweep.csv")
with open(csv_path, "w", newline="") as f:
    keys = list(tau_results[0].keys())
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader(); w.writerows(tau_results)
print(f"Saved: {csv_path}")

best_tau_row = max(tau_results, key=lambda x: x["dice"])
best_tau = best_tau_row["tau"]
print(f"\nBest tau (by Val Dice): {best_tau}, Dice={best_tau_row['dice']:.6f}")

# ========== Step 3: TTA sweep at tau=0.5 and best_tau ==========
print("\n=== TTA sweep ===")
tta_configs = [None, 'h', 'v', 'hv']
tta_labels = ["none", "flip_h", "flip_v", "flip_hv"]
tta_results = []
for tta, label in zip(tta_configs, tta_labels):
    print(f"TTA={label} @ tau=0.5 and best_tau={best_tau}...")
    if tta is None:
        pms = prob_maps  # already cached
    else:
        pms = []
        for img in images_cache:
            pms.append(predict_probs(img, tta=tta))
    
    r05 = compute_metrics_from_probs(pms, val_pairs, 0.5, diagnostic=(tta is None))
    r_bt = compute_metrics_from_probs(pms, val_pairs, best_tau, diagnostic=(tta is None))
    tta_results.append({
        "tta": label,
        "tau_0.5_dice": r05["dice"],
        "tau_0.5_precision": r05["precision"],
        "tau_0.5_recall": r05["recall"],
        "tau_0.5_thin_dice": r05["thin_dice"],
        f"tau_{best_tau}_dice": r_bt["dice"],
        f"tau_{best_tau}_precision": r_bt["precision"],
        f"tau_{best_tau}_recall": r_bt["recall"],
        f"tau_{best_tau}_thin_dice": r_bt["thin_dice"],
    })
    print(f"  tau=0.5  Dice={r05['dice']:.6f}  ThinDice={r05['thin_dice']:.4f}")
    print(f"  tau={best_tau} Dice={r_bt['dice']:.6f}  ThinDice={r_bt['thin_dice']:.4f}")

tta_csv = os.path.join(OUT_DIR, "tta_sweep.csv")
with open(tta_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(tta_results[0].keys()))
    w.writeheader(); w.writerows(tta_results)
print(f"Saved: {tta_csv}")

# Pick best TTA
best_tta_row = max(tta_results, key=lambda x: x[f"tau_{best_tau}_dice"])
best_tta = best_tta_row["tta"]
print(f"\nBest TTA: {best_tta}, best_tau={best_tau}, Dice={best_tta_row[f'tau_{best_tau}_dice']:.6f}")

# ========== Step 4: Bootstrap (10,000 iter) delta vs tau=0.5, no TTA ==========
print("\n=== Bootstrap delta vs tau=0.5 (10000 iter) ===")
dice_base = np.array(compute_metrics_from_probs(prob_maps, val_pairs, 0.5)["per_image_dice"])
dice_best = np.array(compute_metrics_from_probs(prob_maps, val_pairs, best_tau)["per_image_dice"])
delta_arr = dice_best - dice_base
N = len(delta_arr)
rng = np.random.default_rng(42)
boot_deltas = [np.mean(rng.choice(delta_arr, N, replace=True)) for _ in range(10000)]
ci_lo, ci_hi = float(np.percentile(boot_deltas, 2.5)), float(np.percentile(boot_deltas, 97.5))
mean_delta = float(np.mean(delta_arr))
print(f"Mean delta: {mean_delta:.6f}, 95% CI: [{ci_lo:.6f}, {ci_hi:.6f}]")
accepted_tau = ci_lo > 0

# ========== Step 5: Stability — 2-half split, 20 iterations ==========
print("\n=== Stability check: 2-half split, 20 iterations ===")
stability_rows = []
rng2 = np.random.default_rng(42)
N = len(val_pairs)
for rep in range(20):
    seed_i = 42 + rep
    rng3 = np.random.default_rng(seed_i)
    idx = rng3.permutation(N)
    half = N // 2
    idx_A, idx_B = idx[:half], idx[half:]
    
    probs_A = [prob_maps[i] for i in idx_A]
    pairs_A = [val_pairs[i] for i in idx_A]
    probs_B = [prob_maps[i] for i in idx_B]
    pairs_B = [val_pairs[i] for i in idx_B]
    
    # Choose tau on A, evaluate on B
    best_tau_A = max(taus, key=lambda t: compute_metrics_from_probs(probs_A, pairs_A, t)["dice"])
    dice_B = compute_metrics_from_probs(probs_B, pairs_B, best_tau_A)["dice"]
    # Choose tau on B, evaluate on A
    best_tau_B = max(taus, key=lambda t: compute_metrics_from_probs(probs_B, pairs_B, t)["dice"])
    dice_A = compute_metrics_from_probs(probs_A, pairs_A, best_tau_B)["dice"]
    
    stability_rows.append({
        "rep": rep, "seed": seed_i, 
        "tau_A": best_tau_A, "dice_B_given_tauA": dice_B,
        "tau_B": best_tau_B, "dice_A_given_tauB": dice_A
    })
    if rep % 5 == 0:
        print(f"Rep {rep:02d}: tau_A={best_tau_A:.3f} -> DiceB={dice_B:.4f} | tau_B={best_tau_B:.3f} -> DiceA={dice_A:.4f}")

stab_csv = os.path.join(OUT_DIR, "stability.csv")
with open(stab_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(stability_rows[0].keys()))
    w.writeheader(); w.writerows(stability_rows)

all_cross_dices = [r["dice_B_given_tauA"] for r in stability_rows] + [r["dice_A_given_tauB"] for r in stability_rows]
stab_mean = float(np.mean(all_cross_dices))
stab_std  = float(np.std(all_cross_dices))
print(f"Stability cross-eval: mean={stab_mean:.6f}, std={stab_std:.6f}")
print(f"Saved: {stab_csv}")

# ========== Summary =========
summary = {
    "checkpoint": CHECKPOINT,
    "best_tau_overall": best_tau,
    "best_tau_dice": best_tau_row["dice"],
    "baseline_tau05_dice": float(np.mean([r["tau_0.5_dice"] for r in tta_results if r["tta"]=="none"])),
    "bootstrap_mean_delta": mean_delta,
    "bootstrap_ci_lo": ci_lo,
    "bootstrap_ci_hi": ci_hi,
    "tau_change_accepted": accepted_tau,
    "best_tta": best_tta,
    "stability_mean": stab_mean,
    "stability_std": stab_std,
    "tau_results": tau_results,
    "tta_results": tta_results,
}
sum_path = os.path.join(OUT_DIR, "phase1_summary.json")
with open(sum_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved summary: {sum_path}")
print("PHASE 1 COMPLETE")
