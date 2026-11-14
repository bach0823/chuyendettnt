"""
Bootstrap CI cho TTA flip_hv vs no-TTA, tại tau=0.5 và tau=0.6
Dùng prob maps đã cache từ Phase 1 → chạy lại inference
"""
import sys, os, json
import numpy as np
import torch
import cv2
import yaml

sys.path.insert(0, r"d:\truong\SpecialSubjectTTNT\SAGE_LITE")
sys.path.insert(0, r"d:\truong\SpecialSubjectTTNT\SAGE_LITE\scripts")

CHECKPOINT = r"d:\truong\SpecialSubjectTTNT\results\phase7_combination\phase7_s2g_new_pipeline_run2\best_model_b2_stage2.pth"
CONFIG     = r"d:\truong\SpecialSubjectTTNT\configs\p3_ablation\phase7\phase7_b2_s2g_b1_b2_end_to_end.yaml"
OUT_DIR    = r"d:\truong\SpecialSubjectTTNT\results\final_protocol\01_inference_only"

from sage.networks import create_b2_unet
from evaluate_crack_official import get_image_mask_pairs, predict_full_image_direct

with open(CONFIG) as f:
    config = yaml.safe_load(f)
config['root_dir'] = r"d:\truong\SpecialSubjectTTNT\datasets\Crack500_ready"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

val_pairs = get_image_mask_pairs(config, 'val')
print(f"Val: {len(val_pairs)} images")

def predict_probs(image, tta=None):
    logits_orig = predict_full_image_direct(model, image, device, target_size=448)
    prob_orig = 1.0 / (1.0 + np.exp(-logits_orig))
    if tta is None:
        return prob_orig
    probs = [prob_orig]
    if 'h' in tta:
        img_h = cv2.flip(image, 1)
        logits_h = predict_full_image_direct(model, img_h, device, target_size=448)
        probs.append(cv2.flip((1.0/(1.0+np.exp(-logits_h))).astype(np.float32), 1))
    if 'v' in tta:
        img_v = cv2.flip(image, 0)
        logits_v = predict_full_image_direct(model, img_v, device, target_size=448)
        probs.append(cv2.flip((1.0/(1.0+np.exp(-logits_v))).astype(np.float32), 0))
    return np.mean(probs, axis=0)

def per_image_dice(prob_maps, pairs, tau):
    dices = []
    for prob, (img_p, mask_p) in zip(prob_maps, pairs):
        pred = (prob >= tau).astype(np.uint8)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        target = (mask > 127).astype(np.uint8)
        if pred.shape != target.shape:
            pred = cv2.resize(pred.astype(np.float32),(target.shape[1],target.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        tp = int(np.sum((pred==1)&(target==1)))
        fp = int(np.sum((pred==1)&(target==0)))
        fn = int(np.sum((pred==0)&(target==1)))
        dices.append(2*tp/(2*tp+fp+fn+1e-5))
    return np.array(dices)

print("Computing no-TTA prob maps...")
probs_base = []
images = []
for img_p, _ in val_pairs:
    img = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
    images.append(img)
    probs_base.append(predict_probs(img, tta=None))

print("Computing flip_hv TTA prob maps...")
probs_hv = [predict_probs(img, tta='hv') for img in images]

results = {}
rng = np.random.default_rng(42)
N = len(val_pairs)

for tau in [0.5, 0.6]:
    dice_base = per_image_dice(probs_base, val_pairs, tau)
    dice_hv   = per_image_dice(probs_hv,   val_pairs, tau)
    delta = dice_hv - dice_base
    boot = [np.mean(rng.choice(delta, N, replace=True)) for _ in range(10000)]
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    mean_d = float(np.mean(delta))
    accepted = ci_lo > 0
    print(f"tau={tau}: mean_delta={mean_d:.6f}, 95%CI=[{ci_lo:.6f},{ci_hi:.6f}], accepted={accepted}")
    results[f"tau_{tau}"] = {
        "mean_delta_tta_hv_vs_none": mean_d,
        "ci_lo": ci_lo, "ci_hi": ci_hi, "accepted": accepted,
        "dice_base_mean": float(np.mean(dice_base)),
        "dice_hv_mean": float(np.mean(dice_hv)),
    }

out = os.path.join(OUT_DIR, "tta_bootstrap.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {out}")
