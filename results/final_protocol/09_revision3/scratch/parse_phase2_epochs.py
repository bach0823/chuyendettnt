"""Parse Phase 2 epoch-by-epoch val dice from train.log for all 3 seeds."""
import re

LOG_DIR = r"d:\truong\SpecialSubjectTTNT\results\final_protocol\phase2_results"

for seed in [42, 43, 44]:
    log_path = f"{LOG_DIR}/phase2_seed{seed}/train.log"
    try:
        with open(log_path, encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"seed{seed}: NOT FOUND")
        continue

    # Pattern: Epoch X/18 ... Val Dice: 0.XXXX
    pattern = re.compile(r'Epoch (\d+)/\d+.*?Val Dice: ([0-9.]+)')
    epochs = {}
    for line in lines:
        m = pattern.search(line)
        if m:
            ep = int(m.group(1))
            vd = float(m.group(2))
            epochs[ep] = vd

    if not epochs:
        print(f"seed{seed}: no epoch val dice found")
        continue

    best_ep = max(epochs, key=lambda e: epochs[e])
    best_dice = epochs[best_ep]
    all_dice = [epochs[e] for e in sorted(epochs)]
    mean_all = sum(all_dice) / len(all_dice)
    # Selection bias: best - mean of all epochs
    bias = best_dice - mean_all

    print(f"\n=== seed{seed} ===")
    print(f"Epochs found: {sorted(epochs.keys())}")
    for ep in sorted(epochs):
        mark = " <-- BEST" if ep == best_ep else ""
        print(f"  Epoch {ep:2d}: Val Dice = {epochs[ep]:.4f}{mark}")
    print(f"Best epoch: {best_ep}, best dice: {best_dice:.4f}")
    print(f"Mean all epochs: {mean_all:.4f}")
    print(f"Selection bias (best - mean): {bias:.4f} = {bias*100:.2f} diem")
    # Also compute: best - mean of last 5 epochs
    last_5 = sorted(epochs.keys())[-5:]
    mean_last5 = sum(epochs[e] for e in last_5) / len(last_5)
    bias_last5 = best_dice - mean_last5
    print(f"Mean last {len(last_5)} epochs ({last_5}): {mean_last5:.4f}")
    print(f"Selection bias (best - mean_last5): {bias_last5:.4f} = {bias_last5*100:.2f} diem")
