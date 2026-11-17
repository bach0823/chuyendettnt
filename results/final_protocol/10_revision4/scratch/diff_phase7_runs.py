import os, json, re

dirs = [
    ("b1_b2_e2e", r"d:\truong\SpecialSubjectTTNT\results\phase7_combination\phase7_s2g_b1_b2_end_to_end"),
    ("run1", r"d:\truong\SpecialSubjectTTNT\results\phase7_combination\phase7_s2g_new_pipeline_run1"),
    ("run2", r"d:\truong\SpecialSubjectTTNT\results\phase7_combination\phase7_s2g_new_pipeline_run2"),
]

print("=== CONFIG & RUN DIFF: b1_b2_end_to_end vs run1 vs run2 ===")

for name, d in dirs:
    log_p = os.path.join(d, "train.log")
    print(f"\n[{name}] ({d})")
    if os.path.exists(log_p):
        with open(log_p, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        
        # Check seed, val protocol, lr, batch_size, epochs
        info_lines = [l.strip() for l in lines if "[INFO ]" in l][:30]
        for l in info_lines:
            if any(k in l for k in ["Evaluation protocol", "batch_size", "lr", "seed", "epochs", "loss", "optimizer", "scheduler", "augment"]):
                print("  ", l)
        
        # Stage 2 training line
        stage2_lines = [l.strip() for l in lines if "Stage 2" in l or "stage2" in l.lower()][:10]
        for l in stage2_lines[:5]:
            print("   [S2]", l)
            
    # Check test metrics
    diag_p = os.path.join(d, "diagnostics", "final_test_metrics.json")
    if os.path.exists(diag_p):
        with open(diag_p) as f:
            tm = json.load(f)
            meta = tm.get("metadata", {})
            sa = tm.get("setting_a_official", {}).get("mean_dice")
            sb = tm.get("setting_b_reference", {}).get("mean_dice")
            print(f"  Test Setting A: {sa:.4f}, Setting B: {sb:.4f}")
            print(f"  Checkpoint metadata: {meta.get('checkpoint_metadata')}")
