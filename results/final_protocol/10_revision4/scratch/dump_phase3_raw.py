import json, os, glob

folders = [
    "phase3_variantD_seed42",
    "phase3_variantD_seed43",
    "phase3_variantR_seed42",
    "phase3_variantR_seed43",
    "phase3_variantN_seed42",
    "phase3_variantN_seed43",
]

base_dir = r"d:\truong\SpecialSubjectTTNT\results\final_protocol"

print("=== RAW PHASE 3 RUN METADATA ===")
for fld in folders:
    meta_p = os.path.join(base_dir, fld, "run_metadata.json")
    if os.path.exists(meta_p):
        with open(meta_p) as f:
            d = json.load(f)
        # Find checkpoint files
        ckpts = glob.glob(os.path.join(base_dir, fld, "*.pth"))
        ckpt_names = [os.path.basename(c) for c in ckpts]
        print(f"\n[{fld}]")
        for k in ['variant', 'seed', 'best_global_dice', 'best_s2_dice', 'best_s1_dice', 'delta_vs_phase2_mean', 'mde_exceeded']:
            if k in d:
                print(f"  {k}: {d[k]}")
        print(f"  ckpts: {ckpt_names}")
    else:
        print(f"\n[{fld}] NOT FOUND")
