import json, os

paths = [
    r"d:\truong\SpecialSubjectTTNT\results\final_protocol\phase3_variantD_seed42\run_metadata.json",
    r"d:\truong\SpecialSubjectTTNT\results\final_protocol\phase3_variantD_seed43\run_metadata.json",
    r"d:\truong\SpecialSubjectTTNT\results\final_protocol\phase3_variantR_seed42\run_metadata.json",
    r"d:\truong\SpecialSubjectTTNT\results\final_protocol\phase3_variantR_seed43\run_metadata.json",
    r"d:\truong\SpecialSubjectTTNT\results\final_protocol\phase3_variantN_seed42\run_metadata.json",
    r"d:\truong\SpecialSubjectTTNT\results\final_protocol\phase3_variantN_seed43\run_metadata.json",
    r"d:\truong\SpecialSubjectTTNT\results\final_protocol\phase2_results\phase2_seed42\run_metadata.json",
    r"d:\truong\SpecialSubjectTTNT\results\final_protocol\phase2_results\phase2_seed43\run_metadata.json",
    r"d:\truong\SpecialSubjectTTNT\results\final_protocol\phase2_results\phase2_seed44\run_metadata.json",
]

print("=== RUN METADATA PHASE 3 & CANONICAL ===")
for p in paths:
    folder = p.split(os.sep)[-2]
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
            val = d.get("val_result", {})
            sha = d.get("checkpoint_sha256", "NONE")
            dice = val.get("dice", None)
            seed = d.get("seed", None)
            print(f"{folder:25s} | Seed {seed} | SHA: {sha[:16]}... | Dice: {dice}")
    else:
        print(f"{folder:25s} | NOT FOUND: {p}")
