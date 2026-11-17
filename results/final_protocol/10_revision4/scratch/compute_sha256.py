import hashlib, os, glob

base_dir = r"d:\truong\SpecialSubjectTTNT\results\final_protocol"
folders = [
    "phase3_variantD_seed42",
    "phase3_variantD_seed43",
    "phase3_variantR_seed42",
    "phase3_variantR_seed43",
    "phase3_variantN_seed42",
    "phase3_variantN_seed43",
    r"phase2_results\phase2_seed42",
    r"phase2_results\phase2_seed43",
    r"phase2_results\phase2_seed44",
]

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()

print("=== CHECKPOINT SHA-256 ===")
for fld in folders:
    fld_path = os.path.join(base_dir, fld)
    for ckpt_name in ["best_model_b2_global.pth", "best_model_b2_stage2.pth"]:
        p = os.path.join(fld_path, ckpt_name)
        if os.path.exists(p):
            sha = get_sha256(p)
            print(f"{fld:30s} | {ckpt_name:25s} | {sha}")
