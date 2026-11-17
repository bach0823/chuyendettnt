import json
from pathlib import Path

for seed in [42, 43, 44]:
    p = Path(fr'd:\truong\SpecialSubjectTTNT\results\final_protocol\phase2_results\phase2_seed{seed}')
    files = list(p.glob('*'))
    print(f'seed{seed}: {[f.name for f in files]}')
    rp = p / 'run_metadata.json'
    if rp.exists():
        with open(rp) as f:
            d = json.load(f)
        keys = list(d.keys())
        print(f'  run_metadata keys: {keys}')
        for k in ['epoch_val_dice', 'stage2_val_dice', 'val_history', 'history']:
            if k in d:
                print(f'  {k}: {d[k][-10:]}')
