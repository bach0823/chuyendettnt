import re
from pathlib import Path

results = {}

for seed in [42, 43, 44]:
    log_path = Path(fr'd:\truong\SpecialSubjectTTNT\results\final_protocol\phase2_results\phase2_seed{seed}\train.log')
    with open(log_path, encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Find epoch-level val dice — look for INFO lines with Stage 2 epoch completions
    # Pattern: "Stage 2, Epoch X/Y ... val_dice = Z" or "Epoch X ... Val Dice: Z"
    # Let's scan for lines containing epoch number + val dice
    
    # First find stage2 section
    stage2_idx = content.find('Stage 2')
    if stage2_idx == -1:
        stage2_idx = 0
    stage2_content = content[stage2_idx:]
    
    # Try to find "Epoch X completed" or "Epoch X/Y" with val_dice
    patterns = [
        r'Epoch\s*(\d+)/\d+.*?val_dice[=:\s]+(\d+\.\d+)',
        r'\[INFO\s*\].*?Epoch\s*(\d+).*?val_dice[=:\s]+(\d+\.\d+)',
        r'Epoch\s*(\d+).*?best_dice[=:\s]+(\d+\.\d+)',
        r'val_dice.*?(\d+\.\d{4,})',
        r'best_dice.*?(\d+\.\d{4,})',
    ]
    
    found = False
    for pat in patterns:
        matches = re.findall(pat, stage2_content, re.IGNORECASE | re.DOTALL)
        if matches and len(matches) > 1:
            print(f'seed{seed} with pattern {pat[:30]}: {len(matches)} matches, first 5: {matches[:5]}')
            found = True
            break
    
    if not found:
        # Print lines that have 'dice' and are from INFO logger in stage2
        info_lines = [l for l in stage2_content.split('\n') if '[INFO' in l and 'dice' in l.lower()]
        print(f'seed{seed}: {len(info_lines)} INFO lines with dice, first 5:')
        for l in info_lines[:5]:
            print(f'  {repr(l[:200])}')
