import re
from pathlib import Path

# Find the log pattern
log_path = Path(r'd:\truong\SpecialSubjectTTNT\results\final_protocol\phase2_results\phase2_seed42\train.log')
with open(log_path, encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

print(f'Total lines: {len(lines)}')
# Print first 20 non-tqdm lines
count = 0
for line in lines:
    stripped = line.strip()
    if not stripped or '|' in stripped or 'it/s' in stripped:
        continue
    print(repr(stripped[:200]))
    count += 1
    if count >= 30:
        break
