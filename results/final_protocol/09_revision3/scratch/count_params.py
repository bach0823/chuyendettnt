"""Count parameters by module for SAGE B2 model."""
import sys
sys.path.insert(0, r'd:\truong\SpecialSubjectTTNT\SAGE_LITE')

import torch
import yaml

# Load config
config_path = r'd:\truong\SpecialSubjectTTNT\SAGE_LITE\configs\sage_b2_crack500.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Build model
from sage.networks import create_b2_unet
model = create_b2_unet(config)
model.eval()

def count_params(module):
    return sum(p.numel() for p in module.parameters())

total = count_params(model)
print(f"Total params: {total:,}")

# Count by top-level submodule
for name, module in model.named_children():
    n = count_params(module)
    print(f"  {name}: {n:,} ({n/total*100:.1f}%)")

# Try to find backbone, decoder, sage/expert components
print("\n--- Named modules with 'expert' or 'router' or 'sage' ---")
for name, module in model.named_modules():
    if any(k in name.lower() for k in ['expert', 'router', 'sage_layer', 'gate']):
        n = count_params(module)
        if n > 0:
            print(f"  {name}: {n:,}")

print("\n--- Named modules with 'backbone' or 'encoder' ---")
for name, module in model.named_modules():
    if any(k in name.lower() for k in ['backbone', 'encoder']) and '.' not in name:
        n = count_params(module)
        if n > 0:
            print(f"  {name}: {n:,}")
