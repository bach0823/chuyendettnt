"""
Script kiểm tra chi tiết VIỆC 1:
1. Chứng minh id(param) của expert_pool trùng với main_block (ConvNeXt stages & ViT blocks).
2. Phân loại tham số theo sở hữu độc quyền (không trùng lặp):
   (a) Backbone pretrain (ConvNeXt stages + ViT blocks, các khối làm expert)
   (b) Thành phần riêng của SAGE (Router, SA-Hub / format adapters, P3 refinement nếu có)
   (c) Decoder + head + S2-Gate + Projections
   -> Kiểm tra tổng khớp 10,205,755
3. Đo grad-norm sau 1 batch với canonical (residual_scale=0.1) và Variant N (residual_scale=0.0).
"""
import sys, os, yaml
sys.path.insert(0, r"d:\truong\SpecialSubjectTTNT\SAGE_LITE")

import torch
import torch.nn as nn
from sage.networks import create_b2_unet
from sage.components.sage_layer import SageLayer

CONFIG = r"d:\truong\SpecialSubjectTTNT\configs\p3_ablation\phase7\phase7_b2_s2g_b1_b2_end_to_end.yaml"
with open(CONFIG) as f:
    config = yaml.safe_load(f)

sage_cfg = config.get('sage_config', {})
model = create_b2_unet(
    num_transformer_layers=int(config.get('num_transformer_layers', 4)),
    pretrained=False,
    sage_config=sage_cfg,
    p3_mode=config.get('p3_mode'),
    use_plu_head=config.get('use_plu_head', False),
    use_s2_gate=config.get('use_s2_gate', False),
    s2_gate_kernel_size=config.get('s2_gate_kernel_size', 3),
    use_s2_gate_block2=config.get('use_s2_gate_block2', False),
    s2_gate_block2_kernel_size=config.get('s2_gate_block2_kernel_size', 3),
    s2_gate_block2_dilation=config.get('s2_gate_block2_dilation', 1),
)

total_unique_params = sum(p.numel() for p in model.parameters())
print(f"Total model.parameters() [unique tensors]: {total_unique_params:,}")

# 1. Kiểm tra id(param) giữa expert_pool và main_block
print("\n--- 1. Kiểm tra ID tham số giữa expert_pool và main_block ---")
backbone = model.backbone
convnext = backbone.convnext
transformer_blocks = backbone.transformer_blocks
expert_pool = model.expert_pool

id_matches = []
for idx, expert in enumerate(expert_pool):
    exp_name = getattr(expert, 'expert_name', f'expert_{idx}')
    exp_params = list(expert.parameters())
    if idx < len(convnext.stages):
        mb = convnext.stages[idx].main_block
        mb_name = f"convnext.stages[{idx}].main_block"
    else:
        tb_idx = idx - len(convnext.stages)
        mb = transformer_blocks[tb_idx].main_block
        mb_name = f"transformer_blocks[{tb_idx}].main_block"
    
    mb_params = list(mb.parameters())
    all_same = len(exp_params) == len(mb_params) and all(id(p1) == id(p2) for p1, p2 in zip(exp_params, mb_params))
    first_p_id_exp = id(exp_params[0]) if exp_params else None
    first_p_id_mb = id(mb_params[0]) if mb_params else None
    print(f"Expert {idx} ({exp_name}): params={len(exp_params)}, matches {mb_name}? {all_same} (p0_id: {first_p_id_exp} vs {first_p_id_mb})")
    id_matches.append(all_same)

print(f"\nTất cả {len(id_matches)} expert có id(param) TRÙNG NHAU 100% với main_block: {all(id_matches)}")

# 2. Phân loại tham số theo sở hữu độc quyền (không trùng lặp)
print("\n--- 2. Phân loại tham số theo sở hữu độc quyền ---")

def categorize_model_parameters(m):
    assigned_ids = set()
    bb = m.backbone
    cnext = bb.convnext
    tblocks = bb.transformer_blocks
    
    # (a) Backbone pretrain: Stem + ConvNeXt stages (main_blocks) + Transformer blocks (main_blocks)
    backbone_params = []
    for p in cnext.stem.parameters():
        if id(p) not in assigned_ids:
            assigned_ids.add(id(p))
            backbone_params.append(('stem', p))
    for idx, st in enumerate(cnext.stages):
        if hasattr(st, 'main_block'):
            for p in st.main_block.parameters():
                if id(p) not in assigned_ids:
                    assigned_ids.add(id(p))
                    backbone_params.append((f'stage_{idx}_main_block', p))
    for idx, tb in enumerate(tblocks):
        if hasattr(tb, 'main_block'):
            for p in tb.main_block.parameters():
                if id(p) not in assigned_ids:
                    assigned_ids.add(id(p))
                    backbone_params.append((f'transformer_{idx}_main_block', p))
    
    # (b) Riêng của SAGE: Router, SA-Hub / format adapters, P3 refinement (nếu có)
    sage_params = []
    for st_idx, st in enumerate(cnext.stages):
        if hasattr(st, 'router'):
            for p in st.router.parameters():
                if id(p) not in assigned_ids:
                    assigned_ids.add(id(p))
                    sage_params.append((f"convnext.stages[{st_idx}].router", p))
        if hasattr(st, 'sa_hub'):
            for p in st.sa_hub.parameters():
                if id(p) not in assigned_ids:
                    assigned_ids.add(id(p))
                    sage_params.append((f"convnext.stages[{st_idx}].sa_hub", p))
        if hasattr(st, 'p3_refinement') and st.p3_refinement is not None:
            for p in st.p3_refinement.parameters():
                if id(p) not in assigned_ids:
                    assigned_ids.add(id(p))
                    sage_params.append((f"convnext.stages[{st_idx}].p3_refinement", p))
                    
    for tb_idx, tb in enumerate(tblocks):
        if hasattr(tb, 'router'):
            for p in tb.router.parameters():
                if id(p) not in assigned_ids:
                    assigned_ids.add(id(p))
                    sage_params.append((f"transformer_blocks[{tb_idx}].router", p))
        if hasattr(tb, 'sa_hub'):
            for p in tb.sa_hub.parameters():
                if id(p) not in assigned_ids:
                    assigned_ids.add(id(p))
                    sage_params.append((f"transformer_blocks[{tb_idx}].sa_hub", p))
                    
    if hasattr(bb, 'master_adapters'):
        for p in bb.master_adapters.parameters():
            if id(p) not in assigned_ids:
                assigned_ids.add(id(p))
                sage_params.append(('backbone.master_adapters', p))
                
    # (c) Decoder + Head + S2-Gate + Projections / Positional Embeddings / Norms
    c_params = []
    for name, p in m.named_parameters():
        if id(p) not in assigned_ids:
            assigned_ids.add(id(p))
            c_params.append((name, p))
            
    return backbone_params, sage_params, c_params

a_params, b_params, c_params = categorize_model_parameters(model)
count_a = sum(p.numel() for _, p in a_params)
count_b = sum(p.numel() for _, p in b_params)
count_c = sum(p.numel() for _, p in c_params)

print(f"(a) Backbone pretrain (Stem + 4 ConvNeXt stages + 4 ViT blocks): {count_a:,} ({count_a/total_unique_params*100:.2f}%)")
print(f"(b) Riêng của SAGE (8 Routers + SA-Hubs/Adapters):                  {count_b:,} ({count_b/total_unique_params*100:.2f}%)")
print(f"(c) Decoder + Head + S2-Gate + Projections / Norms / PE:           {count_c:,} ({count_c/total_unique_params*100:.2f}%)")
total_sum = count_a + count_b + count_c
print(f"Tổng cộng: {total_sum:,} (khớp 10,205,755? {total_sum == 10205755})")

# Phân rã chi tiết nhóm (a)
print("\nChi tiết nhóm (a) Backbone Pretrain:")
a_by_sub = {}
for name, p in a_params:
    prefix = name.split('_')[0] if '_' in name else name
    a_by_sub[prefix] = a_by_sub.get(prefix, 0) + p.numel()
for k, v in a_by_sub.items():
    print(f"  - {k}: {v:,}")

# Phân rã chi tiết nhóm (b)
print("\nChi tiết nhóm (b) SAGE riêng:")
b_routers = sum(p.numel() for name, p in b_params if 'router' in name)
b_sahub = sum(p.numel() for name, p in b_params if 'sa_hub' in name or 'adapter' in name)
print(f"  - 8 Routers: {b_routers:,}")
print(f"  - SA-Hubs / Format Adapters: {b_sahub:,}")

# Phân rã chi tiết nhóm (c)
print("\nChi tiết nhóm (c) Decoder / Head / S2-Gate / Projections:")
c_groups = {}
for name, p in c_params:
    mod_name = name.split('.')[0]
    if mod_name == 'backbone':
        mod_name = '.'.join(name.split('.')[:2])
    c_groups[mod_name] = c_groups.get(mod_name, 0) + p.numel()
for k, v in sorted(c_groups.items(), key=lambda x: -x[1]):
    print(f"  - {k}: {v:,}")

# 3. Đo grad-norm sau 1 batch
print("\n--- 3. Đo Gradient Norm sau 1 batch ---")

def test_grad_norms(residual_scale_val):
    m = create_b2_unet(
        num_transformer_layers=int(config.get('num_transformer_layers', 4)),
        pretrained=False,
        sage_config=sage_cfg,
        p3_mode=config.get('p3_mode'),
        use_plu_head=config.get('use_plu_head', False),
        use_s2_gate=config.get('use_s2_gate', False),
        s2_gate_kernel_size=config.get('s2_gate_kernel_size', 3),
        use_s2_gate_block2=config.get('use_s2_gate_block2', False),
        s2_gate_block2_kernel_size=config.get('s2_gate_block2_kernel_size', 3),
        s2_gate_block2_dilation=config.get('s2_gate_block2_dilation', 1),
    )
    
    # Set residual scale
    for mod in m.modules():
        if isinstance(mod, SageLayer):
            mod.residual_scale = float(residual_scale_val)
            
    m.train()
    
    a_p, b_p, c_p = categorize_model_parameters(m)
    id_set_a = set(id(p) for _, p in a_p)
    id_set_b = set(id(p) for _, p in b_p)
    id_set_c = set(id(p) for _, p in c_p)
    
    torch.manual_seed(42)
    x = torch.randn(2, 3, 448, 448)
    target = torch.randint(0, 2, (2, 1, 448, 448)).float()
    
    criterion = nn.BCEWithLogitsLoss()
    out = m(x)
    loss = criterion(out, target)
    loss.backward()
    
    def calc_norm(id_set):
        grads = []
        zero_cnt = 0
        total_cnt = 0
        for p in m.parameters():
            if id(p) in id_set:
                if p.grad is not None:
                    g = p.grad.detach().flatten()
                    grads.append(g)
                    zero_cnt += int(torch.sum(g == 0).item())
                    total_cnt += g.numel()
                else:
                    total_cnt += p.numel()
                    zero_cnt += p.numel()
        if not grads:
            return 0.0, total_cnt, zero_cnt
        cat_grads = torch.cat(grads)
        norm = float(torch.norm(cat_grads).item())
        return norm, total_cnt, zero_cnt
        
    norm_a, tot_a, z_a = calc_norm(id_set_a)
    norm_b, tot_b, z_b = calc_norm(id_set_b)
    norm_c, tot_c, z_c = calc_norm(id_set_c)
    
    print(f"\nKết quả với residual_scale = {residual_scale_val}:")
    print(f"  (a) Backbone pretrain : Grad Norm = {norm_a:.6f} (tổng params: {tot_a:,}, params có grad==0: {z_a:,})")
    print(f"  (b) SAGE riêng        : Grad Norm = {norm_b:.6f} (tổng params: {tot_b:,}, params có grad==0: {z_b:,})")
    print(f"  (c) Decoder + Head    : Grad Norm = {norm_c:.6f} (tổng params: {tot_c:,}, params có grad==0: {z_c:,})")
    return (norm_a, norm_b, norm_c)

print("\n--- THỰC THI KIỂM CHỨNG GRAD NORM ---")
test_grad_norms(0.1)
test_grad_norms(0.0)
