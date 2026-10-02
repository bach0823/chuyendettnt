#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_routing_bridge_association.py

Task D: Routing Association Diagnostic for False Bridges
Answers:
- Do the 101 consensus bridge cases show anomalous expert selections, unusual gate weights (g_s),
  or router ambivalence compared to the 228 consensus clean cases?
- Did Phase 6-D.2 alter routing on the 101 persistent cases, or did routing remain frozen?
- What distinguishes the routing signature of the 7 cured cases vs the 101 persistent cases?

Data sources:
- Base routing: results/P3_C_Routing_Diagnostics_D4_K2_H64_Phase5_SAGELR2e-4/diagnostics/full_val/routing_statistics.json
- D2 routing: results/P3_C_Routing_Diagnostics_Phase6_D2_Pure_Sep_D4_K2_H64/diagnostics/full_val/routing_statistics.json
- Bridge consensus: results/diagnostics/phase6_bridge_localization/bridge_consensus_per_image.csv
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    out_dir = 'results/diagnostics/phase6_bridge_localization_v2'
    os.makedirs(out_dir, exist_ok=True)

    base_json_path = 'results/P3_C_Routing_Diagnostics_D4_K2_H64_Phase5_SAGELR2e-4/diagnostics/full_val/routing_statistics.json'
    d2_json_path = 'results/P3_C_Routing_Diagnostics_Phase6_D2_Pure_Sep_D4_K2_H64/diagnostics/full_val/routing_statistics.json'
    meta_csv = 'results/diagnostics/phase6_bridge_localization/bridge_consensus_per_image.csv'

    assert os.path.exists(base_json_path), f"Missing {base_json_path}"
    assert os.path.exists(d2_json_path), f"Missing {d2_json_path}"
    assert os.path.exists(meta_csv), f"Missing {meta_csv}"

    with open(base_json_path, 'r') as f:
        base_data = json.load(f)
    with open(d2_json_path, 'r') as f:
        d2_data = json.load(f)

    df_meta = pd.read_csv(meta_csv)
    # Map case_name to metadata
    case_meta = df_meta.set_index('image_id').to_dict(orient='index')

    base_records = {r['case_name']: r for r in base_data.get('sample_routing_records', [])}
    d2_records = {r['case_name']: r for r in d2_data.get('sample_routing_records', [])}

    print(f"Loaded {len(base_records)} Base routing records and {len(d2_records)} D2 routing records.")
    assert len(base_records) == 348, f"Expected 348 Base records, got {len(base_records)}"

    # 1. Build per-sample routing feature table
    router_names = [
        'convnext.stage_0', 'convnext.stage_1', 'convnext.stage_2', 'convnext.stage_3',
        'transformer.block_0', 'transformer.block_1', 'transformer.block_2', 'transformer.block_3'
    ]

    per_sample_rows = []
    for case_name, meta in case_meta.items():
        b_rec = base_records.get(case_name, {})
        d2_rec = d2_records.get(case_name, {})

        is_c7 = (meta['n_models_with_bridge'] == 7)
        is_clean228 = (meta['n_models_with_bridge'] == 0)
        d2_trans = meta['d2_transition_type']

        row = {
            'image_id': case_name,
            'is_7_consensus': int(is_c7),
            'is_clean_consensus': int(is_clean228),
            'd2_transition': d2_trans,
            'category': meta['category'],
            'gt_cc': meta['gt_cc'],
            'Base_dice': meta['Base_dice'],
        }

        # Base router features
        b_gs_list = []
        b_margin_list = []
        for r_entry in b_rec.get('routing', []):
            r_name = r_entry['router']
            gs = r_entry['g_s']
            sc = r_entry['selected_scores']
            margin = abs(sc[0] - sc[1]) if len(sc) >= 2 else 0.0
            row[f'base_{r_name}_gs'] = round(gs, 4)
            row[f'base_{r_name}_exp0'] = r_entry['selected_experts'][0]
            row[f'base_{r_name}_exp1'] = r_entry['selected_experts'][1]
            row[f'base_{r_name}_margin'] = round(margin, 4)
            b_gs_list.append(gs)
            b_margin_list.append(margin)

        row['base_mean_gs'] = round(float(np.mean(b_gs_list)), 4) if b_gs_list else np.nan
        row['base_mean_margin'] = round(float(np.mean(b_margin_list)), 4) if b_margin_list else np.nan

        # D2 router features and comparison with Base
        identical_expert_count = 0
        d2_gs_list = []
        for i_r, r_entry in enumerate(d2_rec.get('routing', [])):
            r_name = r_entry['router']
            gs = r_entry['g_s']
            row[f'd2_{r_name}_gs'] = round(gs, 4)
            d2_gs_list.append(gs)

            # Check if selected experts changed
            if i_r < len(b_rec.get('routing', [])):
                b_exps = set(b_rec['routing'][i_r]['selected_experts'])
                d2_exps = set(r_entry['selected_experts'])
                if b_exps == d2_exps:
                    identical_expert_count += 1

        row['d2_mean_gs'] = round(float(np.mean(d2_gs_list)), 4) if d2_gs_list else np.nan
        row['d2_vs_base_identical_routers_out_of_8'] = identical_expert_count
        row['d2_vs_base_gs_diff'] = round(float(row['d2_mean_gs'] - row['base_mean_gs']), 4) if not np.isnan(row['d2_mean_gs']) and not np.isnan(row['base_mean_gs']) else np.nan

        per_sample_rows.append(row)

    df_per_sample = pd.DataFrame(per_sample_rows)
    out_sample_path = os.path.join(out_dir, 'routing_bridge_association.csv')
    df_per_sample.to_csv(out_sample_path, index=False)
    print(f"Saved: {out_sample_path} ({len(df_per_sample)} rows)")

    # 2. Statistical Comparison: 101 Consensus Bridge vs 228 Consensus Clean
    sub_bridge101 = df_per_sample[df_per_sample['is_7_consensus'] == 1]
    sub_clean228 = df_per_sample[df_per_sample['is_clean_consensus'] == 1]
    sub_cured7 = df_per_sample[df_per_sample['d2_transition'] == 'Cured_by_D2']
    sub_persistent103 = df_per_sample[df_per_sample['d2_transition'] == 'Persistent_Base_to_D2']

    summary_rows = []

    # Features to compare
    test_features = ['base_mean_gs', 'base_mean_margin']
    for r_name in router_names:
        test_features.append(f'base_{r_name}_gs')
        test_features.append(f'base_{r_name}_margin')

    for feat in test_features:
        v_b = sub_bridge101[feat].dropna().values
        v_c = sub_clean228[feat].dropna().values
        v_cur = sub_cured7[feat].dropna().values

        mean_b = float(np.mean(v_b)) if len(v_b) > 0 else np.nan
        std_b = float(np.std(v_b)) if len(v_b) > 0 else np.nan
        mean_c = float(np.mean(v_c)) if len(v_c) > 0 else np.nan
        std_c = float(np.std(v_c)) if len(v_c) > 0 else np.nan
        mean_cur = float(np.mean(v_cur)) if len(v_cur) > 0 else np.nan

        # Mann-Whitney U test (non-parametric) and Welch t-test
        if len(v_b) > 0 and len(v_c) > 0:
            mwu_stat, p_val = stats.mannwhitneyu(v_b, v_c, alternative='two-sided')
            diff = mean_b - mean_c
        else:
            p_val = np.nan
            diff = np.nan

        summary_rows.append({
            'metric': feat,
            'mean_consensus_bridge_101': round(mean_b, 4),
            'std_consensus_bridge_101': round(std_b, 4),
            'mean_consensus_clean_228': round(mean_c, 4),
            'std_consensus_clean_228': round(std_c, 4),
            'difference (bridge - clean)': round(diff, 4),
            'mann_whitney_p_value': round(float(p_val), 4) if not np.isnan(p_val) else np.nan,
            'statistically_significant_05': bool(p_val < 0.05) if not np.isnan(p_val) else False,
            'mean_cured_7cases': round(mean_cur, 4),
        })

    df_summary = pd.DataFrame(summary_rows)
    out_summary_path = os.path.join(out_dir, 'routing_bridge_summary.csv')
    df_summary.to_csv(out_summary_path, index=False)
    print(f"Saved: {out_summary_path}")

    # 3. Router Agreement Matrix: How much did D2 change routing vs Base?
    d2_agree_101 = sub_bridge101['d2_vs_base_identical_routers_out_of_8'].mean()
    d2_agree_clean = sub_clean228['d2_vs_base_identical_routers_out_of_8'].mean()
    d2_agree_cured = sub_cured7['d2_vs_base_identical_routers_out_of_8'].mean()

    print("\n" + "=" * 80)
    print("TASK D: ROUTING ASSOCIATION SUMMARY:")
    print("=" * 80)
    print(f"1. Base Mean Gate Scale (g_s):")
    print(f"   - 101 Consensus Bridge: {df_per_sample[df_per_sample['is_7_consensus']==1]['base_mean_gs'].mean():.4f}")
    print(f"   - 228 Consensus Clean:  {df_per_sample[df_per_sample['is_clean_consensus']==1]['base_mean_gs'].mean():.4f}")
    print(f"   - 7 Cured by D2:        {df_per_sample[df_per_sample['d2_transition']=='Cured_by_D2']['base_mean_gs'].mean():.4f}")
    print(f"2. Base Mean Routing Margin (|top1 - top2| score):")
    print(f"   - 101 Consensus Bridge: {df_per_sample[df_per_sample['is_7_consensus']==1]['base_mean_margin'].mean():.4f}")
    print(f"   - 228 Consensus Clean:  {df_per_sample[df_per_sample['is_clean_consensus']==1]['base_mean_margin'].mean():.4f}")
    print(f"3. D2 vs Base Routing Identity (identical routers out of 8):")
    print(f"   - 101 Consensus Bridge: {d2_agree_101:.2f} / 8 ({d2_agree_101/8*100:.1f}%)")
    print(f"   - 228 Consensus Clean:  {d2_agree_clean:.2f} / 8 ({d2_agree_clean/8*100:.1f}%)")
    print(f"   - 7 Cured by D2:        {d2_agree_cured:.2f} / 8 ({d2_agree_cured/8*100:.1f}%)")

if __name__ == '__main__':
    main()
