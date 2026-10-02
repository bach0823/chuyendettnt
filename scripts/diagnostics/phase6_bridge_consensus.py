#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_bridge_consensus.py

Task A: 7-Model Bridge Consensus Diagnostic
Answers: Are the ~109-115 bridge counts stable because the SAME validation images
repeatedly fail across models, or because different models fail on different images?

Models evaluated:
1. Candidate B (Base)
2. Phase 6-A.1 (B-IoU)
3. Phase 6-A.2 (PLU)
4. Phase 6-B.1 (AB-BPL)
5. Phase 6-C.1 (clDice)
6. Phase 6-D.1 (PLU+AB-BPL)
7. Phase 6-D.2 (Pure Separation)
"""

import os
import sys
import numpy as np
import pandas as pd

def main():
    master_csv = 'results/diagnostics/phase6_d2_topology/topology_7models_master_paired.csv'
    geom_csv = 'results/diagnostics/phase6_d2_geometry/sample_cc_geometry.csv'
    out_dir = 'results/diagnostics/phase6_bridge_localization'
    os.makedirs(out_dir, exist_ok=True)

    out_per_image = os.path.join(out_dir, 'bridge_consensus_per_image.csv')
    out_summary = os.path.join(out_dir, 'bridge_consensus_summary.csv')
    out_jaccard = os.path.join(out_dir, 'bridge_model_pairwise_jaccard.csv')

    df_master = pd.read_csv(master_csv)
    df_geom = pd.read_csv(geom_csv)

    models = ['Base', 'A1', 'A2', 'B1', 'C1', 'D1', 'D2']
    model_names = {
        'Base': 'Candidate B (Base)',
        'A1': 'Phase 6-A.1 (B-IoU)',
        'A2': 'Phase 6-A.2 (PLU)',
        'B1': 'Phase 6-B.1 (AB-BPL)',
        'C1': 'Phase 6-C.1 (clDice)',
        'D1': 'Phase 6-D.1 (PLU+AB-BPL)',
        'D2': 'Phase 6-D.2 (Pure Sep)'
    }

    # Verify sanity checks
    assert len(df_master) == 348, f"Expected 348 rows, found {len(df_master)}"
    expected_counts = {
        'Base': 110, 'A1': 114, 'A2': 111, 'B1': 115,
        'C1': 113, 'D1': 112, 'D2': 109
    }
    actual_counts = {}
    for m in models:
        cnt = int((df_master[f'{m}_bridge_events'] > 0).sum())
        actual_counts[m] = cnt
        assert cnt == expected_counts[m], f"Mismatch for {m}: expected {expected_counts[m]}, got {cnt}"

    print("[Sanity Check PASS] All 7 models have exactly 348 samples and match official bridge totals:")
    for m in models:
        print(f"  - {model_names[m]:26s}: {actual_counts[m]}/348 ({actual_counts[m]/348*100:.2f}%)")

    # 1. Per-image table
    per_image_dict = {
        'image_id': df_master['case_name'],
        'category': df_master['candidate_b_category'],
        'Base_dice': df_master['Base_dice'],
        'gt_area': df_master['gt_area'],
        'gt_cc': df_master['gt_cc']
    }
    for m in models:
        per_image_dict[f'{m}_bridge'] = (df_master[f'{m}_bridge_events'] > 0).astype(int)

    df_per_image = pd.DataFrame(per_image_dict)
    df_per_image['n_models_with_bridge'] = df_per_image[[f'{m}_bridge' for m in models]].sum(axis=1)

    # Merge geometry
    df_per_image = pd.merge(df_per_image, df_geom[['case_name', 'min_gap', 'median_gap']], 
                            left_on='image_id', right_on='case_name', how='left').drop(columns=['case_name'])

    # Categorize transition vs Base & D2
    base_br = df_per_image['Base_bridge'] == 1
    d1_br = df_per_image['D1_bridge'] == 1
    d2_br = df_per_image['D2_bridge'] == 1

    transition_types = []
    for _, row in df_per_image.iterrows():
        b = row['Base_bridge'] == 1
        d1 = row['D1_bridge'] == 1
        d2 = row['D2_bridge'] == 1
        if b and d2:
            transition_types.append('Persistent_Base_to_D2')
        elif b and not d2:
            transition_types.append('Cured_by_D2')
        elif not b and d2:
            transition_types.append('Created_by_D2')
        else:
            transition_types.append('Clean_Base_and_D2')
    df_per_image['d2_transition_type'] = transition_types

    df_per_image.to_csv(out_per_image, index=False)
    print(f"\nSaved per-image table: {out_per_image} ({len(df_per_image)} rows)")

    # 2. Consensus Distribution (0/7 to 7/7)
    consensus_rows = []
    total_samples = len(df_per_image)
    for k in range(8):
        sub = df_per_image[df_per_image['n_models_with_bridge'] == k]
        count = len(sub)
        pct = count / total_samples * 100.0
        mean_dice = sub['Base_dice'].mean() if count > 0 else np.nan
        med_dice = sub['Base_dice'].median() if count > 0 else np.nan
        mean_gap = sub['min_gap'].mean() if count > 0 else np.nan
        med_gap = sub['min_gap'].median() if count > 0 else np.nan
        consensus_rows.append({
            'k_models_with_bridge': f'{k}/7',
            'k_int': k,
            'count': count,
            'pct_of_total': round(pct, 2),
            'base_dice_mean': round(mean_dice, 4) if not np.isnan(mean_dice) else None,
            'base_dice_median': round(med_dice, 4) if not np.isnan(med_dice) else None,
            'min_gap_mean': round(mean_gap, 2) if not np.isnan(mean_gap) else None,
            'min_gap_median': round(med_gap, 2) if not np.isnan(med_gap) else None,
        })
    df_consensus_dist = pd.DataFrame(consensus_rows)

    # 3. Intersections (>=4/7, >=5/7, >=6/7, 7/7)
    intersection_rows = []
    for cutoff, label in [(7, 'All 7 models (7/7)'),
                          (6, 'At least 6 models (>=6/7)'),
                          (5, 'At least 5 models (>=5/7)'),
                          (4, 'At least 4 models (>=4/7)'),
                          (1, 'At least 1 model (>=1/7, Any Bridge)')]:
        sub = df_per_image[df_per_image['n_models_with_bridge'] >= cutoff]
        count = len(sub)
        pct = count / total_samples * 100.0
        mean_dice = sub['Base_dice'].mean()
        med_dice = sub['Base_dice'].median()
        mean_gap = sub['min_gap'].dropna().mean()
        med_gap = sub['min_gap'].dropna().median()
        cats = sub['category'].value_counts().to_dict()
        intersection_rows.append({
            'subset': label,
            'threshold_k': cutoff,
            'count': count,
            'pct_of_total': round(pct, 2),
            'base_dice_mean': round(mean_dice, 4),
            'base_dice_median': round(med_dice, 4),
            'mean_min_gap_px': round(mean_gap, 2),
            'median_min_gap_px': round(med_gap, 2),
            'category_distribution': str(cats)
        })
    df_intersections = pd.DataFrame(intersection_rows)

    # Save summary containing both distribution and intersections
    with open(out_summary, 'w', encoding='utf-8') as f:
        f.write("# 7-Model Bridge Consensus Summary\n\n")
        f.write("## 1. Consensus Distribution (0/7 to 7/7)\n")
        df_consensus_dist.to_csv(f, index=False)
        f.write("\n## 2. Cumulative Intersections\n")
        df_intersections.to_csv(f, index=False)

    print(f"Saved consensus summary: {out_summary}")

    # 4. Pairwise Model Jaccard Overlap
    jaccard_rows = []
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            s1 = set(df_per_image[df_per_image[f'{m1}_bridge'] == 1]['image_id'])
            s2 = set(df_per_image[df_per_image[f'{m2}_bridge'] == 1]['image_id'])
            inter = len(s1 & s2)
            union = len(s1 | s2)
            jaccard = inter / union if union > 0 else 1.0
            jaccard_rows.append({
                'model_A': m1,
                'model_A_name': model_names[m1],
                'model_B': m2,
                'model_B_name': model_names[m2],
                'count_A': len(s1),
                'count_B': len(s2),
                'intersection_count': inter,
                'union_count': union,
                'jaccard_similarity': round(jaccard, 4)
            })
    df_jaccard = pd.DataFrame(jaccard_rows)
    df_jaccard.to_csv(out_jaccard, index=False)
    print(f"Saved pairwise Jaccard matrix: {out_jaccard}")

    # 5. Transition-Consensus Subgroup Analysis
    print("\n" + "=" * 100)
    print("TRANSITION-CONSENSUS SUBGROUPS")
    print("=" * 100)
    # A. Base bridge + all/most models bridge
    sub_base_all = df_per_image[(df_per_image['Base_bridge'] == 1) & (df_per_image['n_models_with_bridge'] == 7)]
    sub_base_ge6 = df_per_image[(df_per_image['Base_bridge'] == 1) & (df_per_image['n_models_with_bridge'] >= 6)]
    # B. Base bridge + D2 bridge (Persistent)
    sub_base_d2_persist = df_per_image[(df_per_image['Base_bridge'] == 1) & (df_per_image['D2_bridge'] == 1)]
    # C. Base bridge + D2 cured
    sub_base_d2_cured = df_per_image[(df_per_image['Base_bridge'] == 1) & (df_per_image['D2_bridge'] == 0)]
    # D. Base clean + D2 created
    sub_base_clean_d2_created = df_per_image[(df_per_image['Base_bridge'] == 0) & (df_per_image['D2_bridge'] == 1)]
    # E. D1 persistent + D2 persistent
    d1_persist_mask = (df_per_image['Base_bridge'] == 1) & (df_per_image['D1_bridge'] == 1)
    sub_d1_persist_d2_persist = df_per_image[d1_persist_mask & (df_per_image['D2_bridge'] == 1)]
    # F. D1 persistent + D2 cured
    sub_d1_persist_d2_cured = df_per_image[d1_persist_mask & (df_per_image['D2_bridge'] == 0)]

    subgroups = [
        ("Base bridge in ALL 7 models (7/7)", sub_base_all),
        ("Base bridge in >=6 models (>=6/7)", sub_base_ge6),
        ("Base bridge + D2 bridge (Persistent vs Base)", sub_base_d2_persist),
        ("Base bridge + D2 cured (Cured by D2)", sub_base_d2_cured),
        ("Base clean + D2 created (Created by D2)", sub_base_clean_d2_created),
        ("D1 persistent + D2 persistent (Persistent under both)", sub_d1_persist_d2_persist),
        ("D1 persistent + D2 cured (Resolved by D2)", sub_d1_persist_d2_cured),
    ]

    for name, sub in subgroups:
        n = len(sub)
        d_mean = sub['Base_dice'].mean() if n > 0 else 0
        g_med = sub['min_gap'].dropna().median() if n > 0 else 0
        g_mean = sub['min_gap'].dropna().mean() if n > 0 else 0
        print(f"  - {name:50s} : N={n:3d} ({n/total_samples*100:5.2f}%) | Base Dice: {d_mean:.4f} | Min Gap (med/mean): {g_med:.2f}/{g_mean:.2f} px")

    # Print Consensus Distribution table
    print("\n" + "=" * 100)
    print("CONSENSUS DISTRIBUTION ACROSS 7 MODELS (N=348)")
    print("=" * 100)
    print(df_consensus_dist.to_string(index=False))

    # Print Cumulative Intersections table
    print("\n" + "=" * 100)
    print("CUMULATIVE INTERSECTIONS")
    print("=" * 100)
    print(df_intersections[['subset', 'count', 'pct_of_total', 'base_dice_mean', 'median_min_gap_px', 'category_distribution']].to_string(index=False))

    # Print Pairwise Jaccard Summary
    print("\n" + "=" * 100)
    print("PAIRWISE JACCARD OVERLAP SUMMARY")
    print("=" * 100)
    pivot_jaccard = df_jaccard.pivot(index='model_A', columns='model_B', values='jaccard_similarity')
    print(pivot_jaccard.to_string())
    print("\nJaccard range across distinct model pairs (A != B):")
    distinct_jaccards = df_jaccard[df_jaccard['model_A'] != df_jaccard['model_B']]['jaccard_similarity']
    print(f"  Min Jaccard: {distinct_jaccards.min():.4f}")
    print(f"  Max Jaccard: {distinct_jaccards.max():.4f}")
    print(f"  Mean Jaccard: {distinct_jaccards.mean():.4f}")
    print(f"  Base vs D2 Jaccard: {pivot_jaccard.loc['Base', 'D2']:.4f} (Inter: {len(sub_base_d2_persist)}, Union: {len(set(df_per_image[df_per_image['Base_bridge']==1]['image_id']) | set(df_per_image[df_per_image['D2_bridge']==1]['image_id']))})")
    print("=" * 100)

if __name__ == '__main__':
    main()
