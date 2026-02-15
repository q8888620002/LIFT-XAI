import json
import os
from pathlib import Path
from collections import defaultdict

# Define cohorts and file patterns
cohorts = ['crash_2', 'ist3', 'sprint', 'accord']
base_path = Path('/homes/gws/mingyulu/shap_IPW/docs/agent')

# Store results
results = {
    'with_shap': defaultdict(lambda: {'support': 0, 'neutral': 0, 'conflict': 0, 'total': 0}),
    'without_shap': defaultdict(lambda: {'support': 0, 'neutral': 0, 'conflict': 0, 'total': 0})
}

# Process each cohort
for cohort in cohorts:
    cohort_path = base_path / cohort

    # Find the pubmed_validation files
    # Use DRLearner_revised for ist3 and crash_2, XLearner for others
    if cohort in ['ist3', 'crash_2']:
        with_shap_file = cohort_path / 'hypotheses_with_shap_DRLearner_revised_pubmed_validation.json'
    else:
        with_shap_file = cohort_path / 'hypotheses_with_shap_XLearner_pubmed_validation.json'
    
    without_shap_file = cohort_path / 'hypotheses_without_shap_baseline_pubmed_validation.json'

    # Process with_shap file
    if with_shap_file.exists():
        with open(with_shap_file, 'r') as f:
            data = json.load(f)
            support = data.get('overall_support_count', 0)
            neutral = data.get('overall_neutral_count', 0)
            conflict = data.get('overall_conflict_count', 0)
            total = support + neutral + conflict

            results['with_shap'][cohort]['support'] = support
            results['with_shap'][cohort]['neutral'] = neutral
            results['with_shap'][cohort]['conflict'] = conflict
            results['with_shap'][cohort]['total'] = total
            results['with_shap']['all']['support'] += support
            results['with_shap']['all']['neutral'] += neutral
            results['with_shap']['all']['conflict'] += conflict
            results['with_shap']['all']['total'] += total
    else:
        print(f"Warning: {with_shap_file} not found")

    # Process without_shap file
    if without_shap_file.exists():
        with open(without_shap_file, 'r') as f:
            data = json.load(f)
            support = data.get('overall_support_count', 0)
            neutral = data.get('overall_neutral_count', 0)
            conflict = data.get('overall_conflict_count', 0)
            total = support + neutral + conflict

            results['without_shap'][cohort]['support'] = support
            results['without_shap'][cohort]['neutral'] = neutral
            results['without_shap'][cohort]['conflict'] = conflict
            results['without_shap'][cohort]['total'] = total
            results['without_shap']['all']['support'] += support
            results['without_shap']['all']['neutral'] += neutral
            results['without_shap']['all']['conflict'] += conflict
            results['without_shap']['all']['total'] += total
    else:
        print(f"Warning: {without_shap_file} not found")

# Print results
print("=" * 80)
print("NEUTRAL RATIO ANALYSIS - PubMed Validation Results")
print("=" * 80)
print()

print("WITH SHAP (AI-Generated Hypotheses)")
print("-" * 80)
for cohort in cohorts + ['all']:
    if cohort == 'all':
        print("-" * 80)
    data = results['with_shap'][cohort]
    if data['total'] > 0:
        neutral_ratio = data['neutral'] / data['total'] * 100
        support_ratio = data['support'] / data['total'] * 100
        conflict_ratio = data['conflict'] / data['total'] * 100
        print(f"{cohort.upper():12} - Support: {data['support']:4d} ({support_ratio:5.1f}%)  "
              f"Neutral: {data['neutral']:4d} ({neutral_ratio:5.1f}%)  "
              f"Conflict: {data['conflict']:4d} ({conflict_ratio:5.1f}%)  "
              f"Total: {data['total']:4d}")

print()
print("WITHOUT SHAP (Literature-Based Baseline)")
print("-" * 80)
for cohort in cohorts + ['all']:
    if cohort == 'all':
        print("-" * 80)
    data = results['without_shap'][cohort]
    if data['total'] > 0:
        neutral_ratio = data['neutral'] / data['total'] * 100
        support_ratio = data['support'] / data['total'] * 100
        conflict_ratio = data['conflict'] / data['total'] * 100
        print(f"{cohort.upper():12} - Support: {data['support']:4d} ({support_ratio:5.1f}%)  "
              f"Neutral: {data['neutral']:4d} ({neutral_ratio:5.1f}%)  "
              f"Conflict: {data['conflict']:4d} ({conflict_ratio:5.1f}%)  "
              f"Total: {data['total']:4d}")

print()
print("=" * 80)
print("SUPPORT RATE BY DATASET")
print("=" * 80)

for cohort in cohorts:
    print(f"\n{cohort.upper()}")
    print("-" * 80)
    
    ws_data = results['with_shap'][cohort]
    wos_data = results['without_shap'][cohort]
    
    if ws_data['total'] > 0 or wos_data['total'] > 0:
        # Option 1: Including neutral
        print("Support Rate (including neutral): support / (support + neutral + conflict)")
        
        if ws_data['total'] > 0:
            ws_rate = ws_data['support'] / ws_data['total'] * 100
            print(f"  WITH SHAP:    {ws_rate:5.1f}% ({ws_data['support']}/{ws_data['total']})")
        else:
            print(f"  WITH SHAP:    N/A (no data)")
            
        if wos_data['total'] > 0:
            wos_rate = wos_data['support'] / wos_data['total'] * 100
            print(f"  WITHOUT SHAP: {wos_rate:5.1f}% ({wos_data['support']}/{wos_data['total']})")
        else:
            print(f"  WITHOUT SHAP: N/A (no data)")
        
        if ws_data['total'] > 0 and wos_data['total'] > 0:
            diff = wos_rate - ws_rate
            print(f"  Difference:   {diff:+5.1f} percentage points")
        
        # Option 2: Excluding neutral
        ws_relevant = ws_data['support'] + ws_data['conflict']
        wos_relevant = wos_data['support'] + wos_data['conflict']
        
        print("\nSupport Rate (excluding neutral): support / (support + conflict)")
        
        if ws_relevant > 0:
            ws_rate_no_neutral = ws_data['support'] / ws_relevant * 100
            print(f"  WITH SHAP:    {ws_rate_no_neutral:5.1f}% ({ws_data['support']}/{ws_relevant})")
        else:
            print(f"  WITH SHAP:    N/A (no relevant abstracts)")
            
        if wos_relevant > 0:
            wos_rate_no_neutral = wos_data['support'] / wos_relevant * 100
            print(f"  WITHOUT SHAP: {wos_rate_no_neutral:5.1f}% ({wos_data['support']}/{wos_relevant})")
        else:
            print(f"  WITHOUT SHAP: N/A (no relevant abstracts)")
        
        if ws_relevant > 0 and wos_relevant > 0:
            diff_no_neutral = wos_rate_no_neutral - ws_rate_no_neutral
            print(f"  Difference:   {diff_no_neutral:+5.1f} percentage points")

print()
print("=" * 80)
print("OVERALL COMPARISON SUMMARY")
print("=" * 80)
with_shap_all = results['with_shap']['all']
without_shap_all = results['without_shap']['all']

if with_shap_all['total'] > 0 and without_shap_all['total'] > 0:
    print("\nNeutral Ratio (neutral / total):")
    print(f"  WITH SHAP:    {with_shap_all['neutral']/with_shap_all['total']*100:.2f}%")
    print(f"  WITHOUT SHAP: {without_shap_all['neutral']/without_shap_all['total']*100:.2f}%")
    print(f"  Difference: {(with_shap_all['neutral']/with_shap_all['total'] - without_shap_all['neutral']/without_shap_all['total'])*100:+.2f} percentage points")

    # Option 1: Support rate including neutral (support / total)
    ws_support_with_neutral = with_shap_all['support'] / with_shap_all['total'] * 100
    wos_support_with_neutral = without_shap_all['support'] / without_shap_all['total'] * 100

    print("\n" + "-" * 80)
    print("OPTION 1: Support Rate (including neutral in denominator)")
    print("          Formula: support / (support + neutral + conflict)")
    print("-" * 80)
    print(f"  WITH SHAP:    {ws_support_with_neutral:.2f}% ({with_shap_all['support']}/{with_shap_all['total']})")
    print(f"  WITHOUT SHAP: {wos_support_with_neutral:.2f}% ({without_shap_all['support']}/{without_shap_all['total']})")
    print(f"  Difference: {wos_support_with_neutral - ws_support_with_neutral:+.2f} percentage points")
    print("  Interpretation: What % of ALL retrieved abstracts support the mechanism?")

    # Option 2: Support rate excluding neutral (support / (support + conflict))
    ws_relevant = with_shap_all['support'] + with_shap_all['conflict']
    wos_relevant = without_shap_all['support'] + without_shap_all['conflict']

    if ws_relevant > 0 and wos_relevant > 0:
        ws_support_no_neutral = with_shap_all['support'] / ws_relevant * 100
        wos_support_no_neutral = without_shap_all['support'] / wos_relevant * 100

        print("\n" + "-" * 80)
        print("OPTION 2: Support Rate (excluding neutral)")
        print("          Formula: support / (support + conflict)")
        print("-" * 80)
        print(f"  WITH SHAP:    {ws_support_no_neutral:.2f}% ({with_shap_all['support']}/{ws_relevant})")
        print(f"  WITHOUT SHAP: {wos_support_no_neutral:.2f}% ({without_shap_all['support']}/{wos_relevant})")
        print(f"  Difference: {wos_support_no_neutral - ws_support_no_neutral:+.2f} percentage points")
        print("  Interpretation: Among abstracts that addressed the mechanism, what % support it?")

        # Conflict rate
        ws_conflict_rate = with_shap_all['conflict'] / ws_relevant * 100
        wos_conflict_rate = without_shap_all['conflict'] / wos_relevant * 100
        print(f"\n  Conflict Rate WITH SHAP:    {ws_conflict_rate:.2f}%")
        print(f"  Conflict Rate WITHOUT SHAP: {wos_conflict_rate:.2f}%")

print()
