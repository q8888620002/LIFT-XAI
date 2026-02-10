#!/usr/bin/env python3
"""compare_judge_versions.py

Compare hypothesis scores BEFORE (original) vs AFTER (revised) judge evaluation.

Usage:
    # Compare single cohort
    python compare_judge_versions.py \
        --original docs/agent/crash_2/hypotheses_with_shap_XLearner_judge_original.json \
        --revised docs/agent/crash_2/hypotheses_with_shap_XLearner_judge_revised.json \
        --out_csv docs/crash_2_original_vs_revised.csv \
        --plot

    # Compare all cohorts (WITH SHAP)
    python compare_judge_versions.py \
        --original \
            docs/agent/crash_2/hypotheses_with_shap_XLearner_judge_original.json \
            docs/agent/ist3/hypotheses_with_shap_XLearner_judge_original.json \
            docs/agent/sprint/hypotheses_with_shap_XLearner_judge_original.json \
            docs/agent/accord/hypotheses_with_shap_XLearner_judge_original.json \
        --revised \
            docs/agent/crash_2/hypotheses_with_shap_XLearner_judge_revised.json \
            docs/agent/ist3/hypotheses_with_shap_XLearner_judge_revised.json \
            docs/agent/sprint/hypotheses_with_shap_XLearner_judge_revised.json \
            docs/agent/accord/hypotheses_with_shap_XLearner_judge_revised.json \
        --out_csv docs/with_shap_original_vs_revised.csv \
        --plot

    # Compare all cohorts (WITHOUT SHAP baseline)
    python compare_judge_versions.py \
        --original \
            docs/agent/crash_2/hypotheses_without_shap_baseline_judge_original.json \
            docs/agent/ist3/hypotheses_without_shap_baseline_judge_original.json \
            docs/agent/sprint/hypotheses_without_shap_baseline_judge_original.json \
            docs/agent/accord/hypotheses_without_shap_baseline_judge_original.json \
        --revised \
            docs/agent/crash_2/hypotheses_without_shap_baseline_judge_revised.json \
            docs/agent/ist3/hypotheses_without_shap_baseline_judge_revised.json \
            docs/agent/sprint/hypotheses_without_shap_baseline_judge_revised.json \
            docs/agent/accord/hypotheses_without_shap_baseline_judge_revised.json \
        --out_csv docs/without_shap_original_vs_revised.csv \
        --plot
"""

import argparse
import json
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_hypothesis_scores(judge_json_path: str) -> Dict:
    """Extract feature-level scores from judge JSON.

    Returns:
        Dictionary with overall metrics and per-feature scores
    """
    with open(judge_json_path, 'r') as f:
        data = json.load(f)

    # Extract scores from scored_features
    feature_scores = []
    for feat in data.get('scored_features', []):
        # Compute overall_score as average of 4 criteria
        mechanism = feat.get('mechanism_plausibility', 0)
        evidence = feat.get('evidence_alignment', 0)
        subgroup = feat.get('subgroup_implications', 0)
        caveat = feat.get('caveat_awareness', 0)
        computed_overall = (mechanism + evidence + subgroup + caveat) / 4.0
        
        feature_scores.append({
            'feature_name': feat.get('feature_name', ''),
            'mechanism_plausibility': mechanism,
            'evidence_alignment': evidence,
            'subgroup_implications': subgroup,
            'caveat_awareness': caveat,
            'overall_score': computed_overall,
            'recommendation': feat.get('recommendation', ''),
        })
    
    # Compute average scores across all features
    if feature_scores:
        avg_scores = {
            'mechanism_plausibility': np.mean([h['mechanism_plausibility'] for h in feature_scores]),
            'evidence_alignment': np.mean([h['evidence_alignment'] for h in feature_scores]),
            'subgroup_implications': np.mean([h['subgroup_implications'] for h in feature_scores]),
            'caveat_awareness': np.mean([h['caveat_awareness'] for h in feature_scores]),
            'overall_score': np.mean([h['overall_score'] for h in feature_scores]),
            'num_hypotheses': len(feature_scores),
        }
    else:
        avg_scores = {
            'mechanism_plausibility': 0,
            'evidence_alignment': 0,
            'subgroup_implications': 0,
            'caveat_awareness': 0,
            'overall_score': 0,
            'num_hypotheses': 0,
        }
    
    return {
        'summary': data.get('summary', ''),
        'avg_scores': avg_scores,
        'hypotheses': feature_scores,
    }


def compare_original_vs_revised(original_paths: List[str], revised_paths: List[str]) -> pd.DataFrame:
    """Compare average hypothesis scores between ORIGINAL and REVISED versions.
    
    Args:
        original_paths: List of judge JSON paths for ORIGINAL version
        revised_paths: List of judge JSON paths for REVISED version
    
    Returns:
        DataFrame with comparison statistics
    """
    # Extract scores from all files
    original_data = [extract_hypothesis_scores(path) for path in original_paths]
    revised_data = [extract_hypothesis_scores(path) for path in revised_paths]
    
    # Aggregate scores
    score_metrics = [
        'mechanism_plausibility',
        'evidence_alignment',
        'subgroup_implications',
        'caveat_awareness',
        'overall_score'
    ]
    
    results = []
    for metric in score_metrics:
        original_scores = [d['avg_scores'][metric] for d in original_data]
        revised_scores = [d['avg_scores'][metric] for d in revised_data]
        
        results.append({
            'metric': metric.replace('_', ' ').title(),
            'original_mean': np.mean(original_scores),
            'original_std': np.std(original_scores),
            'revised_mean': np.mean(revised_scores),
            'revised_std': np.std(revised_scores),
            'improvement': np.mean(revised_scores) - np.mean(original_scores),
            'percent_improvement': ((np.mean(revised_scores) - np.mean(original_scores)) / 
                                   np.mean(original_scores) * 100) if np.mean(original_scores) > 0 else 0,
        })
    
    comparison_df = pd.DataFrame(results)
    return comparison_df


def plot_version_comparison(comparison_df: pd.DataFrame, out_path: str = 'version_comparison.png', 
                            cohort_name: str = None):
    """Plot comparison between ORIGINAL and REVISED versions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Create title suffix with cohort name if provided
    title_suffix = f" ({cohort_name.upper()})" if cohort_name else ""
    
    # 1. Bar chart comparison
    ax = axes[0]
    x = np.arange(len(comparison_df))
    width = 0.35
    
    ax.bar(x - width/2, comparison_df['original_mean'], width, 
           label='Original', alpha=0.8, color='coral',
           yerr=comparison_df['original_std'], capsize=5)
    ax.bar(x + width/2, comparison_df['revised_mean'], width,
           label='Revised', alpha=0.8, color='steelblue',
           yerr=comparison_df['revised_std'], capsize=5)
    
    ax.set_ylabel('Average Score (1-5)')
    ax.set_title(f'Hypothesis Quality: Original vs Revised{title_suffix}')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['metric'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=2.5, color='gray', linestyle='--', alpha=0.5, label='Midpoint')
    ax.set_ylim(0, 5)
    
    # 2. Improvement chart
    ax = axes[1]
    colors = ['green' if x > 0 else 'red' for x in comparison_df['improvement']]
    bars = ax.barh(comparison_df['metric'], comparison_df['improvement'], color=colors, alpha=0.7)
    
    ax.set_xlabel('Score Improvement (Revised - Original)')
    ax.set_title(f'Improvement from Revision{title_suffix}')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare hypothesis scores ORIGINAL vs REVISED",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--original',
        nargs='+',
        required=True,
        help='Path(s) to judge JSON for ORIGINAL version'
    )
    parser.add_argument(
        '--revised',
        nargs='+',
        required=True,
        help='Path(s) to judge JSON for REVISED version'
    )
    parser.add_argument(
        '--out_csv',
        help='Path to save comparison CSV'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--out_plot',
        default=None,
        help='Path to save comparison plot (default: auto-generated based on input)'
    )
    args = parser.parse_args()

    print("="*80)
    print("COMPARING: ORIGINAL vs REVISED JUDGE EVALUATIONS")
    print("="*80)
    print(f"\nORIGINAL files: {args.original}")
    print(f"REVISED files: {args.revised}")
    
    # Validate input counts match
    if len(args.original) != len(args.revised):
        parser.error(f"Number of original files ({len(args.original)}) must match "
                    f"number of revised files ({len(args.revised)})")
    
    comparison_df = compare_original_vs_revised(args.original, args.revised)
    
    print("\n" + "="*80)
    print("HYPOTHESIS QUALITY COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # Determine improvement
    overall_improvement = comparison_df[comparison_df['metric'] == 'Overall Score']['improvement'].values[0]
    print(f"\n{'='*80}")
    if overall_improvement > 0:
        print(f"✓ REVISED version performs BETTER (avg improvement: +{overall_improvement:.2f} points)")
        print(f"  ({comparison_df[comparison_df['metric'] == 'Overall Score']['percent_improvement'].values[0]:.1f}% improvement)")
    elif overall_improvement < 0:
        print(f"✗ ORIGINAL version performs BETTER (avg difference: {overall_improvement:.2f} points)")
    else:
        print("≈ Both versions perform EQUALLY")
    print("="*80)
    
    # Save comparison CSV
    if args.out_csv:
        comparison_df.to_csv(args.out_csv, index=False)
        print(f"\nSaved comparison to: {args.out_csv}")
    
    # Generate comparison plot
    if args.plot:
        print("\nGenerating comparison plot...")
        
        # Extract cohort/condition name from file path if single comparison
        cohort_name = None
        if len(args.original) == 1:
            # Extract cohort name from path like "docs/agent/crash_2/..."
            path_parts = args.original[0].split('/')
            if 'agent' in path_parts:
                cohort_idx = path_parts.index('agent') + 1
                if cohort_idx < len(path_parts):
                    cohort_name = path_parts[cohort_idx]
            
            # Determine if WITH SHAP or WITHOUT SHAP
            if 'with_shap' in args.original[0]:
                condition = 'WITH_SHAP'
            elif 'without_shap' in args.original[0]:
                condition = 'WITHOUT_SHAP'
            else:
                condition = None
        
        # Generate output path if not specified
        if args.out_plot is None:
            if cohort_name:
                if condition:
                    plot_path = f'docs/{cohort_name.upper()}_{condition}_original_vs_revised.png'
                else:
                    plot_path = f'docs/{cohort_name.upper()}_original_vs_revised.png'
            elif len(args.original) > 1:
                # Multiple cohorts
                if 'with_shap' in args.original[0]:
                    plot_path = 'docs/ALL_COHORTS_WITH_SHAP_original_vs_revised.png'
                elif 'without_shap' in args.original[0]:
                    plot_path = 'docs/ALL_COHORTS_WITHOUT_SHAP_original_vs_revised.png'
                else:
                    plot_path = 'docs/ALL_COHORTS_original_vs_revised.png'
            else:
                plot_path = 'docs/original_vs_revised.png'
        else:
            plot_path = args.out_plot
        
        plot_version_comparison(comparison_df, plot_path, cohort_name)


if __name__ == '__main__':
    main()
