#!/usr/bin/env python3
"""summarize_feature_scores.py

Compute average scores for each feature from judge output JSON files.

Usage:
    python summarize_feature_scores.py \
        --judge_json results/agents/hypotheses_baseline_shapley_XLearner_judge_original.json \
        --out_csv results/agents/feature_scores_summary.csv

    # Compare original vs revised
    python summarize_feature_scores.py \
        --judge_json results/agents/hypotheses_baseline_shapley_XLearner_judge_original.json \
        --judge_json_revised results/agents/hypotheses_baseline_shapley_XLearner_judge_revised.json \
        --out_csv results/agents/feature_scores_comparison.csv \
        --plot
"""

import argparse
import json
import re
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def extract_feature_scores(judge_json_path: str) -> pd.DataFrame:
    """Extract feature-level scores from judge JSON.

    Returns:
        DataFrame with columns: feature_name, mechanism_plausibility,
        clinical_interpretation, evidence_alignment, subgroup_implications,
        validation_plan_quality, caveat_awareness, overall_score,
        avg_mechanism_score, recommendation
    """
    with open(judge_json_path, 'r') as f:
        data = json.load(f)

    records = []
    for feature in data['scored_features']:
        # Feature-level scores
        record = {
            'feature_name': feature['feature_name'],
            'mechanism_plausibility': feature['mechanism_plausibility'],
            'clinical_interpretation': feature['clinical_interpretation'],
            'evidence_alignment': feature['evidence_alignment'],
            'subgroup_implications': feature['subgroup_implications'],
            'validation_plan_quality': feature['validation_plan_quality'],
            'caveat_awareness': feature['caveat_awareness'],
            'overall_score': feature['overall_score'],
            'recommendation': feature['recommendation'],
        }

        # Compute average mechanism score if available
        if 'per_mechanism_scores' in feature and feature['per_mechanism_scores']:
            mechanism_scores = [m['overall_score'] for m in feature['per_mechanism_scores']]
            record['avg_mechanism_score'] = np.mean(mechanism_scores)
            record['num_mechanisms'] = len(mechanism_scores)
            record['min_mechanism_score'] = np.min(mechanism_scores)
            record['max_mechanism_score'] = np.max(mechanism_scores)
        else:
            record['avg_mechanism_score'] = None
            record['num_mechanisms'] = 0
            record['min_mechanism_score'] = None
            record['max_mechanism_score'] = None

        records.append(record)

    df = pd.DataFrame(records)
    return df


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics across all features."""
    score_cols = [
        'mechanism_plausibility',
        'clinical_interpretation',
        'evidence_alignment',
        'subgroup_implications',
        'validation_plan_quality',
        'caveat_awareness',
        'overall_score'
    ]

    summary = pd.DataFrame({
        'metric': score_cols,
        'mean': [df[col].mean() for col in score_cols],
        'std': [df[col].std() for col in score_cols],
        'min': [df[col].min() for col in score_cols],
        'max': [df[col].max() for col in score_cols],
        'median': [df[col].median() for col in score_cols],
    })

    return summary


def plot_feature_scores(df: pd.DataFrame, df_revised: Optional[pd.DataFrame] = None,
                       out_path: str = 'feature_scores.png'):
    """Plot feature scores with optional comparison to revised scores."""
    score_cols = [
        'mechanism_plausibility',
        'clinical_interpretation',
        'evidence_alignment',
        'subgroup_implications',
        'validation_plan_quality',
        'caveat_awareness',
        'overall_score'
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Bar chart of overall scores
    ax = axes[0, 0]
    x = np.arange(len(df))
    width = 0.35

    # Extract clean feature names (remove score values and descriptions in parentheses)
    clean_feature_names = [re.sub(r'\s*\([^)]*\)', '', name.split(':')[0]).strip() for name in df['feature_name']]

    ax.bar(x - width/2, df['overall_score'], width, label='Original', alpha=0.8)
    if df_revised is not None:
        ax.bar(x + width/2, df_revised['overall_score'], width, label='Revised', alpha=0.8)

    ax.set_ylabel('Overall Score')
    ax.set_title('Overall Scores per Feature')
    ax.set_xticks(x)
    ax.set_xticklabels(clean_feature_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Heatmap of all scores
    ax = axes[0, 1]
    score_matrix = df[score_cols].T
    # Extract clean feature names (remove score values and descriptions in parentheses)
    clean_feature_names = [re.sub(r'\s*\([^)]*\)', '', name.split(':')[0]).strip() for name in df['feature_name']]
    sns.heatmap(score_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                vmin=1, vmax=10, ax=ax, cbar_kws={'label': 'Score (1-10)'})
    ax.set_xticklabels(clean_feature_names, rotation=45, ha='right')
    ax.set_yticklabels([col.replace('_', ' ').title() for col in score_cols], rotation=0)
    ax.set_title('Feature Score Heatmap')

    # 3. Average scores across dimensions
    ax = axes[1, 0]
    avg_scores = df[score_cols].mean()
    bars = ax.bar(range(len(avg_scores)), avg_scores, alpha=0.8)
    ax.set_ylabel('Average Score')
    ax.set_title('Average Scores Across All Features')
    ax.set_xticks(range(len(avg_scores)))
    ax.set_xticklabels([col.replace('_', '\n').title() for col in score_cols],
                        rotation=45, ha='right')
    ax.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='Midpoint (5)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Color bars by value
    for i, bar in enumerate(bars):
        if avg_scores.iloc[i] >= 7:
            bar.set_color('green')
        elif avg_scores.iloc[i] >= 5:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    # 4. Recommendation distribution
    ax = axes[1, 1]
    rec_counts = df['recommendation'].value_counts()
    colors = {
        'high_priority': 'green',
        'medium_priority': 'orange',
        'low_priority': 'yellow',
        'reconsider': 'red'
    }
    bar_colors = [colors.get(rec, 'gray') for rec in rec_counts.index]
    ax.bar(range(len(rec_counts)), rec_counts.values, color=bar_colors, alpha=0.8)
    ax.set_ylabel('Count')
    ax.set_title('Recommendation Distribution')
    ax.set_xticks(range(len(rec_counts)))
    ax.set_xticklabels([r.replace('_', '\n').title() for r in rec_counts.index])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {out_path}")
    plt.close()


def plot_mechanism_scores(df: pd.DataFrame, judge_json_path: str,
                          out_path: str = 'mechanism_scores.png'):
    """Plot detailed mechanism scores."""
    with open(judge_json_path, 'r') as f:
        data = json.load(f)

    # Collect all mechanism scores
    mechanism_data = []
    for feature in data['scored_features']:
        if 'per_mechanism_scores' not in feature:
            continue
        for mech in feature['per_mechanism_scores']:
            mechanism_data.append({
                'feature': feature['feature_name'],
                'mechanism_type': mech['mechanism_type'],
                'plausibility': mech['plausibility'],
                'evidence_support': mech['evidence_support'],
                'specificity': mech['specificity'],
                'testability': mech['testability'],
                'overall_score': mech['overall_score'],
            })

    if not mechanism_data:
        print("No mechanism scores found. Skipping mechanism plot.")
        return

    mech_df = pd.DataFrame(mechanism_data)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Mechanism scores by feature
    ax = axes[0, 0]
    features = mech_df['feature'].unique()
    # Extract clean feature names (remove score values and descriptions in parentheses)
    clean_features = [re.sub(r'\s*\([^)]*\)', '', str(name).split(':')[0]).strip() for name in features]
    x = np.arange(len(features))
    width = 0.15

    score_types = ['plausibility', 'evidence_support', 'specificity', 'testability']
    for i, score_type in enumerate(score_types):
        avg_scores = [mech_df[mech_df['feature'] == f][score_type].mean() for f in features]
        ax.bar(x + i*width, avg_scores, width, label=score_type.replace('_', ' ').title())

    ax.set_ylabel('Average Score')
    ax.set_title('Mechanism Scores by Feature')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(clean_features, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Mechanism type distribution
    ax = axes[0, 1]
    mech_type_counts = mech_df['mechanism_type'].value_counts()
    ax.barh(range(len(mech_type_counts)), mech_type_counts.values, alpha=0.8)
    ax.set_yticks(range(len(mech_type_counts)))
    ax.set_yticklabels(mech_type_counts.index)
    ax.set_xlabel('Count')
    ax.set_title('Mechanism Type Distribution')
    ax.grid(axis='x', alpha=0.3)

    # 3. Score distributions
    ax = axes[1, 0]
    mech_df[score_types].boxplot(ax=ax)
    ax.set_ylabel('Score')
    ax.set_title('Distribution of Mechanism Scores')
    ax.set_xticklabels([s.replace('_', '\n').title() for s in score_types])
    ax.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='Midpoint (5)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 4. Overall mechanism scores
    ax = axes[1, 1]
    # Extract clean feature names (remove score values and descriptions in parentheses)
    clean_features = [re.sub(r'\s*\([^)]*\)', '', str(name).split(':')[0]).strip() for name in features]
    for i, feature in enumerate(features):
        feature_mechs = mech_df[mech_df['feature'] == feature]
        ax.scatter([clean_features[i]] * len(feature_mechs), feature_mechs['overall_score'],
                  alpha=0.6, s=100, label=clean_features[i])
    ax.set_ylabel('Overall Mechanism Score')
    ax.set_title('Mechanism Overall Scores by Feature')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.axhline(y=5, color='r', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved mechanism plot to: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Summarize feature scores from judge output")
    parser.add_argument(
        '--judge_json',
        required=True,
        help='Path to judge output JSON (original or revised)'
    )
    parser.add_argument(
        '--judge_json_revised',
        help='Path to judge output JSON for revised hypotheses (for comparison)'
    )
    parser.add_argument(
        '--out_csv',
        help='Path to save feature scores CSV'
    )
    parser.add_argument(
        '--out_summary_csv',
        help='Path to save summary statistics CSV'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--out_plot',
        default='feature_scores.png',
        help='Path to save feature scores plot'
    )
    parser.add_argument(
        '--out_mechanism_plot',
        default='mechanism_scores.png',
        help='Path to save mechanism scores plot'
    )
    args = parser.parse_args()

    # Extract scores
    print(f"Loading judge output from: {args.judge_json}")
    df = extract_feature_scores(args.judge_json)

    df_revised = None
    if args.judge_json_revised:
        print(f"Loading revised judge output from: {args.judge_json_revised}")
        df_revised = extract_feature_scores(args.judge_json_revised)

    # Compute summary statistics
    summary = compute_summary_stats(df)

    # Print to console
    print("\n" + "="*80)
    print("FEATURE SCORES SUMMARY")
    print("="*80)
    print("\nPer-Feature Scores:")
    print(df.to_string(index=False))

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(summary.to_string(index=False))

    if df_revised is not None:
        print("\n" + "="*80)
        print("COMPARISON: ORIGINAL vs REVISED")
        print("="*80)
        comparison = pd.DataFrame({
            'feature_name': df['feature_name'],
            'original_score': df['overall_score'],
            'revised_score': df_revised['overall_score'],
            'improvement': df_revised['overall_score'] - df['overall_score']
        })
        print(comparison.to_string(index=False))
        print(f"\nAverage improvement: {comparison['improvement'].mean():.2f}")

    # Save CSVs
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\nSaved feature scores to: {args.out_csv}")

    if args.out_summary_csv:
        summary.to_csv(args.out_summary_csv, index=False)
        print(f"Saved summary statistics to: {args.out_summary_csv}")

    # Generate plots
    if args.plot:
        print("\nGenerating plots...")
        plot_feature_scores(df, df_revised, args.out_plot)
        plot_mechanism_scores(df, args.judge_json, args.out_mechanism_plot)


if __name__ == '__main__':
    main()
