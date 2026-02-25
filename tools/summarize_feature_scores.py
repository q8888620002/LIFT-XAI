#!/usr/bin/env python3
"""summarize_feature_scores.py

Compute average scores comparing hypotheses across methods or WITH SHAP vs WITHOUT SHAP.

Usage:
    # NEW: Aggregate analysis across all methods (SimpleCoT, Baseline, HypoGeniC, ALEX)
    python summarize_feature_scores.py \
        --judge_files \
            docs/agent/crash_2/simple_cot_judge.json \
            docs/agent/crash_2/hypotheses_without_shap_baseline_judge_revised.json \
            docs/agent/crash_2/hypogenic_hypotheses_judge.json \
            docs/agent/crash_2/hypotheses_with_shap_DRLearner_judge_revised.json \
            docs/agent/ist3/simple_cot_judge.json \
            docs/agent/ist3/hypotheses_without_shap_baseline_judge_revised.json \
            docs/agent/ist3/hypogenic_hypotheses_judge.json \
            docs/agent/ist3/hypotheses_with_shap_DRLearner_judge_revised.json \
        --out_csv docs/method_comparison.csv

    # Aggregate analysis for specific model(s), e.g., gpt-5-mini only
    python summarize_feature_scores.py \
        --judge_files docs/agent/*/*/*/hypotheses_judge*.json \
        --model_filter gpt-5-mini \
        --out_csv docs/method_comparison_gpt5mini.csv

    # Compare WITH SHAP vs WITHOUT SHAP for a single cohort
    python summarize_feature_scores.py \
        --judge_with_shap docs/agent/crash_2/hypotheses_with_shap_XLearner_judge_revised.json \
        --judge_without_shap docs/agent/crash_2/hypotheses_without_shap_baseline_judge_revised.json \
        --version revised \
        --out_csv docs/crash_2_shap_comparison.csv \
        --plot

    # Compare all cohorts (ORIGINAL versions)
    python summarize_feature_scores.py \
        --judge_with_shap \
            docs/agent/crash_2/hypotheses_with_shap_XLearner_judge_original.json \
            docs/agent/ist3/hypotheses_with_shap_XLearner_judge_original.json \
            docs/agent/sprint/hypotheses_with_shap_XLearner_judge_original.json \
            docs/agent/accord/hypotheses_with_shap_XLearner_judge_original.json \
        --judge_without_shap \
            docs/agent/crash_2/hypotheses_without_shap_baseline_judge_original.json \
            docs/agent/ist3/hypotheses_without_shap_baseline_judge_original.json \
            docs/agent/sprint/hypotheses_without_shap_baseline_judge_original.json \
            docs/agent/accord/hypotheses_without_shap_baseline_judge_original.json \
        --version original \
        --out_csv docs/shap_comparison_all_cohorts_original.csv \
        --plot

    # Compare all cohorts (REVISED versions)
    python summarize_feature_scores.py \
        --judge_with_shap \
            docs/agent/crash_2/hypotheses_with_shap_XLearner_judge_revised.json \
            docs/agent/ist3/hypotheses_with_shap_XLearner_judge_revised.json \
            docs/agent/sprint/hypotheses_with_shap_XLearner_judge_revised.json \
            docs/agent/accord/hypotheses_with_shap_XLearner_judge_revised.json \
        --judge_without_shap \
            docs/agent/crash_2/hypotheses_without_shap_baseline_judge_revised.json \
            docs/agent/ist3/hypotheses_without_shap_baseline_judge_revised.json \
            docs/agent/sprint/hypotheses_without_shap_baseline_judge_revised.json \
            docs/agent/accord/hypotheses_without_shap_baseline_judge_revised.json \
        --version revised \
        --out_csv docs/shap_comparison_all_cohorts_revised.csv \
        --plot
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def _file_version_priority(file_path: str) -> int:
    """Return priority for file version (lower = higher priority).
    Prefer revised > original > no suffix.
    """
    if 'revised' in file_path.lower():
        return 0
    elif 'original' in file_path.lower():
        return 1
    else:
        return 2


def _alex_variant_priority(file_path: str) -> int:
    """Return ALEX variant priority (lower = higher priority).

    Prefer DRLearner > XLearner > other WITH SHAP variants.
    """
    path_lower = file_path.lower()
    if 'drlearner' in path_lower or 'with_shap_drlearner' in path_lower:
        return 0
    if 'xlearner' in path_lower or 'with_shap_xlearner' in path_lower:
        return 1
    return 2


def _hypogenic_file_priority(file_path: str) -> int:
    """Return HypoGeniC file priority (lower = higher priority).

    Prefer hypotheses_pubmed_input_judge* over hypotheses_judge*.
    """
    path_lower = file_path.lower()
    if 'hypotheses_pubmed_input_judge' in path_lower:
        return 0
    if 'hypotheses_judge' in path_lower:
        return 1
    return 2


def deduplicate_judge_files(judge_paths: List[str]) -> List[str]:
    """Deduplicate judge files by (model, method, dataset, seed), preferring revised versions.

    Args:
        judge_paths: List of judge JSON file paths

    Returns:
        Deduplicated list with one file per (model, method, dataset)
    """
    from collections import defaultdict

    # Group files by (model, method, dataset, seed)
    groups = defaultdict(list)
    for path in judge_paths:
        model = infer_model_name(path)
        method = infer_method_name(path)
        dataset = infer_dataset_name(path)
        seed = infer_seed_name(path)
        groups[(model, method, dataset, seed)].append(path)

    # Select best file for each group
    selected = []
    for (model, method, dataset, seed), files in groups.items():
        if len(files) == 1:
            selected.append(files[0])
        else:
            # Prefer revised first for all methods.
            # For ALEX ties, prefer DRLearner over XLearner.
            # For HypoGeniC ties, prefer hypotheses_pubmed_input_judge* over hypotheses_judge*.
            if method == 'ALEX':
                best = sorted(
                    files,
                    key=lambda path: (
                        _file_version_priority(path),
                        _alex_variant_priority(path),
                        path,
                    ),
                )[0]
            elif method == 'HypoGeniC':
                best = sorted(
                    files,
                    key=lambda path: (
                        _file_version_priority(path),
                        _hypogenic_file_priority(path),
                        path,
                    ),
                )[0]
            else:
                best = sorted(files, key=lambda path: (_file_version_priority(path), path))[0]
            selected.append(best)

    return selected


def infer_seed_name(file_path: str) -> str:
    """Infer seed identifier from path (e.g., seed_0), fallback to no_seed."""
    parts = file_path.replace('\\', '/').split('/')
    for part in parts:
        part_l = part.lower()
        if part_l.startswith('seed_'):
            return part_l
    return 'no_seed'


def filter_judge_paths_by_model(judge_paths: List[str], model_filters: Optional[List[str]]) -> List[str]:
    """Filter judge paths to requested model(s) before further processing."""
    if not model_filters:
        return judge_paths

    requested_models = {m.lower() for m in model_filters}
    return [p for p in judge_paths if infer_model_name(p).lower() in requested_models]


def infer_method_name(file_path: str) -> str:
    """Infer method name from file path."""
    file_lower = file_path.lower()
    if 'hypogenic' in file_lower:
        return 'HypoGeniC'
    elif 'researchagent' in file_lower:
        return 'ResearchAgent'
    elif 'simple_cot' in file_lower or 'simplecot' in file_lower:
        return 'SimpleCoT'
    elif 'with_shap' in file_lower or 'xlearner' in file_lower or 'drlearner' in file_lower:
        return 'ALEX'
    elif 'without_shap' in file_lower or 'baseline' in file_lower:
        return 'Baseline'
    else:
        return 'Unknown'


def infer_dataset_name(file_path: str) -> str:
    """Infer dataset name from file path."""
    # Look for standard dataset names in path
    path_lower = file_path.lower()
    for dataset in ['crash_2', 'ist3', 'sprint', 'accord']:
        if dataset in path_lower:
            return dataset
    return 'unknown'


def infer_model_name(file_path: str) -> str:
    """Infer LLM model name from file path.

    Expected modern layout:
        docs/agent/<dataset>/<model>/<method>/hypotheses_judge_*.json
    Falls back to regex matching for legacy layouts.
    """
    normalized = file_path.replace('\\', '/')
    parts = [p for p in normalized.split('/') if p]

    dataset_names = {'crash_2', 'ist3', 'sprint', 'accord', 'crash2_txa', 'accord_sprint'}
    method_like_names = {
        'hypogenic',
        'simple_cot',
        'without_shap_baseline',
        'with_shap_xlearner',
        'with_shap_drlearner',
        'researchagent',
        'shapley',
    }

    if 'agent' in parts:
        agent_idx = parts.index('agent')
        if agent_idx + 2 < len(parts):
            dataset_candidate = parts[agent_idx + 1].lower()
            model_candidate = parts[agent_idx + 2]
            model_candidate_lower = model_candidate.lower()

            if (
                dataset_candidate in dataset_names
                and model_candidate_lower not in method_like_names
                and not model_candidate_lower.endswith('.json')
            ):
                return model_candidate

    file_lower = normalized.lower()
    model_pattern = (
        r'(gpt-[a-z0-9._-]+|o[0-9][a-z0-9._-]*|google_gemini-[a-z0-9._-]+'
        r'|gemini-[a-z0-9._-]+|claude-[a-z0-9._-]+|llama-[a-z0-9._-]+)'
    )
    match = re.search(model_pattern, file_lower)
    if match:
        return match.group(1)

    return 'unknown_model'


def extract_hypothesis_scores(judge_json_path: str) -> Dict:
    """Extract feature-level scores from judge JSON as proxy for hypothesis quality.

    Returns:
        Dictionary with overall metrics and per-feature scores
    """
    with open(judge_json_path, 'r') as f:
        data = json.load(f)

    # Extract scores - try both 'scored_features' and 'scored_hypotheses' keys
    # HypoGeniC uses 'scored_hypotheses', other methods use 'scored_features'
    items = data.get('scored_features') or data.get('scored_hypotheses', [])

    feature_scores = []
    for feat in items:
        # Handle different field name conventions
        # HypoGeniC uses: scientific_rigor, clinical_plausibility, subgroup_clarity, confounding_awareness
        # Others use: mechanism_plausibility, clinical_interpretation, subgroup_implications, caveat_awareness

        mechanism = feat.get('mechanism_plausibility') or feat.get('scientific_rigor', 0)
        clinical = feat.get('clinical_interpretation') or feat.get('clinical_plausibility', 0)
        evidence = feat.get('evidence_alignment', 0)
        subgroup = feat.get('subgroup_implications') or feat.get('subgroup_clarity', 0)
        validation = feat.get('validation_plan_quality', 0)
        caveat = feat.get('caveat_awareness') or feat.get('confounding_awareness', 0)
        novelty = feat.get('novelty', 0)

        # Compute overall_score across the 5 rubric items
        computed_overall = (mechanism + evidence + subgroup + caveat + novelty) / 5.0

        feature_scores.append({
            'feature_name': feat.get('feature_name') or feat.get('title', ''),
            'mechanism_plausibility': mechanism,
            'clinical_interpretation': clinical,
            'evidence_alignment': evidence,
            'subgroup_implications': subgroup,
            'validation_plan_quality': validation,
            'caveat_awareness': caveat,
            'novelty': novelty,
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
            'novelty': np.mean([h['novelty'] for h in feature_scores]),
            'overall_score': np.mean([h['overall_score'] for h in feature_scores]),
            'num_hypotheses': len(feature_scores),
        }
    else:
        avg_scores = {
            'mechanism_plausibility': 0,
            'evidence_alignment': 0,
            'subgroup_implications': 0,
            'caveat_awareness': 0,
            'novelty': 0,
            'overall_score': 0,
            'num_hypotheses': 0,
        }

    return {
        'summary': data.get('summary', ''),
        'avg_scores': avg_scores,
        'hypotheses': feature_scores,
    }


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
        mechanism = feature.get('mechanism_plausibility', 0)
        evidence = feature.get('evidence_alignment', 0)
        subgroup = feature.get('subgroup_implications', 0)
        caveat = feature.get('caveat_awareness', 0)
        novelty = feature.get('novelty', 0)
        computed_overall = (mechanism + evidence + subgroup + caveat + novelty) / 5.0

        record = {
            'feature_name': feature['feature_name'],
            'mechanism_plausibility': mechanism,
            'evidence_alignment': evidence,
            'subgroup_implications': subgroup,
            'caveat_awareness': caveat,
            'novelty': novelty,
            'overall_score': computed_overall,
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


def aggregate_scores_by_method(judge_paths: List[str]) -> pd.DataFrame:
    """Aggregate scores by model, method, dataset, and seed.

    Args:
        judge_paths: List of judge JSON file paths

    Returns:
        DataFrame with columns: model, method, dataset, seed, and quality metrics
    """
    records = []

    for path in judge_paths:
        model = infer_model_name(path)
        method = infer_method_name(path)
        dataset = infer_dataset_name(path)
        seed = infer_seed_name(path)
        scores = extract_hypothesis_scores(path)

        record = {
            'model': model,
            'method': method,
            'dataset': dataset,
            'seed': seed,
            **scores['avg_scores']
        }
        records.append(record)

    df = pd.DataFrame(records)
    return df


def compute_method_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average and std scores per (model, method) across all seed runs.

    Args:
        df: DataFrame with method, dataset, and quality scores

    Returns:
        DataFrame with model-method rows and quality metrics as mean/std columns
    """
    score_cols = [
        'mechanism_plausibility',
        'evidence_alignment',
        'subgroup_implications',
        'caveat_awareness',
        'novelty',
        'overall_score'
    ]

    # Group by model+method and compute mean/std across seed runs
    method_avgs = (
        df.groupby(['model', 'method'])[score_cols]
        .agg(['mean', 'std'])
        .reset_index()
    )

    # Flatten MultiIndex columns: metric_mean, metric_std
    def _flatten_col(c):
        if isinstance(c, str):
            return c
        left, right = c
        if right is None or right == "":
            return left
        return f"{left}_{right}"

    method_avgs.columns = [_flatten_col(c) for c in method_avgs.columns]

    # Replace NaN std values (single sample) with 0.0
    for col in method_avgs.columns:
        if col.endswith('_std'):
            method_avgs[col] = method_avgs[col].fillna(0.0)

    # Number of runs (typically seeds × datasets available)
    run_counts = (
        df.groupby(['model', 'method'])
        .size()
        .reset_index(name='n_runs')
    )
    method_avgs = method_avgs.merge(run_counts, on=['model', 'method'], how='left')

    # Sort by predefined method order, then any additional methods (e.g., ResearchAgent)
    method_order = ['SimpleCoT', 'Baseline', 'HypoGeniC', 'ALEX']
    present_methods = [m for m in method_avgs['method'].dropna().astype(str).unique() if m not in method_order]
    categories = method_order + sorted(present_methods)
    method_avgs['method'] = pd.Categorical(
        method_avgs['method'],
        categories=categories,
        ordered=True,
    )
    method_avgs['model'] = method_avgs['model'].astype(str)
    method_avgs = method_avgs.sort_values(['model', 'method']).reset_index(drop=True)

    return method_avgs


def compute_per_dataset_averages(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute average and std scores per (model, method) for each dataset across seeds.

    Args:
        df: DataFrame with method, dataset, and quality scores

    Returns:
        Dictionary mapping dataset name to DataFrame of model-method mean/std scores
    """
    score_cols = [
        'mechanism_plausibility',
        'evidence_alignment',
        'subgroup_implications',
        'caveat_awareness',
        'novelty',
        'overall_score'
    ]

    method_order = ['SimpleCoT', 'Baseline', 'HypoGeniC', 'ALEX']

    results = {}
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        dataset_avgs = (
            dataset_df.groupby(['model', 'method'])[score_cols]
            .agg(['mean', 'std'])
            .reset_index()
        )

        def _flatten_col(c):
            if isinstance(c, str):
                return c
            left, right = c
            if right is None or right == "":
                return left
            return f"{left}_{right}"

        dataset_avgs.columns = [_flatten_col(c) for c in dataset_avgs.columns]

        for col in dataset_avgs.columns:
            if col.endswith('_std'):
                dataset_avgs[col] = dataset_avgs[col].fillna(0.0)

        run_counts = (
            dataset_df.groupby(['model', 'method'])
            .size()
            .reset_index(name='n_runs')
        )
        dataset_avgs = dataset_avgs.merge(run_counts, on=['model', 'method'], how='left')

        # Sort by method order, then any additional methods (e.g., ResearchAgent)
        present_methods = [m for m in dataset_avgs['method'].dropna().astype(str).unique() if m not in method_order]
        categories = method_order + sorted(present_methods)
        dataset_avgs['method'] = pd.Categorical(
            dataset_avgs['method'],
            categories=categories,
            ordered=True,
        )
        dataset_avgs['model'] = dataset_avgs['model'].astype(str)
        dataset_avgs = dataset_avgs.sort_values(['model', 'method']).reset_index(drop=True)

        results[dataset] = dataset_avgs

    return results


def compare_shap_vs_baseline(with_shap_paths: List[str], without_shap_paths: List[str]) -> pd.DataFrame:
    """Compare average hypothesis scores between WITH SHAP and WITHOUT SHAP conditions.

    Args:
        with_shap_paths: List of judge JSON paths for WITH SHAP condition
        without_shap_paths: List of judge JSON paths for WITHOUT SHAP condition

    Returns:
        DataFrame with comparison statistics
    """
    # Extract scores from all files
    with_shap_data = [extract_hypothesis_scores(path) for path in with_shap_paths]
    without_shap_data = [extract_hypothesis_scores(path) for path in without_shap_paths]

    # Aggregate scores
    score_metrics = [
        'mechanism_plausibility',
        'evidence_alignment',
        'subgroup_implications',
        'caveat_awareness',
        'novelty',
        'overall_score'
    ]

    results = []
    for metric in score_metrics:
        with_shap_scores = [d['avg_scores'][metric] for d in with_shap_data]
        without_shap_scores = [d['avg_scores'][metric] for d in without_shap_data]

        results.append({
            'metric': metric.replace('_', ' ').title(),
            'with_shap_mean': np.mean(with_shap_scores),
            'with_shap_std': np.std(with_shap_scores),
            'without_shap_mean': np.mean(without_shap_scores),
            'without_shap_std': np.std(without_shap_scores),
            'difference': np.mean(with_shap_scores) - np.mean(without_shap_scores),
            'percent_improvement': ((np.mean(with_shap_scores) - np.mean(without_shap_scores)) /
                                   np.mean(without_shap_scores) * 100) if np.mean(without_shap_scores) > 0 else 0,
        })

    comparison_df = pd.DataFrame(results)
    return comparison_df


def plot_shap_comparison(comparison_df: pd.DataFrame, out_path: str = 'shap_comparison.png', trial_name: str = None):
    """Plot comparison between WITH SHAP and WITHOUT SHAP conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Create title suffix with trial name if provided
    title_suffix = f" ({trial_name.upper()})" if trial_name else ""

    # 1. Bar chart comparison
    ax = axes[0]
    x = np.arange(len(comparison_df))
    width = 0.35

    ax.bar(x - width/2, comparison_df['with_shap_mean'], width,
           label='WITH SHAP', alpha=0.8, color='steelblue',
           yerr=comparison_df['with_shap_std'], capsize=5)
    ax.bar(x + width/2, comparison_df['without_shap_mean'], width,
           label='WITHOUT SHAP (Baseline)', alpha=0.8, color='coral',
           yerr=comparison_df['without_shap_std'], capsize=5)

    ax.set_ylabel('Average Score (1-5)')
    ax.set_title(f'Hypothesis Quality: WITH SHAP vs WITHOUT SHAP{title_suffix}')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['metric'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=2.5, color='gray', linestyle='--', alpha=0.5, label='Midpoint')
    ax.set_ylim(0, 5)

    # 2. Improvement chart
    ax = axes[1]
    colors = ['green' if x > 0 else 'red' for x in comparison_df['difference']]
    bars = ax.barh(comparison_df['metric'], comparison_df['difference'], color=colors, alpha=0.7)

    ax.set_xlabel('Score Difference (WITH SHAP - WITHOUT SHAP)')
    ax.set_title(f'Improvement with SHAP Feature Guidance{title_suffix}')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {out_path}")
    plt.close()


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics across all features."""
    score_cols = [
        'mechanism_plausibility',
        'evidence_alignment',
        'subgroup_implications',
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
        'evidence_alignment',
        'subgroup_implications',
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
    parser = argparse.ArgumentParser(description="Compare hypothesis scores across methods and datasets")
    parser.add_argument(
        '--judge_files',
        nargs='+',
        help='Path(s) to all judge JSON files for aggregate analysis'
    )
    parser.add_argument(
        '--model_filter',
        nargs='+',
        default=None,
        help='Optional model name(s) to include in aggregate mode only (e.g., gpt-5-mini google_gemini-3-flash-preview)'
    )
    parser.add_argument(
        '--judge_with_shap',
        nargs='+',
        help='Path(s) to judge JSON for hypotheses WITH SHAP'
    )
    parser.add_argument(
        '--judge_without_shap',
        nargs='+',
        help='Path(s) to judge JSON for hypotheses WITHOUT SHAP (baseline)'
    )
    parser.add_argument(
        '--judge_json',
        help='(Legacy) Path to judge output JSON'
    )
    parser.add_argument(
        '--judge_json_revised',
        help='(Legacy) Path to judge output JSON for revised hypotheses'
    )
    parser.add_argument(
        '--out_csv',
        help='Path to save comparison CSV'
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
        '--version',
        choices=['original', 'revised'],
        default='revised',
        help='Whether comparing original or revised judge outputs (affects output filename)'
    )
    parser.add_argument(
        '--out_plot',
        default=None,
        help='Path to save comparison plot (default: docs/shap_comparison_{version}.png)'
    )
    parser.add_argument(
        '--out_mechanism_plot',
        default='docs/mechanism_scores.png',
        help='Path to save mechanism scores plot'
    )
    args = parser.parse_args()

    # Auto-discover aggregate judge files when explicit inputs are not provided.
    has_explicit_mode = bool(args.judge_files) or bool(args.judge_json) or (
        bool(args.judge_with_shap) and bool(args.judge_without_shap)
    )
    if not has_explicit_mode:
        docs_agent_root = Path('docs/agent')
        if docs_agent_root.exists():
            discovered = sorted(
                {
                    *docs_agent_root.rglob('hypotheses_judge*.json'),
                    *docs_agent_root.rglob('hypotheses_pubmed_input_judge*.json'),
                },
                key=lambda p: str(p),
            )
            discovered = [str(p) for p in discovered]
            if discovered:
                args.judge_files = discovered
                print(
                    f"Auto-discovered {len(discovered)} judge files under "
                    f"{docs_agent_root.as_posix()} for aggregate analysis."
                )

    # New aggregate mode: analyze all methods across datasets
    if args.judge_files:
        print("="*80)
        print("AGGREGATE ANALYSIS: ALL MODELS × METHODS ACROSS DATASETS")
        print("="*80)
        print(f"\nReceived {len(args.judge_files)} judge files...")

        # Apply model filter first so only matching model files are parsed/deduped.
        filtered_input_paths = filter_judge_paths_by_model(args.judge_files, args.model_filter)
        if args.model_filter:
            requested_models = sorted({m.lower() for m in args.model_filter})
            print(f"After pre-filter by model {requested_models}: {len(filtered_input_paths)} files")

            if not filtered_input_paths:
                parser.error("No judge files match --model_filter. Check model names in paths.")

        # Deduplicate: prefer revised versions when multiple exist
        deduplicated = deduplicate_judge_files(filtered_input_paths)
        print(f"After deduplication by (model, method, dataset, seed) preferring revised: {len(deduplicated)} files\n")

        # Aggregate scores
        df = aggregate_scores_by_method(deduplicated)

        # Compute model+method averages across all datasets
        method_avgs = compute_method_averages(df)

        single_model_filter = bool(args.model_filter) and len(args.model_filter) == 1

        def _compact_display(table: pd.DataFrame) -> pd.DataFrame:
            base_cols = ['method', 'n_runs'] if single_model_filter else ['model', 'method', 'n_runs']
            compact = table[base_cols].copy()
            metric_map = {
                'mechanism_plausibility': 'mech_plaus',
                'evidence_alignment': 'evid_align',
                'subgroup_implications': 'subgrp_impl',
                'caveat_awareness': 'caveat_awr',
                'novelty': 'novelty',
                'overall_score': 'overall',
            }
            for metric, short_name in metric_map.items():
                mean_col = f'{metric}_mean'
                std_col = f'{metric}_std'
                compact[short_name] = table.apply(
                    lambda r: f"{r[mean_col]:.2f}±{r[std_col]:.2f}",
                    axis=1,
                )
            return compact

        display_df = _compact_display(method_avgs)

        print("\n" + "="*80)
        print("AVERAGE SCORES PER MODEL-METHOD (ACROSS SEED RUNS)")
        print("="*80)
        print("Columns show mean±std (std across seed runs); n_runs = number of rows aggregated.")
        print(display_df.to_string(index=False))

        # Compute per-dataset averages
        per_dataset = compute_per_dataset_averages(df)

        for dataset in sorted(per_dataset.keys()):
            print(f"\n{'='*80}")
            print(f"SCORES FOR {dataset.upper()} (BY MODEL-METHOD)")
            print("="*80)
            print("Columns use *_mean and *_std; n_runs = number of seeds/runs for this dataset")

            display_df = _compact_display(per_dataset[dataset])
            print(display_df.to_string(index=False))

        # Save to CSV if requested
        if args.out_csv:
            # Save model-method averages across datasets
            method_csv = args.out_csv.replace('.csv', '_model_method_averages.csv')
            method_avgs.to_csv(method_csv, index=False)
            print(f"\nSaved model-method averages to: {method_csv}")

            # Save per-dataset details
            dataset_csv = args.out_csv.replace('.csv', '_by_dataset.csv')
            df.to_csv(dataset_csv, index=False)
            print(f"Saved per-dataset details to: {dataset_csv}")

        return

    # Main comparison mode: WITH SHAP vs WITHOUT SHAP
    if args.judge_with_shap and args.judge_without_shap:
        print("="*80)
        print("COMPARING: WITH SHAP vs WITHOUT SHAP")
        print("="*80)
        print(f"\nWITH SHAP files: {args.judge_with_shap}")
        print(f"WITHOUT SHAP files: {args.judge_without_shap}")

        comparison_df = compare_shap_vs_baseline(args.judge_with_shap, args.judge_without_shap)

        print("\n" + "="*80)
        print("HYPOTHESIS QUALITY COMPARISON")
        print("="*80)
        print(comparison_df.to_string(index=False))

        # Determine winner
        overall_improvement = comparison_df[comparison_df['metric'] == 'Overall Score']['difference'].values[0]
        print(f"\n{'='*80}")
        if overall_improvement > 0:
            print(f"✓ WITH SHAP performs BETTER (avg improvement: +{overall_improvement:.2f} points)")
        elif overall_improvement < 0:
            print(f"✗ WITHOUT SHAP performs BETTER (avg difference: {overall_improvement:.2f} points)")
        else:
            print("≈ Both conditions perform EQUALLY")
        print("="*80)

        # Save comparison CSV
        if args.out_csv:
            comparison_df.to_csv(args.out_csv, index=False)
            print(f"\nSaved comparison to: {args.out_csv}")

        # Generate comparison plot
        if args.plot:
            print("\nGenerating comparison plot...")

            # Extract trial name from file path if single trial
            trial_name = None
            if len(args.judge_with_shap) == 1:
                # Extract trial name from path like "docs/agent/crash_2/..."
                path_parts = args.judge_with_shap[0].split('/')
                if 'agent' in path_parts:
                    trial_idx = path_parts.index('agent') + 1
                    if trial_idx < len(path_parts):
                        trial_name = path_parts[trial_idx]

            # Use version and trial name in filename if not explicitly specified
            if args.out_plot is None:
                if trial_name:
                    plot_path = f'docs/{trial_name.upper()}_shap_comparison_{args.version}.png'
                else:
                    plot_path = f'docs/ALL_TRIALS_shap_comparison_{args.version}.png'
            else:
                plot_path = args.out_plot

            plot_shap_comparison(comparison_df, plot_path, trial_name)

        return

    # Legacy mode: single judge file analysis
    if args.judge_json:
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
            # Use different default filename for legacy mode
            legacy_plot_path = args.out_plot if args.out_plot != 'docs/shap_comparison.png' else 'docs/feature_scores_original_vs_revised.png'
            plot_feature_scores(df, df_revised, legacy_plot_path)
            plot_mechanism_scores(df, args.judge_json, args.out_mechanism_plot)

        return

    # No valid arguments provided
    parser.error("Must provide either --judge_files OR (--judge_with_shap and --judge_without_shap) OR --judge_json")


if __name__ == '__main__':
    main()
