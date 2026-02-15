#!/usr/bin/env python3
"""
summarize_hypotheses.py

Analyze and visualize hypothesis scores from clinical_agent.py output.
Compares original and revised hypotheses, generates summary statistics and plots.

Usage:
  python summarize_hypotheses.py --judge_original results/ist3/hypotheses_judge_original.json
  python summarize_hypotheses.py --judge_original results/ist3/hypotheses_judge_original.json \
                                  --judge_revised results/ist3/hypotheses_judge_revised.json \
                                  --out_dir results/ist3/analysis
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_judge_report(path: str) -> Dict:
    """Load judge report JSON."""
    with open(path, "r") as f:
        return json.load(f)


def extract_scores_df(judge_report: Dict, label: str = "original") -> pd.DataFrame:
    """Extract scores into a pandas DataFrame."""
    records = []
    for hyp in judge_report["scored_hypotheses"]:
        records.append(
            {
                "title": hyp["title"],
                "label": label,
                "scientific_rigor": hyp["scientific_rigor"],
                "clinical_plausibility": hyp["clinical_plausibility"],
                "evidence_alignment": hyp["evidence_alignment"],
                "subgroup_clarity": hyp["subgroup_clarity"],
                "confounding_awareness": hyp["confounding_awareness"],
                "validation_plan_quality": hyp["validation_plan_quality"],
                "overall_score": hyp["overall_score"],
                "recommendation": hyp["recommendation"],
            }
        )
    return pd.DataFrame(records)


def plot_score_comparison(
    df: pd.DataFrame, out_path: str, figsize: tuple = (14, 8)
) -> None:
    """Create grouped bar chart comparing scores across hypotheses."""
    score_cols = [
        "scientific_rigor",
        "clinical_plausibility",
        "evidence_alignment",
        "subgroup_clarity",
        "confounding_awareness",
        "validation_plan_quality",
        "overall_score",
    ]

    # Prepare data for plotting
    if "label" in df.columns and len(df["label"].unique()) > 1:
        # Compare original vs revised
        fig, axes = plt.subplots(2, 1, figsize=figsize)

        for idx, label in enumerate(["original", "revised"]):
            df_subset = df[df["label"] == label]
            if df_subset.empty:
                continue

            ax = axes[idx]
            x = np.arange(len(df_subset))
            width = 0.12

            for i, col in enumerate(score_cols):
                offset = (i - len(score_cols) / 2) * width
                ax.bar(
                    x + offset,
                    df_subset[col],
                    width,
                    label=col.replace("_", " ").title(),
                )

            ax.set_ylabel("Score (1-10)")
            ax.set_title(f"Hypothesis Scores ({label.title()})")
            ax.set_xticks(x)
            ax.set_xticklabels(
                df_subset["title"], rotation=45, ha="right", fontsize=8
            )
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            ax.set_ylim(0, 10.5)
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
    else:
        # Single set of hypotheses
        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(df))
        width = 0.12

        for i, col in enumerate(score_cols):
            offset = (i - len(score_cols) / 2) * width
            ax.bar(x + offset, df[col], width, label=col.replace("_", " ").title())

        ax.set_ylabel("Score (1-10)")
        ax.set_title("Hypothesis Scores")
        ax.set_xticks(x)
        ax.set_xticklabels(df["title"], rotation=45, ha="right", fontsize=8)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.set_ylim(0, 10.5)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved score comparison plot to: {out_path}")


def plot_radar_charts(
    df: pd.DataFrame, out_path: str, max_plots: int = 4
) -> None:
    """Create radar charts for top hypotheses."""
    score_cols = [
        "scientific_rigor",
        "clinical_plausibility",
        "evidence_alignment",
        "subgroup_clarity",
        "confounding_awareness",
        "validation_plan_quality",
    ]

    # Select top hypotheses by overall score
    df_sorted = df.sort_values("overall_score", ascending=False).head(max_plots)

    num_vars = len(score_cols)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 12), subplot_kw=dict(projection="polar")
    )
    axes = axes.flatten()

    for idx, (_, row) in enumerate(df_sorted.iterrows()):
        if idx >= max_plots:
            break

        ax = axes[idx]
        values = row[score_cols].tolist()
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, "o-", linewidth=2, label=row["title"])
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([col.replace("_", "\n") for col in score_cols], fontsize=8)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_title(f"{row['title'][:50]}...", fontsize=10, pad=20)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved radar charts to: {out_path}")


def plot_heatmap(df: pd.DataFrame, out_path: str) -> None:
    """Create heatmap of scores."""
    score_cols = [
        "scientific_rigor",
        "clinical_plausibility",
        "evidence_alignment",
        "subgroup_clarity",
        "confounding_awareness",
        "validation_plan_quality",
        "overall_score",
    ]

    # Create pivot table
    if "label" in df.columns:
        df["hyp_label"] = df["title"] + " (" + df["label"] + ")"
    else:
        df["hyp_label"] = df["title"]

    heatmap_data = df.set_index("hyp_label")[score_cols]

    plt.figure(figsize=(10, max(6, len(df) * 0.4)))
    sns.heatmap(
        heatmap_data.T,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=1,
        vmax=10,
        cbar_kws={"label": "Score (1-10)"},
        linewidths=0.5,
    )
    plt.xlabel("Hypothesis")
    plt.ylabel("Dimension")
    plt.title("Hypothesis Scoring Heatmap")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(
        range(len(score_cols)),
        [col.replace("_", " ").title() for col in score_cols],
        rotation=0,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to: {out_path}")


def plot_improvement(
    df_original: pd.DataFrame, df_revised: pd.DataFrame, out_path: str
) -> None:
    """Plot score improvements from original to revised."""
    score_cols = [
        "scientific_rigor",
        "clinical_plausibility",
        "evidence_alignment",
        "subgroup_clarity",
        "confounding_awareness",
        "validation_plan_quality",
        "overall_score",
    ]

    # Match hypotheses by title
    improvements = []
    for _, orig_row in df_original.iterrows():
        title = orig_row["title"]
        rev_row = df_revised[df_revised["title"] == title]
        if not rev_row.empty:
            rev_row = rev_row.iloc[0]
            for col in score_cols:
                improvements.append(
                    {
                        "title": title,
                        "dimension": col.replace("_", " ").title(),
                        "improvement": rev_row[col] - orig_row[col],
                    }
                )

    if not improvements:
        print("No matching hypotheses found for improvement plot.")
        return

    imp_df = pd.DataFrame(improvements)

    plt.figure(figsize=(12, 6))
    pivot = imp_df.pivot(index="title", columns="dimension", values="improvement")
    pivot.plot(kind="bar", figsize=(12, 6), width=0.8)
    plt.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    plt.ylabel("Score Change")
    plt.xlabel("Hypothesis")
    plt.title("Score Improvement: Original → Revised")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved improvement plot to: {out_path}")


def generate_summary_report(
    df_original: pd.DataFrame,
    df_revised: Optional[pd.DataFrame],
    judge_original: Dict,
    judge_revised: Optional[Dict],
    out_path: str,
) -> None:
    """Generate text summary report."""
    lines = []
    lines.append("=" * 80)
    lines.append("HYPOTHESIS SCORING SUMMARY REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Overall summary from judge
    lines.append("OVERALL ASSESSMENT:")
    lines.append("-" * 80)
    lines.append(f"Original Hypotheses Summary:\n{judge_original['summary']}")
    lines.append("")

    if judge_revised:
        lines.append(f"Revised Hypotheses Summary:\n{judge_revised['summary']}")
        lines.append("")

    # Score statistics
    lines.append("SCORE STATISTICS (Original):")
    lines.append("-" * 80)
    score_cols = [
        "scientific_rigor",
        "clinical_plausibility",
        "evidence_alignment",
        "subgroup_clarity",
        "confounding_awareness",
        "validation_plan_quality",
        "overall_score",
    ]

    stats = df_original[score_cols].describe().loc[["mean", "std", "min", "max"]]
    lines.append(stats.to_string())
    lines.append("")

    if df_revised is not None:
        lines.append("SCORE STATISTICS (Revised):")
        lines.append("-" * 80)
        stats_rev = df_revised[score_cols].describe().loc[["mean", "std", "min", "max"]]
        lines.append(stats_rev.to_string())
        lines.append("")

        lines.append("AVERAGE IMPROVEMENT (Revised - Original):")
        lines.append("-" * 80)
        improvements = df_revised[score_cols].mean() - df_original[score_cols].mean()
        lines.append(improvements.to_string())
        lines.append("")

    # Top hypotheses
    lines.append("TOP HYPOTHESES (by Overall Score):")
    lines.append("-" * 80)
    top = df_original.nlargest(5, "overall_score")[
        ["title", "overall_score", "recommendation"]
    ]
    for idx, row in top.iterrows():
        lines.append(
            f"{row['overall_score']:.1f} - {row['title']} [{row['recommendation']}]"
        )
    lines.append("")

    # Recommendations breakdown
    lines.append("RECOMMENDATION BREAKDOWN:")
    lines.append("-" * 80)
    rec_counts = df_original["recommendation"].value_counts()
    for rec, count in rec_counts.items():
        pct = (count / len(df_original)) * 100
        lines.append(f"{rec}: {count} ({pct:.1f}%)")
    lines.append("")

    # Methodological concerns
    if judge_original.get("methodological_concerns"):
        lines.append("METHODOLOGICAL CONCERNS:")
        lines.append("-" * 80)
        for concern in judge_original["methodological_concerns"]:
            lines.append(f"- {concern}")
        lines.append("")

    lines.append("=" * 80)

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved summary report to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize hypothesis scores"
    )
    parser.add_argument(
        "--judge_original",
        required=True,
        help="Path to judge_original.json file",
    )
    parser.add_argument(
        "--judge_revised",
        help="Path to judge_revised.json file (optional)",
    )
    parser.add_argument(
        "--out_dir",
        help="Output directory for plots and reports (default: same as judge_original)",
    )
    args = parser.parse_args()

    # Determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(args.judge_original).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading judge reports...")
    judge_original = load_judge_report(args.judge_original)
    df_original = extract_scores_df(judge_original, label="original")

    judge_revised = None
    df_revised = None
    if args.judge_revised:
        judge_revised = load_judge_report(args.judge_revised)
        df_revised = extract_scores_df(judge_revised, label="revised")

    # Combine dataframes if both available
    if df_revised is not None:
        df_combined = pd.concat([df_original, df_revised], ignore_index=True)
    else:
        df_combined = df_original

    # Generate plots
    print("\nGenerating visualizations...")

    plot_score_comparison(
        df_combined, str(out_dir / "score_comparison.png")
    )

    plot_radar_charts(
        df_original, str(out_dir / "radar_charts_original.png")
    )

    if df_revised is not None:
        plot_radar_charts(
            df_revised, str(out_dir / "radar_charts_revised.png")
        )

    plot_heatmap(df_combined, str(out_dir / "score_heatmap.png"))

    if df_revised is not None:
        plot_improvement(
            df_original, df_revised, str(out_dir / "score_improvement.png")
        )

    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(
        df_original,
        df_revised,
        judge_original,
        judge_revised,
        str(out_dir / "summary_report.txt"),
    )

    # Export to CSV
    print("\nExporting data to CSV...")
    df_combined.to_csv(out_dir / "hypothesis_scores.csv", index=False)
    print(f"Saved scores to: {out_dir / 'hypothesis_scores.csv'}")

    print("\n✅ Analysis complete!")
    print(f"All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
