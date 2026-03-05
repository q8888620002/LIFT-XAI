#!/usr/bin/env python3
"""display_logic_scores.py

Aggregate and display gate-based logic scores from judge_evaluation.py output
(*_logic_score.json / *_logic_score_revised.json) across all methods and trials.

Usage:
    # Auto-discover all logic_score files under ALEX/results/
    python tools/display_logic_scores.py

    # Filter to a specific model
    python tools/display_logic_scores.py --model gpt-5-mini

    # Filter to specific trials
    python tools/display_logic_scores.py --trials sprint crash_2

    # Use revised scores and save CSV
    python tools/display_logic_scores.py --version revised --out_csv docs/logic_scores.csv

    # Point at explicit files
    python tools/display_logic_scores.py \
        --files ALEX/results/sprint/gpt-5-mini/cot/seed_0/hypotheses_logic_score.json \
                ALEX/results/sprint/gpt-5-mini/with_shap_drlearner/seed_0/hypotheses_logic_score.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path-inference helpers (consistent with summarize_feature_scores.py)
# ---------------------------------------------------------------------------

DATASET_NAMES = {"crash_2", "ist3", "sprint", "accord"}
_METHOD_TOKENS = {
    "hypogenic", "simple_cot", "without_shap_baseline",
    "with_shap_xlearner", "with_shap_drlearner", "researchagent", "shapley", "cot",
}


def infer_dataset(path: str) -> str:
    p = path.lower().replace("\\", "/")
    for d in DATASET_NAMES:
        if d in p:
            return d
    return "unknown"


def infer_model(path: str) -> str:
    normalized = path.replace("\\", "/")
    parts = [x for x in normalized.split("/") if x]
    for anchor in ("agent", "results"):
        if anchor in parts:
            idx = parts.index(anchor)
            if idx + 2 < len(parts):
                dataset_cand = parts[idx + 1].lower()
                model_cand = parts[idx + 2]
                if (
                    dataset_cand in DATASET_NAMES
                    and model_cand.lower() not in _METHOD_TOKENS
                    and not model_cand.endswith(".json")
                ):
                    return model_cand
    match = re.search(
        r"(gpt-[a-z0-9._-]+|o[0-9][a-z0-9._-]*|gemini-[a-z0-9._-]+|claude-[a-z0-9._-]+|llama-[a-z0-9._-]+)",
        path.lower(),
    )
    return match.group(1) if match else "unknown_model"


def infer_method(path: str) -> str:
    p = path.lower().replace("\\", "/")
    if "hypogenic" in p:
        return "HypoGeniC"
    if "researchagent" in p:
        return "ResearchAgent"
    if "simple_cot" in p or "simplecot" in p:
        return "SimpleCOT"
    if "/cot/" in p:
        return "CoT"
    if "with_shap" in p or "xlearner" in p or "drlearner" in p:
        return "ALEX"
    if "without_shap" in p or "baseline" in p:
        return "NoSHAP"
    return "Unknown"


def infer_seed(path: str) -> str:
    for part in path.replace("\\", "/").split("/"):
        if part.lower().startswith("seed_"):
            return part.lower()
    return "no_seed"


def is_revised(path: str) -> bool:
    return "revised" in path.lower()


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------

GATES = [
    ("G1_observed",     "is_observed_in_data"),
    ("G2_bio_coherent", "is_biologically_coherent"),
    ("G3_causal",       "is_causally_plausible"),
    ("G4_literature",   "is_literature_backed"),
    ("G5_actionable",   "is_clinically_actionable"),
]
GATE_COLS = [g[0] for g in GATES]


def load_logic_score_file(path: str) -> pd.DataFrame:
    """Load a logic_score JSON and return a DataFrame with one row per scored feature."""
    with open(path) as f:
        data = json.load(f)

    records = []
    for feat in data.get("scored_features", []):
        row: Dict = {
            "feature_name": feat.get("feature_name", ""),
            "hypothesis_id": feat.get("hypothesis_id"),
            "overall_score": feat.get("overall_score", 0),
            "is_novel": int(bool(feat.get("is_novel", False))),
        }
        for col, key in GATES:
            row[col] = int(bool(feat.get(key, False)))
        records.append(row)

    return pd.DataFrame(records)


def summarize_file(path: str) -> Dict:
    """Return a single-row summary dict for one logic_score file."""
    df = load_logic_score_file(path)
    if df.empty:
        n = 0
        gate_rates = {c: float("nan") for c in GATE_COLS}
        novel_rate = float("nan")
        mean_score = float("nan")
    else:
        n = len(df)
        gate_rates = {c: df[c].mean() for c in GATE_COLS}
        novel_rate = df["is_novel"].mean()
        mean_score = df["overall_score"].mean()

    return {
        "path": path,
        "model": infer_model(path),
        "method": infer_method(path),
        "dataset": infer_dataset(path),
        "seed": infer_seed(path),
        "version": "revised" if is_revised(path) else "original",
        "n_features": n,
        "mean_score": mean_score,
        **gate_rates,
        "novel_rate": novel_rate,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

METHOD_ORDER = ["CoT", "SimpleCOT", "NoSHAP", "HypoGeniC", "ResearchAgent", "ALEX"]
DISPLAY_COLS = ["mean_score"] + GATE_COLS + ["novel_rate"]


def _fmt(mean: float, std: float) -> str:
    if np.isnan(mean):
        return "  n/a  "
    if np.isnan(std) or std == 0.0:
        return f"{mean:.2f}"
    return f"{mean:.2f}±{std:.2f}"


def aggregate_by_method_dataset(rows: List[Dict]) -> pd.DataFrame:
    """Aggregate per-file rows into mean±std per (model, method, dataset)."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    agg_records = []
    for (model, method, dataset), grp in df.groupby(["model", "method", "dataset"]):
        rec: Dict = {"model": model, "method": method, "dataset": dataset, "n_seeds": len(grp)}
        for col in DISPLAY_COLS:
            rec[f"{col}_mean"] = grp[col].mean()
            rec[f"{col}_std"] = grp[col].std(ddof=0)
        agg_records.append(rec)

    agg = pd.DataFrame(agg_records)

    # Sort by method order then dataset
    present_extra = [m for m in agg["method"].unique() if m not in METHOD_ORDER]
    cat_order = METHOD_ORDER + sorted(present_extra)
    agg["method"] = pd.Categorical(agg["method"], categories=cat_order, ordered=True)
    agg = agg.sort_values(["model", "method", "dataset"]).reset_index(drop=True)
    return agg


def aggregate_by_method(rows: List[Dict]) -> pd.DataFrame:
    """Aggregate per-file rows into mean±std per (model, method) across all datasets+seeds."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    agg_records = []
    for (model, method), grp in df.groupby(["model", "method"]):
        rec: Dict = {"model": model, "method": method, "n_runs": len(grp)}
        for col in DISPLAY_COLS:
            rec[f"{col}_mean"] = grp[col].mean()
            rec[f"{col}_std"] = grp[col].std(ddof=0)
        agg_records.append(rec)

    agg = pd.DataFrame(agg_records)
    present_extra = [m for m in agg["method"].unique() if m not in METHOD_ORDER]
    cat_order = METHOD_ORDER + sorted(present_extra)
    agg["method"] = pd.Categorical(agg["method"], categories=cat_order, ordered=True)
    agg = agg.sort_values(["model", "method"]).reset_index(drop=True)
    return agg


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

COL_HEADERS = {
    "mean_score":     "score",
    "G1_observed":    "G1_obs",
    "G2_bio_coherent":"G2_bio",
    "G3_causal":      "G3_caus",
    "G4_literature":  "G4_lit",
    "G5_actionable":  "G5_act",
    "novel_rate":     "novel",
}


def print_aggregated_table(agg: pd.DataFrame, show_dataset: bool = False) -> None:
    """Pretty-print an aggregated table."""
    if agg.empty:
        print("  (no data)")
        return

    # Determine the runs count column (n_seeds for per-dataset DF, n_runs for overall DF)
    n_col = "n_seeds" if "n_seeds" in agg.columns else "n_runs"

    header_parts = [f"{'model':<14}", f"{'method':<14}"]
    if show_dataset:
        header_parts.append(f"{'dataset':<10}")
    header_parts.append(f"{'runs':>4}")
    for col in DISPLAY_COLS:
        header_parts.append(f"{COL_HEADERS[col]:>12}")
    print("  " + "  ".join(header_parts))
    print("  " + "-" * (14 + 14 + (10 if show_dataset else 0) + 4 + 12 * len(DISPLAY_COLS) + 2 * (3 + len(DISPLAY_COLS))))

    for _, row in agg.iterrows():
        parts = [f"{str(row['model']):<14}", f"{str(row['method']):<14}"]
        if show_dataset:
            parts.append(f"{str(row['dataset']):<10}")
        parts.append(f"{int(row[n_col]):>4}")
        for col in DISPLAY_COLS:
            cell = _fmt(row[f"{col}_mean"], row[f"{col}_std"])
            parts.append(f"{cell:>12}")
        print("  " + "  ".join(parts))


def print_per_seed_table(rows: List[Dict]) -> None:
    """Print individual seed rows for detailed inspection."""
    if not rows:
        print("  (no data)")
        return

    cols = ["model", "method", "dataset", "seed", "n_features", "mean_score"] + GATE_COLS + ["novel_rate"]
    df = pd.DataFrame(rows)[cols]
    present_extra = [m for m in df["method"].unique() if m not in METHOD_ORDER]
    cat_order = METHOD_ORDER + sorted(present_extra)
    df["method"] = pd.Categorical(df["method"], categories=cat_order, ordered=True)
    df = df.sort_values(["model", "method", "dataset", "seed"]).reset_index(drop=True)

    # Format float cols
    for col in ["mean_score"] + GATE_COLS + ["novel_rate"]:
        df[col] = df[col].apply(lambda x: f"{x:.2f}" if not (isinstance(x, float) and np.isnan(x)) else "n/a")

    print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_files(root: Path, version_filter: Optional[str]) -> List[str]:
    """Find all logic_score JSON files under root."""
    all_files: List[Path] = list(root.rglob("*_logic_score.json"))
    all_files += list(root.rglob("*_logic_score_revised.json"))
    # Deduplicate by path string
    seen = set()
    result = []
    for p in sorted(all_files):
        s = str(p)
        if s not in seen:
            seen.add(s)
            result.append(s)

    if version_filter == "original":
        result = [p for p in result if not is_revised(p)]
    elif version_filter == "revised":
        result = [p for p in result if is_revised(p)]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Display gate-based logic scores across methods/trials")
    parser.add_argument("--files", nargs="+", help="Explicit list of logic_score JSON files")
    parser.add_argument("--root", default="ALEX/results", help="Root dir to search (default: ALEX/results)")
    parser.add_argument("--model", nargs="+", default=None, help="Filter by model name(s)")
    parser.add_argument("--trials", nargs="+", default=None, help="Filter by trial/dataset name(s)")
    parser.add_argument("--methods", nargs="+", default=None, help="Filter by method name(s)")
    parser.add_argument(
        "--version",
        choices=["original", "revised", "both"],
        default="original",
        help="Score file version to include (default: original)",
    )
    parser.add_argument("--per_seed", action="store_true", help="Also print per-seed detail table")
    parser.add_argument("--out_csv", help="Save aggregated per-(model,method,dataset) table to CSV")
    parser.add_argument("--out_summary_csv", help="Save cross-dataset method-averaged table to CSV")
    args = parser.parse_args()

    # ---- Collect files ----
    version_filter = None if args.version == "both" else args.version
    if args.files:
        paths = list(args.files)
    else:
        root = Path(args.root)
        if not root.exists():
            parser.error(f"Root directory not found: {root}")
        paths = discover_files(root, version_filter)

    if not paths:
        print(f"No *_logic_score{'_revised' if version_filter == 'revised' else ''}.json files found.")
        return

    print(f"Found {len(paths)} logic_score file(s).")

    # ---- Load summaries ----
    rows: List[Dict] = []
    for p in paths:
        try:
            rows.append(summarize_file(p))
        except Exception as e:
            print(f"  WARNING: skipping {p} — {e}")

    # ---- Apply filters ----
    if args.model:
        requested = {m.lower() for m in args.model}
        rows = [r for r in rows if r["model"].lower() in requested]
    if args.trials:
        requested = {t.lower() for t in args.trials}
        rows = [r for r in rows if r["dataset"].lower() in requested]
    if args.methods:
        requested = {m.lower() for m in args.methods}
        rows = [r for r in rows if r["method"].lower() in requested]

    if not rows:
        print("No data after filtering.")
        return

    print(f"{len(rows)} row(s) after filtering.\n")

    # ---- Per-seed detail ----
    if args.per_seed:
        print("=" * 100)
        print("PER-SEED DETAIL")
        print("=" * 100)
        print_per_seed_table(rows)
        print()

    # ---- Aggregated per (model, method, dataset) ----
    agg_by_dataset = aggregate_by_method_dataset(rows)
    datasets = sorted(agg_by_dataset["dataset"].unique()) if not agg_by_dataset.empty else []

    for dataset in datasets:
        subset = agg_by_dataset[agg_by_dataset["dataset"] == dataset].copy()
        print("=" * 100)
        print(f"DATASET: {dataset.upper()}")
        print("  score=mean overall_score (0–5)  |  G1–G5=gate pass-rate  |  novel=is_novel rate")
        print("  std shown when >1 seed available")
        print("=" * 100)
        print_aggregated_table(subset, show_dataset=False)
        print()

    # ---- Cross-dataset method averages ----
    agg_overall = aggregate_by_method(rows)
    print("=" * 100)
    print("OVERALL: AVERAGE ACROSS ALL DATASETS (mean of per-dataset means)")
    print("  n_runs = total seed×dataset rows averaged")
    print("=" * 100)
    print_aggregated_table(agg_overall, show_dataset=False)
    print()

    # ---- Save CSVs ----
    if args.out_csv and not agg_by_dataset.empty:
        agg_by_dataset.to_csv(args.out_csv, index=False)
        print(f"Saved per-(model,method,dataset) aggregated table to: {args.out_csv}")

    if args.out_summary_csv and not agg_overall.empty:
        agg_overall.to_csv(args.out_summary_csv, index=False)
        print(f"Saved method-averaged summary to: {args.out_summary_csv}")


if __name__ == "__main__":
    main()
