#!/usr/bin/env python3
"""summarize_pubmed_verdicts.py

Roll up PubMed validation abstracts to a per-hypothesis verdict using the
Oxford CEBM hierarchical scoring logic, then summarise across methods,
datasets and (optionally) seeds.

Usage:
    # All methods, all datasets, collapse seeds (default)
    python tools/summarize_pubmed_verdicts.py

    # Filter to one model
    python tools/summarize_pubmed_verdicts.py --model gpt-5-mini

    # Keep seeds separate
    python tools/summarize_pubmed_verdicts.py --keep-seeds

    # Write CSV
    python tools/summarize_pubmed_verdicts.py --out_csv docs/agent/pubmed_verdicts.csv

    # Also include s2 validation files
    python tools/summarize_pubmed_verdicts.py --include-s2
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────
# Label helpers (same pattern as summarize_classification_percentages.py)
# ──────────────────────────────────────────────────────────────────────

def _normalize_label(value: Any) -> str:
    label = str(value or "").strip().upper()
    return label if label else "UNKNOWN"


# ──────────────────────────────────────────────────────────────────────
# CEBM verdict logic (provided by user)
# ──────────────────────────────────────────────────────────────────────

def compute_hypothesis_verdict(analyzed_abstracts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Roll up abstract classifications using strict EBM hierarchical override."""
    counts: Dict[str, int] = {
        "SUPPORT_INTERACTION": 0, "SUPPORT_WEAK": 0, "CONFLICT": 0,
        "NO_INTERACTION": 0, "PROGNOSTIC_MAIN_EFFECT": 0, "IRRELEVANT": 0,
    }

    # Track the highest Oxford CEBM level seen for each classification
    # Level 2 = RCT/Meta-Analysis.  Level 1 = Observational/Other.  0 = None.
    max_evidence_level: Dict[str, int] = defaultdict(int)

    for abstract in analyzed_abstracts:
        cls = _normalize_label(abstract.get("classification", "IRRELEVANT"))
        if cls not in counts:
            cls = "IRRELEVANT"
        design = str(abstract.get("study_design", "UNKNOWN")).upper()

        counts[cls] += 1

        lvl = 2 if design in {"RCT_SECONDARY", "RCT_SECONDARY_ANALYSIS",
                               "SYSTEMATIC_REVIEW_META_ANALYSIS", "META_ANALYSIS"} else 1
        if lvl > max_evidence_level[cls]:
            max_evidence_level[cls] = lvl

    # Weighted hierarchical scores: RCT-level=4/2, Obs-level=2/1
    # Single-paper conflicts are demoted one tier (require >=2 independent papers for full weight)
    positive_score = 0
    if max_evidence_level["SUPPORT_INTERACTION"] == 2:
        positive_score = max(positive_score, 4)
    elif max_evidence_level["SUPPORT_INTERACTION"] == 1:
        positive_score = max(positive_score, 2)
    if max_evidence_level["SUPPORT_WEAK"] == 2:
        positive_score = max(positive_score, 2)
    elif max_evidence_level["SUPPORT_WEAK"] == 1:
        positive_score = max(positive_score, 1)

    negative_score = 0
    if max_evidence_level["CONFLICT"] == 2:
        # Full -4 only if ≥2 independent conflict papers; single paper demoted to -2
        negative_score = min(negative_score, -4 if counts["CONFLICT"] >= 2 else -2)
    elif max_evidence_level["CONFLICT"] == 1:
        negative_score = min(negative_score, -2)
    if max_evidence_level["NO_INTERACTION"] == 2:
        # Full -2 only if ≥2 independent no-interaction papers; single paper demoted to -1
        negative_score = min(negative_score, -2 if counts["NO_INTERACTION"] >= 2 else -1)
    elif max_evidence_level["NO_INTERACTION"] == 1:
        negative_score = min(negative_score, -1)

    # Resolve verdict using max-override logic
    # Exception: RCT-level support (pos=4) is not demoted by a single weak
    # NO_INTERACTION paper (neg=-1).  Requires at least neg<=-2 to enter
    # the mixed branch.
    if positive_score > 0 and negative_score < 0:
        if positive_score == 4 and abs(negative_score) <= 1:
            verdict = "STRONG_SUPPORT"
        elif positive_score > abs(negative_score):
            verdict = "CONTROVERSIAL_BUT_SUPPORTED"
        elif abs(negative_score) > positive_score:
            verdict = "REFUTED_BY_HIGHER_EVIDENCE"
        else:
            # True score tie — use abstract counts as tiebreaker
            pos_count = counts["SUPPORT_INTERACTION"] + counts["SUPPORT_WEAK"]
            neg_count = counts["CONFLICT"] + counts["NO_INTERACTION"]
            if pos_count > neg_count:
                verdict = "CONTROVERSIAL_BUT_SUPPORTED"
            elif neg_count > pos_count:
                verdict = "REFUTED_BY_HIGHER_EVIDENCE"
            else:
                verdict = "CONTESTED_EQUAL_EVIDENCE"
    elif positive_score == 4:
        verdict = "STRONG_SUPPORT"
    elif positive_score == 2:
        verdict = "WEAK_SUPPORT"
    elif positive_score == 1:
        verdict = "ISOLATED_WEAK_FINDING"
    elif negative_score == -4:
        verdict = "STRONG_CONFLICT"
    elif negative_score == -2:
        verdict = "LIKELY_NO_INTERACTION"
    elif negative_score == -1:
        verdict = "WEAK_CONFLICT"
    elif counts["PROGNOSTIC_MAIN_EFFECT"] > 0:
        verdict = "PROGNOSTIC_ONLY"
    else:
        verdict = "INSUFFICIENT_EVIDENCE"

    final_score: float
    if positive_score == abs(negative_score) and positive_score != 0:
        final_score = 0.0  # True tie
    elif positive_score > abs(negative_score):
        final_score = float(positive_score)
    else:
        final_score = float(negative_score)

    return {
        "hypothesis_verdict": verdict,
        "evidence_score": final_score,
        "class_counts": counts,
    }


def map_verdict_to_original_label(verdict_result: Dict[str, Any]) -> str:
    """Map granular EBM verdicts into final meta-analytical hypothesis tiers."""
    verdict = verdict_result.get("hypothesis_verdict", "INSUFFICIENT_EVIDENCE")

    if verdict == "STRONG_SUPPORT":
        return "Strongly Corroborated"

    if verdict in ("WEAK_SUPPORT", "ISOLATED_WEAK_FINDING", "CONTROVERSIAL_BUT_SUPPORTED", "CONTESTED_EQUAL_EVIDENCE"):
        return "Weakly Corroborated / Debated"

    if verdict in ("STRONG_CONFLICT", "REFUTED_BY_HIGHER_EVIDENCE", "WEAK_CONFLICT"):
        return "Refuted"

    if verdict == "LIKELY_NO_INTERACTION":
        return "Likely No Interaction"

    if verdict == "PROGNOSTIC_ONLY":
        return "Prognostic Main Effect Only"

    # INSUFFICIENT_EVIDENCE
    return "No Evidence"


# ──────────────────────────────────────────────────────────────────────
# Path inference helpers (mirrors summarize_classification_percentages.py)
# ──────────────────────────────────────────────────────────────────────

def _infer_dataset(file_path: Path, payload: Dict[str, Any]) -> str:
    # Aliases: non-canonical names → canonical
    _DATASET_ALIASES = {
        "accord_bp": "accord",
        "accord-bp": "accord",
        "accordbp":  "accord",
    }
    dataset = payload.get("dataset", "")
    if isinstance(dataset, str) and dataset.strip().lower() not in (
        "", "unknown", "unknown_dataset"
    ):
        # Normalise: strip LLM-appended descriptions like "ACCORD (type 2 diabetes, ...)"
        # Keep only the leading word token and lowercase it.
        import re as _re
        canonical = _re.split(r"[\s(,]", dataset.strip())[0].lower()
        # Resolve known aliases (e.g. accord_bp → accord)
        canonical = _DATASET_ALIASES.get(canonical, canonical)
        if canonical:
            return canonical
    parts = file_path.parts
    for anchor in ("agent", "results"):
        try:
            idx = next(i for i, p in enumerate(parts) if p == anchor)
            return parts[idx + 1]
        except (StopIteration, IndexError):
            continue
    return file_path.parent.name


def _infer_model(file_path: Path) -> str:
    parts = file_path.parts
    for anchor in ("agent", "results"):
        try:
            idx = next(i for i, p in enumerate(parts) if p == anchor)
            return parts[idx + 2]
        except (StopIteration, IndexError):
            continue
    return "unknown"


def _infer_method_folder(file_path: Path) -> str:
    parent = file_path.parent.name.lower()
    if parent.startswith("seed_") and file_path.parent.parent is not None:
        return file_path.parent.parent.name.lower()
    return parent


def _infer_method(file_path: Path) -> str:
    folder = _infer_method_folder(file_path)
    mapping = {
        "hypogenic":            "HypoGeniC",
        "simple_cot":           "ALEX (w/o verifier)",
        "cot":                  "CoT",
        "with_shap_xlearner":   "ALEX",
        "with_shap_drlearner":  "ALEX",
        "without_shap_baseline":"ALEX w/o SHAP",
        "researchagent":        "ResearchAgent",
    }
    return mapping.get(folder, folder)


def _infer_seed(file_path: Path) -> str:
    for part in file_path.parts:
        if part.lower().startswith("seed_"):
            return part.lower()
    return "no_seed"


# ──────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────

ALL_VERDICTS = [
    "STRONG_SUPPORT", "CONTROVERSIAL_BUT_SUPPORTED",
    "WEAK_SUPPORT", "ISOLATED_WEAK_FINDING", "CONTESTED_EQUAL_EVIDENCE",
    "STRONG_CONFLICT", "REFUTED_BY_HIGHER_EVIDENCE", "WEAK_CONFLICT",
    "LIKELY_NO_INTERACTION", "PROGNOSTIC_ONLY", "INSUFFICIENT_EVIDENCE",
]


def process_file(
    path: Path,
) -> List[Dict[str, Any]]:
    """Return one row per *feature* in the file.

    Multiple mechanism_results for the same feature_name (e.g. 3 mechanisms ×
    5 features = 15 raw entries) are collapsed into a single verdict by pooling
    all abstracts_analyzed across mechanisms.  This ensures n_tot = n_features
    per seed file (typically 5), not n_mechanisms.
    """
    try:
        with path.open() as f:
            payload = json.load(f)
    except Exception as e:
        print(f"  Warning: could not load {path}: {e}")
        return []

    dataset = _infer_dataset(path, payload)
    model   = _infer_model(path)
    method  = _infer_method(path)
    seed    = _infer_seed(path)

    # Pool all abstracts_analyzed by feature_name across mechanisms
    feature_abstracts: Dict[str, List] = defaultdict(list)
    feature_order: List[str] = []
    for mech_result in payload.get("mechanism_results", []):
        feature_name = mech_result.get("feature_name", "unknown")
        if feature_name not in feature_abstracts:
            feature_order.append(feature_name)
        feature_abstracts[feature_name].extend(
            mech_result.get("abstracts_analyzed", [])
        )

    rows = []
    for feature_name in feature_order:
        abstracts    = feature_abstracts[feature_name]
        verdict_info = compute_hypothesis_verdict(abstracts)
        label        = map_verdict_to_original_label(verdict_info)

        rows.append({
            "dataset":          dataset,
            "model":            model,
            "method":           method,
            "seed":             seed,
            "feature_name":     feature_name,
            "n_abstracts":      len(abstracts),
            "verdict":          verdict_info["hypothesis_verdict"],
            "evidence_score":   verdict_info["evidence_score"],
            "mapped_label":     label,
            **{f"n_{k.lower()}": v for k, v in verdict_info["class_counts"].items()},
        })
    return rows


# ──────────────────────────────────────────────────────────────────────
# Summarisation
# ──────────────────────────────────────────────────────────────────────

def _group_key(row: Dict[str, Any], keep_seeds: bool) -> Tuple:
    if keep_seeds:
        return (row["dataset"], row["model"], row["method"], row["seed"], row["feature_name"])
    return (row["dataset"], row["model"], row["method"], row["feature_name"])


def aggregate_rows(
    rows: List[Dict[str, Any]],
    keep_seeds: bool,
) -> List[Dict[str, Any]]:
    """Collapse seeds by averaging evidence_score and collecting verdict distribution."""
    if keep_seeds:
        return rows

    # Group by (dataset, model, method, feature_name) and average over seeds
    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["model"], row["method"], row["feature_name"])
        groups[key].append(row)

    out = []
    for (dataset, model, method, feature_name), grp in sorted(groups.items()):
        mean_score = sum(r["evidence_score"] for r in grp) / len(grp)
        n_seeds    = len(grp)

        # Majority verdict across seeds
        verdict_counts: Dict[str, int] = defaultdict(int)
        for r in grp:
            verdict_counts[r["verdict"]] += 1
        majority_verdict = max(verdict_counts, key=lambda v: verdict_counts[v])
        majority_label   = map_verdict_to_original_label({"hypothesis_verdict": majority_verdict})

        # Sum class counts
        cc: Dict[str, int] = defaultdict(int)
        for r in grp:
            for k in ALL_VERDICTS:
                field = f"n_{k.lower()}"
                cc[field] += r.get(field, 0)

        out.append({
            "dataset":        dataset,
            "model":          model,
            "method":         method,
            "feature_name":   feature_name,
            "n_seeds":        n_seeds,
            "mean_evidence_score": round(mean_score, 3),
            "majority_verdict":   majority_verdict,
            "mapped_label":       majority_label,
            "verdict_distribution": dict(verdict_counts),
            **cc,
        })
    return out


ALL_MAPPED_LABELS = [
    "Strongly Corroborated",
    "Weakly Corroborated / Debated",
    "Likely No Interaction",
    "Prognostic Main Effect Only",
    "Refuted",
    "No Evidence",
]


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def method_summary(raw_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-method stats: mean±std **across seeds** of label% and evidence score.

    Pipeline:
    1. For each method, build a (dataset, seed) grid padded to N_FEATURES_PER_SEED.
    2. For each seed, average the per-slot percentages across all datasets.
    3. mean = mean of those per-seed values; std = std of those per-seed values.

    This means the std reflects seed-to-seed reproducibility (the conventional
    definition in papers), not the noisier (dataset * seed) variance.
    """
    N_FEATURES_PER_SEED = 5

    # Full (dataset, seed) grid from all methods combined.
    all_pairs: set = set()
    for r in raw_rows:
        all_pairs.add((r["dataset"], r["seed"]))
    expected_pairs = sorted(all_pairs)
    all_seeds = sorted({s for _, s in expected_pairs})
    all_datasets = sorted({d for d, _ in expected_pairs})

    # Group by method → (dataset, seed) → rows
    method_ds_seed: Dict[str, Dict[tuple, List]] = defaultdict(lambda: defaultdict(list))
    for r in raw_rows:
        method_ds_seed[r["method"]][(r["dataset"], r["seed"])].append(r)

    _no_ev_pct = {k: (100.0 if k == "No Evidence" else 0.0) for k in ALL_MAPPED_LABELS}

    summary = {}
    for method, ds_seed_groups in sorted(method_ds_seed.items()):

        # Step 1: compute per-(dataset, seed) slot percentages and scores
        slot_pct: Dict[tuple, Dict[str, float]] = {}
        slot_score: Dict[tuple, float] = {}
        slot_n: Dict[tuple, int] = {}

        for pair in expected_pairs:
            slot_rows = ds_seed_groups.get(pair, [])
            n_actual = len(slot_rows)
            n_denom = max(n_actual, N_FEATURES_PER_SEED)
            slot_n[pair] = n_denom

            if n_actual == 0:
                slot_pct[pair] = dict(_no_ev_pct)
                slot_score[pair] = 0.0
            else:
                lc: Dict[str, int] = defaultdict(int)
                for rr in slot_rows:
                    lbl = rr.get("mapped_label") or map_verdict_to_original_label(
                        {"hypothesis_verdict": rr.get("majority_verdict") or rr.get("verdict", "INSUFFICIENT_EVIDENCE")}
                    )
                    lc[lbl] += 1
                lc["No Evidence"] += n_denom - n_actual
                slot_pct[pair] = {k: 100.0 * lc.get(k, 0) / n_denom for k in ALL_MAPPED_LABELS}
                score_field = "mean_evidence_score" if "mean_evidence_score" in slot_rows[0] else "evidence_score"
                slot_score[pair] = sum(rr[score_field] for rr in slot_rows) / n_denom

        # Step 2: collapse to per-seed values by averaging across datasets
        per_seed_pct: List[Dict[str, float]] = []
        per_seed_score: List[float] = []
        for seed in all_seeds:
            pairs_this_seed = [(d, seed) for d in all_datasets]
            pcts = [slot_pct[p] for p in pairs_this_seed]
            scores = [slot_score[p] for p in pairs_this_seed]
            per_seed_pct.append({k: sum(p[k] for p in pcts) / len(pcts) for k in ALL_MAPPED_LABELS})
            per_seed_score.append(sum(scores) / len(scores))

        # Step 3: mean and std across seeds
        n_seeds = len(all_seeds)
        mean_score = sum(per_seed_score) / n_seeds
        mean_pct = {k: sum(s[k] for s in per_seed_pct) / n_seeds for k in ALL_MAPPED_LABELS}
        std_pct   = {k: _std([s[k] for s in per_seed_pct]) for k in ALL_MAPPED_LABELS}

        summary[method] = {
            "n_hypotheses":        sum(slot_n.values()),
            "n_seeds":             n_seeds,
            "mean_evidence_score": round(mean_score, 3),
            "std_evidence_score":  round(_std(per_seed_score), 3),
            "mean_label_pct":      mean_pct,
            "std_label_pct":       std_pct,
        }
    return summary


# ──────────────────────────────────────────────────────────────────────
# Terminal table rendering
# ──────────────────────────────────────────────────────────────────────

# Short labels for the 6 mapped classes
_LABEL_SHORT = {
    "Strongly Corroborated":        "Sup+",
    "Weakly Corroborated / Debated": "SupWeak",
    "Likely No Interaction":         "NoInt",
    "Prognostic Main Effect Only":   "Prognostic",
    "Refuted":                       "Conflict",
    "No Evidence":                   "NoEvidence",
}


def _print_verdict_tables(all_rows: List[Dict[str, Any]], agg: List[Dict[str, Any]]) -> None:
    """Print one ASCII table per dataset: rows=methods, columns=mean%±std% across seeds."""
    # Collect dimensions
    datasets   = sorted({r["dataset"] for r in all_rows})
    methods    = sorted({r["method"]  for r in all_rows})
    col_labels = [_LABEL_SHORT[v] for v in ALL_MAPPED_LABELS]

    METHOD_W = max(len(m) for m in methods) + 2
    SCORE_W  = 12  # "+0.45±0.12"
    N_W      = 5   # "nn"
    COL_W    = 13  # "52.3±8.1%  "

    SEP_W = METHOD_W + SCORE_W + N_W + COL_W * len(ALL_MAPPED_LABELS) + len(ALL_MAPPED_LABELS) + 4

    def _hline(char="-"):
        widths = [METHOD_W, SCORE_W, N_W] + [COL_W] * len(ALL_MAPPED_LABELS)
        return "+" + "+".join(char * w for w in widths) + "+"

    def _header_row():
        cells = [
            f"{'Method':<{METHOD_W}}",
            f"{'Score(±std)':>{SCORE_W}}",
            f"{'n_tot':>{N_W}}",
        ] + [f"{lbl:^{COL_W}}" for lbl in col_labels]
        return "|" + "|".join(cells) + "|"

    def _data_row(method: str, info: Dict[str, Any]) -> str:
        n     = info["n_hypotheses"]
        score = info["mean_evidence_score"]
        std_s = info["std_evidence_score"]
        score_str = f"{score:+.2f}±{std_s:.2f}"
        cells = [
            f"{method:<{METHOD_W}}",
            f"{score_str:>{SCORE_W}}",
            f"{n:>{N_W}d}",
        ]
        for v in ALL_MAPPED_LABELS:
            mp  = info["mean_label_pct"].get(v, 0.0)
            sp  = info["std_label_pct"].get(v, 0.0)
            if mp > 0:
                cell = f"{mp:.1f}±{sp:.1f}%"
                cells.append(f"{cell:^{COL_W}}")
            else:
                cells.append(" " * COL_W)
        return "|" + "|".join(cells) + "|"

    # Overall table (all datasets combined)
    print("\n" + "=" * max(80, SEP_W))
    print("OVERALL  (all datasets)")
    print(_hline("="))
    print(_header_row())
    print(_hline("="))

    overall_ms = method_summary(all_rows)
    for method in methods:
        info = overall_ms.get(method)
        if info:
            print(_data_row(method, info))
    print(_hline())

    # Per-dataset tables
    ds_raw: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        ds_raw[r["dataset"]].append(r)

    for dataset in datasets:
        print(f"\n{'=' * max(80, SEP_W)}")
        print(f"DATASET: {dataset.upper()}")
        print(_hline("="))
        print(_header_row())
        print(_hline("="))

        dm = method_summary(ds_raw[dataset])
        for method in methods:
            info = dm.get(method)
            if info:
                print(_data_row(method, info))
        print(_hline())


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarise PubMed validation verdicts per hypothesis")
    p.add_argument("--root",       default="ALEX/results",
                   help="Root results directory (default: ALEX/results)")
    p.add_argument("--model",      default=None,
                   help="Filter to a specific model folder (e.g. gpt-5-mini)")
    p.add_argument("--dataset",    default=None,
                   help="Filter to a specific dataset (e.g. sprint)")
    p.add_argument("--method",     default=None,
                   help="Filter to a specific method folder (e.g. simple_cot)")
    p.add_argument("--keep-seeds", action="store_true",
                   help="Keep seed runs separate instead of collapsing to mean")
    p.add_argument("--with-s2", action="store_true",
                   help="Also include hypotheses_s2_validation.json files (excluded by default)")
    p.add_argument("--judge_model", default="gpt-5-mini",
                   help="Judge model used for PubMed validation (default: gpt-5-mini). "
                        "Selects hypotheses_pubmed_validation.json for the default, "
                        "or hypotheses_pubmed_validation__{judge_model}.json for others.")
    p.add_argument("--out_csv",    default=None,
                   help="Write aggregated rows to this CSV path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    # Collect validation files
    judge_suffix = f"__{args.judge_model}" if args.judge_model != "gpt-5-mini" else ""
    patterns = [f"**/hypotheses_pubmed_validation{judge_suffix}.json"]
    if args.with_s2:
        patterns.append("**/hypotheses_s2_validation.json")

    files: List[Path] = []
    for pat in patterns:
        files.extend(root.glob(pat))
    files = sorted(set(files))

    # Apply filters
    if args.model:
        files = [f for f in files if args.model in f.parts]
    if args.dataset:
        files = [f for f in files if args.dataset in f.parts]
    if args.method:
        files = [f for f in files if args.method in _infer_method_folder(f)]

    # Only show primary methods
    _ALLOWED_FOLDERS = {
        "with_shap_drlearner", "without_shap_baseline",
        "hypogenic", "cot", "simple_cot", "researchagent",
    }
    files = [f for f in files if _infer_method_folder(f) in _ALLOWED_FOLDERS]

    if not files:
        print(f"No validation files found under {root}. Check --root / --model / --dataset.")
        return

    print(f"Found {len(files)} validation file(s).\n")

    # Process
    all_rows: List[Dict[str, Any]] = []
    for f in files:
        all_rows.extend(process_file(f))

    if not all_rows:
        print("No mechanism results found.")
        return

    # Aggregate (collapse seeds unless --keep-seeds)
    agg = aggregate_rows(all_rows, keep_seeds=args.keep_seeds)

    # ── Tables ─────────────────────────────────────────────────────
    _print_verdict_tables(all_rows, agg)

    # ── CSV output ──────────────────────────────────────────────────
    if args.out_csv:
        import csv
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine columns
        all_keys: List[str] = []
        seen: set = set()
        for r in agg:
            for k in r:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        # Move verdict_distribution (dict) to a string column at the end
        stable_keys = [k for k in all_keys if k != "verdict_distribution"]
        if "verdict_distribution" in seen:
            stable_keys.append("verdict_distribution")

        with out_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=stable_keys, extrasaction="ignore")
            writer.writeheader()
            for r in agg:
                row_out = dict(r)
                if "verdict_distribution" in row_out:
                    row_out["verdict_distribution"] = json.dumps(row_out["verdict_distribution"])
                writer.writerow(row_out)

        print(f"\nCSV written to: {out_path}  ({len(agg)} rows)")


if __name__ == "__main__":
    main()
