#!/usr/bin/env python3
"""Summarize label percentages for each method across datasets.

This script is intended for abstract-level labels in PubMed validation outputs,
especially `abstracts_analyzed[].classification` (e.g., NO_INTERACTION,
SUPPORT_INTERACTION, IRRELEVANT).

Usage:
    python tools/summarize_classification_percentages.py
    python tools/summarize_classification_percentages.py --label-field stance
    python tools/summarize_classification_percentages.py --out_csv docs/agent/classification_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize abstract-level label percentages by method across datasets"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="docs/agent",
        help="Root directory containing result JSON files (default: docs/agent)",
    )
    parser.add_argument(
        "--source",
        choices=["pubmed", "judge"],
        default="pubmed",
        help="Which result type to aggregate (default: pubmed)",
    )
    parser.add_argument(
        "--label-field",
        choices=["classification", "stance", "recommendation"],
        default="classification",
        help=(
            "Field to summarize. For pubmed, use classification (default) or stance. "
            "For judge, use recommendation."
        ),
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Optional output CSV path for aggregated summary",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Filter results to a specific model folder name (e.g. gpt-5-mini, "
            "google_gemini-3-flash-preview). If omitted, all models are included "
            "and the model name is shown in output."
        ),
    )
    parser.add_argument(
        "--keep-seeds",
        action="store_true",
        help=(
            "Keep seed runs separate (model/seed_x). Default behavior collapses "
            "seeds per model and reports mean±std across seeds."
        ),
    )
    return parser.parse_args()


def infer_model(file_path: Path) -> str:
    """Extract the model folder name from the new path structure.

    New structure: docs/agent/<cohort>/<model>/<method>/hypotheses_*.json
    The model is 2 levels up from the file.
    """
    parts = file_path.parts
    try:
        agent_idx = next(i for i, p in enumerate(parts) if p == "agent")
        return parts[agent_idx + 2]  # cohort=+1, model=+2
    except (StopIteration, IndexError):
        return "unknown"


def infer_method(file_path: Path) -> str:
    # New structure:
    # - docs/agent/<cohort>/<model>/<method>/hypotheses_*.json
    # - docs/agent/<cohort>/<model>/<method>/seed_<n>/hypotheses_*.json
    method_folder = infer_method_folder(file_path)

    if method_folder == "hypogenic":
        return "HypoGeniC"
    if method_folder == "simple_cot":
        return "SimpleCoT"
    if method_folder in ("with_shap_xlearner", "with_shap_drlearner"):
        return "ALEX"
    if method_folder == "without_shap_baseline":
        return "Baseline"

    # Fallback: try filename for legacy flat structure
    name = file_path.name.lower()
    if "hypogenic" in name:
        return "HypoGeniC"
    if "simple_cot" in name:
        return "SimpleCoT"
    if "with_shap_xlearner" in name or "with_shap_drlearner" in name:
        return "ALEX"
    if "without_shap_baseline" in name:
        return "Baseline"
    if "with_shap" in name:
        return "WithSHAP"
    if "without_shap" in name:
        return "WithoutSHAP"

    return method_folder


def infer_method_folder(file_path: Path) -> str:
    """Return raw method folder name, handling optional seed subfolders."""
    parent = file_path.parent.name.lower()
    if parent.startswith("seed_") and file_path.parent.parent is not None:
        return file_path.parent.parent.name.lower()
    return parent


def infer_seed(file_path: Path) -> str:
    """Extract seed identifier from path (e.g., seed_0), else 'no_seed'."""
    for part in file_path.parts:
        part_l = part.lower()
        if part_l.startswith("seed_"):
            return part_l
    return "no_seed"


def infer_dataset(file_path: Path, payload: Dict[str, Any]) -> str:
    dataset = payload.get("dataset")
    if isinstance(dataset, str) and dataset.strip():
        dataset_clean = dataset.strip()
        if dataset_clean.lower() not in {"unknown", "unknown_dataset", "unknown_datase"}:
            return dataset_clean

    # New structure: docs/agent/<cohort>/<model>/<method>/hypotheses_*.json
    # Cohort is 3 levels up from the file
    parts = file_path.parts
    try:
        agent_idx = next(i for i, p in enumerate(parts) if p == "agent")
        cohort = parts[agent_idx + 1]
        return cohort
    except (StopIteration, IndexError):
        pass

    # Fallback: parent folder name (legacy flat structure)
    return file_path.parent.name


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def _normalize_label(value: Any) -> str:
    label = str(value or "").strip()
    if not label:
        return "UNKNOWN"
    return label.upper()


def parse_pubmed_labels(payload: Dict[str, Any], label_field: str) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    mechanism_results = payload.get("mechanism_results", [])
    if not isinstance(mechanism_results, list):
        return counts

    for mechanism in mechanism_results:
        abstracts = mechanism.get("abstracts_analyzed", [])
        if not isinstance(abstracts, list):
            continue
        for abstract in abstracts:
            if not isinstance(abstract, dict):
                continue
            label = abstract.get(label_field)
            if label is None:
                continue
            counts[_normalize_label(label)] += 1
    return counts


# Preferred display order for classification labels (most positive → least).
# Any label not listed here will be appended alphabetically at the end.
PREFERRED_LABEL_ORDER = [
    "SUPPORT_INTERACTION",
    "SUPPORT_WEAK",
    "PROGNOSTIC_MAIN_EFFECT",
    "NO_INTERACTION",
    "CONFLICT",
    "IRRELEVANT",
]


_LABEL_PRIORITY: Dict[str, int] = {
    lbl: i for i, lbl in enumerate(PREFERRED_LABEL_ORDER)
}


def _dominant_classification(abstracts: List[Dict[str, Any]], label_field: str) -> str:
    """Return the mechanism-level label using priority-first (≥1 count) strategy.

    Among non-IRRELEVANT abstract labels, assigns the highest-priority label
    that appears at least once (per PREFERRED_LABEL_ORDER).
    Falls back to IRRELEVANT if all abstracts are IRRELEVANT.
    Returns None if no abstracts have a valid label.
    """
    freq: Dict[str, int] = defaultdict(int)
    for ab in abstracts:
        if not isinstance(ab, dict):
            continue
        raw = ab.get(label_field)
        if raw is None:
            continue
        freq[_normalize_label(raw)] += 1
    if not freq:
        return "IRRELEVANT"  # no abstracts retrieved → treat as no evidence

    # Check each preferred label in priority order — return first one present
    for lbl in PREFERRED_LABEL_ORDER:
        if lbl != "IRRELEVANT" and freq.get(lbl, 0) >= 1:
            return lbl

    # All abstracts are IRRELEVANT (or only unlisted labels)
    return "IRRELEVANT" if freq.get("IRRELEVANT", 0) > 0 else next(iter(freq))


def parse_mechanism_level_labels(payload: Dict[str, Any], label_field: str) -> Dict[str, int]:
    """Classify each mechanism and tally counts.

    For `classification`, applies the verdict rule internally and maps verdicts
    back to the original label set only.
    For other label fields, uses dominant abstract-level label.
    """
    counts: Dict[str, int] = defaultdict(int)
    mechanism_results = payload.get("mechanism_results", [])
    if not isinstance(mechanism_results, list):
        return counts

    for mechanism in mechanism_results:
        abstracts = mechanism.get("abstracts_analyzed", [])
        if label_field == "classification":
            verdict_result = compute_hypothesis_verdict(abstracts)
            mapped_label = map_verdict_to_original_label(verdict_result)
            counts[mapped_label] += 1
        else:
            dominant = _dominant_classification(abstracts, label_field)
            counts[dominant] += 1
    return counts


# def compute_hypothesis_verdict(analyzed_abstracts: List[Dict[str, Any]]) -> Dict[str, Any]:
#     """Roll up abstract classifications into a verdict + score (user rule)."""
#     counts = {
#         'SUPPORT_INTERACTION': 0,
#         'SUPPORT_WEAK': 0,
#         'CONFLICT': 0,
#         'NO_INTERACTION': 0,
#         'PROGNOSTIC_MAIN_EFFECT': 0,
#         'IRRELEVANT': 0,
#     }

#     for abstract in analyzed_abstracts:
#         cls = _normalize_label(abstract.get('classification', 'IRRELEVANT'))
#         if cls in counts:
#             counts[cls] += 1

#     score = (
#         (counts['SUPPORT_INTERACTION'] * 2.0)
#         + (counts['SUPPORT_WEAK'] * 1.0)
#         - (counts['CONFLICT'] * 2.0)
#         - (counts['NO_INTERACTION'] * 0.5)
#     )

#     if counts['SUPPORT_INTERACTION'] > 0 and counts['CONFLICT'] == 0:
#         verdict = "STRONG_SUPPORT"
#     elif (counts['SUPPORT_INTERACTION'] > 0 or counts['SUPPORT_WEAK'] > 0) and counts['CONFLICT'] > 0:
#         verdict = "MIXED_EVIDENCE"
#     elif counts['SUPPORT_WEAK'] > 0 and counts['CONFLICT'] == 0:
#         verdict = "WEAK_SUPPORT"
#     elif counts['CONFLICT'] > 0 and counts['SUPPORT_INTERACTION'] == 0 and counts['SUPPORT_WEAK'] == 0:
#         verdict = "STRONG_CONFLICT"
#     elif counts['NO_INTERACTION'] > 0 and counts['SUPPORT_INTERACTION'] == 0 and counts['SUPPORT_WEAK'] == 0:
#         verdict = "LIKELY_NO_INTERACTION"
#     elif counts['PROGNOSTIC_MAIN_EFFECT'] > 0:
#         verdict = "PROGNOSTIC_ONLY"
#     else:
#         verdict = "INSUFFICIENT_EVIDENCE"

#     return {
#         "hypothesis_verdict": verdict,
#         "evidence_score": score,
#         "class_counts": counts,
#     }


# def map_verdict_to_original_label(verdict_result: Dict[str, Any]) -> str:
#     """Map verdict categories back into original classification labels only."""
#     verdict = verdict_result.get("hypothesis_verdict", "INSUFFICIENT_EVIDENCE")
#     counts = verdict_result.get("class_counts", {})
#     score = float(verdict_result.get("evidence_score", 0.0))

#     if verdict == "STRONG_SUPPORT":
#         return "SUPPORT_INTERACTION"
#     if verdict == "WEAK_SUPPORT":
#         return "SUPPORT_WEAK"
#     if verdict == "STRONG_CONFLICT":
#         return "CONFLICT"
#     if verdict == "LIKELY_NO_INTERACTION":
#         return "NO_INTERACTION"
#     if verdict == "PROGNOSTIC_ONLY":
#         return "PROGNOSTIC_MAIN_EFFECT"
#     if verdict == "INSUFFICIENT_EVIDENCE":
#         return "IRRELEVANT"

#     if verdict == "MIXED_EVIDENCE":
#         if counts.get("SUPPORT_INTERACTION", 0) > 0 and score >= 0:
#             return "SUPPORT_WEAK"
#         if counts.get("SUPPORT_WEAK", 0) > 0 and score >= 0:
#             return "SUPPORT_WEAK"
#         return "CONFLICT"

#     return "IRRELEVANT"

def compute_hypothesis_verdict(analyzed_abstracts: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {
        'SUPPORT_INTERACTION': 0,
        'SUPPORT_WEAK': 0,
        'CONFLICT': 0,
        'NO_INTERACTION': 0,
        'PROGNOSTIC_MAIN_EFFECT': 0,
        'IRRELEVANT': 0,
    }

    for abstract in analyzed_abstracts:
        cls = _normalize_label(abstract.get('classification', 'IRRELEVANT'))
        if cls in counts:
            counts[cls] += 1

    # Same base scoring for downstream ranking
    score = (
        (counts['SUPPORT_INTERACTION'] * 2.0)
        + (counts['SUPPORT_WEAK'] * 1.0)
        - (counts['CONFLICT'] * 2.0)
        - (counts['NO_INTERACTION'] * 0.5)
    )

    # 1. Check for the rare but strong signals first
    if counts['SUPPORT_INTERACTION'] > 0:
        if counts['CONFLICT'] > 0:
            # The signal exists, but other papers actively dispute it
            verdict = "CONTROVERSIAL_SUPPORT" 
        elif counts['NO_INTERACTION'] >= 3: 
            # 1 paper found it, but several specifically failed to find it
            # (You can adjust this threshold > 3 depending on your dataset)
            verdict = "ISOLATED_FINDING" 
        else:
            # Clean, undisputed strong evidence
            verdict = "STRONG_SUPPORT"

    # 2. Check for weak support
    elif counts['SUPPORT_WEAK'] > 0:
        if counts['CONFLICT'] > 0:
            verdict = "MIXED_WEAK_EVIDENCE"
        else:
            verdict = "WEAK_SUPPORT"

    # 3. Check for active disconfirmation
    elif counts['CONFLICT'] > 0:
        verdict = "STRONG_CONFLICT"

    # 4. Check for explicit lack of interaction
    elif counts['NO_INTERACTION'] > 0:
        verdict = "LIKELY_NO_INTERACTION"

    # 5. Check for prognostic only or insufficient data
    elif counts['PROGNOSTIC_MAIN_EFFECT'] > 0:
        verdict = "PROGNOSTIC_ONLY"
    else:
        verdict = "INSUFFICIENT_EVIDENCE"

    return {
        "hypothesis_verdict": verdict,
        "evidence_score": score,
        "class_counts": counts,
    }

def map_verdict_to_original_label(verdict_result: Dict[str, Any]) -> str:
    """Map granular PubMed verdicts back into original classification labels."""
    verdict = verdict_result.get("hypothesis_verdict", "INSUFFICIENT_EVIDENCE")
    score = float(verdict_result.get("evidence_score", 0.0))

    if verdict == "STRONG_SUPPORT":
        return "SUPPORT_INTERACTION"

    if verdict in ("WEAK_SUPPORT", "ISOLATED_FINDING"):
        # An isolated strong finding in a sea of negatives dilutes confidence, 
        # so we downgrade it to WEAK rather than losing the positive signal entirely.
        return "SUPPORT_WEAK"

    if verdict == "CONTROVERSIAL_SUPPORT":
        # Papers are actively fighting (Support vs Conflict). 
        # We use the overall evidence score to break the tie.
        # If the positive/weak signals outweigh or tie the negative, retain as weak support.
        if score >= 0:
            return "SUPPORT_WEAK"
        return "CONFLICT"

    if verdict == "MIXED_WEAK_EVIDENCE":
        # Weak support fighting with active conflict.
        if score >= 0:
            return "SUPPORT_WEAK"
        return "CONFLICT"

    if verdict == "STRONG_CONFLICT":
        return "CONFLICT"

    if verdict == "LIKELY_NO_INTERACTION":
        return "NO_INTERACTION"

    if verdict == "PROGNOSTIC_ONLY":
        return "PROGNOSTIC_MAIN_EFFECT"

    # Fallback for INSUFFICIENT_EVIDENCE
    return "IRRELEVANT"


def parse_judge_labels(payload: Dict[str, Any], label_field: str) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    if label_field != "recommendation":
        return counts

    entries: List[Dict[str, Any]] = []
    if isinstance(payload.get("scored_hypotheses"), list):
        entries = payload["scored_hypotheses"]
    elif isinstance(payload.get("scored_features"), list):
        entries = payload["scored_features"]

    for entry in entries:
        label = entry.get("recommendation")
        if label is None:
            continue
        counts[_normalize_label(label)] += 1
    return counts


def percentage(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100.0


# When multiple files map to the same (method, dataset), a preferred file
# pattern takes priority.  Earlier entries in the list win.
_FILE_PREFERENCE = [
    "drlearner_revised",   # prefer DRLearner_revised over plain XLearner for ALEX
    "drlearner",
    "xlearner",
]


def _file_priority(path: Path) -> int:
    name = path.name.lower()
    for i, pat in enumerate(_FILE_PREFERENCE):
        if pat in name:
            return i
    return len(_FILE_PREFERENCE)


def parse_feature_level_labels(payload: Dict[str, Any], label_field: str) -> Dict[str, int]:
    """Classify each feature by aggregating labels across its mechanisms.

    Returns counts of features per label (one label per feature).
    """
    counts: Dict[str, int] = defaultdict(int)
    mechanism_results = payload.get("mechanism_results", [])
    if not isinstance(mechanism_results, list):
        return counts

    feature_to_label_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for mechanism in mechanism_results:
        if not isinstance(mechanism, dict):
            continue
        feature_name = str(mechanism.get("feature_name") or "UNKNOWN_FEATURE")
        abstracts = mechanism.get("abstracts_analyzed", [])

        if label_field == "classification":
            verdict_result = compute_hypothesis_verdict(abstracts)
            label = map_verdict_to_original_label(verdict_result)
        else:
            label = _dominant_classification(abstracts, label_field)

        feature_to_label_counts[feature_name][label] += 1

    for _, label_counts in feature_to_label_counts.items():
        if not label_counts:
            continue
        # dominant label per feature: highest count, then preferred label order
        dominant = sorted(
            label_counts.items(),
            key=lambda x: (-x[1], _LABEL_PRIORITY.get(x[0], 10_000), x[0]),
        )[0][0]
        counts[dominant] += 1

    return counts


def collect_records(
    root: Path, source: str, label_field: str, model_filter: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collect abstract-level, mechanism-level, and feature-level records.

    When multiple files resolve to the same (method, dataset, model) tuple, only
    the highest-priority file is kept (see _FILE_PREFERENCE).

    Args:
        model_filter: If set, only include files whose model folder matches this string.

    Returns:
        (abstract_records, mechanism_records, feature_records)
    """
    pattern = "**/*pubmed_validation*.json" if source == "pubmed" else "**/*judge*.json"

    # Deduplicate: best path per (method, dataset, model, seed)
    best: Dict[Tuple[str, str, str], Path] = {}
    for path in root.glob(pattern):
        if "cross_cohort" in path.name.lower():
            continue
        method_folder = infer_method_folder(path)
        # Only use drlearner for ALEX — skip xlearner files entirely
        if method_folder == "with_shap_xlearner":
            continue
        payload = load_json(path)
        if not payload:
            continue
        model = infer_model(path)
        seed = infer_seed(path)
        model_seed = model if seed == "no_seed" else f"{model}/{seed}"
        if model_filter and model != model_filter:
            continue
        key = (infer_method(path), infer_dataset(path, payload), model_seed)
        if key not in best or _file_priority(path) < _file_priority(best[key]):
            best[key] = path

    abstract_records: List[Dict[str, Any]] = []
    mechanism_records: List[Dict[str, Any]] = []
    feature_records: List[Dict[str, Any]] = []

    for (method, dataset, model), path in sorted(best.items()):
        payload = load_json(path)
        if not payload:
            continue

        if source == "pubmed":
            ab_counts   = parse_pubmed_labels(payload, label_field=label_field)
            mech_counts = parse_mechanism_level_labels(payload, label_field=label_field)
            feat_counts = parse_feature_level_labels(payload, label_field=label_field)
        else:
            ab_counts   = parse_judge_labels(payload, label_field=label_field)
            mech_counts = {}
            feat_counts = {}

        ab_total = sum(ab_counts.values())
        if ab_total > 0:
            abstract_records.append(
                {"file": str(path), "dataset": dataset, "method": method, "model": model,
                 "label_counts": dict(ab_counts), "total": ab_total}
            )

        mech_total = sum(mech_counts.values())
        if mech_total > 0:
            mechanism_records.append(
                {"file": str(path), "dataset": dataset, "method": method, "model": model,
                 "label_counts": dict(mech_counts), "total": mech_total}
            )

        feat_total = sum(feat_counts.values())
        # ResearchAgent exports one feature with many mechanisms by design.
        # For fair feature-level comparison with 5-feature methods, pad missing
        # features as IRRELEVANT so denominator is fixed at 5 per dataset.
        if method.lower() == "researchagent" and feat_total < 5:
            feat_counts["IRRELEVANT"] += (5 - feat_total)
            feat_total = 5
        if feat_total > 0:
            feature_records.append(
                {"file": str(path), "dataset": dataset, "method": method, "model": model,
                 "label_counts": dict(feat_counts), "total": feat_total}
            )

    return abstract_records, mechanism_records, feature_records


def aggregate(records: Iterable[Dict[str, Any]], group_by_model: bool = False) -> List[Dict[str, Any]]:
    # Key is (method, dataset) or (method, dataset, model) depending on flag
    grouped_counts: Dict[tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    grouped_totals: Dict[tuple, int] = defaultdict(int)
    grouped_model: Dict[tuple, str] = {}

    for r in records:
        model = r.get("model", "unknown")
        key = (r["method"], r["dataset"], model) if group_by_model else (r["method"], r["dataset"])
        grouped_totals[key] += r["total"]
        for label, count in r["label_counts"].items():
            grouped_counts[key][label] += count
        if group_by_model:
            grouped_model[key] = model

        all_key = (r["method"], "ALL_DATASETS", model) if group_by_model else (r["method"], "ALL_DATASETS")
        grouped_totals[all_key] += r["total"]
        for label, count in r["label_counts"].items():
            grouped_counts[all_key][label] += count
        if group_by_model:
            grouped_model[all_key] = model

    rows: List[Dict[str, Any]] = []
    for key, label_counts in grouped_counts.items():
        method, dataset = key[0], key[1]
        model = key[2] if group_by_model else grouped_model.get(key, "")
        total = grouped_totals[key]
        for label, count in sorted(label_counts.items(), key=lambda x: (-x[1], x[0])):
            rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "model": model,
                    "label": label,
                    "count": count,
                    "total_count": total,
                    "pct": percentage(count, total),
                }
            )

    rows.sort(
        key=lambda x: (
            x.get("model", ""),
            x["method"],
            x["dataset"] != "ALL_DATASETS",
            x["dataset"],
            -x["count"],
            x["label"],
        )
    )
    return rows


def split_model_seed(model_key: str) -> Tuple[str, str]:
    if "/seed_" in model_key:
        base_model, seed = model_key.rsplit("/", 1)
        return base_model, seed
    return model_key, "no_seed"


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def collapse_seed_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse per-seed rows into model-level mean/std summaries."""
    if not rows:
        return rows

    seeds_by_group: Dict[Tuple[str, str, str], set] = defaultdict(set)
    totals_by_seed: Dict[Tuple[str, str, str, str], int] = {}
    counts_by_seed_label: Dict[Tuple[str, str, str, str], Dict[str, int]] = defaultdict(dict)
    labels_by_group: Dict[Tuple[str, str, str], set] = defaultdict(set)

    for row in rows:
        model_key = str(row.get("model", ""))
        base_model, seed = split_model_seed(model_key)
        group = (base_model, row["method"], row["dataset"])
        seed_key = (base_model, row["method"], row["dataset"], seed)

        seeds_by_group[group].add(seed)
        totals_by_seed[seed_key] = int(row.get("total_count", 0))
        counts_by_seed_label[seed_key][row["label"]] = int(row.get("count", 0))
        labels_by_group[group].add(row["label"])

    collapsed: List[Dict[str, Any]] = []
    for (base_model, method, dataset), seed_set in seeds_by_group.items():
        seeds = sorted(seed_set)
        labels = sort_labels(labels_by_group[(base_model, method, dataset)])

        totals = [
            float(totals_by_seed.get((base_model, method, dataset, seed), 0))
            for seed in seeds
        ]
        total_mean, total_std = _mean_std(totals)

        for label in labels:
            counts: List[float] = []
            pcts: List[float] = []
            for seed in seeds:
                sk = (base_model, method, dataset, seed)
                count = float(counts_by_seed_label.get(sk, {}).get(label, 0))
                total = float(totals_by_seed.get(sk, 0))
                pct = (count / total * 100.0) if total > 0 else 0.0
                counts.append(count)
                pcts.append(pct)

            count_mean, count_std = _mean_std(counts)
            pct_mean, pct_std = _mean_std(pcts)

            collapsed.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "model": base_model,
                    "label": label,
                    "count": count_mean,
                    "count_std": count_std,
                    "total_count": total_mean,
                    "total_std": total_std,
                    "pct": pct_mean,
                    "pct_std": pct_std,
                    "n_seeds": len(seeds),
                }
            )

    collapsed.sort(
        key=lambda x: (
            x.get("model", ""),
            x["method"],
            x["dataset"] != "ALL_DATASETS",
            x["dataset"],
            -x.get("pct", 0.0),
            x["label"],
        )
    )
    return collapsed


def make_abbrevs(labels: List[str], col_w: int = 8) -> Dict[str, str]:
    """Create unique short column headers for potentially long label names."""
    abbrevs: Dict[str, str] = {}
    used: set = set()
    for lbl in labels:
        # Build candidate from capitalised initials of underscore-separated words
        parts = lbl.split("_")
        candidate = "".join(p[:2] for p in parts)[:col_w].upper()
        base = candidate
        suffix = 2
        while candidate in used:
            candidate = f"{base[:col_w - 1]}{suffix}"
            suffix += 1
        abbrevs[lbl] = candidate
        used.add(candidate)
    return abbrevs


def sort_labels(labels: Iterable[str]) -> List[str]:
    """Sort labels by PREFERRED_LABEL_ORDER, then alphabetically for unknowns."""
    preferred = [lbl for lbl in PREFERRED_LABEL_ORDER if lbl in labels]
    rest = sorted(lbl for lbl in labels if lbl not in PREFERRED_LABEL_ORDER)
    return preferred + rest


def pivot_rows(rows: List[Dict[str, Any]]) -> Tuple[
    Dict[str, Dict[str, Dict[str, Any]]],  # method -> dataset -> {label: {pct, count}, _total}
    List[str],                              # ordered label columns
]:
    """Pivot flat rows into method -> dataset -> {label -> {pct, count}, _total} mapping."""
    # Collect all unique labels
    all_label_set: set = set()
    for r in rows:
        all_label_set.add(r["label"])
    all_labels = sort_labels(all_label_set)

    # Build nested dict storing both pct and count per label
    pivot: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        cell = pivot[r["method"]][r["dataset"]]
        value = {"pct": r["pct"], "count": r["count"]}
        if "pct_std" in r:
            value["pct_std"] = r["pct_std"]
        if "count_std" in r:
            value["count_std"] = r["count_std"]
        cell[r["label"]] = value
        cell["_total"] = r["total_count"]
        if "total_std" in r:
            cell["_total_std"] = r["total_std"]
        if "n_seeds" in r:
            cell["_n_seeds"] = r["n_seeds"]

    return pivot, all_labels


def _print_single_table(
    pivot: Dict[str, Dict[str, Dict[str, Any]]],
    method: str,
    all_labels: List[str],
    abbrevs: Dict[str, str],
    title: str,
    ds_col_w: int = 16,
    total_col_w: int = 7,
    cell_w: int = 11,
    col_sep: str = " ",
) -> None:
    """Print one pivot table for a single method."""
    short_labels = [abbrevs[lbl] for lbl in all_labels]
    label_header = col_sep.join(f"{s:>{cell_w}}" for s in short_labels)
    header_line = f"{'Dataset':<{ds_col_w}} {'Total':>{total_col_w}}{col_sep}{label_header}"
    rule = "=" * len(header_line)
    thin = "-" * len(header_line)

    print(rule)
    print(f"  {title}")
    print(rule)
    print(header_line)
    print(thin)

    datasets = pivot.get(method, {})
    sorted_datasets = sorted(datasets.keys(), key=lambda d: (d == "ALL_DATASETS", d))
    for i, dataset in enumerate(sorted_datasets):
        cell = datasets[dataset]
        total = cell.get("_total", 0)
        total_std = cell.get("_total_std", None)
        n_seeds = cell.get("_n_seeds", None)
        pct_values = col_sep.join(
            _fmt_cell(cell.get(lbl, None), cell_w) for lbl in all_labels
        )
        if dataset == "ALL_DATASETS" and i > 0:
            print(thin)
        if total_std is not None:
            total_s = f"{total:.1f}±{total_std:.1f}"
            if n_seeds is not None:
                total_s += f"[{n_seeds}]"
            print(f"{dataset:<{ds_col_w}} {total_s:>{total_col_w}}{col_sep}{pct_values}")
        else:
            print(f"{dataset:<{ds_col_w}} {int(total):>{total_col_w}d}{col_sep}{pct_values}")
    print()


def print_tables(
    rows: List[Dict[str, Any]],
    hyp_rows: List[Dict[str, Any]],
    feat_rows: List[Dict[str, Any]],
    source: str,
    label_field: str,
    model_filter: Optional[str] = None,
) -> None:
    # Group rows by model for separate display when no filter is applied
    def _rows_for_model(all_rows, model):
        return [r for r in all_rows if r.get("model", "") == model]

    all_models = sorted(
        {r.get("model", "") for r in rows}
        | {r.get("model", "") for r in hyp_rows}
        | {r.get("model", "") for r in feat_rows}
    )

    for model in all_models:
        m_rows     = _rows_for_model(rows, model)
        m_hyp_rows = _rows_for_model(hyp_rows, model)
        m_feat_rows = _rows_for_model(feat_rows, model)

        if not m_rows and not m_hyp_rows and not m_feat_rows:
            continue

        ab_pivot,  ab_labels  = pivot_rows(m_rows)
        hyp_pivot, hyp_labels = pivot_rows(m_hyp_rows) if m_hyp_rows else ({}, [])
        feat_pivot, feat_labels = pivot_rows(m_feat_rows) if m_feat_rows else ({}, [])

        all_labels = sort_labels(set(ab_labels) | set(hyp_labels) | set(feat_labels))
        abbrevs = make_abbrevs(all_labels, col_w=8)

        collapsed_mode = any("pct_std" in r for r in (m_rows + m_hyp_rows + m_feat_rows))
        if collapsed_mode:
            ds_col_w, total_col_w, cell_w = 12, 12, 12
        else:
            ds_col_w, total_col_w, cell_w = 16, 7, 11
        col_sep = " "

        legend_width = max(16 + 7 + (cell_w + 1) * len(all_labels), 40)
        thin_legend = "-" * min(legend_width, 80)
        print()
        print(f"{'='*60}")
        print(f"  MODEL: {model}")
        print(f"{'='*60}")
        print("COLUMN LEGEND")
        print(thin_legend)
        for lbl in all_labels:
            print(f"  {abbrevs[lbl]:>8}  {lbl}")
        print()

        _METHOD_ORDER = ["SimpleCoT", "Baseline", "HypoGeniC", "ALEX"]
        all_methods_set = set(ab_pivot) | set(hyp_pivot)
        all_methods_set |= set(feat_pivot)
        all_methods = [m for m in _METHOD_ORDER if m in all_methods_set] + \
                      sorted(all_methods_set - set(_METHOD_ORDER))
        for method in all_methods:
            if method in ab_pivot:
                _print_single_table(
                    ab_pivot, method, ab_labels, abbrevs,
                    title=f"Method: {method} | ABSTRACT-level ({source.upper()} | {label_field})",
                    ds_col_w=ds_col_w,
                    total_col_w=total_col_w,
                    cell_w=cell_w, col_sep=col_sep,
                )
            if method in hyp_pivot:
                _print_single_table(
                    hyp_pivot, method, hyp_labels, abbrevs,
                    title=f"Method: {method} | MECHANISM-level (dominant label per mechanism)",
                    ds_col_w=ds_col_w,
                    total_col_w=total_col_w,
                    cell_w=cell_w, col_sep=col_sep,
                )
            if method in feat_pivot:
                _print_single_table(
                    feat_pivot, method, feat_labels, abbrevs,
                    title=f"Method: {method} | FEATURE-level (number of features by dominant classification)",
                    ds_col_w=ds_col_w,
                    total_col_w=total_col_w,
                    cell_w=cell_w, col_sep=col_sep,
                )


def _fmt_cell(data: Optional[Any], width: int) -> str:
    """Format a pivot cell right-aligned to width.

    - Per-seed rows: pct%(count)
    - Collapsed rows: pct±std%[avg_count]
    """
    if data is None:
        return f"{'—':>{width}}"
    pct = data["pct"]
    cnt = data["count"]
    if "pct_std" in data:
        s = f"{pct:.1f}±{data['pct_std']:.1f}|{cnt:.0f}"
    else:
        s = f"{pct:.1f}%({int(cnt)})"
    return f"{s:>{width}}"


def build_pivot_csv_rows(
    rows: List[Dict[str, Any]], level: str = "abstract"
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Build wide-format rows for CSV: level, model, method, dataset, total, <label_pct>, <label_count>..."""
    pivot, all_labels = pivot_rows(rows)
    pct_fields = [f"{lbl}_pct" for lbl in all_labels]
    pct_std_fields = [f"{lbl}_pct_std" for lbl in all_labels]
    cnt_fields = [f"{lbl}_count" for lbl in all_labels]
    cnt_std_fields = [f"{lbl}_count_std" for lbl in all_labels]
    fieldnames = ["level", "model", "method", "dataset", "total", "total_std", "n_seeds"] + pct_fields + pct_std_fields + cnt_fields + cnt_std_fields
    # Collect model per (method, dataset) from rows
    model_lookup = {(r["method"], r["dataset"]): r.get("model", "") for r in rows}
    _METHOD_ORDER = ["SimpleCoT", "Baseline", "HypoGeniC", "ALEX"]
    ordered_methods = [m for m in _METHOD_ORDER if m in pivot] + \
                      sorted(set(pivot) - set(_METHOD_ORDER))
    out_rows: List[Dict[str, Any]] = []
    for method in ordered_methods:
        datasets = pivot[method]
        sorted_datasets = sorted(datasets.keys(), key=lambda d: (d == "ALL_DATASETS", d))
        for dataset in sorted_datasets:
            cell = datasets[dataset]
            row: Dict[str, Any] = {
                "level": level,
                "model": model_lookup.get((method, dataset), ""),
                "method": method,
                "dataset": dataset,
                "total": cell.get("_total", 0),
                "total_std": cell.get("_total_std", ""),
                "n_seeds": cell.get("_n_seeds", ""),
            }
            for lbl in all_labels:
                data = cell.get(lbl)
                row[f"{lbl}_pct"]   = f"{data['pct']:.1f}%" if data else ""
                row[f"{lbl}_pct_std"] = f"{data['pct_std']:.1f}%" if data and "pct_std" in data else ""
                row[f"{lbl}_count"] = data["count"] if data else 0
                row[f"{lbl}_count_std"] = data["count_std"] if data and "count_std" in data else ""
            out_rows.append(row)
    return fieldnames, out_rows


def write_csv(
    rows: List[Dict[str, Any]],
    hyp_rows: List[Dict[str, Any]],
    feat_rows: List[Dict[str, Any]],
    out_csv: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ab_fieldnames,  ab_out  = build_pivot_csv_rows(rows,     level="abstract")
    mech_fieldnames, mech_out = build_pivot_csv_rows(hyp_rows, level="mechanism") if hyp_rows else (ab_fieldnames, [])
    feat_fieldnames, feat_out = build_pivot_csv_rows(feat_rows, level="feature") if feat_rows else (ab_fieldnames, [])
    # Union of field names (mechanism table may have extra/fewer labels)
    all_fields: List[str] = list(dict.fromkeys(
        ab_fieldnames
        + [f for f in mech_fieldnames if f not in ab_fieldnames]
        + [f for f in feat_fieldnames if f not in ab_fieldnames and f not in mech_fieldnames]
    ))
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ab_out)
        writer.writerows(mech_out)
        writer.writerows(feat_out)
    print(f"\nSaved CSV summary to: {out_csv}")


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    if not root.exists():
        raise FileNotFoundError(f"Root path not found: {root}")

    if args.source == "pubmed" and args.label_field == "recommendation":
        raise ValueError("For --source pubmed, use --label-field classification or stance")
    if args.source == "judge" and args.label_field != "recommendation":
        raise ValueError("For --source judge, use --label-field recommendation")

    abstract_records, mechanism_records, feature_records = collect_records(
        root=root, source=args.source, label_field=args.label_field,
        model_filter=args.model,
    )
    if not abstract_records:
        model_msg = f" for model '{args.model}'" if args.model else ""
        print(f"No {args.source} records found under: {root}{model_msg}")
        return

    # Group by model key first; if seeds are present and --keep-seeds is not set,
    # collapse model/seed runs into per-model mean±std summaries.
    group_by_model = True
    rows     = aggregate(abstract_records,  group_by_model=group_by_model)
    hyp_rows = aggregate(mechanism_records, group_by_model=group_by_model)
    feat_rows = aggregate(feature_records, group_by_model=group_by_model)

    if not args.keep_seeds:
        rows = collapse_seed_rows(rows)
        hyp_rows = collapse_seed_rows(hyp_rows)
        feat_rows = collapse_seed_rows(feat_rows)

    print_tables(rows, hyp_rows, feat_rows, source=args.source, label_field=args.label_field,
                 model_filter=args.model)

    if args.out_csv:
        write_csv(rows, hyp_rows, feat_rows, Path(args.out_csv))


if __name__ == "__main__":
    main()
