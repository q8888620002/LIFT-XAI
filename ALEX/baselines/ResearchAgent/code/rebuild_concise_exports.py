"""Rebuild ResearchAgent hypotheses.json exports as concise one-sentence
mechanisms grouped by per-sentence feature attribution.

The raw export dumps full problem/method/experiment rationales (4-17k chars)
as mechanism descriptions, which is unreadable on the rating site and mixes
several candidate features under one label. This script re-distills each
cohort's saved agent context (ideas.jsonl) into short hypotheses and
attributes each to the trial feature its own text names.

Usage:
  python rebuild_concise_exports.py --ideas-dir <dir with ideas_<cohort>.jsonl> \
      [--out-root <ALEX/results root>] [--num-hypotheses 15]
"""
import argparse
import json
import os
import re

from main import (
    FEATURE_CUES_BY_TRIAL,
    DEFAULT_FEATURE_BY_TRIAL,
    _expand_hypotheses_with_llm,
    infer_effect_direction,
)
from models.openai import OpenAIClient

COHORT_TO_IDEAS = {
    'crash_2': 'ideas_crash2.jsonl',
    'ist3': 'ideas_ist3.jsonl',
    'sprint': 'ideas_sprint.jsonl',
    'accord': 'ideas_accord.jsonl',
    'accord_glycemia': 'ideas_accord_glycemia.jsonl',
}


DASHES = re.compile(r'[-‐-―−]')


def attribute_feature(text: str, cohort: str) -> str | None:
    cues = FEATURE_CUES_BY_TRIAL.get(cohort, {})
    normalized = DASHES.sub(' ', text.lower())
    best, best_score = None, 0
    for feature_key, feature_cues in cues.items():
        score = sum(
            len(re.findall(r'\b' + re.escape(c.lower().replace('-', ' ')) + r'\b', normalized))
            for c in feature_cues
        )
        if score > best_score:
            best, best_score = feature_key, score
    return best


def rebuild(cohort: str, ideas_path: str, out_path: str, client: OpenAIClient, n: int) -> None:
    ideas = [json.loads(line) for line in open(ideas_path)]
    context = ideas[-1]

    sentences: list[str] = []
    seen = set()
    attempts = 0
    while len(sentences) < n and attempts < 4:
        batch = _expand_hypotheses_with_llm(context=context, api_client=client, target_count=n)
        for s in batch:
            key = s.strip().lower()
            if key and key not in seen:
                seen.add(key)
                sentences.append(s.strip())
        attempts += 1
    sentences = sentences[:n]
    if len(sentences) < n:
        print(f'[{cohort}] WARNING: only {len(sentences)}/{n} unique hypotheses generated')

    default_feature = DEFAULT_FEATURE_BY_TRIAL.get(cohort, 'feature')
    by_feature: dict[str, list[str]] = {}
    for s in sentences:
        feature = attribute_feature(s, cohort) or default_feature
        by_feature.setdefault(feature, []).append(s)

    feature_hypotheses = []
    for rank, (feature, texts) in enumerate(
        sorted(by_feature.items(), key=lambda kv: -len(kv[1])), start=1
    ):
        feature_hypotheses.append({
            'feature_name': feature,
            'importance_rank': rank,
            'mechanisms': [
                {
                    'mechanism_type': 'hypothesis_mechanism',
                    'description': text,
                    'evidence_level': 'hypothesis_generating',
                    'effect_direction': infer_effect_direction(text),
                }
                for text in texts
            ],
        })

    existing = json.load(open(out_path)) if os.path.exists(out_path) else {}
    existing.update({
        'dataset': cohort,
        'model': client.model,
        'summary': (
            f'{len(sentences)} concise trial-paper-only hypotheses distilled from the '
            f'ResearchAgent run for {cohort}, grouped by per-hypothesis feature attribution.'
        ),
        'feature_hypotheses': feature_hypotheses,
    })
    json.dump(existing, open(out_path, 'w'), indent=2)
    dist = {f['feature_name']: len(f['mechanisms']) for f in feature_hypotheses}
    print(f'[{cohort}] features: {dist}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ideas-dir', required=True)
    parser.add_argument('--out-root', default=os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results'))
    parser.add_argument('--model-name', default='gpt-5-mini')
    parser.add_argument('--num-hypotheses', type=int, default=15)
    args = parser.parse_args()

    client = OpenAIClient(model=args.model_name)
    for cohort, ideas_file in COHORT_TO_IDEAS.items():
        ideas_path = os.path.join(args.ideas_dir, ideas_file)
        if not os.path.exists(ideas_path):
            print(f'[{cohort}] skipped — no {ideas_path}')
            continue
        out_path = os.path.join(
            args.out_root, cohort, args.model_name, 'researchagent_fixed', 'seed_0', 'hypotheses.json'
        )
        rebuild(cohort, ideas_path, out_path, client, args.num_hypotheses)


if __name__ == '__main__':
    main()
