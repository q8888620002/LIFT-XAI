#!/usr/bin/env python3
"""hypogenic_baseline.py

Implement HypoGeniC (Hypothesis Generation in Context) algorithm for clinical trials.
This is an iterative hypothesis generation baseline that:
1. Generates initial hypotheses using LLM
2. Tests hypotheses on training samples
3. Refines hypotheses based on prediction errors
4. Generates new hypotheses from difficult samples

Based on: "Hypothesis Generation with Large Language Models"

Requires:
  pip install openai pydantic numpy pandas scikit-learn

Example:
  python hypogenic_baseline.py \
    --trial_name ist3 \
    --out_json docs/results/ist3/hypogenic_hypotheses.json \
    --num_init 20 \
    --top_k 10 \
    --alpha 0.5 \
    --update_batch_size 5 \
    --num_hypotheses_to_update 5

  Use custom number of samples:
  python hypogenic_baseline.py \
    --trial_name ist3 \
    --out_json docs/results/ist3/hypogenic_hypotheses.json \
    --max_samples 500

    With PubMed validation:
  python hypogenic_baseline.py \
    --trial_name ist3 \
    --out_json docs/results/ist3/hypogenic_hypotheses.json \
    --enable_pubmed_validation \
    --max_abstracts 50
"""

import argparse
import json
import math
import os
import sys
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Literal, Optional, Dict, Any
from dataclasses import dataclass, asdict

from openai import OpenAI
from pydantic import BaseModel, Field

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.dataset import Dataset


FIXED_TARGET_FEATURES = 5
FIXED_HYPOTHESES_PER_FEATURE = 3
FIXED_TOTAL_HYPOTHESES = FIXED_TARGET_FEATURES * FIXED_HYPOTHESES_PER_FEATURE


def load_local_env(env_path: Optional[str] = None) -> None:
    """Load KEY=VALUE pairs from a local .env file into os.environ.

    Existing environment variables are not overwritten.
    """
    candidate_paths = []
    if env_path:
        candidate_paths.append(Path(env_path))
    else:
        script_dir = Path(__file__).resolve().parent
        candidate_paths.extend([
            Path.cwd() / ".env",
            script_dir / ".env",
        ])

    env_file = next((path for path in candidate_paths if path.exists()), None)
    if not env_file:
        return

    try:
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError as exc:
        print(f"Warning: unable to read .env file: {exc}")


def resolve_seeded_output_path(output_path: str, seed: int) -> str:
    """Insert seed_<seed> folder before filename unless already present."""
    path_obj = Path(output_path)
    if any(part.startswith("seed_") for part in path_obj.parts):
        return str(path_obj)

    parent = path_obj.parent
    seeded_parent = parent / f"seed_{seed}"
    return str(seeded_parent / path_obj.name)


# -----------------------------
# Pydantic schemas (compatible with clinical_agent.py)
# -----------------------------

class SubgroupDefinition(BaseModel):
    feature: str = Field(
        ..., description="Feature name used to define a subgroup/effect modifier."
    )
    split_rule: str = Field(
        ...,
        description="Human-readable subgroup rule (e.g., 'age >= 75', 'lactate > 2').",
    )
    notes: Optional[str] = Field(
        None, description="Any nuance about encoding, bins, or clinical interpretation."
    )


class ValidationPlan(BaseModel):
    analyses: List[str] = Field(
        ...,
        description=(
            "Concrete follow-up analyses to validate the hypothesis, e.g., "
            "DR estimator within strata, interaction term, sensitivity checks."
        ),
    )
    negative_controls: Optional[List[str]] = Field(
        None, description="Optional negative control ideas / falsification tests."
    )
    robustness: Optional[List[str]] = Field(
        None,
        description="Optional robustness checks (baseline sensitivity, subgroup stability, etc.).",
    )


class ClinicalHypothesis(BaseModel):
    title: str = Field(..., description="Short, specific hypothesis title.")
    hypothesis: str = Field(
        ...,
        description="A testable statement about treatment-effect heterogeneity or subgroup benefit/harm.",
    )
    expected_direction: Literal[
        "higher_benefit", "lower_benefit", "higher_harm", "lower_harm", "ambiguous"
    ] = Field(
        ...,
        description="Direction of effect modification relative to the subgroup rule.",
    )
    subgroup: SubgroupDefinition
    rationale: List[str] = Field(
        ...,
        description="Bullet-like rationales grounded in features + plausible clinical mechanism.",
    )
    key_features: List[str] = Field(
        ...,
        description="Top features that support this hypothesis.",
    )
    confounders_and_bias_risks: List[str] = Field(
        ...,
        description="Potential confounding, bias, measurement error, or collider risks.",
    )
    validation: ValidationPlan
    caveats: Optional[List[str]] = Field(
        None,
        description="Any cautions about interpretation (attribution ≠ causality, encoding, baseline sensitivity).",
    )


# Internal representation for HypoGeniC algorithm
class SubgroupRule(BaseModel):
    feature: str
    operator: Literal[">=", "<=", ">", "<", "==", "!="]
    threshold: Optional[float] = None
    category: Optional[str] = None
    description: str


class TreatmentRecommendation(BaseModel):
    subgroup_rule: SubgroupRule
    recommendation: Literal["treat", "control", "unclear"]
    expected_benefit: Literal["high", "moderate", "low", "none", "harm"]
    rationale: str


class InternalHypothesis(BaseModel):
    """Internal hypothesis format used during HypoGeniC iteration."""
    hypothesis_id: str
    title: str
    hypothesis_statement: str
    treatment_recommendation: TreatmentRecommendation
    mechanism: str
    evidence_basis: List[str]
    testable_prediction: str


class HypothesisSet(BaseModel):
    dataset: str
    learner: str
    treatment: str
    outcome: str
    population: str
    source_explainer: str
    hypotheses: List[ClinicalHypothesis]


class InternalHypothesisSet(BaseModel):
    """Internal hypothesis set used during HypoGeniC iteration."""
    hypotheses: List[InternalHypothesis]
    generation_context: str
    iteration: int


class RefinementContext(BaseModel):
    difficult_samples: List[Dict[str, Any]]
    common_patterns: str
    what_went_wrong: str


class RefinedHypothesisSet(BaseModel):
    new_hypotheses: List[InternalHypothesis]
    refinement_rationale: str
    addresses_patterns: str


# Import judge-related schemas from clinical_agent.py
class HypothesisScore(BaseModel):
    title: str = Field(..., description="Hypothesis title being scored")
    scientific_rigor: int = Field(
        ...,
        ge=1,
        le=5,
        description="Scientific rigor (1-5): testability, operationalizability, falsifiability",
    )
    clinical_plausibility: int = Field(
        ...,
        ge=1,
        le=5,
        description="Clinical plausibility (1-5): biological mechanism, clinical coherence",
    )
    evidence_alignment: int = Field(
        ...,
        ge=1,
        le=5,
        description="Evidence alignment (1-5): how well feature evidence supports the hypothesis",
    )
    subgroup_clarity: int = Field(
        ...,
        ge=1,
        le=5,
        description="Subgroup clarity (1-5): how clear and actionable the subgroup rule is",
    )
    confounding_awareness: int = Field(
        ...,
        ge=1,
        le=5,
        description="Confounding awareness (1-5): thoroughness of bias/confounding discussion",
    )
    validation_plan_quality: int = Field(
        ...,
        ge=1,
        le=5,
        description="Validation plan quality (1-5): concreteness and appropriateness of proposed validation",
    )
    novelty: int = Field(
        ...,
        ge=1,
        le=5,
        description="Novelty (1-5): originality and potential for new insights beyond existing literature",
    )
    overall_score: int = Field(
        ..., ge=1, le=5, description="Overall score (1-5): holistic assessment"
    )
    strengths: List[str] = Field(..., description="Key strengths of this hypothesis")
    weaknesses: List[str] = Field(..., description="Key weaknesses or limitations")
    recommendation: Literal[
        "high_priority", "medium_priority", "low_priority", "reconsider"
    ] = Field(..., description="Recommendation for follow-up research")
    justification: str = Field(
        ..., description="Brief justification for the scores and recommendation"
    )


class JudgeOutput(BaseModel):
    summary: str = Field(
        ..., description="Overall assessment summary across all hypotheses"
    )
    scored_hypotheses: List[HypothesisScore]
    top_hypotheses: List[str] = Field(
        ..., description="Titles of top-ranked hypotheses (by overall_score)"
    )
    methodological_concerns: Optional[List[str]] = Field(
        None, description="Any cross-cutting methodological or interpretive concerns"
    )


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class HypothesisWithReward:
    """Tracks a hypothesis and its UCB-style statistics, mirroring the original
    SummaryInformation class in ChicagoHAI/hypothesis-generation."""

    hypothesis: InternalHypothesis
    acc: float = 0.0        # running accuracy (0-1), incremental average
    num_visits: int = 0     # number of times the hypothesis has been tested
    reward: float = 0.0     # UCB reward: acc + alpha * sqrt(log(n) / visits)

    def update_info_if_useful(self, current_sample: int, alpha: float) -> None:
        """Called when the hypothesis made a correct prediction."""
        self.acc = (self.acc * self.num_visits + 1) / (self.num_visits + 1)
        self.num_visits += 1
        self._update_reward(alpha, current_sample)

    def update_info_if_not_useful(self, current_sample: int, alpha: float) -> None:
        """Called when the hypothesis made a wrong prediction."""
        self.acc = (self.acc * self.num_visits) / (self.num_visits + 1)
        self.num_visits += 1
        self._update_reward(alpha, current_sample)

    def _update_reward(self, alpha: float, num_examples: int) -> None:
        """UCB reward: acc + alpha * sqrt(log(num_examples) / num_visits)."""
        if self.num_visits > 0 and num_examples > 1:
            self.reward = self.acc + alpha * math.sqrt(
                math.log(num_examples) / self.num_visits
            )

    def to_dict(self):
        return {
            "hypothesis": self.hypothesis.model_dump(),
            "acc": float(self.acc),
            "num_visits": int(self.num_visits),
            "reward": float(self.reward),
        }


# -----------------------------
# Helper functions
# -----------------------------

def get_trial_metadata(trial_name: str) -> dict:
    """Return treatment/outcome/population metadata for known clinical trials."""
    trial_map = {
        "ist3": {
            "treatment": "IV alteplase (recombinant tissue plasminogen activator)",
            "outcome": "Alive and independent (Oxford Handicap Score 0-2) at 6 months",
            "population": "Acute ischemic stroke patients within 6 hours of symptom onset",
        },
        "crash_2": {
            "treatment": "Tranexamic acid (TXA)",
            "outcome": "All-cause mortality at 28 days or in-hospital death",
            "population": "Trauma patients with significant bleeding or at risk of significant hemorrhage",
        },
        "sprint": {
            "treatment": "Intensive blood pressure control (systolic BP target <120 mmHg)",
            "outcome": "Composite of major cardiovascular events",
            "population": "Non-diabetic adults aged ≥50 with hypertension and increased cardiovascular risk",
        },
        "accord": {
            "treatment": "Intensive glucose control (HbA1c target <6.0%)",
            "outcome": "Major cardiovascular events (nonfatal MI, nonfatal stroke, cardiovascular death)",
            "population": "Adults with type 2 diabetes and high cardiovascular risk",
        },
    }

    trial_lower = trial_name.lower()
    if trial_lower not in trial_map:
        raise ValueError(
            f"Unknown trial: {trial_name}. Supported trials: {', '.join(trial_map.keys())}.\n"
            "Use --treatment, --outcome, --population arguments instead for custom trials."
        )
    return trial_map[trial_lower]


def load_trial_data_from_dataset(cohort_name: str, random_state: int = 42, max_samples: Optional[int] = None) -> tuple[Dataset, pd.DataFrame]:
    """Load trial data using Dataset class.

    Args:
        cohort_name: Name of the cohort to load
        random_state: Random state for reproducibility
        max_samples: Maximum number of samples to use (None = use all)

    Returns:
        (Dataset object, DataFrame with training samples)
    """
    dataset = Dataset(cohort_name=cohort_name, random_state=random_state, shuffle=False)

    # Reconstruct DataFrame from training data
    X_tr = dataset.x_train
    W_tr = dataset.w_train
    Y_tr = dataset.y_train

    # Get feature names (excluding treatment and outcome)
    feature_cols = [col for col in dataset.data.columns
                   if col not in [dataset.treatment, dataset.outcome]]

    # Create DataFrame
    df = pd.DataFrame(X_tr, columns=feature_cols)
    df[dataset.treatment] = W_tr
    df[dataset.outcome] = Y_tr

    # Limit to max_samples if specified
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_state)
        print(f"Sampled {max_samples} samples from {len(X_tr)} training samples for {cohort_name}")
    else:
        print(f"Loaded {len(df)} training samples for {cohort_name}")

    print(f"  Treatment column: {dataset.treatment}")
    print(f"  Outcome column: {dataset.outcome}")
    print(f"  Features: {len(feature_cols)}")

    return dataset, df


# -----------------------------
# HypoGeniC Algorithm Components
# -----------------------------

def generate_initial_hypotheses(
    study_context: dict,
    available_features: List[str],
    num_hypotheses: int,
    client: OpenAI,
    model_name: str = "gpt-4o-2024-08-06",
    target_features: Optional[int] = None,
    hypotheses_per_feature: Optional[int] = None,
) -> List[InternalHypothesis]:
    """Generate initial hypotheses using LLM (Algorithm 1, Line 2)."""

    system_prompt = (
        "You are a clinical research expert generating testable hypotheses about "
        "treatment effect heterogeneity. Generate hypotheses that:\n"
        "1. Specify clear subgroup rules based on patient characteristics\n"
        "2. Make concrete treatment recommendations for those subgroups\n"
        "3. Provide biological/clinical mechanisms\n"
        "4. Are testable with available data\n"
        "\n"
        "Focus on clinically meaningful subgroups that could inform treatment decisions."
    )

    instructions = [
        f"Generate {num_hypotheses} diverse hypotheses about treatment effect heterogeneity",
        "Each hypothesis should define a subgroup and predict treatment benefit/harm",
        "Base hypotheses on clinical literature and biological plausibility",
        "Make hypotheses testable with the available features",
    ]

    if target_features and hypotheses_per_feature:
        instructions += [
            f"IMPORTANT: Focus on exactly {target_features} features only - select the most clinically important ones",
            f"Generate exactly {hypotheses_per_feature} hypotheses per feature (different subgroup rules/thresholds or directions)",
            f"Total: {target_features} features × {hypotheses_per_feature} hypotheses = {num_hypotheses} hypotheses",
            "Do NOT use more than the specified number of unique features",
        ]
    else:
        instructions.append("Ensure diversity - cover different features and mechanisms")

    user_prompt = {
        "task": "Generate initial clinical hypotheses",
        "study_context": study_context,
        "available_features": available_features,
        "num_hypotheses": num_hypotheses,
        "instructions": instructions,
    }

    try:
        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, indent=2)},
            ],
            response_format=InternalHypothesisSet,
        )
        result = completion.choices[0].message.parsed
        # Reassign hypothesis_id with globally unique UUIDs to prevent
        # ID collisions between calls (LLM tends to reuse generic IDs like "hypothesis_1")
        for hyp in result.hypotheses:
            hyp.hypothesis_id = str(uuid.uuid4())
        return result.hypotheses
    except Exception as e:
        print(f"Error generating hypotheses: {e}")
        return []


def inference(
    hypothesis: InternalHypothesis,
    sample: pd.Series,
) -> str:
    """Make treatment recommendation based on hypothesis (Algorithm 1, Line 7)."""

    rule = hypothesis.treatment_recommendation.subgroup_rule
    feature_value = sample.get(rule.feature)

    if pd.isna(feature_value):
        return "unclear"

    # Evaluate subgroup rule
    in_subgroup = False
    try:
        if rule.operator == ">=":
            in_subgroup = float(feature_value) >= float(rule.threshold)
        elif rule.operator == "<=":
            in_subgroup = float(feature_value) <= float(rule.threshold)
        elif rule.operator == ">":
            in_subgroup = float(feature_value) > float(rule.threshold)
        elif rule.operator == "<":
            in_subgroup = float(feature_value) < float(rule.threshold)
        elif rule.operator == "==":
            if rule.category is not None:
                in_subgroup = str(feature_value) == str(rule.category)
            else:
                in_subgroup = float(feature_value) == float(rule.threshold)
        elif rule.operator == "!=":
            if rule.category is not None:
                in_subgroup = str(feature_value) != str(rule.category)
            else:
                in_subgroup = float(feature_value) != float(rule.threshold)
    except (ValueError, TypeError):
        return "unclear"

    # Return recommendation based on whether sample is in subgroup
    if in_subgroup:
        return hypothesis.treatment_recommendation.recommendation
    else:
        # Opposite recommendation for out-of-subgroup
        rec = hypothesis.treatment_recommendation.recommendation
        if rec == "treat":
            return "control"
        elif rec == "control":
            return "treat"
        else:
            return "unclear"


def is_correct_prediction(
    hypothesis: InternalHypothesis,
    sample: pd.Series,
    actual_treatment: int,
    actual_outcome: int,
) -> bool:
    """Check if the hypothesis correctly predicts treatment benefit for this sample.

    Mirrors the original's pred != label comparison: the hypothesis is considered
    correct if its treatment recommendation is consistent with the observed
    treatment-outcome relationship.
    """
    recommendation = inference(hypothesis, sample)

    if recommendation == "unclear":
        return False

    predicted_treat = 1 if recommendation == "treat" else 0

    # Correct if: recommended treatment matches actual treatment AND outcome was good,
    # or recommended opposite treatment AND outcome was bad (treatment wasn't helpful).
    if predicted_treat == actual_treatment:
        return actual_outcome == 1
    else:
        return actual_outcome == 0


def is_wrong_prediction(
    hypothesis: InternalHypothesis,
    sample: pd.Series,
    actual_treatment: int,
    actual_outcome: int,
) -> bool:
    """Check if hypothesis made wrong prediction (Algorithm 1, Line 9)."""
    return not is_correct_prediction(hypothesis, sample, actual_treatment, actual_outcome)


def select_balanced_hypotheses(
    hypothesis_bank: dict,
    target_features: int,
    hypotheses_per_feature: int,
) -> tuple[List[str], Dict[str, List["HypothesisWithReward"]]]:
    """Select top N features and their best hypotheses from the bank.

    Returns:
        (top_feature_names, feature_groups_dict) for further processing
    """
    # Group hypotheses by feature
    feature_groups: Dict[str, List[HypothesisWithReward]] = {}
    for h in hypothesis_bank.values():
        feature = h.hypothesis.treatment_recommendation.subgroup_rule.feature
        if feature not in feature_groups:
            feature_groups[feature] = []
        feature_groups[feature].append(h)

    # Sort hypotheses within each feature by reward (descending)
    for feature in feature_groups:
        feature_groups[feature].sort(key=lambda x: x.reward, reverse=True)

    # Rank features by their best hypothesis reward, then total reward
    feature_scores = [
        (feature, max(h.reward for h in hyps), sum(h.reward for h in hyps))
        for feature, hyps in feature_groups.items()
    ]
    feature_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    top_features = [f[0] for f in feature_scores[:target_features]]

    print(f"\nSelecting balanced hypotheses:")
    print(f"  Top {target_features} features by reward: {top_features}")
    for feature in top_features:
        hyps = feature_groups[feature]
        print(f"  - {feature}: {len(hyps)} hypothesis(es) in bank (rewards: {[h.reward for h in hyps[:hypotheses_per_feature]]})")

    return top_features, feature_groups


def generate_hypotheses_for_feature(
    feature: str,
    count_needed: int,
    existing_hyps: List[InternalHypothesis],
    study_context: dict,
    available_features: List[str],
    client: OpenAI,
    model_name: str = "gpt-4o-2024-08-06",
) -> List[InternalHypothesis]:
    """Generate additional hypotheses for a specific feature to fill gaps."""

    existing_rules = [h.treatment_recommendation.subgroup_rule.description for h in existing_hyps]

    system_prompt = (
        "You are a clinical research expert generating additional hypotheses for a specific feature. "
        "Generate hypotheses that are DISTINCT from the existing ones "
        "(use different thresholds, operators, or directions of effect)."
    )

    user_prompt = {
        "task": f"Generate {count_needed} additional hypothesis(es) for feature '{feature}'",
        "study_context": study_context,
        "target_feature": feature,
        "count_needed": count_needed,
        "existing_hypotheses_for_this_feature": existing_rules,
        "available_features": available_features,
        "instructions": [
            f"Generate exactly {count_needed} new hypothesis(es) using ONLY the feature '{feature}'",
            "Each hypothesis MUST set subgroup_rule.feature = '" + feature + "'",
            "Use a different threshold or operator than the existing hypotheses listed above",
            "Cover complementary perspectives: e.g., high vs low values, moderate range, or opposite direction",
            "Make each hypothesis clinically meaningful and internally consistent",
        ],
    }

    try:
        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, indent=2)},
            ],
            response_format=InternalHypothesisSet,
        )
        result = completion.choices[0].message.parsed
        # Prefer hypotheses that correctly use the target feature
        on_target = [h for h in result.hypotheses
                     if h.treatment_recommendation.subgroup_rule.feature == feature]
        # Fall back to any returned if LLM deviated
        candidates = on_target if on_target else result.hypotheses
        # Force-correct the feature field if needed
        for h in candidates:
            h.treatment_recommendation.subgroup_rule.feature = feature
        return candidates[:count_needed]
    except Exception as e:
        print(f"  Error generating hypotheses for feature '{feature}': {e}")
        return []


def build_fallback_hypothesis_for_feature(
    feature: str,
    ordinal: int,
    study_context: dict,
) -> InternalHypothesis:
    """Create a minimal deterministic hypothesis for a feature as a last-resort filler."""
    treatment = study_context.get("treatment", "treatment")
    outcome = study_context.get("outcome", "outcome")
    title = f"Fallback hypothesis {ordinal} for {feature}"
    statement = (
        f"{feature} may modify the effect of {treatment} on {outcome}; "
        "this placeholder should be replaced by model-generated content when available."
    )

    return InternalHypothesis(
        hypothesis_id=str(uuid.uuid4()),
        title=title,
        hypothesis_statement=statement,
        treatment_recommendation=TreatmentRecommendation(
            subgroup_rule=SubgroupRule(
                feature=feature,
                operator=">=",
                threshold=0.0,
                category=None,
                description=f"{feature} >= 0.0 (fallback placeholder rule)",
            ),
            recommendation="unclear",
            expected_benefit="low",
            rationale="Fallback placeholder due to insufficient feature-specific generation.",
        ),
        mechanism=(
            f"Potential pathway for {feature} requires validation; "
            "this mechanism is a conservative placeholder."
        ),
        evidence_basis=[
            "Placeholder hypothesis generated to satisfy fixed feature/hypothesis cardinality",
        ],
        testable_prediction=(
            f"Test treatment-by-{feature} interaction for {treatment} on {outcome} "
            "in pre-specified subgroup analysis."
        ),
    )


def ensure_balanced_hypotheses(
    hypothesis_bank: dict,
    target_features: int,
    hypotheses_per_feature: int,
    study_context: dict,
    available_features: List[str],
    client: OpenAI,
    model_name: str = "gpt-4o-2024-08-06",
) -> List[HypothesisWithReward]:
    """Build a strictly balanced list of target_features * hypotheses_per_feature hypotheses.

    Fills gaps by calling the LLM to generate additional hypotheses for any
    feature that has fewer than hypotheses_per_feature entries in the bank.
    """
    target_total = target_features * hypotheses_per_feature

    ranked_features, feature_groups = select_balanced_hypotheses(
        hypothesis_bank, target_features, hypotheses_per_feature
    )

    # If hypothesis bank has fewer unique features than requested, backfill from
    # available dataset features so we can still enforce fixed coverage.
    top_features = list(ranked_features)
    if len(top_features) < target_features:
        missing = target_features - len(top_features)
        backfill = [f for f in available_features if f not in set(top_features)][:missing]
        top_features.extend(backfill)
        print(
            f"  Backfilling {len(backfill)} feature(s) to reach target coverage: {backfill}"
        )

    selected: List[HypothesisWithReward] = []
    for feature in top_features:
        existing = feature_groups.get(feature, [])
        have = existing[:hypotheses_per_feature]
        selected.extend(have)

        gap = hypotheses_per_feature - len(have)
        if gap > 0:
            print(f"  Gap for '{feature}': have {len(have)}, need {gap} more — generating...")
            generated: List[InternalHypothesis] = []
            attempts = 0
            max_attempts = 4
            while len(generated) < gap and attempts < max_attempts:
                attempts += 1
                need = gap - len(generated)
                new_internal = generate_hypotheses_for_feature(
                    feature=feature,
                    count_needed=need,
                    existing_hyps=[h.hypothesis for h in have] + generated,
                    study_context=study_context,
                    available_features=available_features,
                    client=client,
                    model_name=model_name,
                )
                if not new_internal:
                    print(f"    Attempt {attempts}/{max_attempts}: no hypotheses returned for '{feature}'")
                    continue

                # Deduplicate by subgroup rule signature to avoid near-duplicates.
                existing_signatures = {
                    (
                        h.treatment_recommendation.subgroup_rule.operator,
                        h.treatment_recommendation.subgroup_rule.threshold,
                        h.treatment_recommendation.subgroup_rule.category,
                        h.treatment_recommendation.subgroup_rule.description,
                    )
                    for h in ([x.hypothesis for x in have] + generated)
                }
                for h in new_internal:
                    rule = h.treatment_recommendation.subgroup_rule
                    signature = (rule.operator, rule.threshold, rule.category, rule.description)
                    if signature in existing_signatures:
                        continue
                    generated.append(h)
                    existing_signatures.add(signature)
                    if len(generated) >= gap:
                        break

            if len(generated) < gap:
                print(
                    f"    Could only generate {len(generated)}/{gap} missing hypotheses for '{feature}' "
                    f"after {max_attempts} attempts"
                )
                before_fill = len(generated)
                for idx in range(len(generated) + 1, gap + 1):
                    generated.append(
                        build_fallback_hypothesis_for_feature(
                            feature=feature,
                            ordinal=idx,
                            study_context=study_context,
                        )
                    )
                print(
                    f"    Added {len(generated) - before_fill} fallback hypothesis(es) for '{feature}'"
                )

            for h in generated:
                selected.append(HypothesisWithReward(hypothesis=h))

    print(f"\nFinal balanced selection: {len(selected)}/{target_total} hypotheses "
          f"across {len(top_features)} features")
    if len(selected) != target_total:
        print(f"  WARNING: Expected {target_total} hypotheses but got {len(selected)}. "
              "Some gap-filling calls may have returned fewer than requested.")
    else:
        print(f"  OK: exactly {target_total} hypotheses across {len(top_features)} features")
    return selected


def generate_new_hypotheses_from_difficult_samples(
    difficult_samples: List[pd.Series],
    study_context: dict,
    available_features: List[str],
    num_hypotheses: int,
    client: OpenAI,
    model_name: str = "gpt-4o-2024-08-06",
) -> List[InternalHypothesis]:
    """Generate new hypotheses from difficult samples (Algorithm 1, Line 13)."""

    # Analyze difficult samples to find patterns
    if not difficult_samples:
        return []

    df_difficult = pd.DataFrame(difficult_samples)

    # Summarize patterns in difficult samples
    patterns = []
    for col in available_features:
        if col in df_difficult.columns:
            if df_difficult[col].dtype in ['int64', 'float64']:
                patterns.append({
                    "feature": col,
                    "mean": float(df_difficult[col].mean()) if not df_difficult[col].isna().all() else None,
                    "median": float(df_difficult[col].median()) if not df_difficult[col].isna().all() else None,
                    "min": float(df_difficult[col].min()) if not df_difficult[col].isna().all() else None,
                    "max": float(df_difficult[col].max()) if not df_difficult[col].isna().all() else None,
                })

    system_prompt = (
        "You are a clinical research expert refining hypotheses based on difficult cases. "
        "Analyze the patterns in samples where current hypotheses failed and generate "
        "NEW hypotheses that better explain treatment effect heterogeneity in these cases.\n"
        "\n"
        "Focus on:\n"
        "1. Features that distinguish difficult samples\n"
        "2. Alternative subgroup definitions\n"
        "3. Novel mechanisms not covered by previous hypotheses\n"
        "4. Interactions between features"
    )

    user_prompt = {
        "task": "Generate refined hypotheses from difficult samples",
        "study_context": study_context,
        "difficult_sample_patterns": patterns,
        "num_difficult_samples": len(difficult_samples),
        "available_features": available_features,
        "num_new_hypotheses": num_hypotheses,
        "instructions": [
            f"Analyze patterns in {len(difficult_samples)} samples where hypotheses failed",
            f"Generate {num_hypotheses} NEW hypotheses targeting these difficult cases",
            "Focus on features and thresholds that distinguish these samples",
            "Propose mechanisms that explain the complex patterns",
        ],
    }

    try:
        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, indent=2)},
            ],
            response_format=RefinedHypothesisSet,
        )
        result = completion.choices[0].message.parsed
        return result.new_hypotheses
    except Exception as e:
        print(f"Error generating refined hypotheses: {e}")
        return []


def hypogenic_algorithm(
    data: pd.DataFrame,
    study_context: dict,
    num_init: int,
    top_k: int,
    treatment_col: str,
    outcome_col: str,
    client: OpenAI,
    model_name: str = "gpt-4o-2024-08-06",
    alpha: float = 0.5,
    num_wrong_scale: float = 0.8,
    update_batch_size: int = 5,
    num_hypotheses_to_update: int = 5,
    update_hypotheses_per_batch: int = 5,
    target_features: Optional[int] = None,
    hypotheses_per_feature: Optional[int] = None,
) -> List[HypothesisWithReward]:
    """
    Implement HypoGeniC algorithm aligned with the original ChicagoHAI implementation.

    Key design decisions matching the original:
    - UCB reward: acc + alpha * sqrt(log(current_sample) / num_visits)
    - Adaptive wrong threshold: grows as len(top_k) * (i / n) * num_wrong_scale
    - Set-based wrong example accumulation: trigger generation when
      |wrong_example_ids| == update_batch_size * num_hypotheses_to_update
    - DefaultReplace policy: merge new + existing bank, keep top-k by reward

    Args:
        data: Training samples
        study_context: Study metadata
        num_init: Number of initial hypotheses
        top_k: Max size of hypothesis bank (max_num_hypotheses in original)
        treatment_col: Treatment assignment column
        outcome_col: Outcome column
        client: OpenAI client
        model_name: Model name
        alpha: UCB exploration constant (default 0.5, matches original)
        num_wrong_scale: Scale for adaptive wrong threshold (default 0.8)
        update_batch_size: Examples per update batch (default 5)
        num_hypotheses_to_update: Number of generation rounds when batch fills (default 5)
        update_hypotheses_per_batch: Hypotheses generated per round (default 5)
        target_features: If set, constrain final output to this many unique features
        hypotheses_per_feature: If set, select this many hypotheses per feature

    Returns:
        List of top hypotheses with UCB rewards
    """

    available_features = [col for col in data.columns
                         if col not in [treatment_col, outcome_col]]
    num_train_examples = len(data)

    # Initialize hypothesis bank
    print(f"Generating {num_init} initial hypotheses...")
    if target_features and hypotheses_per_feature:
        print(f"  Constrained to {target_features} features × {hypotheses_per_feature} hypotheses each")
    initial_hypotheses = generate_initial_hypotheses(
        study_context=study_context,
        available_features=available_features,
        num_hypotheses=num_init,
        client=client,
        model_name=model_name,
        target_features=target_features,
        hypotheses_per_feature=hypotheses_per_feature,
    )

    # hypothesis bank: {hypothesis_id -> HypothesisWithReward}
    H: Dict[str, HypothesisWithReward] = {
        h.hypothesis_id: HypothesisWithReward(hypothesis=h)
        for h in initial_hypotheses
    }

    # Set of training example indices where top-k hypotheses disagreed (wrong examples)
    wrong_example_ids: set = set()

    # Iterate over training examples (mirrors original DefaultUpdate.update loop)
    for idx, (_, sample) in enumerate(data.iterrows()):
        current_sample = idx + 1  # 1-based like the original

        if idx % 100 == 0:
            print(f"Processing sample {idx}/{num_train_examples}...")

        actual_treatment = int(sample[treatment_col])
        actual_outcome = int(sample[outcome_col])

        # Get top-k hypotheses sorted by UCB reward
        top_k_keys = sorted(H.keys(), key=lambda x: H[x].reward, reverse=True)[:top_k]

        # Adaptive wrong threshold: grows over time, matching original num_wrong_to_add_bank
        num_wrong_to_add_bank = (
            len(top_k_keys) * idx / num_train_examples
        ) * num_wrong_scale if num_wrong_scale > 0 else 0

        num_wrong_hypotheses = 0

        # For each top hypothesis: compute prediction, update UCB stats
        for h_id in top_k_keys:
            h = H[h_id]
            correct = is_correct_prediction(
                h.hypothesis, sample, actual_treatment, actual_outcome
            )
            if correct:
                h.update_info_if_useful(current_sample, alpha)
            else:
                h.update_info_if_not_useful(current_sample, alpha)
                num_wrong_hypotheses += 1

        # If enough hypotheses were wrong (adaptive threshold), record this example
        if num_wrong_hypotheses >= num_wrong_to_add_bank or len(top_k_keys) == 0:
            wrong_example_ids.add(idx)

        # When wrong-example set is full, generate new hypotheses (DefaultUpdate style)
        if len(wrong_example_ids) == update_batch_size * num_hypotheses_to_update:
            print(
                f"\nGenerating hypotheses from {len(wrong_example_ids)} difficult examples "
                f"(sample {current_sample}/{num_train_examples})..."
            )

            difficult_samples = [data.iloc[i] for i in wrong_example_ids]
            new_hyp_bank: Dict[str, HypothesisWithReward] = {}

            # Multiple rounds of generation, mirroring num_hypotheses_to_update iterations
            for _ in range(num_hypotheses_to_update):
                new_hypotheses = generate_new_hypotheses_from_difficult_samples(
                    difficult_samples=difficult_samples,
                    study_context=study_context,
                    available_features=available_features,
                    num_hypotheses=update_hypotheses_per_batch,
                    client=client,
                    model_name=model_name,
                )
                for h in new_hypotheses:
                    new_hyp_bank[h.hypothesis_id] = HypothesisWithReward(hypothesis=h)

            # Reset wrong-example set
            wrong_example_ids = set()

            # DefaultReplace: merge new hypotheses with existing bank, keep top-k by reward
            merged = {**new_hyp_bank, **H}
            H = dict(
                sorted(merged.items(), key=lambda x: x[1].reward, reverse=True)[:top_k]
            )

            print(f"  Updated hypothesis bank size: {len(H)}")

    # Return top-k hypotheses, optionally balanced across features
    print(f"\nCompleted HypoGeniC algorithm. Final bank size: {len(H)}")

    if target_features and hypotheses_per_feature:
        final_hypotheses = ensure_balanced_hypotheses(
            H,
            target_features,
            hypotheses_per_feature,
            study_context=study_context,
            available_features=available_features,
            client=client,
            model_name=model_name,
        )
        expected_total = target_features * hypotheses_per_feature
        if len(final_hypotheses) != expected_total:
            raise RuntimeError(
                f"Expected exactly {expected_total} hypotheses "
                f"({target_features} features × {hypotheses_per_feature}), "
                f"but got {len(final_hypotheses)}."
            )
    else:
        final_hypotheses = sorted(H.values(), key=lambda x: x.reward, reverse=True)[:top_k]

    # Gap-fill only for unconstrained mode.
    if not (target_features and hypotheses_per_feature):
        while len(final_hypotheses) < top_k:
            gap = top_k - len(final_hypotheses)
            print(f"  Gap-fill: have {len(final_hypotheses)}, need {gap} more hypotheses...")
            extra = generate_initial_hypotheses(
                study_context=study_context,
                available_features=available_features,
                num_hypotheses=gap * 2,  # over-request slightly to account for duplicates
                client=client,
                model_name=model_name,
            )
            existing_ids = {h.hypothesis.hypothesis_id for h in final_hypotheses}
            for h in extra:
                if h.hypothesis_id not in existing_ids and len(final_hypotheses) < top_k:
                    final_hypotheses.append(HypothesisWithReward(hypothesis=h))
                    existing_ids.add(h.hypothesis_id)
            # Safety break: if LLM keeps returning nothing, stop
            if not extra:
                print(f"  Warning: could not fill gap, returning {len(final_hypotheses)} hypotheses")
                break

    print(f"  Returning {len(final_hypotheses)} hypotheses")
    return final_hypotheses


def convert_to_clinical_hypothesis(
    internal_hyp: InternalHypothesis,
    study_context: dict,
    client: OpenAI,
    model_name: str = "gpt-4o-2024-08-06",
) -> Optional[ClinicalHypothesis]:
    """Convert InternalHypothesis to ClinicalHypothesis format for judging/validation.

    Uses LLM to expand the internal hypothesis into the full clinical format.
    """

    system_prompt = (
        "You are a clinical research expert converting a hypothesis into a structured "
        "clinical hypothesis format. Expand the provided hypothesis with:\n"
        "1. Detailed rationale points (3-5 bullet points)\n"
        "2. Confounders and bias risks\n"
        "3. Validation plan with specific analyses\n"
        "4. Caveats about interpretation\n"
        "\n"
        "Base all additions on clinical evidence and maintain scientific rigor."
    )

    rule = internal_hyp.treatment_recommendation.subgroup_rule

    user_prompt = {
        "task": "Convert to full clinical hypothesis format",
        "study_context": study_context,
        "hypothesis": {
            "title": internal_hyp.title,
            "statement": internal_hyp.hypothesis_statement,
            "subgroup_rule": rule.description,
            "feature": rule.feature,
            "operator": rule.operator,
            "threshold": rule.threshold,
            "category": rule.category,
            "mechanism": internal_hyp.mechanism,
            "evidence_basis": internal_hyp.evidence_basis,
            "recommendation": internal_hyp.treatment_recommendation.recommendation,
            "expected_benefit": internal_hyp.treatment_recommendation.expected_benefit,
        },
        "instructions": [
            "Expand the hypothesis into full clinical format",
            "Provide 3-5 detailed rationale points based on mechanism and evidence",
            "Identify key confounders and bias risks",
            "Propose concrete validation analyses",
            "Add appropriate caveats about causality and interpretation",
        ],
    }

    try:
        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, indent=2)},
            ],
            response_format=ClinicalHypothesis,
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f"Error converting hypothesis '{internal_hyp.title}': {e}")
        return None


def score_hypotheses(
    hypotheses: List[ClinicalHypothesis],
    study_context: dict,
    client: OpenAI,
    model_name: str = "gpt-4o-2024-08-06",
) -> Optional[JudgeOutput]:
    """Score hypotheses using an independent judge (similar to clinical_agent.py)."""

    judge_system = (
        "You are an independent scientific judge evaluating clinical hypotheses.\n"
        "Evaluate each hypothesis on multiple dimensions using a 1-5 scale.\n"
        "Be objective, fair, and constructive.\n"
        "\n"
        "SCORING CRITERIA:\n"
        "\n"
        "1. Scientific Rigor (1-5):\n"
        "   - Testability and falsifiability of the hypothesis\n"
        "   - Operationalizability of the subgroup rule (can it be applied in practice?)\n"
        "   - Clarity and specificity of hypothesis statement\n"
        "   - Avoidance of vague or overly broad claims\n"
        "   Score 5: Highly testable, clear operational definition, specific and falsifiable\n"
        "   Score 3: Moderately testable, some ambiguity in subgroup definition\n"
        "   Score 1: Untestable, vague, or unfalsifiable\n"
        "\n"
        "2. Clinical Plausibility (1-5):\n"
        "   - Biological/physiological coherence of proposed mechanisms\n"
        "   - Alignment with established pathophysiology and pharmacology\n"
        "   - Clinical relevance and actionability of findings\n"
        "   - Specificity to the treatment and outcome context\n"
        "   Score 5: Strong biological basis, well-established pathways, clinically actionable\n"
        "   Score 3: Reasonable but speculative, some supporting evidence\n"
        "   Score 1: Implausible, contradicts known biology, not clinically meaningful\n"
        "\n"
        "3. Evidence Alignment (1-5):\n"
        "   - Support from clinical literature and prior trials\n"
        "   - Consistency with meta-analyses and systematic reviews\n"
        "   - Strength of mechanistic evidence from basic science\n"
        "   - Whether proposed subgroups have been validated in other studies\n"
        "   - Appropriate recognition when evidence is sparse or speculative\n"
        "   Score 5: Strong literature support, validated in multiple trials\n"
        "   Score 3: Plausible with some supporting evidence, but limited validation\n"
        "   Score 1: Contradicts existing evidence or lacks any supporting literature\n"
        "\n"
        "4. Subgroup Clarity (1-5):\n"
        "   - Clarity and precision of subgroup definition\n"
        "   - Feasibility of identifying subgroup members in clinical practice\n"
        "   - Clinical meaningfulness and actionability of the subgroup\n"
        "   - Avoidance of arbitrary or data-driven cutpoints without justification\n"
        "   Score 5: Clear, clinically meaningful, easily identifiable subgroup\n"
        "   Score 3: Reasonable but somewhat vague or difficult to operationalize\n"
        "   Score 1: Unclear, arbitrary, or clinically meaningless stratification\n"
        "\n"
        "5. Confounding Awareness (1-5):\n"
        "   - Thoroughness of bias and confounding discussion\n"
        "   - Recognition of alternative explanations and competing hypotheses\n"
        "   - Acknowledgment of measurement error and data limitations\n"
        "   - Appropriate epistemic humility (avoiding overclaiming)\n"
        "   - Recognition that association ≠ causation\n"
        "   Score 5: Comprehensive discussion of limitations, honest about uncertainty\n"
        "   Score 3: Some caveats mentioned but incomplete\n"
        "   Score 1: Overclaiming, ignoring limitations, false certainty\n"
        "\n"
        "6. Validation Plan Quality (1-5):\n"
        "   - Concreteness and specificity of proposed analyses\n"
        "   - Appropriateness of statistical methods for the hypothesis\n"
        "   - Feasibility with available data and resources\n"
        "   - Inclusion of sensitivity analyses and robustness checks\n"
        "   - Consideration of negative controls and falsification tests\n"
        "   Score 5: Detailed, appropriate, feasible validation plan\n"
        "   Score 3: General validation ideas but lacking specificity\n"
        "   Score 1: Vague, inappropriate, or infeasible validation plan\n"
        "\n"
        "7. Novelty (1-5):\n"
        "   - Originality of the hypothesis beyond existing clinical literature\n"
        "   - Potential to generate new insights or challenge existing paradigms\n"
        "   - Whether the hypothesis identifies underexplored treatment effect modifiers\n"
        "   - Balance between novelty and plausibility (novel but not implausible)\n"
        "   Score 5: Highly original, identifies underexplored mechanisms, potential paradigm shift\n"
        "   Score 3: Moderately novel, extends existing knowledge in meaningful ways\n"
        "   Score 1: Reiterates well-established findings, no new insights\n"
        "\n"
        "Provide honest, rigorous critique in strengths/weaknesses. Use the full 1-5 range."
    )

    judge_prompt = {
        "study_context": study_context,
        "hypotheses_to_score": [h.model_dump() for h in hypotheses],
        "instructions": [
            "Score each hypothesis on all dimensions",
            "Provide strengths and weaknesses",
            "Give overall assessment and recommendation",
            "Rank hypotheses by overall_score",
        ],
    }

    try:
        result = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": judge_system},
                {"role": "user", "content": json.dumps(judge_prompt, indent=2)},
            ],
            response_format=JudgeOutput,
        )
        return result.choices[0].message.parsed
    except Exception as e:
        print(f"Error scoring hypotheses: {e}")
        return None


def convert_clinical_to_feature_format(clinical_hypotheses: List[ClinicalHypothesis], study_context: dict, max_features: int = 5, max_mechanisms_per_feature: int = 3) -> dict:
    """Convert ClinicalHypothesis format to FeatureHypothesis format for PubMed validation.

    Groups hypotheses by feature and limits to top N features with up to M mechanisms each
    for fair comparison with other baseline methods.

    Args:
        clinical_hypotheses: List of ClinicalHypothesis objects
        study_context: Study context dict
        max_features: Maximum number of features to include (default: 5)
        max_mechanisms_per_feature: Maximum mechanisms per feature (default: 3)

    Returns:
        Dictionary in FeatureHypothesis format compatible with pubmed_mechanism_validator
    """
    # Group hypotheses by feature
    feature_groups = {}
    for hyp in clinical_hypotheses:
        feature_name = hyp.subgroup.feature
        if feature_name not in feature_groups:
            feature_groups[feature_name] = []
        feature_groups[feature_name].append(hyp)

    # Limit to top N features (by number of hypotheses, then alphabetically for ties)
    sorted_features = sorted(
        feature_groups.items(),
        key=lambda x: (-len(x[1]), x[0])  # More hypotheses first, then alphabetical
    )[:max_features]

    print(f"\nFiltering hypotheses for PubMed validation:")
    print(f"  Total unique features: {len(feature_groups)}")
    print(f"  Limiting to top {max_features} features")
    print(f"  Max mechanisms per feature: {max_mechanisms_per_feature}")

    # Create feature hypotheses - one entry per feature with multiple mechanisms
    feature_hypotheses = []
    for feature_name, hypotheses in sorted_features:
        # Limit to max_mechanisms_per_feature hypotheses for this feature
        selected_hyps = hypotheses[:max_mechanisms_per_feature]

        # Create mechanisms from selected hypotheses
        mechanisms = []
        for hyp in selected_hyps:
            rationale_text = " ".join(hyp.rationale) if hyp.rationale else hyp.hypothesis
            mechanism = {
                "mechanism_type": "biological",  # Default type
                "description": rationale_text,
                "evidence_level": "moderate",  # Default level
            }
            mechanisms.append(mechanism)

        # Use the first hypothesis for main metadata
        primary_hyp = selected_hyps[0]

        # Create feature hypothesis entry
        feature_hyp = {
            "feature_name": feature_name,
            "importance_rank": len(feature_hypotheses) + 1,
            "shap_value": 0.0,  # Not available from HypoGeniC
            "effect_direction": primary_hyp.expected_direction,
            "clinical_interpretation": primary_hyp.hypothesis,
            "why_important": f"Treatment effect modifier identified by HypoGeniC algorithm ({len(selected_hyps)} mechanisms)",
            "mechanisms": mechanisms,  # Multiple mechanisms per feature
            "subgroup_implications": primary_hyp.subgroup.split_rule,
            "validation_suggestions": primary_hyp.validation.analyses,
            "caveats": primary_hyp.caveats if primary_hyp.caveats else [],
        }
        feature_hypotheses.append(feature_hyp)

        print(f"  - {feature_name}: {len(selected_hyps)} mechanism(s)")

    total_mechanisms = sum(len(f['mechanisms']) for f in feature_hypotheses)
    print(f"  Total mechanisms for validation: {total_mechanisms}")

    return {
        "dataset": study_context.get("dataset", "unknown"),
        "model": "HypoGeniC",
        "summary": f"Top {len(feature_hypotheses)} features with {total_mechanisms} mechanisms generated by HypoGeniC algorithm for {study_context.get('dataset', 'unknown')}",
        "feature_hypotheses": feature_hypotheses,
        "cross_feature_patterns": None,
    }


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HypoGeniC: Iterative hypothesis generation for clinical trials"
    )

    # Data arguments
    parser.add_argument(
        "--trial_name",
        required=True,
        help="Trial name (ist3, crash_2, sprint, accord) to load data from Dataset class",
    )
    parser.add_argument(
        "--out_json",
        required=True,
        help="Output path for generated hypotheses JSON",
    )

    # Optional arguments
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for data splitting",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Maximum number of training samples to use (default: 200)",
    )

    # HypoGeniC algorithm parameters
    parser.add_argument(
        "--num_init",
        type=int,
        default=20,
        help="Number of initial hypotheses to generate",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=FIXED_TOTAL_HYPOTHESES,
        help=f"Number of top hypotheses to maintain (fixed at {FIXED_TOTAL_HYPOTHESES})",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="UCB exploration constant (default: 0.5, matches original HypoGeniC)",
    )
    parser.add_argument(
        "--num_wrong_scale",
        type=float,
        default=0.8,
        help="Scale for adaptive wrong-prediction threshold (default: 0.8)",
    )
    parser.add_argument(
        "--update_batch_size",
        type=int,
        default=5,
        help="Number of wrong examples to accumulate per update batch (default: 5)",
    )
    parser.add_argument(
        "--num_hypotheses_to_update",
        type=int,
        default=5,
        help="Generation rounds per update batch (default: 5)",
    )
    parser.add_argument(
        "--update_hypotheses_per_batch",
        type=int,
        default=5,
        help="Hypotheses generated per generation round (default: 5)",
    )
    parser.add_argument(
        "--target_features",
        type=int,
        default=FIXED_TARGET_FEATURES,
        help=f"Number of unique features to focus on (fixed at {FIXED_TARGET_FEATURES})",
    )
    parser.add_argument(
        "--hypotheses_per_feature",
        type=int,
        default=FIXED_HYPOTHESES_PER_FEATURE,
        help=f"Number of hypotheses per feature (fixed at {FIXED_HYPOTHESES_PER_FEATURE})",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="OpenAI model name",
    )

    # Evaluation arguments
    parser.add_argument(
        "--enable_pubmed_validation",
        action="store_true",
        help="Enable PubMed literature validation of mechanisms",
    )
    parser.add_argument(
        "--max_abstracts",
        type=int,
        default=30,
        help="Maximum number of PubMed abstracts to retrieve per hypothesis (default: 30)",
    )
    parser.add_argument(
        "--api_provider",
        type=str,
        default="openai",
        choices=["openai", "openrouter"],
        help="API provider to use (default: openai)",
    )
    parser.add_argument(
        "--api_base_url",
        type=str,
        default=None,
        help="Optional API base URL override",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed index used to place outputs under seed_<seed> subfolder (default: 0)",
    )

    args = parser.parse_args()

    # Enforce fixed balanced output: exactly 5 features × 3 hypotheses = 15 hypotheses.
    if args.top_k != FIXED_TOTAL_HYPOTHESES:
        print(
            f"Overriding --top_k={args.top_k} to fixed value {FIXED_TOTAL_HYPOTHESES} "
            f"({FIXED_TARGET_FEATURES}x{FIXED_HYPOTHESES_PER_FEATURE})."
        )
    if args.target_features != FIXED_TARGET_FEATURES:
        print(
            f"Overriding --target_features={args.target_features} to fixed value {FIXED_TARGET_FEATURES}."
        )
    if args.hypotheses_per_feature != FIXED_HYPOTHESES_PER_FEATURE:
        print(
            f"Overriding --hypotheses_per_feature={args.hypotheses_per_feature} to fixed value {FIXED_HYPOTHESES_PER_FEATURE}."
        )

    args.top_k = FIXED_TOTAL_HYPOTHESES
    args.target_features = FIXED_TARGET_FEATURES
    args.hypotheses_per_feature = FIXED_HYPOTHESES_PER_FEATURE

    resolved_out_json = resolve_seeded_output_path(args.out_json, args.seed)
    if resolved_out_json != args.out_json:
        print(f"Resolved output path with seed folder: {resolved_out_json}")
    args.out_json = resolved_out_json

    # Load .env (if present), then resolve API key
    load_local_env()

    # Get API key and build client
    if args.api_provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. Set it in your environment or .env file."
            )
        base_url = args.api_base_url or "https://openrouter.ai/api/v1"
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY in your environment "
                "or in a local .env file."
            )
        kwargs = {"api_key": api_key}
        if args.api_base_url:
            kwargs["base_url"] = args.api_base_url
        client = OpenAI(**kwargs)

    # Get trial metadata and load data using Dataset class
    trial_meta = get_trial_metadata(args.trial_name)
    treatment = trial_meta["treatment"]
    outcome = trial_meta["outcome"]
    population = trial_meta["population"]

    # Load data from Dataset class
    print(f"Loading data for {args.trial_name}...")
    dataset, data = load_trial_data_from_dataset(
        args.trial_name,
        args.random_state,
        max_samples=args.max_samples
    )

    # Get treatment and outcome column names from dataset
    treatment_col = dataset.treatment
    outcome_col = dataset.outcome

    # Prepare study context
    study_context = {
        "dataset": args.trial_name,
        "treatment": treatment,
        "outcome": outcome,
        "population": population,
        "sample_size": len(data),
        "method": "HypoGeniC",
    }

    # Run HypoGeniC algorithm
    print("=" * 80)
    print("Running HypoGeniC Algorithm")
    print("=" * 80)

    # Derive num_init from target_features * hypotheses_per_feature if constrained
    num_init = args.num_init
    if args.target_features and args.hypotheses_per_feature:
        constrained_total = args.target_features * args.hypotheses_per_feature
        # Generate more initially to ensure enough diversity, then select balanced
        num_init = max(args.num_init, constrained_total * 2)
        print(f"Constrained mode: {args.target_features} features × {args.hypotheses_per_feature} hypotheses = {constrained_total} total")
        print(f"Generating {num_init} initial hypotheses to ensure coverage")

    final_hypotheses = hypogenic_algorithm(
        data=data,
        study_context=study_context,
        num_init=num_init,
        top_k=args.top_k,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        client=client,
        model_name=args.model,
        alpha=args.alpha,
        num_wrong_scale=args.num_wrong_scale,
        update_batch_size=args.update_batch_size,
        num_hypotheses_to_update=args.num_hypotheses_to_update,
        update_hypotheses_per_batch=args.update_hypotheses_per_batch,
        target_features=args.target_features,
        hypotheses_per_feature=args.hypotheses_per_feature,
    )

    print("\n" + "=" * 80)
    print("HypoGeniC Algorithm Completed - Starting Post-Processing")
    print("=" * 80)

    # Convert hypotheses to clinical format if PubMed validation is enabled
    clinical_hypotheses = None
    if args.enable_pubmed_validation:
        print("\nConverting hypotheses to clinical format...")
        clinical_hypotheses = []
        for h_with_reward in final_hypotheses:
            clinical_h = convert_to_clinical_hypothesis(
                h_with_reward.hypothesis,
                study_context,
                client,
                args.model,
            )
            if clinical_h:
                clinical_hypotheses.append(clinical_h)
        print(f"Converted {len(clinical_hypotheses)}/{len(final_hypotheses)} hypotheses")

    # Prepare output
    output = {
        "method": "HypoGeniC",
        "study_context": study_context,
        "algorithm_parameters": {
            "num_init": args.num_init,
            "top_k": args.top_k,
            "alpha": args.alpha,
            "num_wrong_scale": args.num_wrong_scale,
            "update_batch_size": args.update_batch_size,
            "num_hypotheses_to_update": args.num_hypotheses_to_update,
            "update_hypotheses_per_batch": args.update_hypotheses_per_batch,
        },
        "hypotheses": [h.to_dict() for h in final_hypotheses],
        "summary": {
            "total_hypotheses": len(final_hypotheses),
            "avg_reward": float(np.mean([h.reward for h in final_hypotheses])),
            "avg_accuracy": float(np.mean([h.acc for h in final_hypotheses])),
        },
    }

    # Save internal format output
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote HypoGeniC hypotheses (internal format) to: {args.out_json}")

    # Save clinical format for compatibility with judge/PubMed validator
    if clinical_hypotheses:
        clinical_output = {
            "dataset": study_context.get("dataset", args.trial_name or "unknown"),
            "learner": "HypoGeniC",
            "treatment": treatment,
            "outcome": outcome,
            "population": population,
            "source_explainer": "HypoGeniC_iterative",
            "hypotheses": [h.model_dump() for h in clinical_hypotheses],
        }

        clinical_path = os.path.splitext(args.out_json)[0] + "_clinical_format.json"
        with open(clinical_path, "w") as f:
            json.dump(clinical_output, f, indent=2)
        print(f"Wrote clinical format hypotheses to: {clinical_path}")

    # ---------- PUBMED VALIDATION (optional) ----------
    if args.enable_pubmed_validation and clinical_hypotheses:
        print("\n" + "=" * 80)
        print("POST-ITERATION VALIDATION: PubMed Literature Evidence")
        print("=" * 80)

        # Convert to FeatureHypothesis format for PubMed validator
        # Keep the same fixed structure used during generation: 5 features × 3 mechanisms.
        pubmed_compatible_format = convert_clinical_to_feature_format(
            clinical_hypotheses,
            study_context,
            max_features=FIXED_TARGET_FEATURES,
            max_mechanisms_per_feature=FIXED_HYPOTHESES_PER_FEATURE,
        )

        # Count mechanisms for user info
        total_mechanisms = sum(len(f['mechanisms']) for f in pubmed_compatible_format['feature_hypotheses'])
        num_features = len(pubmed_compatible_format['feature_hypotheses'])

        print(f"\nPreparing PubMed validation...")
        print(f"  Max abstracts per mechanism: {args.max_abstracts}")
        print(f"\nNote: Each mechanism requires PubMed search + abstract retrieval + LLM analysis")
        print(f"      This may take 5-10 minutes depending on API rate limits\n")

        pubmed_input_path = os.path.splitext(args.out_json)[0] + "_pubmed_input.json"
        with open(pubmed_input_path, "w") as f:
            json.dump(pubmed_compatible_format, f, indent=2)
        print(f"Created PubMed-compatible input: {pubmed_input_path}")
        print(f"Run pubmed_mechanism_validator.py separately with this file.")

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"HypoGeniC Algorithm Summary")
    print(f"{'=' * 80}")
    print(f"\nTop {len(final_hypotheses)} Hypotheses:")
    for i, h in enumerate(final_hypotheses, 1):
        print(f"  {i}. {h.hypothesis.title}")
        print(f"     Reward: {h.reward:.4f}, Acc: {h.acc:.2%}, Visits: {h.num_visits}")

if __name__ == "__main__":
    main()
