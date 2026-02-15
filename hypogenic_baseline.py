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
    --w_max 100 \
    --max_iterations 5

  Use custom number of samples:
  python hypogenic_baseline.py \
    --trial_name ist3 \
    --out_json docs/results/ist3/hypogenic_hypotheses.json \
    --max_samples 500

  With judge and PubMed validation:
  python hypogenic_baseline.py \
    --trial_name ist3 \
    --out_json docs/results/ist3/hypogenic_hypotheses.json \
    --enable_judge \
    --enable_pubmed_validation \
    --max_abstracts 50
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from typing import List, Literal, Optional, Dict, Any
from dataclasses import dataclass, asdict

from openai import OpenAI
from pydantic import BaseModel, Field

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.dataset import Dataset


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
    hypothesis: InternalHypothesis
    reward: float
    correct_predictions: int
    total_predictions: int

    def to_dict(self):
        return {
            "hypothesis": self.hypothesis.model_dump(),
            "reward": float(self.reward),
            "correct_predictions": int(self.correct_predictions),
            "total_predictions": int(self.total_predictions),
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

    user_prompt = {
        "task": "Generate initial clinical hypotheses",
        "study_context": study_context,
        "available_features": available_features,
        "num_hypotheses": num_hypotheses,
        "instructions": [
            f"Generate {num_hypotheses} diverse hypotheses about treatment effect heterogeneity",
            "Each hypothesis should define a subgroup and predict treatment benefit/harm",
            "Base hypotheses on clinical literature and biological plausibility",
            "Make hypotheses testable with the available features",
            "Ensure diversity - cover different features and mechanisms",
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


def update_reward(
    hypothesis: InternalHypothesis,
    sample: pd.Series,
    actual_treatment: int,
    actual_outcome: int,
) -> tuple[bool, float]:
    """Update reward based on prediction accuracy (Algorithm 1, Line 8).

    Returns:
        (is_correct, reward_delta)
    """

    recommendation = inference(hypothesis, sample)

    if recommendation == "unclear":
        return False, 0.0

    # Check if recommendation aligns with beneficial outcome
    # This is simplified - in reality, would need causal effect estimation
    predicted_treat = 1 if recommendation == "treat" else 0

    # Reward correct treatment recommendations
    # Simplification: assume treated patients with good outcomes validate "treat" recommendation
    is_correct = False
    reward_delta = 0.0

    if predicted_treat == actual_treatment:
        # Predicted treatment matches actual treatment
        if actual_outcome == 1:  # Good outcome
            is_correct = True
            reward_delta = 1.0
        else:  # Bad outcome
            is_correct = False
            reward_delta = -0.5
    else:
        # Predicted different treatment than actual
        if actual_outcome == 1:  # Good outcome despite our recommendation
            is_correct = False
            reward_delta = -0.5
        else:  # Bad outcome, maybe our recommendation would have helped
            is_correct = True
            reward_delta = 0.5

    return is_correct, reward_delta


def is_wrong_prediction(
    hypothesis: InternalHypothesis,
    sample: pd.Series,
    actual_treatment: int,
    actual_outcome: int,
) -> bool:
    """Check if hypothesis made wrong prediction (Algorithm 1, Line 9)."""
    is_correct, _ = update_reward(hypothesis, sample, actual_treatment, actual_outcome)
    return not is_correct


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
    w_max: int,
    max_iterations: int,
    treatment_col: str,
    outcome_col: str,
    client: OpenAI,
    model_name: str = "gpt-4o-2024-08-06",
) -> List[HypothesisWithReward]:
    """
    Implement HypoGeniC algorithm (Algorithm 1).

    Args:
        data: Training samples
        study_context: Study metadata
        num_init: Number of initial hypotheses
        top_k: Number of top hypotheses to keep
        w_max: Maximum size of difficult sample set before regenerating
        max_iterations: Maximum refinement iterations
        treatment_col: Treatment assignment column
        outcome_col: Outcome column
        client: OpenAI client
        model_name: Model name

    Returns:
        List of top hypotheses with rewards
    """

    available_features = [col for col in data.columns
                         if col not in [treatment_col, outcome_col]]

    # Line 2: Initialize hypothesis bank
    print(f"Generating {num_init} initial hypotheses...")
    initial_hypotheses = generate_initial_hypotheses(
        study_context=study_context,
        available_features=available_features,
        num_hypotheses=num_init,
        client=client,
        model_name=model_name,
    )

    # Track hypotheses with rewards
    H = {h.hypothesis_id: HypothesisWithReward(h, 0.0, 0, 0)
         for h in initial_hypotheses}

    # Line 3: Initialize difficult sample set
    W = []

    iteration = 0

    # Line 4: For each sample
    for idx, (_, sample) in enumerate(data.iterrows()):
        if idx % 100 == 0:
            print(f"Processing sample {idx}/{len(data)}...")

        actual_treatment = int(sample[treatment_col])
        actual_outcome = int(sample[outcome_col])

        # Line 5: Get top-k hypotheses by reward
        H_top = sorted(H.values(), key=lambda x: x.reward, reverse=True)[:top_k]

        wrong_count = 0

        # Line 6-8: For each top hypothesis, make inference and update reward
        for h_with_reward in H_top:
            is_correct, reward_delta = update_reward(
                h_with_reward.hypothesis,
                sample,
                actual_treatment,
                actual_outcome,
            )

            h_with_reward.reward += reward_delta
            h_with_reward.total_predictions += 1
            if is_correct:
                h_with_reward.correct_predictions += 1
            else:
                wrong_count += 1

        # Line 9-11: If too many wrong predictions, add to difficult set
        if wrong_count >= len(H_top) * 0.6:  # w_hyp threshold (60% wrong)
            W.append(sample)

        # Line 12-15: If difficult set is full, generate new hypotheses
        if len(W) >= w_max and iteration < max_iterations:
            iteration += 1
            print(f"\nIteration {iteration}: Generating new hypotheses from {len(W)} difficult samples...")

            new_hypotheses = generate_new_hypotheses_from_difficult_samples(
                difficult_samples=W,
                study_context=study_context,
                available_features=available_features,
                num_hypotheses=num_init // 2,  # Generate fewer new hypotheses
                client=client,
                model_name=model_name,
            )

            # Add new hypotheses to bank
            for h in new_hypotheses:
                if h.hypothesis_id not in H:
                    H[h.hypothesis_id] = HypothesisWithReward(h, 0.0, 0, 0)

            # Reset difficult set
            W = []

            # Keep top-k from combined set
            all_hypotheses = sorted(H.values(), key=lambda x: x.reward, reverse=True)
            H = {h.hypothesis.hypothesis_id: h for h in all_hypotheses[:top_k * 2]}

            print(f"  Current hypothesis bank size: {len(H)}")

    # Line 16: Return top-k hypotheses
    final_hypotheses = sorted(H.values(), key=lambda x: x.reward, reverse=True)[:top_k]

    print(f"\nCompleted HypoGeniC algorithm:")
    print(f"  Total iterations: {iteration}")
    print(f"  Final hypothesis bank size: {len(H)}")
    print(f"  Returning top {len(final_hypotheses)} hypotheses")

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


def convert_clinical_to_feature_format(clinical_hypotheses: List[ClinicalHypothesis], study_context: dict) -> dict:
    """Convert ClinicalHypothesis format to FeatureHypothesis format for PubMed validation.

    Note: To reduce PubMed validation runtime, this function:
    - Uses only the primary (first) hypothesis per feature
    - Limits to 3 mechanisms per feature (first 3 rationale points)
    This prevents excessive API calls while still validating core mechanisms.

    Args:
        clinical_hypotheses: List of ClinicalHypothesis objects
        study_context: Study context dict

    Returns:
        Dictionary in FeatureHypothesis format compatible with pubmed_mechanism_validator
    """
    # Group hypotheses by feature
    feature_groups = {}
    for hyp in clinical_hypotheses:
        feature = hyp.subgroup.feature
        if feature not in feature_groups:
            feature_groups[feature] = []
        feature_groups[feature].append(hyp)

    # Create feature hypotheses
    feature_hypotheses = []
    for feature_name, hyps in feature_groups.items():
        # Only use first hypothesis per feature to limit PubMed queries
        # Each mechanism requires multiple PubMed searches and LLM analyses
        primary_hyp = hyps[0] if hyps else None
        if not primary_hyp:
            continue

        # Convert rationale points to mechanisms (limit to first 3 to reduce runtime)
        mechanisms = []
        for j, rationale_point in enumerate(primary_hyp.rationale[:3]):  # Limit to 3 mechanisms
            mechanism = {
                "mechanism_type": "biological",  # Default type
                "description": rationale_point,
                "evidence_level": "moderate",  # Default level
            }
            mechanisms.append(mechanism)

        # Create feature hypothesis entry
        feature_hyp = {
            "feature_name": feature_name,
            "importance_rank": len(feature_hypotheses) + 1,
            "shap_value": 0.0,  # Not available from HypoGeniC
            "effect_direction": primary_hyp.expected_direction,
            "clinical_interpretation": primary_hyp.hypothesis,
            "why_important": f"Treatment effect modifier identified by HypoGeniC algorithm",
            "mechanisms": mechanisms,
            "subgroup_implications": primary_hyp.subgroup.split_rule,
            "validation_suggestions": primary_hyp.validation.analyses,
            "caveats": primary_hyp.caveats if primary_hyp.caveats else [],
        }
        feature_hypotheses.append(feature_hyp)

    return {
        "dataset": study_context.get("dataset", "unknown"),
        "model": "HypoGeniC",
        "summary": f"Hypotheses generated by HypoGeniC algorithm for {study_context.get('dataset', 'unknown')}",
        "feature_hypotheses": feature_hypotheses,
        "cross_feature_patterns": None,
    }


def run_pubmed_validation(
    hypotheses_json: str,
    output_path: str,
    trial_name: Optional[str] = None,
    max_abstracts: int = 30,
) -> bool:
    """Run PubMed validation using pubmed_mechanism_validator.py.

    Args:
        hypotheses_json: Path to hypotheses JSON file
        output_path: Path for validation output
        trial_name: Optional trial name for better context
        max_abstracts: Maximum number of PubMed abstracts to retrieve

    Returns:
        True if validation succeeded, False otherwise
    """
    try:
        import subprocess

        cmd = [
            "python",
            "pubmed_mechanism_validator.py",
            "--input", hypotheses_json,
            "--output", output_path,
        ]

        if trial_name:
            cmd.extend(["--cohort", trial_name])

        cmd.extend(["--max-abstracts", str(max_abstracts)])

        print(f"Running PubMed validation: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))

        # Print stdout for debugging
        if result.stdout:
            print("PubMed validator output:")
            print(result.stdout)

        if result.returncode == 0:
            print(f"PubMed validation completed successfully")
            return True
        else:
            print(f"PubMed validation failed with return code: {result.returncode}")
            if result.stderr:
                print(f"Error output:\n{result.stderr}")
            return False
    except Exception as e:
        print(f"Error running PubMed validation: {e}")
        return False


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
        default=10,
        help="Number of top hypotheses to maintain (k in algorithm)",
    )
    parser.add_argument(
        "--w_max",
        type=int,
        default=100,
        help="Maximum difficult samples before regenerating hypotheses",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=5,
        help="Maximum number of refinement iterations",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        default="gpt-4o-2024-08-06",
        help="OpenAI model name",
    )

    # Evaluation arguments
    parser.add_argument(
        "--enable_judge",
        action="store_true",
        help="Enable independent judging/scoring of hypotheses",
    )
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

    args = parser.parse_args()

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            from src.constants import openai_api_key
            api_key = openai_api_key
        except ImportError:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or create src/constants.py with openai_api_key defined."
            )
    client = OpenAI(api_key=api_key)

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

    final_hypotheses = hypogenic_algorithm(
        data=data,
        study_context=study_context,
        num_init=args.num_init,
        top_k=args.top_k,
        w_max=args.w_max,
        max_iterations=args.max_iterations,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        client=client,
        model_name=args.model,
    )

    print("\n" + "=" * 80)
    print("HypoGeniC Algorithm Completed - Starting Post-Processing")
    print("=" * 80)

    # Convert hypotheses to clinical format if judging or PubMed validation enabled
    clinical_hypotheses = None
    if args.enable_judge or args.enable_pubmed_validation:
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
            "w_max": args.w_max,
            "max_iterations": args.max_iterations,
        },
        "hypotheses": [h.to_dict() for h in final_hypotheses],
        "summary": {
            "total_hypotheses": len(final_hypotheses),
            "avg_reward": float(np.mean([h.reward for h in final_hypotheses])),
            "avg_accuracy": float(np.mean([
                h.correct_predictions / h.total_predictions
                if h.total_predictions > 0 else 0.0
                for h in final_hypotheses
            ])),
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

    # ---------- JUDGE (optional) ----------
    judge_report = None
    avg_scores = None
    rec_stats = None

    if args.enable_judge and clinical_hypotheses:
        print("\n" + "=" * 80)
        print("POST-ITERATION VALIDATION: Independent Judge Scoring")
        print("=" * 80)
        print("Scoring hypotheses with independent judge...")
        judge_report = score_hypotheses(
            hypotheses=clinical_hypotheses,
            study_context=study_context,
            client=client,
            model_name=args.model,
        )

        if judge_report:
            # Calculate summary statistics
            total = len(judge_report.scored_hypotheses)
            avg_scores = {
                "scientific_rigor": float(np.mean([s.scientific_rigor for s in judge_report.scored_hypotheses])),
                "clinical_plausibility": float(np.mean([s.clinical_plausibility for s in judge_report.scored_hypotheses])),
                "evidence_alignment": float(np.mean([s.evidence_alignment for s in judge_report.scored_hypotheses])),
                "subgroup_clarity": float(np.mean([s.subgroup_clarity for s in judge_report.scored_hypotheses])),
                "confounding_awareness": float(np.mean([s.confounding_awareness for s in judge_report.scored_hypotheses])),
                "validation_plan_quality": float(np.mean([s.validation_plan_quality for s in judge_report.scored_hypotheses])),
                "overall_score": float(np.mean([s.overall_score for s in judge_report.scored_hypotheses])),
            }

            high_priority = sum(1 for s in judge_report.scored_hypotheses if s.recommendation == "high_priority")
            medium_priority = sum(1 for s in judge_report.scored_hypotheses if s.recommendation == "medium_priority")
            low_priority = sum(1 for s in judge_report.scored_hypotheses if s.recommendation == "low_priority")
            reconsider = sum(1 for s in judge_report.scored_hypotheses if s.recommendation == "reconsider")

            rec_stats = {
                "high_priority": high_priority,
                "medium_priority": medium_priority,
                "low_priority": low_priority,
                "reconsider": reconsider,
                "support_rate": float((high_priority + medium_priority) / total if total > 0 else 0),
                "neutral_rate": float(low_priority / total if total > 0 else 0),
                "conflict_rate": float(reconsider / total if total > 0 else 0),
            }

            # Add statistics to judge report
            judge_output = judge_report.model_dump()
            judge_output["statistics"] = {
                "average_scores": avg_scores,
                "recommendations": rec_stats,
            }

            judge_path = os.path.splitext(args.out_json)[0] + "_judge.json"
            with open(judge_path, "w") as f:
                json.dump(judge_output, f, indent=2)
            print(f"Wrote judge report to: {judge_path}")

            # Print detailed scores for each hypothesis
            print(f"\nJudge Scores (1-5 scale):")
            print(f"{'='*80}")
            for score in judge_report.scored_hypotheses:
                print(f"\n{score.title}")
                print(f"  Scientific Rigor: {score.scientific_rigor}/5")
                print(f"  Clinical Plausibility: {score.clinical_plausibility}/5")
                print(f"  Evidence Alignment: {score.evidence_alignment}/5")
                print(f"  Subgroup Clarity: {score.subgroup_clarity}/5")
                print(f"  Confounding Awareness: {score.confounding_awareness}/5")
                print(f"  Validation Plan Quality: {score.validation_plan_quality}/5")
                print(f"  Overall Score: {score.overall_score}/5")
                print(f"  Recommendation: {score.recommendation}")

            # Print top hypotheses
            print(f"\n{'='*80}")
            print(f"Top hypotheses by judge score:")
            for i, title in enumerate(judge_report.top_hypotheses[:5], 1):
                print(f"  {i}. {title}")

    # ---------- PUBMED VALIDATION (optional) ----------
    if args.enable_pubmed_validation and clinical_hypotheses:
        print("\n" + "=" * 80)
        print("POST-ITERATION VALIDATION: PubMed Literature Evidence")
        print("=" * 80)

        # Convert to FeatureHypothesis format for PubMed validator
        pubmed_compatible_format = convert_clinical_to_feature_format(clinical_hypotheses, study_context)

        # Count mechanisms for user info
        total_mechanisms = sum(len(f['mechanisms']) for f in pubmed_compatible_format['feature_hypotheses'])
        num_features = len(pubmed_compatible_format['feature_hypotheses'])

        print(f"Preparing PubMed validation...")
        print(f"  Features to validate: {num_features}")
        print(f"  Mechanisms per feature: {total_mechanisms / num_features:.1f} avg (max 3)")
        print(f"  Total mechanisms: {total_mechanisms}")
        print(f"  Max abstracts per mechanism: {args.max_abstracts}")
        print(f"\nNote: Each mechanism requires PubMed search + abstract retrieval + LLM analysis")
        print(f"      This may take 5-10 minutes depending on API rate limits\n")

        pubmed_input_path = os.path.splitext(args.out_json)[0] + "_pubmed_input.json"
        with open(pubmed_input_path, "w") as f:
            json.dump(pubmed_compatible_format, f, indent=2)
        print(f"Created PubMed-compatible format: {pubmed_input_path}")

        pubmed_path = os.path.splitext(args.out_json)[0] + "_pubmed_validation.json"

        success = run_pubmed_validation(
            hypotheses_json=pubmed_input_path,
            output_path=pubmed_path,
            trial_name=args.trial_name,
            max_abstracts=args.max_abstracts,
        )

        if success:
            print(f"PubMed validation results saved to: {pubmed_path}")

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"HypoGeniC Algorithm Summary")
    print(f"{'=' * 80}")
    print(f"\nTop {len(final_hypotheses)} Hypotheses:")
    for i, h in enumerate(final_hypotheses, 1):
        accuracy = h.correct_predictions / h.total_predictions if h.total_predictions > 0 else 0.0
        print(f"  {i}. {h.hypothesis.title}")
        print(f"     Reward: {h.reward:.2f}, Accuracy: {accuracy:.2%} ({h.correct_predictions}/{h.total_predictions})")

    if judge_report:
        print(f"\n{'=' * 80}")
        print(f"Judge Assessment Summary:")
        print(f"{'=' * 80}")
        print(f"{judge_report.summary}")

        # Print average scores (already calculated above)
        print(f"\nAverage Judge Scores (1-5 scale):")
        print(f"  Scientific Rigor: {avg_scores['scientific_rigor']:.2f}")
        print(f"  Clinical Plausibility: {avg_scores['clinical_plausibility']:.2f}")
        print(f"  Evidence Alignment: {avg_scores['evidence_alignment']:.2f}")
        print(f"  Subgroup Clarity: {avg_scores['subgroup_clarity']:.2f}")
        print(f"  Confounding Awareness: {avg_scores['confounding_awareness']:.2f}")
        print(f"  Validation Plan Quality: {avg_scores['validation_plan_quality']:.2f}")
        print(f"  Overall Score: {avg_scores['overall_score']:.2f}")

        # Print recommendation distribution (already calculated above)
        total = len(judge_report.scored_hypotheses)

        print(f"\nRecommendation Distribution:")
        print(f"  High Priority: {rec_stats['high_priority']} ({rec_stats['high_priority']/total*100:.1f}%)")
        print(f"  Medium Priority: {rec_stats['medium_priority']} ({rec_stats['medium_priority']/total*100:.1f}%)")
        print(f"  Low Priority: {rec_stats['low_priority']} ({rec_stats['low_priority']/total*100:.1f}%)")
        print(f"  Reconsider: {rec_stats['reconsider']} ({rec_stats['reconsider']/total*100:.1f}%)")
        print(f"\nAggregated Rates:")
        print(f"  Support Rate (High+Medium): {rec_stats['support_rate']*100:.1f}%")
        print(f"  Neutral Rate (Low): {rec_stats['neutral_rate']*100:.1f}%")
        print(f"  Conflict Rate (Reconsider): {rec_stats['conflict_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
