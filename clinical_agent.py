#!/usr/bin/env python3
"""
generate_clinical_hypotheses.py

Generate clinical research hypotheses from a Shapley summary JSON using OpenAI's API,
returning structured JSON output via Structured Outputs (Pydantic schema).

Requires:
  pip install openai pydantic

Auth:
  export OPENAI_API_KEY="..."

Example:
  python clinical_agent.py \
    --shap_json results/ist3/baseline_shapley_value_sampling_summary_shuffle_True_RLearner_zero_baseline_True.json \
    --out_json results/ist3/hypotheses_baseline_shapley_RLearner.json \
    --trial_name ist3 \
    --n_features 15 \
    --n_hypotheses 8
    
  Or with manual metadata:
  python clinical_agent.py \
    --shap_json results/custom/shap_summary.json \
    --out_json results/custom/hypotheses.json \
    --treatment "Custom treatment" \
    --outcome "Custom outcome" \
    --population "Custom population" \
    --n_features 15 \
    --n_hypotheses 8
"""

import argparse
import json
import os
from typing import List, Optional, Literal

from openai import OpenAI
from pydantic import BaseModel, Field

# -----------------------------
# Structured output schema
# -----------------------------

class SubgroupDefinition(BaseModel):
    feature: str = Field(..., description="Feature name used to define a subgroup/effect modifier.")
    split_rule: str = Field(..., description="Human-readable subgroup rule (e.g., 'age >= 75', 'lactate > 2').")
    notes: Optional[str] = Field(None, description="Any nuance about encoding, bins, or clinical interpretation.")


class ValidationPlan(BaseModel):
    analyses: List[str] = Field(
        ...,
        description=(
            "Concrete follow-up analyses to validate the hypothesis, e.g., "
            "DR estimator within strata, interaction term, sensitivity checks."
        ),
    )
    negative_controls: Optional[List[str]] = Field(
        None,
        description="Optional negative control ideas / falsification tests."
    )
    robustness: Optional[List[str]] = Field(
        None,
        description="Optional robustness checks (baseline sensitivity, subgroup stability, etc.)."
    )


class ClinicalHypothesis(BaseModel):
    title: str = Field(..., description="Short, specific hypothesis title.")
    hypothesis: str = Field(
        ...,
        description="A testable statement about treatment-effect heterogeneity or subgroup benefit/harm."
    )
    expected_direction: Literal["higher_benefit", "lower_benefit", "higher_harm", "lower_harm", "ambiguous"] = Field(
        ...,
        description="Direction of effect modification relative to the subgroup rule."
    )
    subgroup: SubgroupDefinition
    rationale: List[str] = Field(
        ...,
        description="Bullet-like rationales grounded in features + plausible clinical mechanism."
    )
    key_features: List[str] = Field(
        ...,
        description="Top features (from Shapley summary) that support this hypothesis."
    )
    confounders_and_bias_risks: List[str] = Field(
        ...,
        description="Potential confounding, bias, measurement error, or collider risks."
    )
    validation: ValidationPlan
    caveats: Optional[List[str]] = Field(
        None,
        description="Any cautions about interpretation (attribution ≠ causality, encoding, baseline sensitivity)."
    )


class HypothesisSet(BaseModel):
    dataset: str
    learner: str
    treatment: str
    outcome: str
    population: str
    source_explainer: str
    hypotheses: List[ClinicalHypothesis]


# -----------------------------
# Helpers
# -----------------------------

def load_top_features(shap_json_path: str, n_features: int) -> dict:
    with open(shap_json_path, "r") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    explainer = meta.get("explainer", "unknown_explainer")
    dataset = meta.get("dataset", "unknown_dataset")
    learner = meta.get("learner", "unknown_learner")

    # Prefer pre-sorted list by mean_abs if present
    features = data.get("features", [])
    if not features:
        raise ValueError("No 'features' array found in Shapley JSON.")

    # Sort defensively by shap_mean_abs desc
    features_sorted = sorted(features, key=lambda x: float(x.get("shap_mean_abs", 0.0)), reverse=True)
    top = features_sorted[:n_features]

    # Return compact evidence for prompting
    top_evidence = [
        {
            "feature": f.get("feature"),
            "feature_index": f.get("feature_index"),
            "topN_frequency_pct": f.get("topN_frequency_pct"),
            "shap_mean_abs": f.get("shap_mean_abs"),
            "shap_mean": f.get("shap_mean"),
            "pearson_sign_pos_frac": f.get("pearson_sign_pos_frac"),
            "pearson_sign_neg_frac": f.get("pearson_sign_neg_frac"),
        }
        for f in top
    ]

    return {
        "dataset": dataset,
        "learner": learner,
        "explainer": explainer,
        "top_feature_evidence": top_evidence,
    }


def ensure_out_dir(path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


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
            "outcome": "Composite of major cardiovascular events (MI, stroke, heart failure, cardiovascular death)",
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


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shap_json", required=True, help="Path to Shapley summary JSON created earlier.")
    parser.add_argument("--out_json", required=True, help="Where to write generated hypotheses JSON.")
    
    # Option 1: Use trial name for automatic metadata lookup
    parser.add_argument("--trial_name", help="Trial name (ist3, crash_2, sprint, accord) - auto-populates metadata.")
    
    # Option 2: Manual metadata (used if --trial_name not provided)
    parser.add_argument("--treatment", help="Treatment/exposure description (required if no --trial_name).")
    parser.add_argument("--outcome", help="Outcome description (required if no --trial_name).")
    parser.add_argument("--population", help="Population/cohort description (required if no --trial_name).")

    parser.add_argument("--n_features", type=int, default=15, help="Number of top Shapley features to include.")
    parser.add_argument("--n_hypotheses", type=int, default=8, help="How many hypotheses to generate.")
    parser.add_argument("--model", default="gpt-4o-2024-08-06", help="Model name supporting structured outputs (e.g., gpt-4o-2024-08-06).")

    args = parser.parse_args()
    
    # Determine treatment/outcome/population
    if args.trial_name:
        trial_meta = get_trial_metadata(args.trial_name)
        treatment = trial_meta["treatment"]
        outcome = trial_meta["outcome"]
        population = trial_meta["population"]
    else:
        if not all([args.treatment, args.outcome, args.population]):
            parser.error("Must provide either --trial_name OR all of (--treatment, --outcome, --population)")
        treatment = args.treatment
        outcome = args.outcome
        population = args.population

    evidence = load_top_features(args.shap_json, args.n_features)

    system_instructions = (
        "You are a clinical research assistant. Generate testable *research hypotheses* about "
        "treatment-effect heterogeneity (effect modification) from feature-importance evidence. "
        "Do NOT provide patient-specific medical advice. Do NOT claim causality as proven; "
        "phrase as hypotheses and propose validation analyses."
    )

    user_prompt = {
        "task": "Generate clinical research hypotheses from Shapley evidence.",
        "constraints": {
            "num_hypotheses": args.n_hypotheses,
            "must_be_testable": True,
            "avoid_overclaiming": True,
            "prefer_effect_modifiers": True,
        },
        "study_context": {
            "dataset": evidence["dataset"],
            "learner": evidence["learner"],
            "population": population,
            "treatment": treatment,
            "outcome": outcome,
            "source_explainer": evidence["explainer"],
        },
        "top_feature_evidence": evidence["top_feature_evidence"],
        "guidance": [
            "Each hypothesis should pick a *single primary subgroup rule* based on one feature, "
            "and may reference other features as supporting context.",
            "Include plausible mechanism/rationale, confounding risks, and a concrete validation plan.",
            "If a feature name is not clinically interpretable, still propose a careful subgroup rule "
            "and note the ambiguity.",
        ],
    }

    client = OpenAI(api_key="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")

    # Use Structured Outputs via beta.chat.completions.parse with Pydantic schema
    completion = client.beta.chat.completions.parse(
        model=args.model,
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": json.dumps(user_prompt, indent=2)},
        ],
        response_format=HypothesisSet,
    )

    parsed: HypothesisSet = completion.choices[0].message.parsed

    ensure_out_dir(args.out_json)
    with open(args.out_json, "w") as f:
        json.dump(parsed.model_dump(), f, indent=2)

    print(f"Wrote hypotheses to: {args.out_json}")


if __name__ == "__main__":
    main()
