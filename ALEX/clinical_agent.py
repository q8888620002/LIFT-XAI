#!/usr/bin/env python3
"""clinical_agent.py

Generate clinical research hypotheses from a Shapley summary JSON using OpenAI's API
or OpenRouter, returning structured JSON output via Structured Outputs (Pydantic schema).
Optionally verify hypotheses with a separate verifier model.

Requires:
  pip install openai pydantic

API Keys:
  - OpenAI: Set OPENAI_API_KEY environment variable
  - OpenRouter: Set OPENROUTER_API_KEY environment variable

Example:
  python clinical_agent.py \
    --shap_json results/ist3/baseline_shapley_value_sampling_summary_shuffle_True_RLearner_zero_baseline_True.json \
    --out_json results/ist3/hypotheses_baseline_shapley_RLearner.json \
    --trial_name ist3 \
    --n_features 15 \
    --n_hypotheses 8

  Or with OpenRouter:
  export OPENROUTER_API_KEY=your_key_here
  python clinical_agent.py \
    --shap_json results/ist3/shap_summary.json \
    --out_json results/ist3/hypotheses.json \
    --trial_name ist3 \
    --model anthropic/claude-3.5-sonnet \
        --api_provider openrouter

  Or with manual metadata:
  python clinical_agent.py \
    --shap_json results/custom/shap_summary.json \
    --out_json results/custom/hypotheses.json \
    --treatment "Custom treatment" \
    --outcome "Custom outcome" \
    --population "Custom population" \
    --n_features 15 \
    --n_hypotheses 8

    With verification:
  python clinical_agent.py \
    --shap_json results/ist3/shap_summary.json \
    --out_json results/ist3/hypotheses.json \
    --trial_name ist3 \
        --enable_verifier
"""

import argparse
import json
import os
from typing import List, Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field
from src.agent_utils import (
    get_model_client,
    get_trial_metadata,
    load_top_features,
    load_local_env,
    resolve_seeded_output_path,
    search_and_extract_article,
    write_json_file,
)

# -----------------------------
# Structured output schema
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
        description="Top features (from Shapley summary) that support this hypothesis.",
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


class HypothesisSet(BaseModel):
    dataset: str
    learner: str
    treatment: str
    outcome: str
    population: str
    source_explainer: str
    hypotheses: List[ClinicalHypothesis]


class HypothesisIssue(BaseModel):
    type: Literal[
        "overclaiming_causality",
        "not_testable",
        "subgroup_rule_ambiguous",
        "direction_not_supported",
        "feature_not_in_evidence",
        "confounding_missing",
        "validation_plan_weak",
        "clinical_implausible",
        "other",
    ]
    severity: Literal["low", "medium", "high"]
    message: str


class HypothesisReview(BaseModel):
    title: str
    verdict: Literal["approve", "revise", "reject"]
    issues: List[HypothesisIssue] = Field(default_factory=list)
    suggested_edits: Optional[str] = Field(
        None,
        description="Concrete rewrite guidance or a rewritten version for the hypothesis text/subgroup rule.",
    )
    evidence_alignment: Literal["strong", "moderate", "weak"] = "moderate"
    confidence: Literal["low", "medium", "high"] = "medium"


class VerificationOutput(BaseModel):
    overall_verdict: Literal["approve", "revise", "reject"]
    summary: str
    per_hypothesis: List[HypothesisReview]
    revised: Optional[HypothesisSet] = Field(
        None,
        description="If overall_verdict is revise, provide a corrected HypothesisSet.",
    )


class MechanismHypothesis(BaseModel):
    mechanism_type: Literal[
        "biological", "physiological", "pharmacological", "behavioral", "statistical"
    ] = Field(..., description="Type of mechanism")
    description: str = Field(
        ..., description="Detailed explanation of the mechanism"
    )
    evidence_level: Literal["strong", "moderate", "weak", "speculative"] = Field(
        ..., description="Strength of supporting evidence"
    )

class FeatureHypothesis(BaseModel):
    feature_name: str = Field(..., description="Name of the feature")
    importance_rank: int = Field(..., description="Rank by SHAP importance (1=most important)")
    shap_value: float = Field(..., description="Mean absolute SHAP value")
    effect_direction: Literal["positive", "negative", "bidirectional", "unclear"] = Field(
        ..., description="Direction of feature's influence on treatment effect"
    )
    clinical_interpretation: str = Field(
        ..., description="What this feature represents clinically"
    )
    why_important: str = Field(
        ...,
        description="Why this feature is important for treatment effect heterogeneity"
    )
    mechanisms: List[MechanismHypothesis] = Field(
        ..., description="Possible mechanisms explaining importance"
    )
    subgroup_implications: str = Field(
        ...,
        description="What subgroups this suggests might have differential treatment effects"
    )
    validation_suggestions: List[str] = Field(
        ..., description="How to test these hypotheses"
    )
    caveats: List[str] = Field(
        ..., description="Limitations and alternative explanations"
    )


class FeatureHypothesesSet(BaseModel):
    dataset: str
    model: str
    summary: str = Field(
        ..., description="Overall summary of feature importance patterns"
    )
    feature_hypotheses: List[FeatureHypothesis]
    cross_feature_patterns: Optional[str] = Field(
        None, description="Patterns across multiple features"
    )


class FeatureHypothesisIssue(BaseModel):
    type: Literal[
        "mechanism_implausible",
        "clinical_interpretation_wrong",
        "effect_direction_unsupported",
        "validation_plan_weak",
        "missing_caveats",
        "overclaiming_certainty",
        "other",
    ]
    severity: Literal["low", "medium", "high"]
    message: str


class MechanismReview(BaseModel):
    mechanism_type: str = Field(..., description="Type of mechanism being reviewed")
    verdict: Literal["approve", "revise", "reject"]
    plausibility: Literal["high", "moderate", "low", "implausible"]
    evidence_level_appropriate: bool = Field(
        ..., description="Whether the claimed evidence level matches the actual support"
    )
    comments: str = Field(
        ..., description="Detailed comments on this specific mechanism"
    )
    suggested_revision: Optional[str] = Field(
        None, description="Suggested revision for the mechanism description if needed"
    )


class FeatureHypothesisReview(BaseModel):
    feature_name: str
    verdict: Literal["approve", "revise", "reject"]
    issues: List[FeatureHypothesisIssue] = Field(default_factory=list)
    per_mechanism: List[MechanismReview] = Field(
        default_factory=list,
        description="Review of each individual mechanism hypothesis for this feature"
    )
    suggested_edits: Optional[str] = Field(
        None,
        description="Concrete suggestions for improving this feature hypothesis.",
    )
    mechanism_quality: Literal["strong", "moderate", "weak"] = "moderate"
    confidence: Literal["low", "medium", "high"] = "medium"


class FeatureVerificationOutput(BaseModel):
    overall_verdict: Literal["approve", "revise", "reject"]
    summary: str
    per_feature: List[FeatureHypothesisReview]
    revised: Optional[FeatureHypothesesSet] = Field(
        None,
        description="If overall_verdict is revise, provide a corrected FeatureHypothesesSet.",
    )


# -----------------------------
# Helpers
# -----------------------------


def _fill_mechanisms_for_feature(
    feature_hypothesis: "FeatureHypothesis",
    count_needed: int,
    study_context: dict,
    client: OpenAI,
    model_name: str,
) -> List["MechanismHypothesis"]:
    """Request additional mechanisms for a feature that was under-generated."""

    existing_descriptions = [m.description for m in (feature_hypothesis.mechanisms or [])]
    existing_types = [m.mechanism_type for m in (feature_hypothesis.mechanisms or [])]

    class _MechanismList(BaseModel):
        mechanisms: List[MechanismHypothesis]

    system = (
        "You are a clinical research expert. Generate additional distinct mechanism hypotheses "
        "for a specific feature explaining treatment effect heterogeneity. "
        "Each mechanism must be different in type or focus from the existing ones."
    )
    user_prompt = {
        "task": f"Generate {count_needed} additional mechanism(s) for feature '{feature_hypothesis.feature_name}'",
        "study_context": study_context,
        "feature_name": feature_hypothesis.feature_name,
        "clinical_interpretation": feature_hypothesis.clinical_interpretation,
        "existing_mechanisms": [
            {"type": t, "description": d}
            for t, d in zip(existing_types, existing_descriptions)
        ],
        "count_needed": count_needed,
        "instructions": [
            f"Generate exactly {count_needed} new mechanism(s) DISTINCT from the existing ones above",
            "Prefer unused mechanism types: biological, pharmacological, statistical/proxy",
            "Each mechanism must explain how this feature modifies the TREATMENT EFFECT (not just prognosis)",
        ],
    }
    try:
        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_prompt, indent=2)},
            ],
            response_format=_MechanismList,
        )
        return (completion.choices[0].message.parsed.mechanisms or [])[:count_needed]
    except Exception as e:
        print(f"    Error in mechanism gap-fill for '{feature_hypothesis.feature_name}': {e}")
        return []


def generate_feature_hypotheses(
    top_features: List[dict],
    study_context: dict,
    client: OpenAI,
    model_name: str = "gpt-4o-2024-08-06"
) -> Optional[FeatureHypothesesSet]:

    feature_label_map = {}
    for item in top_features:
        raw_name = (item or {}).get("feature_raw")
        mapped_name = (item or {}).get("feature")
        if raw_name and mapped_name:
            feature_label_map[str(raw_name).strip().lower()] = mapped_name
            feature_label_map[str(mapped_name).strip().lower()] = mapped_name

    # 1. CORE DEFINITIONS (Strict Definitions to prevent drift)
    definitions = (
        "DEFINITIONS:\n"
        "- PROGNOSTIC FACTOR: A feature that predicts the outcome regardless of treatment (e.g., 'Age increases mortality'). "
        "-> IGNORE these unless they also modify treatment response.\n"
        "- PREDICTIVE FACTOR (EFFECT MODIFIER): A feature that changes the MAGNITUDE or DIRECTION of the treatment benefit "
        "(e.g., 'Drug works better in Young people than Old'). -> FOCUS on these.\n"
    )

    # 2. MECHANISM DIVERSITY INSTRUCTION
    diversity_instruction = (
        "For each feature, you must propose distinct mechanism types if possible:\n"
        "   - 'biological': Direct pathophysiological interaction.\n"
        "   - 'pharmacological': PK/PD, metabolism, drug clearance.\n"
        "   - 'statistical/proxy': If the feature is likely a proxy for an unmeasured confounder (e.g., 'zip code' -> 'socioeconomic status').\n"
    )

    # 3. CONTEXTUAL LOGIC
    use_data_summary = study_context.get("use_data_summary", False)

    if len(top_features) > 0:
        # MODE 1: WITH SHAP (Interpretation)
        role_type = "Clinical Expert"
        directive = (
            "You are analyzing feature attribution from a conditional average treatment effect model using trial metadata (treatment/outcome/population).\n"
            "TASK: Hypothesize WHY these specific features could modify treatment response.\n"
            "Treat SHAP-ranked features as model signals to interpret, not proof of causality.\n"
            "If a feature is coded/ambiguous or not strongly established in literature, keep it and explain it as a plausible proxy or exploratory/novel modifier with explicit uncertainty.\n"
        )
    elif use_data_summary:
        # MODE 2: BLINDED PREDICTION (Available Features Only)
        role_type = "Clinical Expert"
        available_cols = study_context.get("available_features", [])
        directive = (
            f"You know only the study design and the list of measured features: {available_cols}.\n"
            "TASK: hypothesize which of these available features would modify the treatment response.\n"
        )
    else:
        # MODE 3: PURE THEORY (Literature Only)
        role_type = "Clinical Expert"
        directive = (
            "Based ONLY on the trial metadata (Treatment/Outcome), "
            "TASK: hypothesize which patient characteristics would modify the treatment response."
        )

    # 4. ASSEMBLE SYSTEM PROMPT
    system_instructions = (
        f"You are a {role_type}.\n\n"
        f"{definitions}\n"
        f"{directive}\n\n"
        f"{diversity_instruction}\n"
        "REQUIREMENTS:\n"
        "1. Focus STRICTLY on Heterogeneous Treatment Effects (Interaction), not just main effects.\n"
        "2. If a feature has a bidirectional effect (e.g., good for some, bad for others), specify that.\n"
        "3. grounding: Cite known trials or physiological principles.\n"
        "4. When data_to_interpret provides mapped clinical feature labels, use those labels exactly in feature_name; do not output raw codes alone.\n"
        "5. For each feature, distinguish whether support is established vs exploratory; include caveats when evidence is limited rather than dropping the feature.\n"
    )

    # 5. USER PROMPT
    n_mechanisms = study_context.get("n_hypotheses_per_feature", 3)
    n_features = study_context.get("n_features", len(top_features) if top_features else 5)

    user_prompt = {
        "study_context": study_context,
        "data_to_interpret": top_features if top_features else "NONE (Blinded Mode)",
        "task_constraints": [
            f"Generate hypotheses for the top {n_features} features.",
            f"Provide exactly {n_mechanisms} distinct mechanisms per feature.",
            "Ensure 'effect_direction' describes how the feature changes the TREATMENT BENEFIT (e.g., 'Positive' = Feature increases benefit)."
        ]
    }

    try:
        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": json.dumps(user_prompt, indent=2)},
            ],
            response_format=FeatureHypothesesSet,
        )
        result = completion.choices[0].message.parsed
    except Exception as e:
        print(f"Error in hypothesis generation: {e}")
        return None

    # --- Gap-fill: ensure every feature has exactly n_mechanisms mechanisms ---
    if result and result.feature_hypotheses:
        # Normalize any raw feature names back to mapped clinical labels.
        for fh in result.feature_hypotheses:
            current_name = (fh.feature_name or "").strip()
            mapped_name = feature_label_map.get(current_name.lower())
            if mapped_name:
                fh.feature_name = mapped_name

        for fh in result.feature_hypotheses:
            existing = fh.mechanisms or []
            gap = n_mechanisms - len(existing)
            if gap <= 0:
                continue
            print(f"  Gap-fill '{fh.feature_name}': have {len(existing)}, need {gap} more mechanisms...")
            extra = _fill_mechanisms_for_feature(
                feature_hypothesis=fh,
                count_needed=gap,
                study_context=study_context,
                client=client,
                model_name=model_name,
            )
            fh.mechanisms = existing + extra
            if len(fh.mechanisms) < n_mechanisms:
                print(f"    Warning: still only {len(fh.mechanisms)}/{n_mechanisms} after fill")

    return result

# -----------------------------
# Main
# -----------------------------


def main():
    load_local_env()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shap_json",
        required=True,
        help="Path to Shapley summary JSON created earlier.",
    )
    parser.add_argument(
        "--out_json", required=True, help="Where to write generated hypotheses JSON."
    )

    # Option 1: Use trial name for automatic metadata lookup
    parser.add_argument(
        "--trial_name",
        help="Trial name (ist3, crash_2, sprint, accord, txa) - auto-populates metadata.",
    )

    # Option 2: Manual metadata (used if --trial_name not provided)
    parser.add_argument(
        "--treatment",
        help="Treatment/exposure description (required if no --trial_name).",
    )
    parser.add_argument(
        "--outcome", help="Outcome description (required if no --trial_name)."
    )
    parser.add_argument(
        "--population",
        help="Population/cohort description (required if no --trial_name).",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset name (overrides metadata from SHAP JSON; defaults to trial_name if provided).",
    )

    parser.add_argument(
        "--n_features",
        type=int,
        default=15,
        help="Number of top Shapley features to include.",
    )
    parser.add_argument(
        "--n_hypotheses", type=int, default=8, help="How many hypotheses to generate."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed identifier used to create output subfolder seed_<seed>.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="Model name supporting structured outputs (e.g., gpt-5-mini for OpenAI, openai/gpt-4o for OpenRouter).",
    )
    parser.add_argument(
        "--api_provider",
        default="openai",
        choices=["openai", "openrouter"],
        help="API provider to use (default: openai).",
    )
    parser.add_argument(
        "--api_base_url",
        help="Custom API base URL (e.g., https://openrouter.ai/api/v1 for OpenRouter).",
    )
    parser.add_argument(
        "--enable_verifier",
        action="store_true",
        help="Enable verification/refinement pass using the model specified by --model.",
    )
    parser.add_argument(
        "--use_data_summary",
        action="store_true",
        help="Use data summary baseline: provide available features but no SHAP values (intermediate between with_shap and without_shap).",
    )
    parser.add_argument(
        "--verifier_iterations",
        type=int,
        default=1,
        help="Number of refinement iterations with verifier (default: 1)",
    )
    parser.add_argument(
        "--retrieve_article",
        action="store_true",
        help="Search for and extract information from the original trial article.",
    )
    parser.add_argument(
        "--fail_on_reject",
        action="store_true",
        help="Exit non-zero if verifier rejects.",
    )
    args = parser.parse_args()

    seeded_out_json = resolve_seeded_output_path(args.out_json, args.seed)
    if seeded_out_json != args.out_json:
        print(f"Using seeded output path: {seeded_out_json}")
    args.out_json = seeded_out_json

    # Set verifier model based on flags
    verifier_model = args.model if args.enable_verifier else None

    client = get_model_client(args.api_provider, args.api_base_url)

    # Determine treatment/outcome/population and fetch trial_meta once
    trial_meta = None
    if args.trial_name:
        trial_meta = get_trial_metadata(args.trial_name)
        treatment = trial_meta["treatment"]
        outcome = trial_meta["outcome"]
        population = trial_meta["population"]
    else:
        if not all([args.treatment, args.outcome, args.population]):
            parser.error(
                "Must provide either --trial_name OR all of (--treatment, --outcome, --population)"
            )
        treatment = args.treatment
        outcome = args.outcome
        population = args.population

    # Determine dataset name: explicit --dataset > trial_name > SHAP JSON metadata
    dataset_name = args.dataset if args.dataset else (args.trial_name.lower() if args.trial_name else None)
    evidence = load_top_features(args.shap_json, args.n_features, dataset_override=dataset_name)

    # ---------- RETRIEVE ARTICLE (optional) ----------
    article_extraction = None
    if args.retrieve_article and args.trial_name:
        print(f"Retrieving article information for {args.trial_name}...")
        article_query = trial_meta.get("article_query", f"{args.trial_name} clinical trial")

        article_extraction = search_and_extract_article(
            article_query, args.trial_name, client, model_name=args.model
        )
        if article_extraction:
            print("Successfully extracted article information.")
    elif args.retrieve_article and not args.trial_name:
        print("Warning: --retrieve_article requires --trial_name. Skipping article retrieval.")

    # ---------- GENERATE FEATURE HYPOTHESES ----------
    print("Generating mechanistic hypotheses for individual features...")

    study_context = {
        "dataset": evidence["dataset"],
        "learner": evidence["learner"],
        "population": population,
        "treatment": treatment,
        "outcome": outcome,
        "source_explainer": evidence["explainer"],
        "n_hypotheses_per_feature": args.n_hypotheses,
        "n_features": args.n_features,  # Pass n_features so generator knows how many to propose if no SHAP
        "available_features": evidence.get("available_features", []),
        "use_data_summary": args.use_data_summary,  # New baseline mode flag
    }

    feature_hypotheses = generate_feature_hypotheses(
        top_features=evidence["top_feature_evidence"],
        study_context=study_context,
        client=client,
        model_name=args.model,
    )

    if not feature_hypotheses:
        raise RuntimeError("Failed to generate feature hypotheses")

    print(f"Generated feature-level hypotheses for {len(feature_hypotheses.feature_hypotheses)} features.")

    final_output = feature_hypotheses
    verification_report = None

    # Save original hypotheses before refinement
    original_output = feature_hypotheses

    # ---------- VERIFY (optional) ----------
    if verifier_model:
        print(f"Refining feature hypotheses with verifier ({args.verifier_iterations} iteration(s))...")

        for iteration in range(args.verifier_iterations):
            if iteration > 0:
                print(f"  Refinement iteration {iteration + 1}/{args.verifier_iterations}...")

            verifier_system = (
                "You are a collaborative scientific advisor helping to REFINE feature-level mechanistic hypotheses.\n"
                "Your role is to IMPROVE the hypotheses, not just evaluate them.\n"
                "\n"
                "For each hypothesis:\n"
                "1. Identify strengths worth preserving\n"
                "2. Spot weaknesses that need improvement\n"
                "3. Provide constructive refinement suggestions\n"
                "4. ALWAYS return a revised, improved version\n"
                "\n"
                "Focus on:\n"
                "- Strengthening mechanism plausibility with more specific biological/clinical details\n"
                "- Sharpening clinical interpretation to be more precise\n"
                "- Better aligning with SHAP evidence (direction, magnitude)\n"
                "- Making subgroup implications more actionable\n"
                "- Enhancing validation plans with concrete, feasible steps\n"
                "- Adding important caveats and alternative explanations\n"
                "\n"
                "For EACH mechanism:\n"
                "- Assess plausibility (high/moderate/low/implausible)\n"
                "- Check if claimed evidence level matches actual support\n"
                "- Suggest specific improvements to mechanism description\n"
                "- Mark for revision if implausible or poorly supported\n"
                "\n"
                "If trial article context provided:\n"
                "- Refine mechanisms to align with known trial physiology\n"
                "- Adjust interpretations to match population characteristics\n"
                "- Tailor validation plans to be feasible within trial design\n"
                "- Flag any contradictions with trial findings\n"
                "\n"
                "IMPORTANT: Your goal is to help create the BEST possible hypotheses.\n"
                "Always provide a complete revised FeatureHypothesesSet with improvements,\n"
                "even if changes are minor. Build on strengths and fix weaknesses.\n"
                "Stay grounded in evidence - improve but don't add unsupported claims.\n"
                "\n"
                "CRITICAL: You MUST preserve ALL mechanisms for every feature. Never reduce the number of\n"
                "mechanisms. If the input has 3 mechanisms for a feature, the output must also have exactly\n"
                "3 mechanisms. Refine or rewrite mechanisms, but do NOT drop or omit any."
                )

            # Use current hypotheses (either original or from previous iteration)
            current_hypotheses = final_output if iteration > 0 else feature_hypotheses

            verifier_prompt = {
                "iteration": iteration + 1,
                "total_iterations": args.verifier_iterations,
                "evidence": {
                    "study_context": study_context,
                    "top_feature_evidence": evidence["top_feature_evidence"],
                },
                "current_hypotheses": current_hypotheses.model_dump(),
                "refinement_goals": {
                    "strengthen_mechanisms": True,
                    "sharpen_clinical_interpretation": True,
                    "improve_evidence_alignment": True,
                    "make_subgroups_actionable": True,
                    "enhance_validation_plans": True,
                    "add_important_caveats": True,
                },
                "instructions": [
                    "Review each feature hypothesis and its mechanisms",
                    "Identify specific improvements to make",
                    "Provide detailed per-mechanism reviews",
                    "Return a complete revised FeatureHypothesesSet",
                    "Build on what works, fix what doesn't",
                    "Make hypotheses more specific, actionable, and evidence-based",
                ],
            }

            # Add article context if available
            if article_extraction is not None:
                verifier_prompt["trial_article_context"] = article_extraction.model_dump()
                verifier_prompt["refinement_goals"]["align_with_trial_context"] = (
                    "Use trial article to refine mechanisms and interpretations to match trial physiology"
                )

            v = client.beta.chat.completions.parse(
                model=verifier_model,
                messages=[
                    {"role": "system", "content": verifier_system},
                    {"role": "user", "content": json.dumps(verifier_prompt, indent=2)},
                ],
                response_format=FeatureVerificationOutput,
            )
            verification_report: FeatureVerificationOutput = v.choices[0].message.parsed

            # Track refinement progress
            if verification_report.revised is not None:
                final_output = verification_report.revised
                num_revisions = sum(
                    1 for review in verification_report.per_feature
                    if review.verdict in ["revise", "reject"]
                )
                print(f"    Refined {num_revisions}/{len(verification_report.per_feature)} features")
            else:
                print(f"    No revisions produced in iteration {iteration + 1}")

        print(f"Completed {args.verifier_iterations} refinement iteration(s). Using final refined hypotheses.")

    # ---------- WRITE OUTPUTS ----------
    # Always save original hypotheses
    write_json_file(args.out_json, original_output.model_dump())
    print(f"Wrote original feature hypotheses to: {args.out_json}")

    # If refined, save revised version separately
    if verification_report is not None:
        revised_path = os.path.splitext(args.out_json)[0] + "_revised.json"
        write_json_file(revised_path, final_output.model_dump())
        print(f"Wrote revised feature hypotheses to: {revised_path}")

        # Write verifier report
        report_path = os.path.splitext(args.out_json)[0] + "_verification.json"
        write_json_file(report_path, verification_report.model_dump())
        print(f"Wrote verifier report to: {report_path}")

    # Write article extraction if available
    if article_extraction is not None:
        article_path = os.path.splitext(args.out_json)[0] + "_article_context.json"
        write_json_file(article_path, article_extraction.model_dump())
        print(f"Wrote article context to: {article_path}")

    if (
        args.fail_on_reject
        and verification_report is not None
        and verification_report.overall_verdict == "reject"
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
