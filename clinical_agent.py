#!/usr/bin/env python3
"""clinical_agent.py

Generate clinical research hypotheses from a Shapley summary JSON using OpenAI's API,
returning structured JSON output via Structured Outputs (Pydantic schema).
Optionally verify and independently score hypotheses with separate LLM judges.

Requires:
  pip install openai pydantic

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

  With verification and independent judging:
  python clinical_agent.py \
    --shap_json results/ist3/shap_summary.json \
    --out_json results/ist3/hypotheses.json \
    --trial_name ist3 \
    --enable_verifier \
    --enable_judge
"""

import argparse
import json
import os
from typing import List, Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field


def load_local_env(env_path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file into os.environ.

    Existing environment variables are not overwritten.
    """
    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass

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


class HypothesisScore(BaseModel):
    title: str = Field(..., description="Hypothesis title being scored")
    scientific_rigor: int = Field(
        ...,
        ge=1,
        le=10,
        description="Scientific rigor (1-10): testability, operationalizability, falsifiability",
    )
    clinical_plausibility: int = Field(
        ...,
        ge=1,
        le=10,
        description="Clinical plausibility (1-10): biological mechanism, clinical coherence",
    )
    evidence_alignment: int = Field(
        ...,
        ge=1,
        le=10,
        description="Evidence alignment (1-10): how well feature evidence supports the hypothesis",
    )
    subgroup_clarity: int = Field(
        ...,
        ge=1,
        le=10,
        description="Subgroup clarity (1-10): how clear and actionable the subgroup rule is",
    )
    confounding_awareness: int = Field(
        ...,
        ge=1,
        le=10,
        description="Confounding awareness (1-10): thoroughness of bias/confounding discussion",
    )
    validation_plan_quality: int = Field(
        ...,
        ge=1,
        le=10,
        description="Validation plan quality (1-10): concreteness and appropriateness of proposed validation",
    )
    overall_score: int = Field(
        ..., ge=1, le=10, description="Overall score (1-10): holistic assessment"
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


class ArticleMetadata(BaseModel):
    title: str = Field(..., description="Article title")
    authors: Optional[str] = Field(None, description="Authors (if found)")
    journal: Optional[str] = Field(None, description="Journal name")
    year: Optional[int] = Field(None, description="Publication year")
    doi: Optional[str] = Field(None, description="DOI")
    pmid: Optional[str] = Field(None, description="PubMed ID")
    url: str = Field(..., description="URL to the article")


class TrialCharacteristics(BaseModel):
    sample_size: Optional[int] = Field(None, description="Total sample size")
    intervention_description: str = Field(..., description="Description of intervention")
    control_description: str = Field(..., description="Description of control/comparator")
    primary_outcome: str = Field(..., description="Primary outcome measure")
    inclusion_criteria: List[str] = Field(
        ..., description="Key inclusion criteria"
    )
    exclusion_criteria: Optional[List[str]] = Field(
        None, description="Key exclusion criteria"
    )
    baseline_characteristics: Optional[str] = Field(
        None, description="Summary of baseline characteristics"
    )
    randomization_method: Optional[str] = Field(
        None, description="Randomization method if described"
    )


class TrialResults(BaseModel):
    primary_outcome_result: str = Field(
        ..., description="Result for primary outcome (effect size, p-value, CI)"
    )
    subgroup_analyses_reported: bool = Field(
        ..., description="Whether subgroup analyses were reported"
    )
    subgroups_analyzed: Optional[List[str]] = Field(
        None, description="Which subgroups were analyzed (if any)"
    )
    heterogeneity_findings: Optional[str] = Field(
        None, description="Any reported treatment effect heterogeneity"
    )
    adverse_events: Optional[str] = Field(
        None, description="Key adverse events or safety findings"
    )


class ArticleExtraction(BaseModel):
    metadata: ArticleMetadata
    trial_characteristics: TrialCharacteristics
    results: TrialResults
    study_limitations: List[str] = Field(
        ..., description="Limitations noted in the article or evident from design"
    )
    relevant_to_hypotheses: str = Field(
        ...,
        description="How this trial context relates to the generated hypotheses"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ..., description="Confidence in the extraction accuracy"
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


class FeatureHypothesisScore(BaseModel):
    feature_name: str = Field(..., description="Feature being scored")
    mechanism_plausibility: int = Field(
        ...,
        ge=1,
        le=10,
        description="Mechanism plausibility (1-10): biological/clinical coherence of proposed mechanisms",
    )
    clinical_interpretation: int = Field(
        ...,
        ge=1,
        le=10,
        description="Clinical interpretation (1-10): accuracy and clarity of what the feature represents",
    )
    evidence_alignment: int = Field(
        ...,
        ge=1,
        le=10,
        description="Alignment with existing literature (1-10): how well grounded in published clinical evidence",
    )
    subgroup_implications: int = Field(
        ...,
        ge=1,
        le=10,
        description="Subgroup implications (1-10): clarity and usefulness of suggested subgroups",
    )
    validation_plan_quality: int = Field(
        ...,
        ge=1,
        le=10,
        description="Validation quality (1-10): concreteness and appropriateness of validation suggestions",
    )
    caveat_awareness: int = Field(
        ...,
        ge=1,
        le=10,
        description="Caveat awareness (1-10): thoroughness in acknowledging limitations and alternatives",
    )
    overall_score: int = Field(
        ..., ge=1, le=10, description="Overall score (1-10): holistic assessment"
    )
    strengths: List[str] = Field(..., description="Key strengths")
    weaknesses: List[str] = Field(..., description="Key weaknesses")
    recommendation: Literal[
        "high_priority", "medium_priority", "low_priority", "reconsider"
    ] = Field(..., description="Recommendation for follow-up")
    justification: str = Field(
        ..., description="Brief justification for scores and recommendation"
    )


class MechanismScore(BaseModel):
    mechanism_type: str = Field(..., description="Type of mechanism being scored")
    plausibility: int = Field(
        ...,
        ge=1,
        le=5,
        description="Plausibility (1-5): how believable is this mechanism"
    )
    evidence_support: int = Field(
        ...,
        ge=1,
        le=5,
        description="Evidence support (1-5): how well is this mechanism supported by evidence"
    )
    specificity: int = Field(
        ...,
        ge=1,
        le=5,
        description="Specificity (1-5): how specific and detailed is the mechanism description"
    )
    testability: int = Field(
        ...,
        ge=1,
        le=5,
        description="Testability (1-5): how testable/falsifiable is this mechanism"
    )
    overall_score: int = Field(
        ..., ge=1, le=5, description="Overall score (1-5) for this mechanism"
    )
    comments: str = Field(
        ..., description="Brief comments on this mechanism's strengths and weaknesses"
    )


class FeatureHypothesisScoreWithMechanisms(BaseModel):
    feature_name: str = Field(..., description="Feature being scored")
    per_mechanism_scores: List[MechanismScore] = Field(
        ..., description="Scores for each individual mechanism"
    )
    mechanism_plausibility: int = Field(
        ...,
        ge=1,
        le=5,
        description="Mechanism plausibility (1-5): biological/clinical coherence of proposed mechanisms",
    )
    clinical_interpretation: int = Field(
        ...,
        ge=1,
        le=5,
        description="Clinical interpretation (1-5): accuracy and clarity of what the feature represents",
    )
    evidence_alignment: int = Field(
        ...,
        ge=1,
        le=5,
        description="Alignment with existing literature (1-5): how well grounded in published clinical evidence",
    )
    subgroup_implications: int = Field(
        ...,
        ge=1,
        le=5,
        description="Subgroup implications (1-5): clarity and usefulness of suggested subgroups",
    )
    validation_plan_quality: int = Field(
        ...,
        ge=1,
        le=5,
        description="Validation quality (1-5): concreteness and appropriateness of validation suggestions",
    )
    caveat_awareness: int = Field(
        ...,
        ge=1,
        le=5,
        description="Caveat awareness (1-5): thoroughness in acknowledging limitations and alternatives",
    )
    overall_score: int = Field(
        ..., ge=1, le=5, description="Overall score (1-5): holistic assessment"
    )
    strengths: List[str] = Field(..., description="Key strengths")
    weaknesses: List[str] = Field(..., description="Key weaknesses")
    recommendation: Literal[
        "high_priority", "medium_priority", "low_priority", "reconsider"
    ] = Field(..., description="Recommendation for follow-up")
    justification: str = Field(
        ..., description="Brief justification for scores and recommendation"
    )


class FeatureJudgeOutput(BaseModel):
    summary: str = Field(
        ..., description="Overall assessment summary across all feature hypotheses"
    )
    scored_features: List[FeatureHypothesisScoreWithMechanisms]
    top_features: List[str] = Field(
        ..., description="Names of top-ranked features (by overall_score)"
    )
    methodological_concerns: Optional[List[str]] = Field(
        None, description="Any cross-cutting concerns about the analysis"
    )


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
    
    # Get all available feature names for WITHOUT SHAP condition
    all_feature_names = [f.get("feature") for f in features if f.get("feature")]
    
    # If features is empty, return baseline mode (no SHAP data available)
    # This allows baseline experiments where generator proposes features from literature
    if not features:
        return {
            "dataset": dataset,
            "learner": learner,
            "explainer": "baseline_no_shap",
            "top_feature_evidence": [],
            "available_features": [],
        }

    # Sort defensively by shap_mean_abs desc
    features_sorted = sorted(
        features, key=lambda x: float(x.get("shap_mean_abs", 0.0)), reverse=True
    )
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
        "available_features": all_feature_names,
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
            "article_query": "IST-3 trial alteplase stroke Sandercock 2012",
        },
        "crash_2": {
            "treatment": "Tranexamic acid (TXA)",
            "outcome": "All-cause mortality at 28 days or in-hospital death",
            "population": "Trauma patients with significant bleeding or at risk of significant hemorrhage",
            "article_query": "CRASH-2 trial tranexamic acid trauma 2010",
        },
        "sprint": {
            "treatment": "Intensive blood pressure control (systolic BP target <120 mmHg)",
            "outcome": "Composite of major cardiovascular events (MI, stroke, heart failure, cardiovascular death)",
            "population": "Non-diabetic adults aged ≥50 with hypertension and increased cardiovascular risk",
            "article_query": "SPRINT trial intensive blood pressure control 2015",
        },
        "accord": {
            "treatment": "Intensive glucose control (HbA1c target <6.0%)",
            "outcome": "Major cardiovascular events (nonfatal MI, nonfatal stroke, cardiovascular death)",
            "population": "Adults with type 2 diabetes and high cardiovascular risk",
            "article_query": "ACCORD trial intensive glucose control diabetes 2008",
        },
        "txa": {
            "treatment": "Pre-hospital tranexamic acid (TXA) administration",
            "outcome": "Survival (in-hospital mortality status)",
            "population": "Adult trauma patients in a pre-hospital TXA cohort",
            "article_query": "pre-hospital TXA trauma cohort retrospective study",
        },
    }

    trial_lower = trial_name.lower()
    if trial_lower not in trial_map:
        raise ValueError(
            f"Unknown trial: {trial_name}. Supported trials: {', '.join(trial_map.keys())}.\n"
            "Use --treatment, --outcome, --population arguments instead for custom trials."
        )
    return trial_map[trial_lower]


def search_and_extract_article(
    query: str, trial_name: str, client: OpenAI, model_name: str = "gpt-4o-2024-08-06"
) -> Optional[ArticleExtraction]:
    """Search for trial article and extract key information.

    Args:
        query: Search query for the article
        trial_name: Name of the clinical trial
        client: OpenAI client
        model_name: Model to use for extraction

    Returns:
        ArticleExtraction or None if error

    Note: This is a simplified implementation. In production, you'd want to:
    - Use a proper academic search API (PubMed, Semantic Scholar, etc.)
    - Implement web scraping with appropriate permissions
    - Use OpenAI's web browsing capability if available
    """

    # For now, provide known URLs for the major trials
    known_articles = {
        "ist3": "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(12)60768-5/fulltext",
        "crash_2": "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(10)60835-5/fulltext",
        "sprint": "https://www.nejm.org/doi/full/10.1056/NEJMoa1511939",
        "accord": "https://www.nejm.org/doi/full/10.1056/NEJMoa0802743",
    }

    trial_lower = trial_name.lower()
    if trial_lower not in known_articles:
        print(f"Warning: No known article URL for trial '{trial_name}'. Skipping article retrieval.")
        return None

    article_url = known_articles[trial_lower]

    # Prompt LLM to extract information
    # Note: In practice, you'd need to fetch the actual article content
    # This is a simplified version that relies on the LLM's training data
    extraction_system = (
        "You are a clinical research extraction assistant. Extract key information "
        "from a clinical trial article to provide context for hypothesis generation. "
        "Be accurate and cite only what is typically reported in such trials. "
        "If you don't know specific details, use 'not specified' or mark fields as null."
    )

    extraction_prompt = {
        "task": "Extract trial characteristics and results",
        "trial_name": trial_name,
        "query": query,
        "article_url": article_url,
        "instructions": [
            "Extract metadata (title, authors, journal, year, DOI)",
            "Extract trial design (sample size, intervention, control, outcomes)",
            "Extract key results including any subgroup analyses",
            "Note study limitations",
            "Explain how this context relates to ML-generated hypotheses about treatment heterogeneity",
        ],
        "note": "Use your knowledge of this published trial. Be conservative - don't invent details."
    }

    try:
        extraction = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": extraction_system},
                {"role": "user", "content": json.dumps(extraction_prompt, indent=2)},
            ],
            response_format=ArticleExtraction,
        )
        return extraction.choices[0].message.parsed
    except Exception as e:
        print(f"Error extracting article information: {e}")
        return None


def generate_feature_hypotheses(
    top_features: List[dict],
    study_context: dict,
    client: OpenAI,
    model_name: str = "gpt-4o-2024-08-06"
) -> Optional[FeatureHypothesesSet]:
    
    # 1. SHARED KNOWLEDGE BASE (Literature access for both)
    literature_grounding = (
        "Your reasoning MUST be grounded in the latest clinical literature, including:\n"
        "- Known pathophysiology and pharmacology (MoA)\n"
        "- Published results from phase III trials and meta-analyses\n"
        "- Standard-of-care clinical guidelines (e.g., ACC/AHA, ASCO, etc.)\n"
        "- Biological plausibility and pharmacokinetic principles\n"
    )

    # 2. DIRECTIONAL LOGIC
    use_data_summary = study_context.get("use_data_summary", False)
    
    if len(top_features) > 0:
        # MODE 1: WITH SHAP - Data -> Literature
        role_type = "Forensic Clinical Interpreter"
        directive = (
            "You have been provided with SHAP values from a machine learning model. "
            "Your task is to use the literature to EXPLAIN why these specific features "
            "were found to be important. Do not ignore the data in favor of generic "
            "mechanisms; justify the observed signal using science."
        )
    elif use_data_summary:
        # MODE 2: WITH DATA SUMMARY - Literature + Available Features
        role_type = "Informed Clinical Expert"
        available_cols = study_context.get("available_features", [])
        directive = (
            f"You are blinded to the model importance scores, but you know which features "
            f"were measured in this study: {available_cols}. Based on the trial context and "
            f"these available features, use the literature to PREDICT which characteristics "
            f"are most likely to modify treatment effects. Prioritize the most biologically "
            f"plausible candidates from the available features."
        )
    else:
        # MODE 3: WITHOUT SHAP - Literature Only
        role_type = "Theoretical Clinical Expert"
        directive = (
            f"You are blinded to both the model results AND the dataset features. Based ONLY on "
            f"the trial information (treatment, outcome, population), use the literature to "
            f"PREDICT which patient characteristics are most likely to modify treatment effects. "
            f"Nominate the most biologically plausible candidates based on established clinical "
            f"evidence, without reference to what was measured in the study."
        )

    # 3. ASSEMBLE SYSTEM PROMPT
    system_instructions = (
        f"You are a {role_type}.\n\n"
        f"{directive}\n\n"
        f"{literature_grounding}\n"
        "For each characteristic, provide:\n"
        "1. Clinical definition\n"
        "2. Biological mechanism (up to 3)\n"
        "3. Implied subgroups\n"
        "\nUse concrete names. Be honest about speculative vs. established links."
    )

    # 4. TASK PARAMETERS
    n_mechanisms = study_context.get("n_hypotheses_per_feature", 3)
    n_features = study_context.get("n_features", len(top_features) if top_features else 5)
    
    user_prompt = {
        "study_context": study_context,
        "observed_data": top_features if top_features else "NONE (Blinded Mode)",
        "instructions": [
            f"Generate hypotheses for {n_features} features.",
            f"Propose {n_mechanisms} mechanisms each.",
            "Cite common clinical trials or physiological laws where applicable."
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
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f"Error: {e}")
        return None

def score_feature_hypotheses(
    hypotheses: List[dict],
    study_context: dict,
    evidence: dict,
    client: OpenAI,
    model_name: str,
    article_extraction: Optional[ArticleExtraction] = None,
    is_revised: bool = False,
) -> Optional[FeatureJudgeOutput]:
    """Score feature hypotheses using an independent judge.

    Args:
        hypotheses: List of feature hypothesis dicts to score
        study_context: Study context dict
        evidence: Top feature evidence dict
        client: OpenAI client
        model_name: Model to use for judging
        article_extraction: Optional article context
        is_revised: Whether these are revised hypotheses

    Returns:
        FeatureJudgeOutput with scores, or None if error
    """
    # Use same judging criteria for both WITH and WITHOUT SHAP conditions
    judge_system = (
        "You are an independent scientific judge for mechanistic hypotheses.\n"
        "Evaluate each hypothesis on multiple dimensions using a 1-5 scale.\n"
        "Be objective, fair, and constructive.\n"
        "\n"
        "IMPORTANT: Each hypothesis includes an 'importance_rank' field (1=most important feature).\n"
        "When evaluating Evidence Alignment, consider whether the RELATIVE RANKING aligns with\n"
        "clinical knowledge. Features ranked higher should be more established effect modifiers\n"
        "according to literature. A mismatch between model ranking and clinical knowledge should\n"
        "lower the Evidence Alignment score.\n"
        "\n"
        "SCORING CRITERIA:\n"
        "\n"
        "1. Mechanism Plausibility (1-5):\n"
        "   - Biological/physiological coherence of proposed mechanisms\n"
        "   - Consistency with established pathophysiology and pharmacology\n"
        "   - Specificity to the treatment and outcome context\n"
        "   - Avoidance of generic mechanisms that could apply to any feature\n"
        "   Score 5: Strong biological basis, well-established pathways, specific to context\n"
        "   Score 3: Reasonable but speculative, some supporting literature, moderately specific\n"
        "   Score 1: Implausible, contradicts known biology, purely generic\n"
        "\n"
        "2. Evidence Alignment (1-5):\n"
        "   - Feature importance: Whether identified features are known effect modifiers from published trials\n"
        "   - Ranking accuracy: Does the feature's importance_rank align with clinical literature?\n"
        "     * Higher-ranked features (rank 1, 2, 3) should be well-established modifiers\n"
        "     * Lower-ranked features may be plausible but less critical\n"
        "     * Penalize if a weakly-supported feature ranks above a well-established one\n"
        "     * Penalize if proposed features were not measured/available in the original trial\n"
        "   - Correctness of feature selection/ranking based on existing clinical literature\n"
        "   - Consistency with meta-analyses and systematic reviews on treatment heterogeneity\n"
        "   - Mechanistic grounding: Citations and references to established treatment effect heterogeneity\n"
        "   - Connection to known biological markers and risk stratification literature\n"
        "   - Appropriate recognition when evidence is sparse or speculative\n"
        "   Score 5: Feature is a well-established modifier AND ranking position matches clinical importance\n"
        "   Score 3: Plausible modifier with some literature support, OR ranking doesn't match expected clinical priority\n"
        "   Score 1: Unlikely modifier, contradicts literature, OR inappropriate ranking (weak feature ranked too high)\n"
        "\n"
        "3. Subgroup Implications (1-5):\n"
        "   - Clarity and actionability of proposed subgroups\n"
        "   - Feasibility of defining subgroups in practice (available data, clear cutpoints)\n"
        "   - Clinical utility - would these subgroups inform treatment decisions?\n"
        "   - Avoidance of arbitrary or clinically meaningless stratifications\n"
        "   Score 5: Clear, actionable, clinically meaningful subgroups\n"
        "   Score 3: Reasonable but vague or difficult to operationalize\n"
        "   Score 1: Unclear, arbitrary, or clinically meaningless\n"
        "\n"
        "4. Caveat Awareness (1-5):\n"
        "   - Thoroughness in acknowledging limitations and alternative explanations\n"
        "   - Recognition of potential confounding, bias, measurement error\n"
        "   - Appropriate epistemic humility (avoiding overclaiming)\n"
        "   - Acknowledgment when evidence is weak or mechanisms speculative\n"
        "   Score 5: Comprehensive caveats, honest about limitations\n"
        "   Score 3: Some caveats but incomplete or superficial\n"
        "   Score 1: Overclaiming, ignoring limitations, false certainty\n"
        "\n"
        "For EACH individual mechanism, also score:\n"
        "- Plausibility (1-10): biological believability of this specific mechanism\n"
        "- Evidence support (1-10): how well clinical literature supports this mechanism\n"
        "- Specificity (1-10): how detailed and connected to clinical/biological reasoning\n"
        "- Testability (1-10): how testable/falsifiable with available data and methods\n"
        "- Overall score (1-10): holistic assessment of this mechanism\n"
        "- Brief comments: strengths, weaknesses, any concerns\n"
        "\n"
        "If original trial article context is provided, use it to:\n"
        "- Assess mechanism plausibility based on trial physiology\n"
        "- Judge interpretation accuracy against population characteristics\n"
        "- Evaluate validation feasibility given trial design\n"
        "- Cross-check if proposed mechanisms contradict known trial findings\n"
        "\n"
        "Provide honest, rigorous critique in strengths/weaknesses. Use the full 1-10 range."
    )

    judge_prompt = {
        "evidence": {
            "study_context": study_context,
        },
        "feature_hypotheses_to_score": hypotheses,
        "scoring_instructions": {
            "be_objective": True,
            "score_range": "1-5 for each dimension",
            "provide_justification": True,
        },
    }

    if is_revised:
        judge_prompt["scoring_instructions"]["note"] = "These are REVISED hypotheses after verification"

    if article_extraction is not None:
        judge_prompt["trial_article_context"] = article_extraction.model_dump()
        judge_prompt["scoring_instructions"]["use_article_context"] = (
            "Use trial article context to inform scoring, especially for "
            "mechanism plausibility and clinical interpretation accuracy."
        )

    try:
        result = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": judge_system},
                {"role": "user", "content": json.dumps(judge_prompt, indent=2)},
            ],
            response_format=FeatureJudgeOutput,
        )
        return result.choices[0].message.parsed
    except Exception as e:
        print(f"Error scoring feature hypotheses: {e}")
        return None


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
        "--n_features",
        type=int,
        default=15,
        help="Number of top Shapley features to include.",
    )
    parser.add_argument(
        "--n_hypotheses", type=int, default=8, help="How many hypotheses to generate."
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-2024-08-06",
        help="Model name supporting structured outputs (e.g., gpt-4o-2024-08-06).",
    )
    parser.add_argument(
        "--enable_verifier",
        action="store_true",
        help="Enable verification/refinement pass using the model specified by --model.",
    )
    parser.add_argument(
        "--enable_judge",
        action="store_true",
        help="Enable independent judging/scoring pass using the model specified by --model.",
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

    # Set verifier and judge models based on flags
    verifier_model = args.model if args.enable_verifier else None
    judge_model = args.model if args.enable_judge else None

    # Initialize API key and OpenAI client once
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

    evidence = load_top_features(args.shap_json, args.n_features)

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
                "Stay grounded in evidence - improve but don't add unsupported claims."
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

    # ---------- JUDGE (optional) ----------
    judge_report = None
    judge_report_revised = None

    if judge_model:
        print("Scoring feature hypotheses...")
        judge_report = score_feature_hypotheses(
            hypotheses=feature_hypotheses.model_dump()["feature_hypotheses"],
            study_context=study_context,
            evidence=evidence,
            client=client,
            model_name=judge_model,
            article_extraction=article_extraction,
            is_revised=False,
        )

        # If there are revised hypotheses, score those too
        if verification_report is not None and verification_report.revised is not None:
            print("Scoring revised feature hypotheses...")
            judge_report_revised = score_feature_hypotheses(
                hypotheses=verification_report.revised.model_dump()["feature_hypotheses"],
                study_context=study_context,
                evidence=evidence,
                client=client,
                model_name=judge_model,
                article_extraction=article_extraction,
                is_revised=True,
            )

    # ---------- WRITE OUTPUTS ----------
    ensure_out_dir(args.out_json)
    
    # Always save original hypotheses
    with open(args.out_json, "w") as f:
        json.dump(original_output.model_dump(), f, indent=2)
    print(f"Wrote original feature hypotheses to: {args.out_json}")
    
    # If refined, save revised version separately
    if verification_report is not None:
        revised_path = os.path.splitext(args.out_json)[0] + "_revised.json"
        with open(revised_path, "w") as f:
            json.dump(final_output.model_dump(), f, indent=2)
        print(f"Wrote revised feature hypotheses to: {revised_path}")
        
        # Write verifier report
        report_path = os.path.splitext(args.out_json)[0] + "_verification.json"
        with open(report_path, "w") as f:
            json.dump(verification_report.model_dump(), f, indent=2)
        print(f"Wrote verifier report to: {report_path}")

    # Write judge reports (original and revised if applicable)
    if judge_report is not None:
        judge_path = os.path.splitext(args.out_json)[0] + "_judge_original.json"
        with open(judge_path, "w") as f:
            json.dump(judge_report.model_dump(), f, indent=2)
        print(f"Wrote judge report (original) to: {judge_path}")

        if judge_report_revised is not None:
            judge_path_rev = os.path.splitext(args.out_json)[0] + "_judge_revised.json"
            with open(judge_path_rev, "w") as f:
                json.dump(judge_report_revised.model_dump(), f, indent=2)
            print(f"Wrote judge report (revised) to: {judge_path_rev}")

    # Write article extraction if available
    if article_extraction is not None:
        article_path = os.path.splitext(args.out_json)[0] + "_article_context.json"
        with open(article_path, "w") as f:
            json.dump(article_extraction.model_dump(), f, indent=2)
        print(f"Wrote article context to: {article_path}")

    if (
        args.fail_on_reject
        and verification_report is not None
        and verification_report.overall_verdict == "reject"
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
