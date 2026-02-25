import os
import re
import json
from typing import Optional

from src.agent_schemas import ArticleExtraction


COMMON_FEATURE_LABEL_MAP = {
    "age": "Age",
    "sbp": "Systolic blood pressure",
    "dbp": "Diastolic blood pressure",
    "egfr": "Estimated glomerular filtration rate",
    "screat": "Serum creatinine",
    "creat": "Serum creatinine",
    "bmi": "Body mass index",
    "hr": "Heart rate",
    "sex": "Sex",
    "male": "Male sex",
    "female": "Female sex",
    "uacr": "Urine albumin-to-creatinine ratio",
    "hba1c": "Hemoglobin A1c",
    "ldl": "LDL cholesterol",
    "hdl": "HDL cholesterol",
    "tg": "Triglycerides",
    "chr": "Total cholesterol / HDL ratio",
}


DATASET_FEATURE_LABEL_MAP = {
    "crash_2": {
        "ninjurytime": "Time from injury to treatment",
        "injurytime": "Time from injury to treatment",
        "icc": "Injury classification code",
    },
    "ist3": {
        "dbprand": "Randomization/baseline diastolic BP variable",
    },
}


def map_feature_label(raw_feature: str, dataset: Optional[str] = None) -> str:
    """Map raw feature names to clinician-readable labels while preserving raw codes."""
    if not raw_feature:
        return raw_feature

    raw = str(raw_feature).strip()
    key = raw.lower()
    dataset_key = (dataset or "").lower()

    mapped = None
    if dataset_key in DATASET_FEATURE_LABEL_MAP:
        mapped = DATASET_FEATURE_LABEL_MAP[dataset_key].get(key)
    if not mapped:
        mapped = COMMON_FEATURE_LABEL_MAP.get(key)

    if not mapped:
        if "_" in raw:
            humanized = raw.replace("_", " ").strip()
            humanized = re.sub(r"\s+", " ", humanized)
            if humanized and humanized.lower() != key:
                mapped = humanized[:1].upper() + humanized[1:]

    if mapped and mapped.lower() != key:
        return f"{mapped} ({raw})"
    return raw


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


def ensure_out_dir(path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def load_json_file(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def write_json_file(path: str, payload: dict) -> None:
    ensure_out_dir(path)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def infer_revised_path(original_path: str) -> Optional[str]:
    base, ext = os.path.splitext(original_path)
    candidate = f"{base}_revised{ext}"
    return candidate if os.path.exists(candidate) else None


def resolve_seeded_output_path(path: str, seed: int) -> str:
    """Insert seed subfolder into output path unless one already exists."""
    normalized = path.replace("\\", "/")
    directory, filename = os.path.split(normalized)

    directory_parts = [part for part in directory.split("/") if part]
    if any(part.startswith("seed_") for part in directory_parts):
        return path

    seeded_directory = os.path.join(directory, f"seed_{seed}") if directory else f"seed_{seed}"
    return os.path.join(seeded_directory, filename)


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


def get_model_client(api_provider: str, api_base_url: Optional[str] = None):
    """Create an OpenAI-compatible client for OpenAI or OpenRouter."""
    from openai import OpenAI

    if api_provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY.")
        base_url = api_base_url or "https://openrouter.ai/api/v1"
        print(f"Using OpenRouter API with base URL: {base_url}")
        return OpenAI(api_key=api_key, base_url=base_url)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            from src.constants import openai_api_key

            api_key = openai_api_key
        except ImportError:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY or define src/constants.py:openai_api_key"
            )

    if api_base_url:
        print(f"Using custom OpenAI-compatible API with base URL: {api_base_url}")
        return OpenAI(api_key=api_key, base_url=api_base_url)

    print("Using OpenAI API")
    return OpenAI(api_key=api_key)


def load_top_features(
    shap_json_path: str, n_features: int, dataset_override: str = None
) -> dict:
    with open(shap_json_path, "r") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    explainer = meta.get("explainer", "unknown_explainer")
    dataset = dataset_override if dataset_override else meta.get("dataset", "unknown_dataset")
    learner = meta.get("learner", "unknown_learner")

    features = data.get("features", [])

    all_feature_names = [
        map_feature_label(f.get("feature"), dataset) for f in features if f.get("feature")
    ]

    if not features:
        return {
            "dataset": dataset,
            "learner": learner,
            "explainer": "baseline_no_shap",
            "top_feature_evidence": [],
            "available_features": [],
        }

    features_sorted = sorted(
        features, key=lambda x: float(x.get("shap_mean_abs", 0.0)), reverse=True
    )
    top = features_sorted[:n_features]

    top_evidence = [
        {
            "feature": map_feature_label(f.get("feature"), dataset),
            "feature_raw": f.get("feature"),
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


def search_and_extract_article(
    query: str,
    trial_name: str,
    client,
    model_name: str = "gpt-4o-2024-08-06",
) -> Optional[ArticleExtraction]:
    """Search for trial article and extract key information."""

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
        "note": "Use your knowledge of this published trial. Be conservative - don't invent details.",
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