# Explaining Conditional Average Treatment Effect

This repository contains code for [CODE-XAI](https://www.medrxiv.org/content/10.1101/2024.09.04.24312866v2), explaining CATE models with attribution techniques and downstream hypothesis validation workflows.

## Prerequisites

Core CATE models are based on [CATENets](https://github.com/AliciaCurth/CATENets), which provides Torch/Jax-based sklearn-style CATE estimators.

## Core Scripts

### `single_cohort_analysis.py`

Computes SHAP values for CATE models on a single cohort using bootstrapped trials and exports JSON summaries compatible with `clinical_agent.py`.

Example:

```bash
python single_cohort_analysis.py \
    --num_trials 20 \
    --cohort_name crash_2 \
    --baseline \
    --wandb \
    --relative_change_threshold 0.05 \
    --top_n_features 15
```

### `clinical_agent.py`

Generates clinical mechanism hypotheses from SHAP summaries.

Example:

```bash
python clinical_agent.py \
    --shap_json results/crash_2/shapley/crash_2_shap_summary_True.json \
    --out_json docs/agent/crash_2/hypotheses_with_shap_XLearner.json \
    --trial_name crash_2 \
    --n_features 15 \
    --n_hypotheses 8
```

### `run_experiment_clinical_data.py`

Runs ensemble explanation experiments with knowledge distillation.

Example:

```bash
python run_experiment_clinical_data.py \
    --dataset crash_2 \
    --shuffle \
    --num_trials 10 \
    --learner XLearner \
    --top_n_features 10
```

### `summarize_feature_scores.py`

Summarizes and visualizes feature scores from clinical agent outputs.

## PubMed Mechanism Validator

`pubmed_mechanism_validator.py` validates hypothesis mechanisms against PubMed literature by:

1. Searching PubMed for relevant abstracts
2. Classifying abstracts as support/conflict/neutral
3. Producing summary and detailed JSON reports

### Installation

```bash
pip install -r pubmed_requirements.txt
```

### Basic Usage

```bash
python pubmed_mechanism_validator.py --cohort ist3
python pubmed_mechanism_validator.py --cohort accord
python pubmed_mechanism_validator.py --cohort crash_2
python pubmed_mechanism_validator.py --cohort sprint
```

### Advanced Usage

```bash
# Custom input file
python pubmed_mechanism_validator.py --input docs/agent/ist3/hypotheses_with_shap_XLearner.json

# Custom output
python pubmed_mechanism_validator.py --cohort ist3 --output my_validation.json

# LLM analysis (reads OPENAI_API_KEY from environment or .env)
python pubmed_mechanism_validator.py --cohort ist3

# Explicit API key override
python pubmed_mechanism_validator.py --cohort ist3 --api-key "your-api-key-here"

# Keyword-only mode
python pubmed_mechanism_validator.py --cohort ist3 --no-llm

# More abstracts per mechanism
python pubmed_mechanism_validator.py --cohort ist3 --max-abstracts 50
```

### Analysis Modes

- **LLM-based analysis** (recommended): more nuanced support/conflict classification.
- **Keyword-based analysis**: no API key required, faster but less precise.

### Output

Outputs `<input_basename>_pubmed_validation.json` (unless `--output` is provided), containing:

- dataset-level totals (`overall_support_count`, `overall_conflict_count`, `overall_neutral_count`)
- per-mechanism query + abstract counts
- per-abstract stance/reasoning

### Best Practices

1. Use LLM mode for final reporting.
2. Use keyword mode for quick screening.
3. Tune `--max-abstracts` for depth vs speed.
4. Review constructed queries when retrieval quality is low.

### Troubleshooting

- **No abstracts found**: check query specificity and internet access.
- **LLM errors / auth failures**: verify `OPENAI_API_KEY` (or `--api-key`) and account status.
- **Too many neutral results**: increase query specificity or mechanism detail.
