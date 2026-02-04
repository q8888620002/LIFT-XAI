# Explaining Conditional Average Treatment Effect

This is a repository for [CODE-XAI](https://www.medrxiv.org/content/10.1101/2024.09.04.24312866v2), explaining CATE models with attribution techniques.

<<<<<<< HEAD
## Prerequisites

CATE models are based on [CATENets](https://github.com/AliciaCurth/CATENets), which is a repo that contains Torch/Jax-based, sklearn-style implementations of Neural Network-based Conditional Average Treatment Effect (CATE) Estimators by Alicia Curth.

## Scripts

### `single_cohort_analysis.py`

Computes SHAP values for CATE models on a single cohort/dataset using XLearner. The script performs bootstrapped SHAP computation across multiple trials and generates a JSON summary compatible with the clinical agent for hypothesis generation.

**Example command:**
```bash
python single_cohort_analysis.py \
    --num_trials 20 \
    --cohort_name crash_2 \
    --baseline \
    --wandb \
    --relative_change_threshold 0.05 \
    --top_n_features 15
```

**Arguments:**
- `--num_trials` (required): Number of bootstrap trials to run for SHAP computation
- `--cohort_name` (required): Name of the dataset (e.g., crash_2, ist3, sprint, accord)
- `--baseline`: Use random sample baseline (default: True). If not set, uses median baseline
- `--wandb`: Enable Weights & Biases logging (default: True)
- `--relative_change_threshold`: Threshold for early stopping based on SHAP convergence (default: 0.05)
- `--top_n_features`: Number of top features to include in summary (default: 10)

**Outputs:**
- `{cohort_name}_shap_summary_{baseline}.json`: JSON summary with SHAP statistics, compatible with clinical_agent.py
- `{cohort_name}_shap_bootstrapped_{baseline}.pkl`: Raw SHAP values across all trials
- `{cohort_name}_predict_results_{baseline}.pkl`: CATE predictions across all trials

### `run_experiments.py`

Contains an experiment pipeline for synthetics data analysis, the script is modified based on

### `run_experiment_clinical_data.py`

Contains experiments for examining ensemble explanations with knowledge distillation.

**Example command:**
```bash
python run_experiment_clinical_data.py \
    --dataset crash_2 \
    --shuffle \
    --num_trials 10 \
    --learner XLearner \
    --top_n_features 10
```

**Arguments:**
- `--dataset`: Dataset name
- `--shuffle`: Whether to shuffle data, only active for training set
- `--num_trials`: Number of ensemble models
- `--learner`: Types of CATE learner, e.g. X-Learner, DR-Learner
- `--top_n_features`: Whether to report top n features across runs

### `clinical_agent.py`

Generate clinical research hypotheses from SHAP summary JSON using OpenAI's API. Requires SHAP summary output from `single_cohort_analysis.py`.

**Example command:**
```bash
python clinical_agent.py \
    --shap_json results/crash_2/shapley/crash_2_shap_summary_True.json \
    --out_json results/crash_2/hypotheses_baseline_shapley_XLearner.json \
    --trial_name crash_2 \
    --n_features 15 \
    --n_hypotheses 8
```

### `summarize_feature_scores.py`

Summarizes and visualizes feature scores from clinical agent hypothesis outputs.

**Example command:**
```bash
python summarize_feature_scores.py \
    --judge_json results/agents/hypotheses_baseline_shapley_XLearner_judge_original.json \
    --out_csv results/agents/feature_scores_summary.csv \
    --plot \
    --out_plot results/agents/feature_scores.png
=======
Prerequisites

CATE models are based on [CATENets](https://github.com/AliciaCurth/CATENets), which is a repo that contains Torch/Jax-based, sklearn-style implementations of Neural Network-based Conditional Average Treatment Effect (CATE) Estimators by Alicia Curth.

```run_experiments.py``` contains an experiment pipeline for synthetics data analysis, the script is modified based on

```run_experiment_clinical_data.py```contains experiments for examining ensemble explanations with knowledge distillation. An example command is as follows
```
run_experiment_clinical_data.py
--dataset          # dataset name
--shuffle          # whether to shuffle data, only active for training set
--num_trials       # number of ensemble models
--learner          # types of CATE learner, e.g. X-Learner, DR-Learner
--top_n_features   # whether to report top n features across runs.
>>>>>>> a0ff67ef55955080ceda52732dad9b5ee4c1c750
```
