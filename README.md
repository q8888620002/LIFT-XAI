# Explaining Conditional Average Treatment Effect

This is a repository for [CODE-XAI](https://www.medrxiv.org/content/10.1101/2024.09.04.24312866v2), explaining CATE models with attribution techniques.

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
```
