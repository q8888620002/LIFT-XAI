# Explaining Conditional Average Treatment Effect

This is a repository for [LIFT-XAI](https://www.medrxiv.org/content/10.1101/2024.09.04.24312866v2), explaining CATE models with attribution techniques.



## 1. System Requirements

### 1.1 Software dependencies
- **Operating Systems:** Linux, macOS, Windows  
- **Python:** 3.9–3.11 (tested: 3.10)  
- **Core packages**
  - `torch` (tested: 2.2.1, CUDA 12.1)
  - `numpy`, `pandas`, `scikit-learn`, `scipy`
  - `tqdm`, `matplotlib`, `seaborn`
- **Optional**
  - CUDA toolkit for GPU acceleration

> Full environment definitions: `requirements.txt` and `environment.yml`

### 1.2 Versions tested on
- Ubuntu 22.04, macOS 14  
- Python 3.10  
- PyTorch 2.2.1 (CUDA 12.1)

### 1.3 Non-standard hardware
- **None required**  
- Optional: NVIDIA GPU (≥ 8 GB VRAM) for faster training/inference

---

## 2. Installation Guide

**Typical install time:** 5–10 minutes on a normal desktop computer.

### Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate liftxai

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
