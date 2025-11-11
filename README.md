# LIFT-XAI: Explaining Conditional Average Treatment Effect (CATE)

This repository accompanies **[LIFT-XAI](https://www.medrxiv.org/content/10.1101/2024.09.04.24312866v2)** — a framework for **explaining Conditional Average Treatment Effect (CATE) models** using attribution and interpretability techniques.

## 1. System Requirements

### 1.1 Software dependencies
- **Operating Systems:** Linux, macOS, Windows  
- **Python:** 3.9–3.11 (tested: 3.10)  
- **Core packages**
  - `torch` (tested: 2.2.1, CUDA 12.1)
  - `numpy`, `pandas`, `scikit-learn`, `scipy`
  - `tqdm`, `matplotlib`, `seaborn`
  - CATE models are based on [CATENets](https://github.com/AliciaCurth/CATENets), which is a repo that contains Torch/Jax-based implementations of Neural Network-based Conditional Average Treatment Effect (CATE) Estimators by Alicia Curth.
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
```

## 3. Demo (Synthetic Experiment)

LIFT-XAI includes a synthetic data pipeline for demonstration and validation. Run the demo:
```bash
python run_synthetic_experiments.py \
  --num_trials 3 \
  --learner x_learner \
  --top_n_features 10
```

### Expected output
 Results are saved in ```outputs/synthetic/``` and include:
 * metrics.csv — Performance metrics (ATE, CATE, PEHE)
 * feature_importance.csv — Attribution analysis
 * plots/ — Visual summaries (effect estimation, error curves)

#### Expected runtime:
* CPU-only: 5–8 min
* GPU: 2–3 min

## 4. Running on Clinical Data
To reproduce experiments on clinical datasets, please obtain the data from the following sources (requires appropriate permissions):
* [CRASH-2](https://freebird.lshtm.ac.uk/index.php/available-trials/,)
* [IST-3](https://datashare.ed.ac.uk/handle/10283/1931)
* [ACCORD & SPRINT](https://biolincc.nhlbi.nih.gov/home/)

### Clinical experiment script
```run_experiment_clinical_data.py```performs ensemble explanations with knowledge distillation.. An example command is as follows
```
run_experiment_clinical_data.py
--dataset          # dataset name
--shuffle          # whether to shuffle data, only active for training set
--num_trials       # number of ensemble models
--learner          # types of CATE learner, e.g. X-Learner, DR-Learner
--top_n_features   # whether to report top n features across runs.
```
## Bibltex
if you find this project useful in your research, please consider citing our paper
```
@article{liftxai2024,
  title   = {LIFT-XAI: Explaining Conditional Average Treatment Effect Models with Attribution Techniques},
  year    = {2024},
  note    = {medRxiv preprint},
  url     = {https://www.medrxiv.org/content/10.1101/2024.09.04.24312866v2}
}
```
