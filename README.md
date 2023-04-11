# Explaining Conditional Average Treatment Effect 

Main script that generates synthetic dataset and test spearman correlation between Kernel SHAP and shapley-regression with masks.

- experiment_missingness.py

```python  experiment_missingness.py -n number_of_samples -d feature_dimension -r random_seed```

[CATENets](https://github.com/AliciaCurth/CATENets) is a repo contains Torch/Jax-based, sklearn-style implementations of Neural Network-based Conditional Average Treatment Effect (CATE) Estimators by Alicia Curth. 

Model modifications for explanation with mask are in ```CATENets/catenets/models/torch``` 

- ```pseudo_outcome_nets.py``` It contains abstract class for PseudoOutcomeLearner e.g. RA, DR, and PW-learner and Learner with Masks e.g. DRLearnerMask.
- ```base.py``` This script contains the prediction model e.g. propensity score model (PropensityNet), treatment effect (BasicNet) and their masks version. 
- ```utils/model_utlis.py```

Shapley Value Calculation is in ```shapley-regression/shapreg```
- ```CateGame() in games.py```
