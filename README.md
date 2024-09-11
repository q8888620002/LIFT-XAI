# Explaining Conditional Average Treatment Effect 

This is a repository for CODE-XAI, explaining CATE models with attribution techniques. 

```run_experiment_clinical_data```contains experiments for examining ensemble explanations with knowledge distillation. An example command is as follow
```
run_experiment_clinical_data.py 
--shuffle          # whether to shuffle data, only active for training set
--num_trials       # number of ensemble models
--learner          # types of CATE learner
--top_n_features   # whether to report top n features across runs.

``` 




[CATENets](https://github.com/AliciaCurth/CATENets) is a repo contains Torch/Jax-based, sklearn-style implementations of Neural Network-based Conditional Average Treatment Effect (CATE) Estimators by Alicia Curth. 

Model modifications for explanation with mask are in ```CATENets/catenets/models/torch``` 

- ```pseudo_outcome_nets.py``` It contains abstract class for PseudoOutcomeLearner e.g. RA, DR, and PW-learner and Learner with Masks e.g. DRLearnerMask.
- ```base.py``` This script contains the prediction model e.g. propensity score model (PropensityNet), treatment effect (BasicNet) and their masks version. 
- ```utils/model_utlis.py```

Shapley Value Calculation is in ```shapley-regression/shapreg```
- ```CateGame() in games.py```
