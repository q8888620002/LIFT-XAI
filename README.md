#### This is the repository for explaning Conditional Treatment Effect Model. 

Main script that runs experiments

- '''experiment_missingness.py'''

Model modifications for training with mask are in CATENets/catenets/models/ 

- pseudo_outcome_nets_mask.py (It contains abstract class for PseudoOutcomeLearner e.g. RA, DR, and PW-learner)
- torch/base_mask.py (This script contains the actual model e.g. BasicNet, PropensityNet ,and RepresentationNet.)
- torch/utils/model_utlis.py
