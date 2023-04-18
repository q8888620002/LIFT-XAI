import numpy as np
import pandas as pd

import os
import sys
import collections
import pickle
import torch

from scipy import stats
from shapreg import shapley, games, removal, shapley_sampling
from sklearn.impute import SimpleImputer
from sklearn import preprocessing, model_selection

from captum.attr import (
    DeepLift,
    FeatureAblation,
    FeaturePermutation,
    IntegratedGradients,
    KernelShap,
    Lime,
    ShapleyValueSampling,
    GradientShap,
)

module_path = os.path.abspath(os.path.join('CATENets/'))
if module_path not in sys.path:
    sys.path.append(module_path)

import catenets.models.torch.pseudo_outcome_nets as pseudo_outcome_nets


#### filtering out procedure



def normalize_data(X_train):
    
    X_normalized_train = (X_train - np.min(X_train, axis=0)) / (np.max(X_train, axis=0) - np.min(X_train, axis=0))

    return X_normalized_train


fluid_cohort = pd.read_pickle("data/low_bp_survival.pkl")

#
fluid_cohort = fluid_cohort[fluid_cohort.columns.drop(list(fluid_cohort.filter(regex='proc')))]
fluid_cohort = fluid_cohort[fluid_cohort.columns.drop(list(fluid_cohort.filter(regex='ethnicity')))]
fluid_cohort = fluid_cohort[fluid_cohort.columns.drop(list(fluid_cohort.filter(regex='residencestate')))]
fluid_cohort = fluid_cohort[fluid_cohort.columns.drop(list(fluid_cohort.filter(regex='toxicologyresults')))]


x = fluid_cohort.loc[:, ~fluid_cohort.columns.isin(["registryid",
                                                    "COV",
                                                    "TT",
                                                    "scenegcsmotor",
                                                    "scenegcseye",
                                                    "scenegcsverbal",
                                                    "edgcsmotor",
                                                    "edgcseye",
                                                    "edgcsverbal",
                                                    "outcome",
                                                    "sex_F",
                                                    "traumatype_P",
                                                    "traumatype_other"
                                                ])
                    ]

n, feature_size = x.shape
names = x.drop(["treated"], axis=1).columns
treatment_index = x.columns.get_loc("treated")
sex_index = x.columns.get_loc("sex_M")

var_index = [i for i in range(feature_size) if i != treatment_index]

x_norm = normalize_data(x)

## impute missing value

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(x_norm)
x_train_scaled = imp.transform(x_norm)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
                                             x_train_scaled,  
                                             fluid_cohort["outcome"], 
                                             test_size=0.2, 
                                             random_state=10,
                                             stratify=fluid_cohort["treated"]
                                    )

w_train = X_train[:, treatment_index]
w_test =  X_test[:, treatment_index]
X_train = X_train[:,var_index]
X_test = X_test[:, var_index]


device = "cpu"

learner_explanations = {}
explainers = ["ig", "shapley_sampling_0", "shap"]
#### Getting top 10 features from multiple runs. 
top_k_results = {"ig":[],
                 "shapley_sampling_0":[],
                 "shap":[]}

trials = 5
results_train = np.zeros((trials, len(X_train)))
results_test = np.zeros((trials, len(X_test)))
results_sign = np.zeros((trials,feature_size))

for i in range(trials):
    model = pseudo_outcome_nets.XLearner(  
                                            X_train.shape[1],
                                            binary_y=(len(np.unique(y_train)) == 2),
                                            n_layers_out=2,
                                            n_units_out=100,
                                            batch_size=128,
                                            n_iter=3000,
                                            nonlin="relu",
                                            device=device,
                                            )

    model.fit(X_train, y_train, w_train)
    
    results_train[i] = model(X_train).detach().cpu().numpy().flatten()
    results_test[i] = model(X_test).detach().cpu().numpy().flatten()

    ig = IntegratedGradients(model)
    print("explaining with IG")

    learner_explanations["ig"] = ig.attribute(
                                        torch.from_numpy(X_test).to(device).requires_grad_(),
                                        n_steps=500,
                                ).detach().cpu().numpy()
    print("explaining with shapley sampling -0")
    shapley_value_sampling_model = ShapleyValueSampling(model)
    learner_explanations["shapley_sampling_0"] = shapley_value_sampling_model.attribute(
                                                    torch.from_numpy(X_test).to(device).requires_grad_(),
                                                    n_samples=500,
                                                    perturbations_per_eval=10,
                                                ).detach().cpu().numpy()
    print("explaining with shapley sampling - marginal distribution")

    marginal_extension = removal.MarginalExtension(X_test, model)
    shap_values = np.zeros((X_test.shape))

    for test_ind in range(len(X_test)):
        instance = X_test[test_ind]
        game = games.PredictionGame(marginal_extension, instance)
        explanation = shapley_sampling.ShapleySampling(game, thresh=0.01, batch_size=128)
        shap_values[test_ind] = explanation.values.reshape(-1, X_test.shape[1])
    
    learner_explanations["shap"] = shap_values

    for e in explainers:
        ind = np.argpartition(np.abs(learner_explanations[e]).mean(0).round(2), -5)[-5:]
        top_k_results[e].extend(names[ind].tolist())

for e in explainers:

    results = collections.Counter(top_k_results[e])
    summary = pd.DataFrame(results.items(), columns=['feature', 'count (%)']).sort_values(by="count (%)", ascending=False)
    summary["count (%)"] = np.round(summary["count (%)"]/trials,2)*100
    summary.to_csv("results/massive_trans/"+ e+ "_top_5_fatures_mass_xlearner.csv")

with open(r"results/massive_trans/result_train_xlearner.pkl", "wb") as output_file:    
     pickle.dump(results_train, output_file)

with open(r"results/massive_trans/result_test_xlearner.pkl", "wb") as output_file:
    pickle.dump(results_test, output_file)