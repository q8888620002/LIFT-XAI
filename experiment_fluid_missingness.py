import numpy as np
import pandas as pd
import shap
import os
import sys
import collections
import torch

from scipy import stats
from shapreg import shapley, games
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

module_path = os.path.abspath(os.path.join('CATENets/'))
if module_path not in sys.path:
    sys.path.append(module_path)

import catenets.models.torch.pseudo_outcome_nets as cate_models_masks


#### filtering out procedure

fluid_cohort = pd.read_pickle("data/low_bp_survival.pkl")
#fluid_cohort = pd.read_pickle("data/trauma_team_activated.pkl")
#fluid_cohort = pd.read_pickle("data/trauma_responder.pkl")

#
fluid_cohort = fluid_cohort[fluid_cohort.columns.drop(list(fluid_cohort.filter(regex='proc')))]
fluid_cohort = fluid_cohort[fluid_cohort.columns.drop(list(fluid_cohort.filter(regex='toxicologyresults')))]

x_train = fluid_cohort.loc[:, ~fluid_cohort.columns.isin(["registryid",
                                                            "treated",
                                                            "COV",
                                                            "TT",
                                                            "scenegcsmotor",
                                                            "scenegcseye",
                                                            "scenegcsverbal",
                                                            "edgcsmotor",
                                                            "edgcseye",
                                                            "edgcsverbal",
                                                            "outcome",
                                                            ])]

### normalize x_train 
#x = x_train.values 

min_max_scaler = preprocessing.MinMaxScaler()
x_train_scaled = min_max_scaler.fit_transform(x_train)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(x_train_scaled)
x_train_scaled = imp.transform(x_train_scaled)

n, feature_size = x_train.shape
names = x_train.columns
device = "cuda:0"
rng = np.random.default_rng(0)
inds = np.arange(n)
rng.shuffle(inds)

n_train = int(0.8 * n)

train_inds = inds[:n_train]
test_inds = inds[n_train:]

X_train, X_test = x_train_scaled[train_inds], x_train_scaled[test_inds]
w_train, w_test = fluid_cohort["treated"].values[train_inds], fluid_cohort["treated"].values[test_inds]
y_train, y_test = fluid_cohort["outcome"].values[train_inds], fluid_cohort["outcome"].values[test_inds]

top_k_results = []

#### Getting top 10 features from multiple runs. 

seeds = np.arange(0, 6, 0.5, dtype=int)
results_sign = np.zeros((len(seeds), feature_size))

for i, seed in enumerate(seeds):
    np.random.seed(seed)
    model = cate_models_masks.DRLearnerMask(  
                                            X_train.shape[1],
                                            binary_y=(len(np.unique(y_train)) == 2),
                                            n_layers_out=2,
                                            n_units_out=100,
                                            batch_size=128,
                                            n_iter=3000,
                                            nonlin="relu",
                                            device=device
                                            )

    model.fit(X_train, y_train, w_train)
    feature_values = np.zeros((X_test.shape))

    for test_ind in range(len(X_test)):

        instance = torch.from_numpy(X_test[test_ind, :])[None,:].to(device)
        game  = games.CateGame(instance, model)
        explanation = shapley.ShapleyRegression(game, batch_size=128)
        feature_values[test_ind] = explanation.values

    for col in range(feature_size):
        results_sign[i, col] = stats.pearsonr(X_test[:,col], feature_values[:, col])[0]
        
    ind = np.argpartition(np.abs(feature_values).mean(0).round(2), -10)[-10:]
    top_k_results.extend(names[ind].tolist())

### TODO: calcualte CIs across different seeds

results = collections.Counter(top_k_results)
summary = pd.DataFrame(results.items(), columns=['feature', 'count (%)']).sort_values(by="count (%)", ascending=False)
summary["count (%)"] = np.round(summary["count (%)"]/len(seeds),2)*100

indices = [names.tolist().index(i) for i in summary.feature.tolist()]
summary["sign"] = np.sign(np.mean(results_sign,axis=0)[indices])

summary.to_csv("results/trauma_top_10_fatures_survival_mask.csv")