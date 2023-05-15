import numpy as np
import pandas as pd
import shap
import os
import sys
import collections
import pickle

from sklearn.impute import SimpleImputer
from shapreg import shapley, games, removal, shapley_sampling
from sklearn import preprocessing, model_selection
from scipy import stats
from utilities import normalize_data

module_path = os.path.abspath(os.path.join('CATENets/'))
if module_path not in sys.path:
    sys.path.append(module_path)

import catenets.models.torch.pseudo_outcome_nets as pseudo_outcome_nets

ist3 = pd.read_sas("data/datashare_aug2015.sas7bdat")

continuous_vars = [
                    "gender",
                    "age",
                    "weight",
                    "glucose",
                    "gcs_eye_rand",
                    "gcs_motor_rand",
                    "gcs_verbal_rand",
                    # "gcs_score_rand",   
                     "nihss" ,
                     "sbprand",
                     "dbprand",
                  ]

cate_variables = [
                     # "livealone_rand",
                     # "indepinadl_rand",
                     "infarct",
                     "antiplat_rand",
                     # "atrialfib_rand",
                    #  "liftarms_rand",
                    # "ablewalk_rand",
                    # "weakface_rand",
                    # "weakarm_rand",
                    # "weakleg_rand",
                    # "dysphasia_rand",
                    # "hemianopia_rand",
                    # "visuospat_rand",
                    # "brainstemsigns_rand",
                    # "otherdeficit_rand",
                    "stroketype"
                 ]

outcomes = ["dead7","dead6mo","aliveind6"]
treatment = ["itt_treat"]


### normalize x_train 

x = ist3[continuous_vars + cate_variables + treatment]

x = pd.get_dummies(x, columns=cate_variables)

n, feature_size = x.shape

names = x.drop(["itt_treat"], axis=1).columns
treatment_index = x.columns.get_loc("itt_treat")
var_index = [i for i in range(feature_size) if i != treatment_index]
feature_size = len(var_index)

x_norm, x_min = normalize_data(x)

## impute missing value

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(x_norm)
x_train_scaled = imp.transform(x_norm)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
                                             x_train_scaled,  
                                             ist3["aliveind6"], 
                                             test_size=0.2, 
                                             random_state=10,
                                    )


w_train = X_train[:, treatment_index] == 0
w_test =  X_test[:, treatment_index] == 0

X_train = X_train[:,var_index]
X_test = X_test[:, var_index]

y_train = y_train ==0
y_test = y_test ==0

top_k_results = []

model = pseudo_outcome_nets.XLearner(
                                        X_train.shape[1],
                                        binary_y=(len(np.unique(y_train)) == 2),
                                        n_layers_out=2,
                                        n_units_out=100,
                                        batch_size=128,
                                        nonlin="relu",
                                        device="cuda"
                                        )

#### Getting top 10 features from multiple runs. 

seeds = np.arange(0, 6, 1, dtype=int)

results_sign = np.zeros((len(seeds),feature_size))
results_train = np.zeros((len(seeds), len(X_train)))
results_test = np.zeros((len(seeds), len(X_test)))

for i, seed in enumerate(seeds):
    np.random.seed(seed)

    model.fit(X_train, y_train, w_train)
    
    results_train[i] = model(X_train).detach().cpu().numpy().flatten()
    results_test[i] = model(X_test).detach().cpu().numpy().flatten()

    #### showing explanation on cate

    shap_values = np.zeros((X_test.shape))
    marginal_extension = removal.MarginalExtension(X_test, model)

    for test_ind in range(len(X_test)):
        instance = X_test[test_ind]
        game = games.PredictionGame(marginal_extension, instance)
        explanation = shapley_sampling.ShapleySampling(game, thresh=0.03, batch_size=128)
        shap_values[test_ind] = explanation.values.reshape(-1, X_test.shape[1])


    for col in range(feature_size):
        results_sign[i, col] = stats.pearsonr(X_test[:,col], shap_values[:, col])[0]

    ind = np.argpartition(np.abs(shap_values).mean(0).round(2), -10)[-10:]
    top_k_results.extend(names[ind].tolist())


results = collections.Counter(top_k_results)
summary = pd.DataFrame(results.items(), columns=['feature', 'count (%)']).sort_values(by="count (%)", ascending=False)
summary["count (%)"] = np.round(summary["count (%)"]/len(seeds),2)*100

indices = [names.tolist().index(i) for i in summary.feature.tolist()]
summary["sign"] = np.sign(np.mean(results_sign, axis=(0))[indices])

summary.to_csv("results/ist3/top_10_features_xlearner.csv")

with open( f"results/ist3/train_xlearner.pkl", "wb") as output_file:    
    pickle.dump(results_train, output_file)

with open( f"results/ist3/test_xlearner.pkl", "wb") as output_file:    
    pickle.dump(results_test, output_file)
