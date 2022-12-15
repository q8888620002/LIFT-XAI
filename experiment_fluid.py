import numpy as np
import pandas as pd
import shap
import os
import sys
import collections

from sklearn.impute import SimpleImputer
from sklearn import preprocessing

module_path = os.path.abspath(os.path.join('CATENets/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from catenets.models.jax import TNet, SNet,SNet1, SNet2, SNet3, DRNet, RANet, PWNet, RNet, XNet


#### filtering out procedure

fluid_cohort = pd.read_pickle("data/low_bp_responder.pkl")

fluid_cohort = fluid_cohort[fluid_cohort.columns.drop(list(fluid_cohort.filter(regex='proc')))]
x_train = fluid_cohort.loc[:, ~fluid_cohort.columns.isin(["registryid","treated","outcome"])]

### normalize x_train 
#x = x_train.values 

min_max_scaler = preprocessing.MinMaxScaler()
x_train_scaled = min_max_scaler.fit_transform(x_train)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(x_train_scaled)
x_train_scaled = imp.transform(x_train_scaled)

n = x_train.shape[0]

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

models = [TNet(), DRNet(), RNet(),RANet(), PWNet(), XNet(), SNet() , SNet1(), SNet2(), SNet3()]

#### Getting top 10 features from multiple runs. 

names = x_train.columns
seeds = np.arange(0, 6, 0.5, dtype=int)

for i in seeds:
    np.random.seed(i)

    for model in models:
        model.fit(X_train, y_train, w_train)
        model_lam = lambda x: model.predict(x)

        explainer = shap.Explainer(model_lam, X_train)
        #### showing explanation on cate
        shap_values = explainer(X_test)
        ind = np.argpartition(np.abs(shap_values.values).mean(0).round(2), -10)[-10:]
        top_k_results.extend(names[ind].tolist())

### TODO: calcualte CIs across different seeds

results = collections.Counter(top_k_results)
summary = pd.DataFrame(results.items(), columns=['feature', 'count (%)']).sort_values(by="count (%)", ascending=False)
summary["count (%)"] = np.round(summary["count (%)"]/(len(models)*len(seeds)),2)*100

summary.to_csv("results/top_10_fatures_responder.csv")