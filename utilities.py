from __future__ import annotations
from typing import List

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn import  model_selection
from sklearn import metrics

def normalize_data(x):

    x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))

    return x

def subgroup_identification(
        rank_indices: np.ndarray,
        x_train: np.ndarray,
        x_test: np.ndarray,
        cate_model: torch.nn.Module
) -> tuple:


    train_pred = cate_model.predict(X=x_train).detach().cpu().numpy()
    test_pred = cate_model.predict(X=x_test).detach().cpu().numpy()

    threshold  = np.mean(train_pred)

    y_true_train = (train_pred > threshold)
    y_true_test = (test_pred > threshold)

    xgb_model = xgb.XGBClassifier()


    xgb_model.fit(x_train[:, rank_indices], y_true_train)

    y_pred = xgb_model.predict(x_test[:, rank_indices])

    ate = np.sum(test_pred[y_pred == 1])/len(test_pred)

    auroc = metrics.roc_auc_score(y_true_test, y_pred)


    return ate, auroc

def insertion_deletion(
    data: Dataset,
    rank_indices:list,
    cate_model: torch.nn.Module,
    x_replacement: np.ndarray,
    selection_types: List[str],
    model_type: str ="CATENets",
    device: str = "cuda:3"
) -> tuple:
    """
    Compute partial average treatment effect (PATE) with feature subsets by insertion and deletion

    Args:
        x_test: testing set for explanation with insertion and deletion
        feature_attributions: feature attribution outputted by a feature importance method
        pate_model: masking models for PATE estimation.
    Returns:
        results of insertion and deletion of PATE.
    """
    ## training plugin estimator on
    x_test, _, _ = data.get_testing_data()

    n, d = x_test.shape
    x_test_del =x_test.copy()
    x_test_ins = np.tile(x_replacement, (n, 1))

    original_cate_model = cate_model

    if model_type == "CATENets":
        cate_model = lambda x: original_cate_model.predict(X=x)
    else:
        cate_model = lambda x: original_cate_model.forward(X=x)

    deletion_results = {selection_type: np.zeros(d+1) for selection_type in selection_types}
    insertion_results = {selection_type: np.zeros(d+1) for selection_type in selection_types}

    for rank_index in range(len(rank_indices) + 1):
        if rank_index > 0:  # Skip this on the first iteration
            col_indices = rank_indices[rank_index - 1]
            x_test_ins[:, col_indices] = x_test[:, col_indices]
            x_test_del[:, col_indices] = x_replacement[col_indices]

        for selection_type in selection_types:
            # For the insertion process
            cate_pred_subset_ins = cate_model(x_test_ins).detach().cpu().numpy().flatten()
            insertion_results[selection_type][rank_index] = calculate_pehe(
                cate_pred_subset_ins,
                data,
                selection_type
            )

            # For the deletion process
            cate_pred_subset_del = cate_model(x_test_del).detach().cpu().numpy().flatten()
            deletion_results[selection_type][rank_index] = calculate_pehe(
                cate_pred_subset_del,
                data,
                selection_type
            )

    return insertion_results, deletion_results

def train_nuisance_models(
    x_val: np.ndarray,
    y_val: np.ndarray,
    w_val: np.ndarray
)-> tuple:

    mu0 = xgb.XGBClassifier()
    mu1 = xgb.XGBClassifier()
    m = xgb.XGBClassifier()
    rf = xgb.XGBClassifier(
        reg_lambda=2,
        max_depth=3,
        colsample_bytree=0.2,
        min_split_loss=10
    )

    x0, x1 = x_val[w_val == 0], x_val[w_val == 1]
    y0, y1 = y_val[w_val == 0], y_val[w_val == 1]

    mu0.fit(x0, y0)
    mu1.fit(x1, y1)
    m.fit(x_val, y_val)
    rf.fit(x_val, w_val)

    return mu0, mu1, rf, m

def calculate_if_pehe(
    w_test: np.ndarray,
    p: np.ndarray,
    prediction: np.ndarray,
    t_plugin: np.ndarray,
    y_test: np.ndarray,
    ident: np.ndarray
)-> np.ndarray:

    EPS = 1e-7
    a = w_test - p
    c = p * (ident - p)
    b = 2 * np.ones(len(w_test)) * w_test * (w_test - p) / (c + EPS)

    plug_in = (t_plugin - prediction) ** 2
    l_de = (ident - b) * t_plugin ** 2 + b * y_test * (t_plugin - prediction) + (- a * (t_plugin - prediction) ** 2 + prediction ** 2)

    return np.sum(plug_in) + np.sum(l_de)

def calculate_pseudo_outcome_pehe_dr(
    w_test: np.ndarray,
    p: np.ndarray,
    prediction: np.ndarray,
    y_test: np.ndarray,
    mu_1: np.ndarray,
    mu_0: np.ndarray
)-> np.ndarray:

    """
    calculating pseudo outcome for DR
    """

    EPS = 1e-7
    w_1 = w_test / (p + EPS)
    w_0 = (1 - w_test) / (EPS + 1 - p)
    pseudo_outcome = (w_1 - w_0) * y_test + ((1 - w_1) * mu_1 - (1 - w_0) * mu_0)

    return np.sqrt(np.mean((prediction - pseudo_outcome) ** 2))

def calculate_pseudo_outcome_pehe_r(
    w_test: np.ndarray,
    p: np.ndarray,
    prediction: np.ndarray,
    y_test: np.ndarray,
    m: np.ndarray
)-> np.ndarray:

    """
    calculating pseudo outcome for R
    """

    y_pseudo = (y_test - m) - (w_test - p)*prediction

    return np.sqrt(np.mean(y_pseudo ** 2))

def calculate_pehe(
    prediction: np.ndarray,
    data: Dataset,
    selection_type: str
) -> np.ndarray:

    x_val, w_val, y_val = data.get_validation_data()
    x_test, w_test, y_test = data.get_testing_data()

    xgb_plugin0, xgb_plugin1, rf, m = train_nuisance_models(x_val, y_val, w_val)

    mu_0 = xgb_plugin0.predict_proba(x_test)[:, 1]
    mu_1 = xgb_plugin1.predict_proba(x_test)[:, 1]

    mu = m.predict_proba(x_test)[:, 1]

    if data.get_cohort_name() == "ist3" or data.get_cohort_name() == "crash_2":
        p = 0.5*np.ones(len(x_test))
    else:
        p = rf.predict_proba(x_test)[:, 1]

    t_plugin = mu_1 - mu_0

    ident = np.ones(len(p))

    selection_types = {
        "if_pehe": calculate_if_pehe,
        "pseudo_outcome_dr": calculate_pseudo_outcome_pehe_dr,
        "pseudo_outcome_r": calculate_pseudo_outcome_pehe_r
    }

    pehe_calculator = selection_types.get(selection_type)

    if pehe_calculator == calculate_if_pehe:
        return pehe_calculator(w_test, p, prediction, t_plugin, y_test, ident)
    elif pehe_calculator == calculate_pseudo_outcome_pehe_dr:
        return pehe_calculator(w_test, p, prediction, y_test, mu_1, mu_0)
    elif pehe_calculator == calculate_pseudo_outcome_pehe_r:
        return pehe_calculator(w_test, p, prediction, y_test, mu)


    raise ValueError(f"Unknown selection_type: {selection_type}")

class Dataset:
    """
    Data wrapper for clinical data.
    """
    def __init__(self, cohort_name, random_state=42, shuffle=False):

        self.cohort_name = cohort_name
        self.shuffle = shuffle
        self.random_state  = random_state

        self.data, self.treatment_col, self.outcome_col = self._load_data(cohort_name)

        self._process_data()

    def _load_data(self, cohort_name):

        if cohort_name in ["massive_trans", "responder"]:
            data = self._load_pickle_data(cohort_name)
            treatment = "treated"
            outcome = "outcome"
        elif cohort_name == "ist3":
            data = self._load_sas_data()
            treatment = "itt_treat"
            outcome = "aliveind6"
        elif cohort_name == "crash_2":
            data = self._load_xlsx_data()
            outcome = "outcome"
            treatment = "treatment_code"
        else:
            raise ValueError(f"Unsupported cohort: {cohort_name}")

        return data, treatment, outcome

    def _load_pickle_data(self, cohort_name):

        if cohort_name == "responder":
            data = pd.read_pickle(f"data/trauma_responder.pkl")
        elif cohort_name == "massive_trans":
            data = pd.read_pickle(f"data/low_bp_survival.pkl")
        else:
            raise ValueError(f"Unsupported cohort: {cohort_name}")

        data = self._filter_data(data)

        return data

    def _filter_data(self, data):
        filter_regex = [
            'proc',
            'ethnicity',
            'residencestate',
            'toxicologyresults',
            "registryid",
            "COV",
            "TT",
            "scenegcsmotor",
            "scenegcseye",
            "scenegcsverbal",
            "edgcsmotor",
            "edgcseye",
            "edgcsverbal",
            "sex_F",
            "traumatype_P",
            "traumatype_other"
        ]
        for regex in filter_regex:
            data = data[data.columns.drop(list(data.filter(regex=regex)))]

        data = self._normalize_data(data)

        return data

    def _load_sas_data(self):
        data = pd.read_sas("data/datashare_aug2015.sas7bdat")

        outcome_col = "aliveind6"
        treatment_col = "itt_treat"

        continuous_vars = [
            "age",
            "weight",
            "glucose",
            "gcs_eye_rand",
            "gcs_motor_rand",
            "gcs_verbal_rand",
            "nihss" ,
            "sbprand",
            "dbprand"
        ]

        cate_variables = [
            "infarct",
            "stroketype"
        ]

        binary_vars = [
            "gender",
            "antiplat_rand"
        ]

        data = data[continuous_vars + cate_variables + binary_vars + [treatment_col]+ [outcome_col]]
        data["antiplat_rand"] = np.where(data["antiplat_rand"]== 2, 0, 1)
        data["gender"] = np.where(data["gender"]== 2, 1, 0)

        data[continuous_vars] = self._normalize_data(data[continuous_vars])

        data = pd.get_dummies(data, columns=cate_variables)
        
        data = data.sample(1500)

        return data

    def _load_xlsx_data(self):

        outcome = "outcome"
        treatment = "treatment_code"

        data = pd.read_excel('data/crash_2.xlsx')
        data[outcome] = np.where(data["icause"].isna(), 1, 0)

        data = data.drop(data[(data[treatment] == "P")|(data[treatment] == "D")].index)

        continuous_vars = [
            "iage",
            'isbp',
            'irr',
            'icc',
            'ihr',
            'ninjurytime',
            # 'igcseye',
            # 'igcsmotor',
            # 'igcsverbal',
            'igcs'
        ]

        cate_variables = [
            "iinjurytype"
        ]

        binary_vars = [
            "isex"
        ]

        data = data[continuous_vars + cate_variables + binary_vars + [treatment]+ [outcome]]
        data["isex"] = np.where(data["isex"]== 2, 0, 1)

        # deal with missing data
        data["irr"] = np.where(data["irr"]== 0, np.nan,data["irr"])
        data["isbp"] = np.where(data["isbp"] == 999, np.nan, data["isbp"])
        data["ninjurytime"] = np.where(data["ninjurytime"] == 999, np.nan, data["ninjurytime"])
        data["ninjurytime"] = np.where(data["ninjurytime"] == 0, np.nan, data["ninjurytime"])

        data[treatment] = np.where(data[treatment] == "Active", 1, 0)

        data = data[data.iinjurytype !=3 ]

        data[continuous_vars] = self._normalize_data(data[continuous_vars])

        data = pd.get_dummies(data, columns=cate_variables)

        data["iinjurytype_1"] = np.where(data["iinjurytype_2"]== 1, 0, 1)
        data.pop("iinjurytype_2")

        # data["iinjurytype_1"] = np.where(data["iinjurytype_3"]== 1, 1, 0)
        # data["iinjurytype_2"] = np.where(data["iinjurytype_3"]== 1, 1, 0)
        # data.pop("iinjurytype_3")

        data = data.sample(5000)

        return data


    def _process_data(self):

        self.n, self.feature_size = self.data.shape
        self.feature_names = self.data.drop([self.treatment_col, self.outcome_col], axis=1).columns

        x_train_scaled = self._impute_missing_values(self.data)

        self._split_data(x_train_scaled)

    def _normalize_data(self, x):

        x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))

        return x

    def _impute_missing_values(self, x_norm):

        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(x_norm)
        x_train_scaled = imp.transform(x_norm)

        return x_train_scaled

    def _split_data(self, x_train_scaled):

        treatment_index = self.data.columns.get_loc(self.treatment_col)
        outcome_index = self.data.columns.get_loc(self.outcome_col)
        var_index = [i for i in range(self.feature_size) if i not in [treatment_index, outcome_index]]

        if self.shuffle:
            random_state = self.random_state
        else:
            random_state = 42

        x_train, x_test, y_train, self.y_test = model_selection.train_test_split(
            x_train_scaled,
            self.data[self.outcome_col],
            test_size=0.2,
            random_state=random_state,
            stratify=self.data[self.treatment_col]
        )

        x_train, x_val, self.y_train, self.y_val = model_selection.train_test_split(
            x_train,
            y_train,
            test_size=0.2,
            random_state=random_state,
            stratify=x_train[:,treatment_index]
        )

        if self.cohort_name == "ist3":
            self.w_train = x_train[:, treatment_index] == 0
            self.w_val = x_val[:, treatment_index] == 0
            self.w_test =  x_test[:, treatment_index] == 0

            self.y_train = self.y_train ==0
            self.y_val = self.y_val ==0
            self.y_test = self.y_test ==0
        else:
            self.w_train = x_train[:, treatment_index]
            self.w_val =  x_val[:, treatment_index]
            self.w_test =  x_test[:, treatment_index]

        self.x_train = x_train[:,var_index]
        self.x_val = x_val[:,var_index]
        self.x_test = x_test[:, var_index]

    def get_training_data(self):
        """
        return training tuples (X,W,Y)
        """
        return self.x_train, self.w_train, self.y_train

    def get_validation_data(self):
        """
        return training tuples (X,W,Y)
        """
        return self.x_val, self.w_val, self.y_val

    def get_testing_data(self):
        """
        return testing tuples (X,W,Y)
        """
        return self.x_test, self.w_test, self.y_test

    def get_feature_names(self):
        """
        return feature names
        """
        return self.feature_names

    def get_cohort_name(self):

        return self.cohort_name

    def get_replacement_value(self):
        """
        return values for insertion & deletion
        """
        # x_replacement = np.zeros(self.x_train.shape[1])
        # x_replacement = np.min(self.x_train, axis=0)

        control = self.x_train
        x_replacement = np.mean(control, axis=0)

        return x_replacement