from __future__ import annotations
from typing import List

import shap
import torch
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn import  model_selection

def normalize_data(x_train):

    x_normalized_train = (x_train - np.min(x_train, axis=0)) / (np.max(x_train, axis=0) - np.min(x_train, axis=0))

    return x_normalized_train, np.min(x_train, axis=0)

def insertion_deletion(
    data: Dataset,
    rank_indices:list,
    cate_model: torch.nn.Module,
    x_replacement: np.ndarray,
    selection_types: List[str]
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
    x_test_del = x_test.copy()
    x_test_ins = np.tile(x_replacement, (n, 1))

    deletion_results = {selection_type: np.zeros(d+1) for selection_type in selection_types}
    insertion_results = {selection_type: np.zeros(d+1) for selection_type in selection_types}

    for rank_index in range(len(rank_indices) + 1):
        if rank_index > 0:  # Skip this on the first iteration
            col_indices = rank_indices[rank_index - 1]
            x_test_ins[:, col_indices] = x_test[:, col_indices]
            x_test_del[:, col_indices] = x_replacement[col_indices]

        for selection_type in selection_types:
            # For the insertion process
            cate_pred_subset_ins = cate_model.predict(X=x_test_ins).detach().cpu().numpy().flatten()
            insertion_results[selection_type][rank_index] = calculate_pehe(
                cate_pred_subset_ins,
                data,
                selection_type
            )

            # For the deletion process
            cate_pred_subset_del = cate_model.predict(X=x_test_del).detach().cpu().numpy().flatten()
            deletion_results[selection_type][rank_index] = calculate_pehe(
                cate_pred_subset_del,
                data,
                selection_type
            )

    return insertion_results, deletion_results

def train_nuisance_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray
)-> tuple:

    mu0 = xgb.XGBClassifier()
    mu1 = xgb.XGBClassifier()
    m = xgb.XGBClassifier()

    rf = xgb.XGBClassifier()

    x0, x1 = x_train[w_train == 0], x_train[w_train == 1]
    y0, y1 = y_train[w_train == 0], y_train[w_train == 1]

    mu0.fit(x0, y0)
    mu1.fit(x1, y1)

    m.fit(np.column_stack([x_train, w_train]), y_train)

    rf.fit(x_train, w_train)

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

    return np.sum(l_de) + np.sum(plug_in)

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

    y_pseudo = (y_test - m) - (prediction)*(w_test - p)

    return np.sqrt(np.mean(y_pseudo** 2))

def calculate_pehe(
    prediction: np.ndarray,
    data: Dataset,
    selection_type: str
) -> np.ndarray:

    x_train, w_train, y_train = data.get_validation_data()
    x_test, w_test, y_test = data.get_testing_data()

    xgb_plugin0, xgb_plugin1, rf, m = train_nuisance_models(x_train, y_train, w_train)

    mu_0 = xgb_plugin0.predict_proba(x_test)[:, 1]
    mu_1 = xgb_plugin1.predict_proba(x_test)[:, 1]

    mu = m.predict_proba(np.column_stack([x_test, w_test]))[:, 1]

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
    Dataset wrapper class for clinical data including massive transfucion, responder, and IST-3

    """

    def __init__(self, name, random_state ):

        if name == "massive_trans":
            data = pd.read_pickle("data/low_bp_survival.pkl")

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
            treatment_col = "treated"
            outcome_col = "outcome"

            for regex in filter_regex:
                data = data[data.columns.drop(list(data.filter(regex=regex)))]


        elif name == "responder":
            data = pd.read_pickle("data/trauma_responder.pkl")
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
            treatment_col = "treated"
            outcome_col = "outcome"

            for regex in filter_regex:
                data = data[data.columns.drop(list(data.filter(regex=regex)))]

        elif name =="ist3":
            data = pd.read_sas("data/datashare_aug2015.sas7bdat")

            outcome_col = "aliveind6"
            treatment_col = "itt_treat"

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

            data = data[continuous_vars + cate_variables + [treatment_col]+ [outcome_col]]
            data = pd.get_dummies(data, columns=cate_variables)

        self.data = data
        self.random_state = random_state
        self.n, self.feature_size = data.shape
        self.names = data.drop([treatment_col, outcome_col], axis=1).columns

        treatment_index = data.columns.get_loc(treatment_col)
        outcome_index = data.columns.get_loc(outcome_col)

        var_index = [i for i in range(self.feature_size) if i not in [treatment_index, outcome_index]]

        x_norm, features_min = normalize_data(data)


        ## impute missing value

        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(x_norm)
        x_train_scaled = imp.transform(x_norm)

        x_train, x_test, y_train, y_test = model_selection.train_test_split(
                                                    x_train_scaled,
                                                    data[outcome_col],
                                                    test_size=0.2,
                                                    random_state=random_state,
                                                    stratify=data[treatment_col]
                                            )

        x_train, x_val, y_train, y_val = model_selection.train_test_split(
                                                    x_train,
                                                    y_train,
                                                    test_size=0.2,
                                                    random_state=random_state,
                                                    stratify=x_train[:,treatment_index]
                                            )

        if name == "ist3":
            w_train = x_train[:, treatment_index] == 0
            w_val = x_val[:, treatment_index] == 0
            w_test =  x_test[:, treatment_index] == 0

            x_train = x_train[:,var_index]
            x_val = x_val[:,var_index]
            x_test = x_test[:, var_index]

            y_train = y_train ==0
            y_val = y_val ==0
            y_test = y_test ==0
        else:
            w_train = x_train[:, treatment_index]
            w_val =  x_val[:, treatment_index]
            w_test =  x_test[:, treatment_index]

            x_train = x_train[:,var_index]
            x_val = x_val[:,var_index]
            x_test = x_test[:, var_index]

        self.features_min = features_min[var_index]

        self.x_train = x_train
        self.w_train = w_train
        self.y_train = y_train

        self.x_val = x_val
        self.w_val = w_val
        self.y_val = y_val

        self.x_test = x_test
        self.w_test = w_test
        self.y_test = y_test

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
        return self.names

    def get_replacement_value(self):
        """
        return values for insertion & deletion
        """
        # x_replacement = np.zeros(self.x_train.shape[1])
        x_replacement = np.mean(self.x_train, axis=0)

        # x_replacement = np.min(self.x_train, axis=0)

        return x_replacement


