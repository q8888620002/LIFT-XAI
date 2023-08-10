from __future__ import annotations
from typing import List

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.linear_model import LogisticRegressionCV
from sklearn.impute import SimpleImputer
from sklearn import  model_selection
from sklearn import metrics


class NuisanceFunctions:
    def __init__(self, rct: bool):

        self.rct = rct
        self.mu0 = xgb.XGBClassifier()
        self.mu1 = xgb.XGBClassifier()
        
        self.m = xgb.XGBClassifier()

        self.rf = xgb.XGBClassifier(
            # reg_lambda=2,
            # max_depth=3,
            # colsample_bytree=0.2,
            # min_split_loss=10
        )
        # self.rf = LogisticRegressionCV(Cs=[0.00001, 0.001, 0.01, 0.1, 1])

    def fit(self, x_val, y_val, w_val):
        
        x0, x1 = x_val[w_val == 0], x_val[w_val == 1]
        y0, y1 = y_val[w_val == 0], y_val[w_val == 1]

        self.mu0.fit(x0, y0)
        self.mu1.fit(x1, y1)
        self.m.fit(x_val, y_val)
        self.rf.fit(x_val, w_val)

    def predict_mu_0(self, x):
        return self.mu0.predict(x)

    def predict_mu_1(self, x):
        return self.mu1.predict(x)

    def predict_propensity(self, x):
        if self.rct:
            p = 0.5*np.ones(x.shape[0])
        else:
            p = self.rf.predict_proba(x)[:, 1]
        return p

    def predict_m(self, x):
        return self.m.predict(x)

def normalize_data(x):

    x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))

    return x

def subgroup_identification(
        rank_indices: np.ndarray,
        x_train: np.ndarray,
        x_test: np.ndarray,
        cate_model: torch.nn.Module
) -> tuple:
    """
    Function for calculating evaluationg metrics for treatment assignment in testing set.

    args: 
        rank_indices: global ranking of features 
        x_train: training data
        x_test: testing data
        cate_model: trained CATE model. 
    return:
        ate: average treatment effect according to the xgb classifier
        auroc: corresponding AUROC for xgb classifier in testing set.

    """
    train_effects = cate_model.predict(X=x_train).detach().cpu().numpy()
    test_effects = cate_model.predict(X=x_test).detach().cpu().numpy()

    threshold  = np.mean(train_effects)
    train_treatment_assignments = (train_effects > threshold)
    test_treatment_assignments = (test_effects > threshold)

    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(x_train[:, rank_indices], train_treatment_assignments)
    y_pred = xgb_model.predict(x_test[:, rank_indices])

    ate = np.sum(test_treatment_assignments[y_pred == 1])/len(test_treatment_assignments)
    auroc = metrics.roc_auc_score(test_treatment_assignments, y_pred)

    return ate, auroc

def attribution_ranking(feature_attributions: np.ndarray) -> list:
    """"
    Compute the ranking of features according to atribution score

    Args:
        feature_attributions: an n x d array of feature attribution scores
    Return:
        a d x n list of indices starting from the highest attribution score
    """

    rank_indices = np.argsort(feature_attributions, axis=1)[:, ::-1]
    rank_indices = list(map(list, zip(*rank_indices)))

    return rank_indices

def insertion_deletion(
    test_data: tuple,
    rank_indices:list,
    cate_model: torch.nn.Module,
    baseline: np.ndarray,
    selection_types: List[str],
    nuisance_functions: NuisanceFunctions,
    model_type: str ="CATENets"
) -> tuple:
    """
    Compute partial average treatment effect (PATE) with with proxy pehe and feature subsets by insertion and deletion

    Args:
        test_data: testing data to calculate pehe 
        rank_indices: local ranking from a given explanation method
        cate_model: trained cate model
        baseline: replacement values for insertion & deletion
        selection_types: approximation methods for estimating ground truth pehe
        nuisance function: Nuisance functions to estimate proxy pehe. 
        model_type: model types for cate package
    Returns:
        results of insertion and deletion of PATE.
    """
    ## training plugin estimator on
    
    x_test,_ ,_ = test_data

    n, d = x_test.shape
    x_test_del = x_test.copy()
    x_test_ins = np.tile(baseline, (n, 1))
    baseline = np.tile(baseline, (n, 1))

    original_cate_model = cate_model

    if model_type == "CATENets":
        cate_model = lambda x: original_cate_model.predict(X=x)
    else:
        cate_model = lambda x: original_cate_model.forward(X=x)

    deletion_results = {selection_type: np.zeros(d+1) for selection_type in selection_types}
    insertion_results = {selection_type: np.zeros(d+1) for selection_type in selection_types}

    for rank_index in range(len(rank_indices) + 1):
        # Skip this on the first iteration

        if rank_index > 0:
            col_indices = rank_indices[rank_index - 1]
            x_test_ins[:, col_indices] = x_test[:, col_indices]
            x_test_del[:, col_indices] = baseline[:, col_indices]

        for selection_type in selection_types:
            # For the insertion process
            cate_pred_subset_ins = cate_model(x_test_ins).detach().cpu().numpy().flatten()

            insertion_results[selection_type][rank_index] = calculate_pehe(
                cate_pred_subset_ins,
                test_data,
                selection_type,
                nuisance_functions
            )

            # For the deletion process
            cate_pred_subset_del = cate_model(x_test_del).detach().cpu().numpy().flatten()
            deletion_results[selection_type][rank_index] = calculate_pehe(
                cate_pred_subset_del,
                test_data,
                selection_type,
                nuisance_functions
            )

    return insertion_results, deletion_results

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
    test_data: tuple,
    selection_type: str,
    nuisance_functions: NuisanceFunctions
) -> np.ndarray:

    x_test, w_test, y_test = test_data

    mu_0 = nuisance_functions.predict_mu_0(x_test)
    mu_1 = nuisance_functions.predict_mu_1(x_test)
    mu = nuisance_functions.predict_m(x_test)
    p = nuisance_functions.predict_propensity(x_test)

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
            "traumatype_Other"
        ]

        treatment_col = "treated"
        outcome_col = "outcome"

        for regex in filter_regex:
            data = data[data.columns.drop(list(data.filter(regex=regex)))]

        binary_vars = [
            "sex_F",
            "traumatype_B",
        ]

        continuous_vars = [
            'age',
            'scenegcs', 'scenefirstbloodpressure', 'scenefirstpulse','scenefirstrespirationrate', 
            'edfirstbp', 'edfirstpulse', 'edfirstrespirationrate', 'edgcs',
            'temps2',  'BD', 'CFSS', 'COHB', 'CREAT', 'FIB', 'FIO2', 'HCT',
            'HGB', 'INR', 'LAC', 'NA', 'PAO2', 'PH', 'PLTS'
        ]

        cate_variables = [
            "causecode"
        ]

        self.categorical_indices = self.get_one_hot_column_indices(
            data.drop(
                [
                    treatment_col,
                    outcome_col
                ],  axis=1
            ), cate_variables
            )
        # import ipdb;ipdb.set_trace()

        data[continuous_vars] = self._normalize_data(data[continuous_vars])

        return data

    def _load_sas_data(self):
        data = pd.read_sas("data/datashare_aug2015.sas7bdat")

        outcome_col = "aliveind6"
        treatment_col = "itt_treat"

        continuous_vars = [
            "age",
            "weight",
            "glucose",
            # "gcs_eye_rand",
            # "gcs_motor_rand",
            # "gcs_verbal_rand",
            "gcs_score_rand",
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
            "antiplat_rand",
            "atrialfib_rand"
        ]

        data = data[continuous_vars + cate_variables + binary_vars + [treatment_col]+ [outcome_col]]
        data["antiplat_rand"] = np.where(data["antiplat_rand"]== 1, 1, 0)
        data["atrialfib_rand"] = np.where(data["atrialfib_rand"]== 1, 1, 0)
        data["gender"] = np.where(data["gender"]== 2, 1, 0)
        
        data[treatment_col] = np.where(data[treatment_col]== 0, 1, 0)
        data[outcome_col] = np.where(data[outcome_col]== 1, 1, 0)
        data[continuous_vars] = self._normalize_data(data[continuous_vars])

        data = pd.get_dummies(data, columns=cate_variables)

        self.categorical_indices = self.get_one_hot_column_indices(
            data.drop(
            [
                treatment_col, 
                outcome_col
            ],  axis=1
            ), cate_variables
            )

        # data = data.sample(2500)

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
        
        self.categorical_indices = self.get_one_hot_column_indices(
            data.drop(
            [
                treatment, 
                outcome
            ],  axis=1
            ), cate_variables
            )
        
        data = pd.get_dummies(data, columns=cate_variables)

        data["iinjurytype_1"] = np.where(data["iinjurytype_2"]== 1, 0, 1)
        data.pop("iinjurytype_2")

        # data["iinjurytype_1"] = np.where(data["iinjurytype_3"]== 1, 1, 0)
        # data["iinjurytype_2"] = np.where(data["iinjurytype_3"]== 1, 1, 0)
        # data.pop("iinjurytype_3")

        # data = data.sample(5000)

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

        self.x = x_train_scaled[:, var_index]
        self.w = x_train_scaled[:, treatment_index]
        self.y = self.data[self.outcome_col]

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
        
        # x_val_eta, x_val, self.y_val_eta, self.y_val = model_selection.train_test_split(
        #     x_val,
        #     self.y_val,
        #     test_size=0.5,
        #     random_state=random_state,
        #     stratify=x_val[:,treatment_index]
        # )


        self.w_train = x_train[:, treatment_index]
        self.w_val =  x_val[:, treatment_index]
        # self.w_val_eta =  x_val_eta[:, treatment_index]
        self.w_test =  x_test[:, treatment_index]

        self.x_train = x_train[:,var_index]
        self.x_val = x_val[:,var_index]
        # self.x_val_eta = x_val_eta[:, var_index]
        self.x_test = x_test[:, var_index]

    def get_data(self):

        return self.x, self.w, self.y

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
    
    def get_validation_eta_data(self):
        """
        return training tuples (X,W,Y)
        """
        return self.x_val_eta, self.w_val_eta, self.y_val_eta

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
    
    def get_one_hot_column_indices(self, df, prefixes):
        """
        Get the indices for one-hot encoded columns for each specified prefix. 
        This function assumes that the DataFrame has been one-hot encoded using 
        pandas' get_dummies method.
        
        Parameters:
        df: pandas DataFrame
        prefixes: list of strings, the prefixes used in the one-hot encoded columns
        
        Returns:
        indices_dict: dictionary where keys are the prefixes and values are lists of 
                    indices representing the position of each category column for that prefix
        """
        indices_dict = {}
        
        for prefix in prefixes:
            # Filter for one-hot encoded columns with the given prefix
            one_hot_cols = [col for col in df.columns if col.startswith(prefix)]
            
            # Get the indices for these columns
            indices_dict[prefix] = [df.columns.get_loc(col) for col in one_hot_cols]
        
        return indices_dict

