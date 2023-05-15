import numpy as np
import math
import shap
import torch
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn import preprocessing, model_selection
from copy import deepcopy

def kl_mvn(m0, S0, m1, S1):
    """
    https://stackoverflow.com/questions/44549369/kullback-leibler-divergence-from-gaussian-pm-pv-to-gaussian-qm-qv
    The following function computes the KL-Divergence between any two 
    multivariate normal distributions 
    (no need for the covariance matrices to be diagonal)
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    - accepts stacks of means, but only one S0 and S1
    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    # 'diagonal' is [1, 2, 3, 4]
    tf.diag(diagonal) ==> [[1, 0, 0, 0]
                          [0, 2, 0, 0]
                          [0, 0, 3, 0]
                          [0, 0, 0, 4]]
    # See wikipedia on KL divergence special case.              
    #KL = 0.5 * tf.reduce_sum(1 + t_log_var - K.square(t_mean) - K.exp(t_log_var), axis=1)   
                if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))                               
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N) 


def KL_divergence(a, b):
    epsilon = 1e-35 
    a += epsilon
    b += epsilon

    return np.mean(a * np.log(a/b))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mse(x1, x2):
    return (np.mean((x1 - x2)**2))

def rmse(x1, x2):
    return np.sqrt(np.mean((x1 - x2)**2))

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def simulation(var_size, n, low , high, random_state, oracle=bool):
    
    np.random.seed(random_state)
    cov = np.random.uniform(low=-1, high=1, size=((var_size,var_size)))
    
    ### control 
    mu0 = np.zeros(var_size)
    x0 = np.random.multivariate_normal(mu0, 0.5*cov, size=int(n/2))
    
    ### treated
    mu1 = np.random.uniform(low = low, high=high, size=(var_size))
    x1 = np.random.multivariate_normal(mu1, 0.5*cov, size=int(n/2))
    
    X = np.concatenate((x0, x1))
    
    #### Treatment assignment
    w_t = np.random.uniform(low=-0.1, high=0.1, size=((var_size,1)))
    n_t = np.random.normal(0, 0.1)
    
    p = sigmoid(np.dot(X, w_t) + n_t)

    w = np.concatenate((np.zeros((int(n/2),1)), np.ones((int(n/2),1 ))), 0).astype(int)
    #### Potential outcome
    if oracle:
        p = 0.5*np.ones((n,1))
        w = np.random.binomial(1, p=p)
    
    w_ty = np.random.uniform(low=-1, high=1, size=((var_size,2)))
    n_ty = np.random.multivariate_normal(np.zeros((2)), 0.1*np.eye(2))    
    y_po = np.dot(X, w_ty) + n_ty

    return X, y_po, w, p, kl_mvn(mu0, 0.5*cov, mu1, 0.5*cov)


def oracle(X_train, w_train, y_train, X_test, model):

    model.fit(X_train, y_train, w_train)
    ### TODOs implement prediction function for different cate
    prediction = model.predict(X_test)

    if torch.is_tensor(X_train):
        X_train = X_train.detach().numpy()
        X_test = X_test.detach().numpy()
        model_lam = lambda x: model.predict(x).detach().numpy()
    else:
        model_lam = lambda x: model.predict(x)


    explainer = shap.Explainer(model_lam, X_train)

    #### showing explanation on oracle cate
    shap_values = explainer(X_test)
    shap_mean = (shap_values.values).mean(0)
    shap_abs_mean = np.abs(shap_values.values).mean(0)
    
    return (shap_mean, shap_abs_mean, model)
    
    
def generate_masks(X):
    
    batch_size = X.shape[0]
    num_features = X.shape[1]
    
    unif = torch.rand(batch_size, num_features)
    ref = torch.rand(batch_size, 1)
    
    return (unif > ref).float()
    
def normalize_data(X_train):
    
    X_normalized_train = (X_train - np.min(X_train, axis=0)) / (np.max(X_train, axis=0) - np.min(X_train, axis=0))

    return X_normalized_train, np.min(X_train, axis=0)

def insertion_deletion_if_pehe(
    X_test: np.ndarray, 
    rank_indices:list,
    cate_model: torch.nn.Module,
    X_replacement: np.ndarray,
    a: np.ndarray, 
    b: np.ndarray, 
    c: np.ndarray, 
    t_plugin: np.ndarray,
    y_test:np.ndarray
) -> tuple:
    """
    Compute partial average treatment effect (PATE) with feature subsets by insertion and deletion

    Args:
        X_test: testing set for explanation with insertion and deletion 
        feature_attributions: feature attribution outputted by a feature importance method
        pate_model: masking models for PATE estimation. 
    Returns:
        results of insertion and deletion of PATE. 
    """

    n_samples, n_features = X_test.shape

    X_test_del = X_test.copy()
    X_test_ins = np.tile(X_replacement, (n_samples, 1))

    deletion_results = np.zeros(n_features+1)
    insertion_results = np.zeros(n_features+1)
    
    cate_pred_full = cate_model.predict(X=X_test_del).detach().cpu().numpy().flatten()

    ident = np.ones(len(c))

    # Calculate IF-PEHE for deletion 

    plug_in = (t_plugin - cate_pred_full)**2
    l_de = (ident - b) * t_plugin**2 + b*y_test*(t_plugin - cate_pred_full) + (- a*(t_plugin - cate_pred_full)**2 + cate_pred_full**2)
    deletion_results[0] = np.sum(l_de) + np.sum(plug_in)

    for rank_index, col_indices in enumerate(rank_indices):
        
        X_test_del[:, col_indices] = X_replacement[col_indices]
        
        cate_pred_subset = cate_model.predict(X=X_test_del).detach().cpu().numpy().flatten()
        
        plug_in = (t_plugin - cate_pred_subset)**2
        l_de = (ident - b) * t_plugin**2 + b*y_test*(t_plugin - cate_pred_subset) + (- a*(t_plugin - cate_pred_subset)**2 + cate_pred_subset**2)

        deletion_results[ rank_index+1] = np.sum(l_de) + np.sum(plug_in)


    # Calculate IF-PEHE for insertion

    cate_pred_null = cate_model.predict(X=X_test_ins).detach().cpu().numpy().flatten()
    plug_in = (t_plugin - cate_pred_null)**2
    l_de = (ident - b) * t_plugin**2 + b*y_test*(t_plugin - cate_pred_null) + (- a*(t_plugin - cate_pred_null)**2 + cate_pred_null**2)

    insertion_results[ 0] = np.sum(l_de) + np.sum(plug_in)

    for rank_index, col_indices in enumerate(rank_indices):
        
        X_test_ins[:, col_indices] = X_test[:, col_indices]

        cate_pred_subset = cate_model.predict(X=X_test_ins).detach().cpu().numpy().flatten()

        plug_in = (t_plugin - cate_pred_subset)**2
        l_de = (ident - b) * t_plugin**2 + b*y_test*(t_plugin - cate_pred_subset) + (- a*(t_plugin - cate_pred_subset)**2 + cate_pred_subset**2)
        insertion_results[rank_index+1] = np.sum(l_de) + np.sum(plug_in)

    return insertion_results, deletion_results


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

        X_train, X_test, y_train, y_test = model_selection.train_test_split(
                                                    x_train_scaled,  
                                                    data[outcome_col], 
                                                    test_size=0.2, 
                                                    random_state=random_state,
                                                    stratify=data[treatment_col]
                                            )

        if name == "ist3":
            w_train = X_train[:, treatment_index] == 0
            w_test =  X_test[:, treatment_index] == 0
            X_train = X_train[:,var_index]
            X_test = X_test[:, var_index]
            y_train = y_train ==0
            y_test = y_test ==0
        else:
            w_train = X_train[:, treatment_index]
            w_test =  X_test[:, treatment_index]
            X_train = X_train[:,var_index]
            X_test = X_test[:, var_index]

        self.features_min = features_min[var_index]

        self.X_train = X_train
        self.w_train = w_train
        self.y_train = y_train

        self.X_test = X_test
        self.w_test = w_test
        self.y_test = y_test

    def get_training_data(self):
        """
        return training tuples (X,W,Y)
        """
        return self.X_train, self.w_train, self.y_train  

    def get_testing_data(self):
        """
        return testing tuples (X,W,Y)
        """
        return self.X_test, self.w_test, self.y_test   

    def get_feature_names(self):
        """
        return feature names
        """
        return self.names
    
    def get_replacement_value(self):
        """
        return values for insertion & deletion
        """
        return np.mean(self.X_train, axis=0) 

