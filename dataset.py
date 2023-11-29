import numpy as np
import pandas as pd

from sklearn import  model_selection
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors



class Dataset:
    """
    Data wrapper for clinical data.
    """
    def __init__(self, cohort_name, random_state=42, shuffle=False):

        self.cohort_name = cohort_name
        self.shuffle = shuffle
        self.random_state  = random_state

        self._load_data(cohort_name)
        self._process_data()

    def _load_data(self, cohort_name):

        if cohort_name in ["massive_trans", "responder"]:
            self.data = self._load_pickle_data(cohort_name)
        elif cohort_name == "ist3":
            self.data = self._load_ist3_data()
        elif cohort_name == "crash_2":
            self.data = self._load_crash2_data()
        elif cohort_name == "txa":
            self.data = self._load_txa_data()
        elif cohort_name == "sprint":
            self.data = self._load_sprint_data()
        elif cohort_name == "accord":
            self.data = self._load_accord_data()
        elif cohort_name == "sprint_filter":
            self.data = self._load_sprint_filter_data()
        elif cohort_name =="accord_filter":
            self.data = self._load_accord_filter_data()
        else:
            raise ValueError(f"Unsupported cohort: {cohort_name}")
        
        self.continuous_indices = [self.data.columns.get_loc(col) for col in self.continuous_vars]

        self.categorical_indices = self.get_one_hot_column_indices(
            self.data.drop(
                [
                    self.treatment,
                    self.outcome
                ],  axis=1
            ),  self.categorical_vars
        )

        self.discrete_indices = self.get_one_hot_column_indices(
            self.data.drop(
                [
                    self.treatment,
                    self.outcome
                ],  axis=1
            ),  self.categorical_vars + self.binary_vars
        )

    def _load_pickle_data(self, cohort_name):

        if cohort_name == "responder":
            data = pd.read_pickle(f"data/trauma_responder.pkl")
        elif cohort_name == "massive_trans":
            data = pd.read_pickle(f"data/low_bp_survival.pkl")
        else:
            raise ValueError(f"Unsupported cohort: {cohort_name}")

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
            "traumatype_OTHER",
            "causecode"
        ]

        self.treatment = "treated"
        self.outcome = "outcome"

        for regex in filter_regex:
            data = data[data.columns.drop(list(data.filter(regex=regex)))]

        self.binary_vars = [
            "sex_M",
            "traumatype_B",
        ]

        self.continuous_vars = [
            'age',
            'scenegcs', 'scenefirstbloodpressure', 'scenefirstpulse','scenefirstrespirationrate', 
            'edfirstbp', 'edfirstpulse', 'edfirstrespirationrate', 'edgcs',
            'temps2',  'BD', 'CFSS', 'COHB', 'CREAT', 'FIB', 'FIO2', 'HCT',
            'HGB', 'INR', 'LAC', 'NA', 'PAO2', 'PH', 'PLTS'
        ]
        
        # self.categorical_vars = [col for col in data.columns if 'causecode' in col]
        self.categorical_vars = []
        
        data = data[self.continuous_vars + self.categorical_vars + self.binary_vars + [self.treatment]+ [self.outcome]]


        return data

    def _load_txa_data(self):

        data = pd.read_pickle(f"data/txa_cohort.pkl")

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
            "traumatype_OTHER",
            "causecode"
        ]

        self.treatment = "treated"
        self.outcome = "outcome"

        for regex in filter_regex:
            data = data[data.columns.drop(list(data.filter(regex=regex)))]

        self.binary_vars = [
            "sex_M",
            "traumatype_B",
        ]

        self.continuous_vars = [
            'age',
            'scenefirstbloodpressure', 
            'scenefirstpulse',
            'scenefirstrespirationrate', 
            'scenegcs'
        ]
        
        self.categorical_vars = []
        # For continuous variables


        data = data[self.continuous_vars + self.categorical_vars + self.binary_vars + [self.treatment]+ [self.outcome]]
        imp_mean = SimpleImputer(strategy='mean')
        data[self.continuous_vars] = imp_mean.fit_transform(data[self.continuous_vars])
        # Instantiate the Matcher class

        X = data.drop(columns=[self.treatment, self.outcome])
        y = data[self.treatment]

        model = LogisticRegression(max_iter=2000)
        model.fit(X, y)

        data["propensity_score"] = model.predict_proba(X)[:, 1]

        treated = data[data[self.treatment] == 1].copy()
        control = data[data[self.treatment] == 0].copy()

        # Fit the nearest neighbors model for 1 neighbors
        nbrs = NearestNeighbors(n_neighbors=1).fit(control[["propensity_score"]])

        # Find the nearest neighbor indices for the treated group
        _, indices = nbrs.kneighbors(treated[["propensity_score"]])

        # Flatten indices for 1:2 matching & Extract matched controls
        
        matched_control = control.iloc[indices.flatten()]
        matched_data = pd.concat([treated, matched_control]).sort_index()
        matched_data = matched_data.sort_index()
        matched_data.drop(columns=["propensity_score"], inplace=True)

        return matched_data


    def _load_ist3_data(self):
        data = pd.read_sas("data/datashare_aug2015.sas7bdat")

        self.outcome = "aliveind6"
        self.treatment = "itt_treat"

        self.continuous_vars = [
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

        self.categorical_vars = [
            "stroketype",
        ]

        self.binary_vars = [
            "gender",
            "antiplat_rand",
            "atrialfib_rand",
            "infarct"

        ]

        data = data[data.stroketype!=5]

        data["antiplat_rand"] = np.where(data["antiplat_rand"]== 1, 1, 0)        
        data["atrialfib_rand"] = np.where(data["atrialfib_rand"]== 1, 1, 0)
        
        data["gender"] = np.where(data["gender"]== 2, 1, 0)

        data["infarct"] =  np.where(data["infarct"] == 0, 0, 1)

        data[self.treatment] = np.where(data[self.treatment]== 0, 1, 0)
        data[self.outcome] = np.where(data[self.outcome]== 1, 1, 0)

        data = data[self.continuous_vars + self.categorical_vars + self.binary_vars + [self.treatment]+ [self.outcome]]

        data = pd.get_dummies(data, columns=self.categorical_vars)

        # data = data.sample(2500)
        
        return data

    def _load_crash2_data(self):

        self.outcome = "outcome"
        self.treatment = "treatment_code"

        data = pd.read_excel('data/crash_2.xlsx')

        self.continuous_vars = [
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

        self.categorical_vars = [
            "iinjurytype"
        ]

        self.binary_vars = [
            "isex"
        ]

        data = data.drop(data[(data[self.treatment] == "P")|(data[self.treatment] == "D")].index)
        
        data = data[data.iinjurytype != 3 ]

        data["isex"] = np.where(data["isex"]== 2, 0, 1)

        # deal with missing data

        data["irr"] = np.where(data["irr"]== 0, np.nan,data["irr"])
        data["isbp"] = np.where(data["isbp"] == 999, np.nan, data["isbp"])
        data["ninjurytime"] = np.where(data["ninjurytime"] == 999, np.nan, data["ninjurytime"])
        data["ninjurytime"] = np.where(data["ninjurytime"] == 0, np.nan, data["ninjurytime"])

        data[self.treatment] = np.where(data[self.treatment] == "Active", 1, 0)
        data[self.outcome] = np.where(data["icause"].isna(), 1, 0)

        data = data[self.continuous_vars + self.categorical_vars + self.binary_vars + [self.treatment]+ [self.outcome]]

        data = pd.get_dummies(data, columns=self.categorical_vars)

        data["iinjurytype_1"] = np.where(data["iinjurytype_2"]== 1, 0, 1)

        # data.pop("iinjurytype_3")
        
        # data.pop("iinjurytype_2")

        # data = data.sample(int(len(data)*0.95))

        return data

    def _load_sprint_data(self):

        self.outcome = "event_primary"
        self.treatment = "intensive"

        outcome = pd.read_csv("data/sprint/outcomes.csv")
        baseline = pd.read_csv("data/sprint/baseline.csv")

        baseline.columns = [x.lower() for x in baseline.columns]
        outcome.columns = [x.lower() for x in outcome.columns]

        data = baseline.merge(outcome, on="maskid", how="inner")
        
        data["smoke_3cat"] = np.where(data["smoke_3cat"] == 4, np.nan, 
                                    np.where(data["smoke_3cat"] == 3, 1, 0))


        self.continuous_vars = [
            "age", 
            "sbp",
            "dbp",
            "n_agents",
            "egfr", 
            "screat",
            "chr",
            "glur",
            "hdl",
            "trr",
            "umalcr",
            "bmi",
            # "risk10yrs"
        ]

        self.binary_vars = [
            "female" ,
            "race_black",
            "smoke_3cat",
            "aspirin",
            "statin",
            "sub_cvd",
            "sub_ckd"
            # "inclusionfrs"
            # "noagents"
        ]

        self.categorical_vars = []

        data = data[self.continuous_vars + self.categorical_vars + self.binary_vars + [self.treatment] + [self.outcome]]

        data[self.outcome] = np.where(data[self.outcome] == 1, 0, 1)
        data = pd.get_dummies(data, columns=self.categorical_vars)

        return data

    def _load_sprint_filter_data(self):

        self.outcome = "event_primary"
        self.treatment = "intensive"

        outcome = pd.read_csv("data/sprint/outcomes.csv")
        baseline = pd.read_csv("data/sprint/baseline.csv")

        baseline.columns = [x.lower() for x in baseline.columns]
        outcome.columns = [x.lower() for x in outcome.columns]

        data = baseline.merge(outcome, on="maskid", how="inner")
        
        data["smoke_3cat"] = np.where(data["smoke_3cat"] == 4, np.nan, 
                                    np.where(data["smoke_3cat"] == 3, 1, 0))

        self.continuous_vars = [
            "age", 
            "sbp",
            "dbp",
            "n_agents",
            "egfr", 
            "screat",
            "chr",
            "glur",
            "hdl",
            "trr",
            "umalcr",
            "bmi",
            # "risk10yrs"
        ]

        self.binary_vars = [
            "female" ,
            "race_black",
            "smoke_3cat",
            "aspirin",
            "statin",
            "sub_cvd",
            # "sub_ckd"
            # "inclusionfrs"
            # "noagents"
        ]
        

        self.categorical_vars = []

        data = data[self.continuous_vars + self.categorical_vars + self.binary_vars + [self.treatment] + [self.outcome]]

        data[self.outcome] = np.where(data[self.outcome] == 1, 0, 1)

        
        data = pd.get_dummies(data, columns=self.categorical_vars)

        return data
    
    def _load_accord_data(self):

        data = pd.read_csv("data/accord/accord.csv")

        self.outcome = "censor_po"
        self.treatment = "treatment"

        self.continuous_vars = [
            'baseline_age', 
            'bmi',
            'sbp', 
            'dbp',
            'hr',
            'fpg', 
            'alt', 
            'cpk',
            'potassium',
            'screat', 
            'gfr',
            # 'ualb', 
            # 'ucreat', 
            'uacr',
            'chol', 
            'trig',
            'vldl',
            'ldl',
            'hdl',
            'bp_med'
        ]

        self.binary_vars = [
            'female',
            'raceclass',
            'cvd_hx_baseline',
            'statin',
            'aspirin',
            'antiarrhythmic',
            'anti_coag',
            # 'dm_med',
            # 'cv_med',
            # 'lipid_med',
            'x4smoke'
        ]

        self.categorical_vars = []

        data["treatment"] = np.where(data["treatment"].str.contains("Intensive BP"), 1, 0)
        data["raceclass"] = np.where(data["raceclass"]== "Black", 1, 0)
        data["x4smoke"] = np.where(data["x4smoke"] == 1, 1, 0)

        data = data[self.continuous_vars + self.categorical_vars + self.binary_vars + [self.treatment] + [self.outcome]]

        data = pd.get_dummies(data, columns=self.categorical_vars)

        return data
    
    def _load_accord_filter_data(self):

        data = pd.read_csv("data/accord/accord.csv")

        self.outcome = "censor_po"
        self.treatment = "treatment"

        self.continuous_vars = [
            'baseline_age', 
            'sbp', 
            'dbp',
            'bp_med',
            'gfr',
            'screat', 
            'chol', 
            'fpg', 
            'hdl',
            'trig',
            'uacr',
            'bmi',
            # 'hr',
            # 'alt', 
            # 'cpk',
            # 'potassium',
            # 'ualb', 
            # 'ucreat', 
            # 'vldl',
            # 'ldl',

        ]

        self.binary_vars = [
            'female',
            'raceclass',
            'x4smoke',
            'aspirin',
            'statin',
            'cvd_hx_baseline'
            # 'antiarrhythmic',
            # 'anti_coag',
            # 'dm_med',
            # 'cv_med',
            # 'lipid_med',
        ]

        self.categorical_vars = []

        data["treatment"] = np.where(data["treatment"].str.contains("Intensive BP"), 1, 0)
        data["raceclass"] = np.where(data["raceclass"]== "Black", 1, 0)
        data["x4smoke"] = np.where(data["x4smoke"] == 1, 1, 0)

        data = data[self.continuous_vars + self.categorical_vars + self.binary_vars + [self.treatment] + [self.outcome]]

        data = pd.get_dummies(data, columns=self.categorical_vars)

        return data

    def _process_data(self):

        self.data[self.continuous_vars] = self._normalize_data(self.data[self.continuous_vars], "minmax")
        
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(self.data)
        self._split_data(imp.transform(self.data))

    def _normalize_data(self, x: np.ndarray, type:str):

        if type == "minmax":
            self.scaler = MinMaxScaler()

        elif type == "standard":
            self.scaler = StandardScaler()

        self.scaler.fit(x.values)
        x = self.scaler.transform(x.values)

        return x

    def _split_data(self, x_train_scaled):

        treatment_index = self.data.columns.get_loc(self.treatment)
        outcome_index = self.data.columns.get_loc(self.outcome)

        var_index = [i for i in range(x_train_scaled.shape[1]) if i not in [treatment_index, outcome_index]]

        if self.shuffle:
            random_state = self.random_state
        else:
            random_state = 42
        
        x_train, x_test, y_train, self.y_test = model_selection.train_test_split(
            x_train_scaled,
            self.data[self.outcome].values,
            test_size=0.2,
            random_state=random_state,
            stratify=self.data[self.treatment]
        )

        x_train, x_val, self.y_train, self.y_val = model_selection.train_test_split(
            x_train,
            y_train,
            test_size=0.2,
            random_state=random_state,
            stratify=x_train[:,treatment_index]
        )

        self.x = x_train_scaled[:, var_index]
        self.w = x_train_scaled[:, treatment_index]
        self.y = self.data[self.outcome].values

        self.w_train = x_train[:, treatment_index]
        self.w_val =  x_val[:, treatment_index]
        self.w_test =  x_test[:, treatment_index]

        self.x_train = x_train[:,var_index]
        self.x_val = x_val[:,var_index]
        self.x_test = x_test[:, var_index]

    def get_data(self, set:str=None):

        if set == "train":
            return self.x_train, self.w_train, self.y_train
        elif set == "val":
            return self.x_val, self.w_val, self.y_val
        elif set == "test":
            return self.x_test, self.w_test, self.y_test
        else:
            return self.x, self.w, self.y

    def get_feature_range(
            self, 
            feature: int
        ) -> np.ndarray:
        """
        return value range for a feature
        """

        x_original  = self.scaler.inverse_transform(self.x[:, self.continuous_indices])

        min = np.min(x_original[:, feature])
        max = np.max(x_original[:, feature])

        return max - min
    
    def get_unnorm_value(
            self,
            x: np.ndarray
        )-> np.ndarray:

        return self.scaler.inverse_transform(x[:, self.continuous_indices])

    def get_norm(
            self, 
            x: np.ndarray,
    ) -> np.ndarray:
        
        return self.scaler.transform(x[:, self.continuous_indices])
    
    def get_feature_names(self):
        """
        return feature names
        """
        return self.data.drop([self.treatment, self.outcome], axis=1).columns
    
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


def obtain_accord_baselines() -> np.ndarray:
    """
    Return normalized baseline of ACCORD dataset with SPRINT value range. 

    Return:
        Tuple containing:
        - Normalized baselines for SPRINT
        - Treatment for SPRINT
        - Outcome for SPRINT
        - Normalized baselines for ACCORD
    """

    accord = pd.read_csv("data/accord/accord.csv")

    continuous_vars_accord = [
        'baseline_age', 'sbp', 'dbp', 'bp_med', 'gfr', 'screat', 
        'chol', 'fpg', 'hdl', 'trig', 'uacr', 'bmi'
    ]

    binary_vars_accord = [
        'female', 'raceclass', 'x4smoke', 'aspirin', 
        'statin', 'cvd_hx_baseline'
    ]

    accord["raceclass"] = np.where(accord["raceclass"]== "Black", 1, 0)
    accord["x4smoke"] = np.where(accord["x4smoke"] == 1, 1, 0)

    outcome = "censor_po"
    treatment = "treatment"
    
    accord = accord[continuous_vars_accord + binary_vars_accord +[treatment] + [outcome]]
    accord["treatment"] = np.where(accord["treatment"].str.contains("Intensive BP"), 1, 0)

    outcome = pd.read_csv("data/sprint/outcomes.csv")
    baseline = pd.read_csv("data/sprint/baseline.csv")

    baseline.columns = [x.lower() for x in baseline.columns]
    outcome.columns = [x.lower() for x in outcome.columns]

    sprint = baseline.merge(outcome, on="maskid", how="inner")

    sprint["smoke_3cat"] = np.where(sprint["smoke_3cat"] == 4, np.nan, 
                                np.where(sprint["smoke_3cat"] == 3, 1, 0))

    continuous_vars_sprint = [
        "age", "sbp", "dbp", "n_agents", "egfr", "screat",
        "chr", "glur", "hdl", "trr", "umalcr", "bmi"
    ]

    binary_vars_sprint = [
        "female", "race_black", "smoke_3cat", 
        "aspirin", "statin", "sub_cvd"
    ]

    treatment = "intensive"
    outcome_col = "event_primary"

    sprint = sprint[continuous_vars_sprint + binary_vars_sprint + [treatment] + [outcome_col]]
    sprint[outcome_col] = np.where(sprint[outcome_col] == 1, 0, 1)

    scaler = MinMaxScaler()

    scaler.fit(
        np.concatenate(
            (
              sprint[continuous_vars_sprint].values,
              accord[continuous_vars_accord].values
            ), axis=0
        )
    )

    sprint[continuous_vars_sprint] = scaler.transform(sprint[continuous_vars_sprint].values)
    accord[continuous_vars_accord] = scaler.transform(accord[continuous_vars_accord].values)

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(sprint)
    sprint = imp.transform(sprint)

    imp.fit(accord)
    accord = imp.transform(accord)
    
    return sprint[:, :-2], sprint[:, -2], sprint[:, -1], accord[:, :-2], accord[:, -2], accord[:, -1]

def obtain_unnorm_accord_baselines() -> np.ndarray:
    """
    Return normalized baseline of ACCORD dataset with SPRINT value range. 

    Return:
        Tuple containing:
        - Normalized baselines for SPRINT
        - Treatment for SPRINT
        - Outcome for SPRINT
        - Normalized baselines for ACCORD
    """

    accord = pd.read_csv("data/accord/accord.csv")

    continuous_vars_accord = [
        'baseline_age', 'sbp', 'dbp', 'bp_med', 'gfr', 'screat', 
        'chol', 'fpg', 'hdl', 'trig', 'uacr', 'bmi'
    ]

    binary_vars_accord = [
        'female', 'raceclass', 'x4smoke', 'aspirin', 
        'statin', 'cvd_hx_baseline'
    ]

    accord["raceclass"] = np.where(accord["raceclass"]== "Black", 1, 0)
    accord["x4smoke"] = np.where(accord["x4smoke"] == 1, 1, 0)

    outcome = "censor_po"
    treatment = "treatment"
    
    accord = accord[continuous_vars_accord + binary_vars_accord +[treatment] + [outcome]]
    accord["treatment"] = np.where(accord["treatment"].str.contains("Intensive BP"), 1, 0)

    outcome = pd.read_csv("data/sprint/outcomes.csv")
    baseline = pd.read_csv("data/sprint/baseline.csv")

    baseline.columns = [x.lower() for x in baseline.columns]
    outcome.columns = [x.lower() for x in outcome.columns]

    sprint = baseline.merge(outcome, on="maskid", how="inner")

    sprint["smoke_3cat"] = np.where(sprint["smoke_3cat"] == 4, np.nan, 
                                np.where(sprint["smoke_3cat"] == 3, 1, 0))

    continuous_vars_sprint = [
        "age", "sbp", "dbp", "n_agents", "egfr", "screat",
        "chr", "glur", "hdl", "trr", "umalcr", "bmi"
    ]

    binary_vars_sprint = [
        "female", "race_black", "smoke_3cat", 
        "aspirin", "statin", "sub_cvd"
    ]

    treatment = "intensive"
    outcome_col = "event_primary"

    sprint = sprint[continuous_vars_sprint + binary_vars_sprint + [treatment] + [outcome_col]]
    sprint[outcome_col] = np.where(sprint[outcome_col] == 1, 0, 1)

    # scaler = MinMaxScaler()

    # scaler.fit(
    #     np.concatenate(
    #         (
    #           sprint[continuous_vars_sprint].values,
    #           accord[continuous_vars_accord].values
    #         ), axis=0
    #     )
    # )

    # sprint[continuous_vars_sprint] = scaler.transform(sprint[continuous_vars_sprint].values)
    # accord[continuous_vars_accord] = scaler.transform(accord[continuous_vars_accord].values)

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(sprint)
    sprint = imp.transform(sprint)

    imp.fit(accord)
    accord = imp.transform(accord)
    
    return sprint[:, :-2], sprint[:, -2], sprint[:, -1], accord[:, :-2], accord[:, -2], accord[:, -1]
