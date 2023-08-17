import numpy as np
import pandas as pd

from sklearn import  model_selection
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler



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

        data[continuous_vars] = self._normalize_data(data[continuous_vars], "minmax")

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
        data = data[data.stroketype!=5]
        
        data["antiplat_rand"] = np.where(data["antiplat_rand"]== 1, 1, 0)
        data["atrialfib_rand"] = np.where(data["atrialfib_rand"]== 1, 1, 0)
        data["gender"] = np.where(data["gender"]== 2, 1, 0)
        
        data[treatment_col] = np.where(data[treatment_col]== 0, 1, 0)
        data[outcome_col] = np.where(data[outcome_col]== 1, 1, 0)
        data[continuous_vars] = self._normalize_data(data[continuous_vars], "minmax")

        data = pd.get_dummies(data, columns=cate_variables)

        self.continuous_indices = [data.columns.get_loc(col) for col in continuous_vars]
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

        data[continuous_vars] = self._normalize_data(data[continuous_vars], "minmax")

        self.continuous_indices = [data.columns.get_loc(col) for col in continuous_vars]
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

    def _normalize_data(self, x: np.ndarray, type:str):

        if type == "minmax":
            self.scaler = MinMaxScaler()

        elif type == "standard":
            self.scaler = StandardScaler()

        self.scaler.fit(x.values)
        x = self.scaler.transform(x.values)

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
            feature:int
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

