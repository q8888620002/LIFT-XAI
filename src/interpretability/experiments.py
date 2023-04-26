import sys
import os
import torch
import pickle as pkl
import xgboost as xgb

from sklearn import metrics

from scipy import stats
from copy import deepcopy

from pathlib import Path

#import catenets.models as cate_models

#### import CATE model
module_path = os.path.abspath(os.path.join('./CATENets/'))

if module_path not in sys.path:
    sys.path.append(module_path)

from catenets.models.jax import TNet, SNet,SNet1, SNet2, SNet3, DRNet, RANet, PWNet, RNet, XNet
import catenets.models as cate_models
import catenets.models.torch.tlearner as tlearner
import catenets.models.torch.pseudo_outcome_nets as pseudo_outcome_nets


import numpy as np
import pandas as pd

import src.interpretability.logger as log
from src.interpretability.explain import Explainer
from src.interpretability.datasets.data_loader import load
from src.interpretability.synthetic_simulate import (
    SyntheticSimulatorLinear,
    SyntheticSimulatorModulatedNonLinear,
)
from src.interpretability.utils import (
    LogisticRegression,
    attribution_accuracy,
    compute_pehe,
    attribution_ranking,
    attribution_insertion_deletion
)


class PredictiveSensitivity:
    """
    Sensitivity analysis for predictive scale.
    """

    def __init__(
        self,
        n_units_hidden: int = 100,
        n_layers: int = 2,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 256,
        n_iter: int = 1000,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        predictive_scales: list = [1e-3, 1e-2, 1e-1, 0.5,  1, 2],
        num_interactions: int = 1,
        synthetic_simulator_type: str = "linear",
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.predictive_scales = predictive_scales
        self.num_interactions = num_interactions
        self.synthetic_simulator_type = synthetic_simulator_type

    def run(
        self,
        dataset: str = "tcga_10",
        train_ratio: float = 0.8,
        num_important_features: int = 2,
        binary_outcome: bool = False,
        random_feature_selection: bool = True,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
            "naive_shap",
            "explain_with_missingness"
        ],
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features}."
        )

        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        if self.synthetic_simulator_type == "linear":
            sim = SyntheticSimulatorLinear(
                X_raw_train,
                num_important_features=num_important_features,
                random_feature_selection=random_feature_selection,
                seed=self.seed,
            )
        else:
            raise Exception("Unknown simulator type.")


        explainability_data = []
        insertion_deletion_data = []

        for predictive_scale in self.predictive_scales:
            log.info(f"Now working with predictive_scale = {predictive_scale}...")
            (
                X_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )

            encoder = cate_models.torch.base.MaskingModel(
                   n_unit_in=X_train.shape[1],
                   n_layers=2,
                   n_units=X_train.shape[1]//2,
                   n_iter=500,
                   batch_size=1024,
                   batch_norm=False,
                   nonlin="relu",
                   mask_dis = "Uniform"
            )

            encoder.fit(X_train)

            
            X_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )

            log.info("Fitting and explaining learners...")
            learners = {
                   # "TLearner": cate_models.torch.TLearner(
                   # X_train.shape[1],
                   # device = "cuda:1",
                   # binary_y=(len(np.unique(Y_train)) == 2),
                   # n_layers_out=2,
                   # n_units_out=100,
                   # batch_size=1024,
                   # n_iter=self.n_iter,
                   # batch_norm=False,
                   # nonlin="relu",
               # ),
               # "SLearner": cate_models.torch.SLearner(
               #     X_train.shape[1],
               #     device = "cuda:1",
               #     binary_y=(len(np.unique(Y_train)) == 2),
               #     n_layers_out=2,
               #     n_units_out=100,
               #     n_iter=self.n_iter,
               #     batch_size=1024,
               #     batch_norm=False,
               #     nonlin="relu",
               # ),
            #    "TARNet": cate_models.torch.TARNet(
            #        X_train.shape[1],
            #        binary_y=(len(np.unique(Y_train)) == 2),
            #        device = "cuda:1",
            #        n_layers_r=1,
            #        n_layers_out=1,
            #        n_units_out=100,
            #        n_units_r=100,
            #        batch_size=1024,
            #        n_iter=self.n_iter,
            #        batch_norm=False,
            #        nonlin="relu",
            #    ),
                 "XLearnerMask": pseudo_outcome_nets.XLearnerMask(
                     X_train.shape[1],
                     binary_y=(len(np.unique(Y_train)) == 2),
                     n_layers_out=2,
                     n_units_out=100,
                     n_iter=10,
                     batch_size=self.batch_size,
                     lr=1e-3,
                     patience=10,
                     batch_norm=False,
                     nonlin="relu",
                     device="cuda:1",
                     encoder = encoder
                 ),
                # "XLearner": pseudo_outcome_nets.XLearner(
                #      X_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=2,
                #      n_units_out=100,
                #      n_iter=self.n_iter,
                #      batch_size=self.batch_size,
                #      batch_norm=False,
                #      lr=1e-3,
                #      patience=10,
                #      nonlin="relu",
                #      device="cuda:1"
                #  ),
                #  "DRLearner": cate_models.torch.DRLearner(
                #      X_train.shape[1],
                #      device = "cuda:1",
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=self.n_layers,
                #      n_units_out=self.n_units_hidden,
                #      n_iter=self.n_iter,
                #      lr=1e-3,
                #      patience=10,
                #      batch_size=self.batch_size,
                #      batch_norm=False,
                #      nonlin="relu"
                #  ),

                #  "DRLearnerMask": pseudo_outcome_nets.DRLearnerMask(  
                #      X_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      device="cuda:1",
                #      n_layers_out=self.n_layers,
                #      n_units_out=self.n_units_hidden,
                #      n_iter=2,
                #      batch_size=256,
                #      batch_norm=False,
                #      lr=1e-3,
                #      patience=10,
                #      nonlin="relu",
                #      mask_dis = "Uniform"
                #      ),
                #  "DRLearnerHalf": pseudo_outcome_nets.DRLearnerMaskHalf(
                #      X_train.shape[1],
                #      device = "cuda:0",
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=self.n_layers,
                #      n_units_out=self.n_units_hidden,
                #      n_iter=self.n_iter,
                #      lr=1e-3,
                #      patience=10,
                #      batch_size=self.batch_size,
                #      batch_norm=False,
                #      nonlin="relu"
                #  ),

            }

            learner_explainers = {}
            learner_explanations = {}
            learner_explaintion_lists = {}

            for learner_name in learners:                    
                if not  "mask" in learner_name.lower():
                    learner_explaintion_lists[learner_name] = [           
                    "integrated_gradients",
                    "shapley_value_sampling",
                    "naive_shap"
                    ]
                elif  "half" in learner_name.lower():
                    learner_explaintion_lists[learner_name] = [
                        "shapley_value_sampling"
                    ]
                else:
                    learner_explaintion_lists[learner_name] = [
                        "explain_with_missingness"
                    ]

                log.info(f"Fitting {learner_name}.")

                # if learner_name == "DRLearnerMask":
                #     pretrained_te = deepcopy(learners["DRLearner"]._te_estimator)
                #     learners[learner_name]._add_units(pretrained_te)

                learners[learner_name].fit(X=X_train, y=Y_train, w=W_train)
                learner_explainers[learner_name] = Explainer(
                    learners[learner_name],
                    feature_names=list(range(X_train.shape[1])),
                    explainer_list=learner_explaintion_lists[learner_name],
                )
                log.info(f"Explaining {learner_name}.")
                learner_explanations[learner_name] = learner_explainers[learner_name].explain(
                    X_test[:self.explainer_limit]
                )

            all_important_features = sim.get_all_important_features()
            pred_features = sim.get_predictive_features()
            prog_features = sim.get_prognostic_features()

            cate_test = sim.te(X_test)

            for learner_name in learners:
                for explainer_name in learner_explaintion_lists[learner_name]:

                    attribution_est = np.abs(
                        learner_explanations[learner_name][explainer_name]
                    )
                    acc_scores_all_features, acc_scores_all_features_score  = attribution_accuracy(
                        all_important_features, attribution_est
                    )
                    acc_scores_predictive_features, acc_scores_predictive_features_score = attribution_accuracy(
                        pred_features, attribution_est
                    )
                    acc_scores_prog_features, acc_scores_prog_features_score = attribution_accuracy(
                        prog_features, attribution_est
                    )
                    # Computing insertion & deletion results

                    if  not "mask" in learner_name.lower():
                        cate_pred = learners[learner_name].predict(X=X_test)
                        pate_model_name = "DRLearnerMask"
                    else:
                        prediction_mask = torch.ones(X_test.shape)
                        cate_pred = learners[learner_name].predict(X=X_test, M=prediction_mask)
                        pate_model_name = learner_name

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                    # Obtain feature importance rank
                    rank_indices = attribution_ranking(learner_explanations[learner_name][explainer_name])

                    # Using PATE to predict CATE
                    insertion_results, deletion_results = attribution_insertion_deletion(
                            X_test[:self.explainer_limit, :],
                            rank_indices,
                            learners[pate_model_name]
                    )
                    
                    insertion_deletion_data.append(
                        [
                            predictive_scale,
                            learner_name,
                            explainer_name,
                            insertion_results,
                            deletion_results,
                            rank_indices
                        ]
                    )

                    explainability_data.append(
                        [
                            predictive_scale,
                            learner_name,
                            explainer_name,
                            acc_scores_all_features,
                            acc_scores_all_features_score,
                            acc_scores_predictive_features,
                            acc_scores_predictive_features_score,
                            acc_scores_prog_features,
                            acc_scores_prog_features_score,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                        ]
                    )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Predictive Scale",
                "Learner",
                "Explainer",
                "All features ACC",
                "All features ACC Score ",
                "Pred features ACC",
                "Pred features ACC Score ",
                "Prog features ACC",
                "Prog features ACC Score ",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],
        )
        
        results_path = self.save_path / "results/drlearner/predictive_sensitivity/model_performance"
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path / f"predictive_scale_{dataset}_{num_important_features}_"
            f"{self.synthetic_simulator_type}_random_{random_feature_selection}_"
            f"binary_{binary_outcome}_seed{self.seed}.csv"
        )
        
        results_path = self.save_path / "results/drlearner/predictive_sensitivity/insertion_deletion"
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        with open( results_path / f"predictive_scale_{dataset}_{num_important_features}_"
            f"{self.synthetic_simulator_type}_random_{random_feature_selection}_"
            f"binary_{binary_outcome}_seed{self.seed}.pkl", 'wb') as handle:
            pkl.dump(insertion_deletion_data , handle)


class PredictiveAssignment:
    """
    Sensitivity analysis for predictive scale.
    """

    def __init__(
        self,
        n_units_hidden: int = 100,
        n_layers: int = 2,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 256,
        n_iter: int = 1000,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        predictive_scales: list = [1e-3, 1e-2, 1e-1, 0.5,  1, 2],
        num_interactions: int = 1,
        synthetic_simulator_type: str = "linear",
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.predictive_scales = predictive_scales
        self.num_interactions = num_interactions
        self.synthetic_simulator_type = synthetic_simulator_type

    def run(
        self,
        dataset: str = "tcga_10",
        train_ratio: float = 0.8,
        num_important_features: int = 2,
        binary_outcome: bool = False,
        random_feature_selection: bool = True,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
            "naive_shap",
            "explain_with_missingness"
        ],
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features}."
        )

        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        if self.synthetic_simulator_type == "linear":
            sim = SyntheticSimulatorLinear(
                X_raw_train,
                num_important_features=num_important_features,
                random_feature_selection=random_feature_selection,
                seed=self.seed,
            )
        else:
            raise Exception("Unknown simulator type.")


        explainability_data = []
        insertion_deletion_data = []

        for predictive_scale in self.predictive_scales:
            log.info(f"Now working with predictive_scale = {predictive_scale}...")
            (
                X_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )

            X_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )

            log.info("Fitting and explaining learners...")
            learners = {
                   # "TLearner": cate_models.torch.TLearner(
                   # X_train.shape[1],
                   # device = "cuda:1",
                   # binary_y=(len(np.unique(Y_train)) == 2),
                   # n_layers_out=2,
                   # n_units_out=100,
                   # batch_size=1024,
                   # n_iter=self.n_iter,
                   # batch_norm=False,
                   # nonlin="relu",
               # ),
               # "SLearner": cate_models.torch.SLearner(
               #     X_train.shape[1],
               #     device = "cuda:1",
               #     binary_y=(len(np.unique(Y_train)) == 2),
               #     n_layers_out=2,
               #     n_units_out=100,
               #     n_iter=self.n_iter,
               #     batch_size=1024,
               #     batch_norm=False,
               #     nonlin="relu",
               # ),
              #  "TARNet": cate_models.torch.TARNet(
              #      X_train.shape[1],
              #      device = "cuda:1",
              #      binary_y=(len(np.unique(Y_train)) == 2),
              #      n_layers_r=1,
              #      n_layers_out=1,
              #      n_units_out=100,
              #      n_units_r=100,
              #      batch_size=1024,
              #      n_iter=self.n_iter,
              #      batch_norm=False,
              #      nonlin="relu",
              #  ),
                #  "XLearnerMask": cate_models_masks.XLearnerMask(
                #      X_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=2,
                #      n_units_out=100,
                #      n_iter=10,
                #      batch_size=self.batch_size,
                #     lr=1e-3,
                #     patience=10,
                #      batch_norm=False,
                #      nonlin="relu",
                #      device="cuda:1"
                #  ),
                "XLearner": cate_models.torch.XLearner(
                     X_train.shape[1],
                     binary_y=(len(np.unique(Y_train)) == 2),
                     n_layers_out=2,
                     n_units_out=100,
                     n_iter=self.n_iter,
                     batch_size=self.batch_size,
                     batch_norm=False,
                     lr=1e-3,
                     patience=10,
                     nonlin="relu",
                     device="cuda:1"
                 ),
                #  "DRLearner": cate_models.torch.DRLearner(
                #      X_train.shape[1],
                #      device = "cuda:1",
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=self.n_layers,
                #      n_units_out=self.n_units_hidden,
                #      n_iter=self.n_iter,
                #      lr=1e-3,
                #      patience=10,
                #      batch_size=self.batch_size,
                #      batch_norm=False,
                #      nonlin="relu"
                #  ),

                #  "DRLearnerMask": pseudo_outcome_nets.DRLearnerMask(  
                #      X_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      device="cuda:1",
                #      n_layers_out=self.n_layers,
                #      n_units_out=self.n_units_hidden,
                #      n_iter=self.n_iter,
                #      batch_size=256,
                #      batch_norm=False,
                #      lr=1e-3,
                #      patience=10,
                #      nonlin="relu",
                #      mask_dis = "Uniform"
                #      ),
                #  "DRLearnerHalf": pseudo_outcome_nets.DRLearnerMaskHalf(
                #      X_train.shape[1],
                #      device = "cuda:0",
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=self.n_layers,
                #      n_units_out=self.n_units_hidden,
                #      n_iter=self.n_iter,
                #      lr=1e-3,
                #      patience=10,
                #      batch_size=self.batch_size,
                #      batch_norm=False,
                #      nonlin="relu"
                #  ),

            }

            learner_explainers = {}
            learner_explanations = {}
            learner_explaintion_lists = {}

            for learner_name in learners:                    
                if not  "mask" in learner_name.lower():
                    learner_explaintion_lists[learner_name] = [  
                    "integrated_gradients",
                    "shapley_value_sampling",
                    "naive_shap",
                    ]
                elif  "half" in learner_name.lower():
                    learner_explaintion_lists[learner_name] = [
                        "shapley_value_sampling"
                    ]
                else:
                    learner_explaintion_lists[learner_name] = [
                        "explain_with_missingness"
                    ]

                log.info(f"Fitting {learner_name}.")

                if learner_name == "DRLearnerMask":
                    pretrained_te = deepcopy(learners["DRLearner"]._te_estimator)
                    learners[learner_name]._add_units(pretrained_te)

                learners[learner_name].fit(X=X_train, y=Y_train, w=W_train)
                learner_explainers[learner_name] = Explainer(
                    learners[learner_name],
                    feature_names=list(range(X_train.shape[1])),
                    explainer_list=learner_explaintion_lists[learner_name],
                )
                log.info(f"Explaining {learner_name}.")
                learner_explanations[learner_name] = learner_explainers[learner_name].explain(
                    X_test[:self.explainer_limit]
                )

            all_important_features = sim.get_all_important_features()
            pred_features = sim.get_predictive_features()
            prog_features = sim.get_prognostic_features()

            cate_test = sim.te(X_test)

            for learner_name in learners:
                for explainer_name in learner_explaintion_lists[learner_name]:

                    attribution_est = np.abs(
                        learner_explanations[learner_name][explainer_name]
                    )
                    acc_scores_all_features, acc_scores_all_features_score  = attribution_accuracy(
                        all_important_features, attribution_est
                    )
                    acc_scores_predictive_features, acc_scores_predictive_features_score = attribution_accuracy(
                        pred_features, attribution_est
                    )
                    acc_scores_prog_features, acc_scores_prog_features_score = attribution_accuracy(
                        prog_features, attribution_est
                    )
                    # Computing insertion & deletion results

                    if  not "mask" in learner_name.lower():
                        cate_pred = learners[learner_name].predict(X=X_test)
                        pate_model_name = "DRLearnerMask"
                    else:
                        prediction_mask = torch.ones(X_test.shape)
                        cate_pred = learners[learner_name].predict(X=X_test, M=prediction_mask)
                        pate_model_name = learner_name

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)
                    # Obtain feature global importance rank
   
                    rank_indices = np.argsort(np.sum(attribution_est, axis=0))[::-1]
                    # Relabel positive outcome

                    new_y_train = learners[learner_name].predict(X=X_train)
                    # setting threshold
                    
                    cate_threshold = torch.mean(new_y_train)
                    new_y_train = (new_y_train > cate_threshold ).float().detach().cpu().numpy()

                    new_y_test = learners[learner_name].predict(X=X_test)
                    new_y_test = (new_y_test > cate_threshold).float().detach().cpu().numpy()

                    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=self.seed)
                    new_X_train = X_train[:,rank_indices[:5]]
                    new_X_test = X_test[:,rank_indices[:5]]
                    xgb_model.fit(new_X_train, new_y_train)

                    y_pred = xgb_model.predict(new_X_test)
                    auroc = metrics.roc_auc_score(new_y_test, y_pred)
                    s_hat = X_test[np.where(y_pred == 1), :]

                    total_num = len(X_raw_test) + len(X_raw_train)
                    utility_index = np.mean(learners[learner_name].predict(X=s_hat).detach().cpu().numpy())*len(s_hat)/total_num

                    # baseline with random features

                    random_xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=self.seed)
                    np.random.shuffle(rank_indices)
                    new_X_train = X_train[:,rank_indices[:5]]
                    new_X_test = X_test[:,rank_indices[:5]]

                    random_xgb_model.fit(new_X_train, new_y_train)
                    y_pred = random_xgb_model.predict(new_X_test)

                    random_auroc = metrics.roc_auc_score(new_y_test, y_pred)
                    
                    s_hat_original = X_test[np.where(W_test == 1), :]

                    s_hat = X_test[np.where(y_pred == 1), :]

                    # if len(s_hat) <= 1:
                    #     random_utility_index = 0
                    # else:

                    random_utility_index = np.mean(learners[learner_name].predict(X=s_hat).detach().cpu().numpy())*len(s_hat)/total_num
                    random_assignment_utility_index = np.mean(learners[learner_name].predict(X=s_hat_original).detach().cpu().numpy())*len(s_hat)/total_num

                    explainability_data.append(
                        [
                            predictive_scale,
                            learner_name,
                            explainer_name,
                            acc_scores_all_features,
                            acc_scores_all_features_score,
                            acc_scores_predictive_features,
                            acc_scores_predictive_features_score,
                            acc_scores_prog_features,
                            acc_scores_prog_features_score,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                            auroc,
                            utility_index,
                            random_utility_index,
                            random_assignment_utility_index
                        ]
                    )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Predictive Scale",
                "Learner",
                "Explainer",
                "All features ACC",
                "All features ACC Score",
                "Pred features ACC",
                "Pred features ACC Score",
                "Prog features ACC",
                "Prog features ACC Score",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
                "AUROC",
                "utility_score",
                "utility_score_random_feature",
                "utility_score_random_assignment"
            ],
        )
        
        results_path = self.save_path / "results/predictive_sensitivity/assignment"
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path / f"predictive_scale_{dataset}_{num_important_features}_"
            f"{self.synthetic_simulator_type}_random_{random_feature_selection}_"
            f"binary_{binary_outcome}_seed{self.seed}.csv"
        )



class PredictiveSensitivityLoss:
    """
    Sensitivity analysis for predictive scale.
    """

    def __init__(
        self,
        n_units_hidden: int = 50,
        n_layers: int = 2,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 256,
        n_iter: int = 1500,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        predictive_scales: list = [1e-3, 1e-2, 1e-1, 0.5,  1, 2],
        num_interactions: int = 1,
        synthetic_simulator_type: str = "linear",
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.predictive_scales = predictive_scales
        self.num_interactions = num_interactions
        self.synthetic_simulator_type = synthetic_simulator_type

    def run(
        self,
        dataset: str = "tcga_10",
        train_ratio: float = 0.8,
        num_important_features: int = 2,
        binary_outcome: bool = False,
        random_feature_selection: bool = True,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
            "naive_shap",
            "explain_with_missingness"
        ],
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features}."
        )

        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        if self.synthetic_simulator_type == "linear":
            sim = SyntheticSimulatorLinear(
                X_raw_train,
                num_important_features=num_important_features,
                random_feature_selection=random_feature_selection,
                seed=self.seed,
            )
        else:
            raise Exception("Unknown simulator type.")


        explainability_data = []
        insertion_deletion_data = []

        for predictive_scale in self.predictive_scales:
            log.info(f"Now working with predictive_scale = {predictive_scale}...")
            (
                X_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )

            X_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )

            log.info("Fitting and explaining learners...")
            learners = {
                   # "TLearner": cate_models.torch.TLearner(
                   # X_train.shape[1],
                   # device = "cuda:1",
                   # binary_y=(len(np.unique(Y_train)) == 2),
                   # n_layers_out=2,
                   # n_units_out=100,
                   # batch_size=1024,
                   # n_iter=self.n_iter,
                   # batch_norm=False,
                   # nonlin="relu",
               # ),
               # "SLearner": cate_models.torch.SLearner(
               #     X_train.shape[1],
               #     device = "cuda:1",
               #     binary_y=(len(np.unique(Y_train)) == 2),
               #     n_layers_out=2,
               #     n_units_out=100,
               #     n_iter=self.n_iter,
               #     batch_size=1024,
               #     batch_norm=False,
               #     nonlin="relu",
               # ),
              #  "TARNet": cate_models.torch.TARNet(
              #      X_train.shape[1],
              #      device = "cuda:1",
              #      binary_y=(len(np.unique(Y_train)) == 2),
              #      n_layers_r=1,
              #      n_layers_out=1,
              #      n_units_out=100,
              #      n_units_r=100,
              #      batch_size=1024,
              #      n_iter=self.n_iter,
              #      batch_norm=False,
              #      nonlin="relu",
              #  ),
                #  "XLearnerMask": cate_models_masks.XLearnerMask(
                #      X_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=2,
                #      n_units_out=100,
                #      n_iter=10,
                #      batch_size=self.batch_size,
                #     lr=1e-3,
                #     patience=10,
                #      batch_norm=False,
                #      nonlin="relu",
                #      device="cuda:1"
                #  ),
                # "XLearner": cate_models.torch.XLearner(
                #      X_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=2,
                #      n_units_out=100,
                #      n_iter=self.n_iter,
                #      batch_size=self.batch_size,
                #      batch_norm=False,
                #      lr=1e-3,
                #      patience=10,
                #      nonlin="relu",
                #      device="cuda:1"
                #  ),
                 "DRLearnerMask": pseudo_outcome_nets.DRLearnerMask(  
                     X_train.shape[1],
                     binary_y=(len(np.unique(Y_train)) == 2),
                     device="cuda:1",
                     n_layers_out=self.n_layers,
                     n_units_out=self.n_units_hidden,
                     n_iter=self.n_iter,
                     batch_size=256,
                     batch_norm=False,
                     lr=1e-3,
                     patience=10,
                     nonlin="relu",
                     mask_dis="Uniform"
                     ),
                 "DRLearnerMaskBeta": pseudo_outcome_nets.DRLearnerMask(  
                     X_train.shape[1],
                     binary_y=(len(np.unique(Y_train)) == 2),
                     device="cuda:1",
                     n_layers_out=self.n_layers,
                     n_units_out=self.n_units_hidden,
                     n_iter=self.n_iter,
                     batch_size=256,
                     batch_norm=False,
                     lr=1e-3,
                     patience=10,
                     nonlin="relu",
                     mask_dis="Beta"
                     ),
                 "DRLearnerMask1": pseudo_outcome_nets.DRLearnerMask1(  
                     X_train.shape[1],
                     binary_y=(len(np.unique(Y_train)) == 2),
                     device="cuda:1",
                     n_layers_out=self.n_layers,
                     n_units_out=self.n_units_hidden,
                     n_iter=self.n_iter,
                     batch_size=256,
                     batch_norm=False,
                     lr=1e-3,
                     patience=10,
                     nonlin="relu",
                     mask_dis="Uniform"
                     ),
                 "DRLearnerMask0": pseudo_outcome_nets.DRLearnerMask0(  
                     X_train.shape[1],
                     binary_y=(len(np.unique(Y_train)) == 2),
                     device="cuda:1",
                     n_layers_out=self.n_layers,
                     n_units_out=self.n_units_hidden,
                     n_iter=self.n_iter,
                     batch_size=256,
                     batch_norm=False,
                     lr=1e-3,
                     patience=10,
                     nonlin="relu",
                     mask_dis="Uniform"
                     ),
                 "DRLearnerHalfMask": pseudo_outcome_nets.DRLearnerMaskHalf(
                     X_train.shape[1],
                     device = "cuda:0",
                     binary_y=(len(np.unique(Y_train)) == 2),
                     n_layers_out=self.n_layers,
                     n_units_out=self.n_units_hidden,
                     n_iter=self.n_iter,
                     lr=1e-3,
                     patience=10,
                     batch_size=self.batch_size,
                     batch_norm=False,
                     nonlin="relu",
                     mask_dis="Uniform"
                 ),
                 "DRLearner": cate_models.torch.DRLearner(
                     X_train.shape[1],
                     device = "cuda:0",
                     binary_y=(len(np.unique(Y_train)) == 2),
                     n_layers_out=self.n_layers,
                     n_units_out=self.n_units_hidden,
                     n_iter=self.n_iter,
                     lr=1e-3,
                     patience=10,
                     batch_size=self.batch_size,
                     batch_norm=False,
                     nonlin="relu"
                 ),
            }

            learner_explainers = {}
            learner_explanations = {}
            learner_explaintion_lists = {}

            for learner_name in learners:                    

                log.info(f"Fitting {learner_name}.")
                learners[learner_name].fit(X=X_train, y=Y_train, w=W_train)
                cate_test = sim.te(X_test)

                if  not "mask" in learner_name.lower():
                    print(learner_name)
                    cate_pred = learners[learner_name].predict(X=X_test)

                else:
                    prediction_mask = torch.ones(X_test.shape)
                    cate_pred = learners[learner_name].predict(X=X_test, M=prediction_mask)

                pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                explainability_data.append(
                    [
                        predictive_scale,
                        learner_name,
                        pehe_test,
                        np.mean(cate_test),
                        np.var(cate_test),
                        pehe_test / np.sqrt(np.var(cate_test)),
                    ]
                )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Predictive Scale",
                "Learner",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],
        )
        
        results_path = self.save_path / "results/losses/default/predictive_sensitivity/model_performance"
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path / f"predictive_scale_{dataset}_{num_important_features}_"
            f"{self.synthetic_simulator_type}_random_{random_feature_selection}_"
            f"binary_{binary_outcome}_seed{self.seed}.csv"
        )

class PredictiveSensitivityHeldOutOne:
    """
    Held out one analysis for predictive scale.
    """

    def __init__(
        self,
        n_units_hidden: int = 50,
        n_layers: int = 2,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 256,
        n_iter: int = 1000,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        predictive_scales: list = [1e-3, 1e-2, 1e-1, 0.5,  1, 2, 5, 10],
        num_interactions: int = 1,
        synthetic_simulator_type: str = "linear",
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.predictive_scales = predictive_scales
        self.num_interactions = num_interactions
        self.synthetic_simulator_type = synthetic_simulator_type

    def run(
        self,
        dataset: str = "tcga_10",
        train_ratio: float = 0.8,
        num_important_features: int = 2,
        binary_outcome: bool = False,
        random_feature_selection: bool = True,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
            "naive_shap",
            "explain_with_missingness"
        ],
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features}."
        )

        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        if self.synthetic_simulator_type == "linear":
            sim = SyntheticSimulatorLinear(
                X_raw_train,
                num_important_features=num_important_features,
                random_feature_selection=random_feature_selection,
                seed=self.seed,
            )
        else:
            raise Exception("Unknown simulator type.")


        explainability_data = []
        held_out_data = []

        for predictive_scale in self.predictive_scales:
            log.info(f"Now working with predictive_scale = {predictive_scale}...")
            (
                X_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )

            X_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )
                
            log.info("Fitting and explaining learners...")


            learners = {
                   "TLearner": tlearner.TLearner(
                   X_train.shape[1],
                   device = "cuda:1",
                   binary_y=(len(np.unique(Y_train)) == 2),
                   n_layers_out=2,
                   n_units_out=100,
                   batch_size=self.batch_size,
                   n_iter=self.n_iter,
                   batch_norm=False,
                   nonlin="relu",
               ),
                   "TLearnerMask": tlearner.TLearnerMask(
                   X_train.shape[1],
                   device = "cuda:1",
                   binary_y=(len(np.unique(Y_train)) == 2),
                   n_layers_out=2,
                   n_units_out=100,
                   batch_size=self.batch_size,
                   n_iter=self.n_iter,
                   batch_norm=False,
                   nonlin="relu",
               ),
               # "SLearner": cate_models.torch.SLearner(
               #     X_train.shape[1],
               #     device = "cuda:1",
               #     binary_y=(len(np.unique(Y_train)) == 2),
               #     n_layers_out=2,
               #     n_units_out=100,
               #     n_iter=self.n_iter,
               #     batch_size=1024,
               #     batch_norm=False,
               #     nonlin="relu",
               # ),
              #  "TARNet": cate_models.torch.TARNet(
              #      X_train.shape[1],
              #      device = "cuda:1",
              #      binary_y=(len(np.unique(Y_train)) == 2),
              #      n_layers_r=1,
              #      n_layers_out=1,
              #      n_units_out=100,
              #      n_units_r=100,
              #      batch_size=1024,
              #      n_iter=self.n_iter,
              #      batch_norm=False,
              #      nonlin="relu",
              #  ),
                #  "XLearnerMask": pseudo_outcome_nets.XLearnerMask(
                #      X_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=2,
                #      n_units_out=100,
                #      n_iter=self.n_iter,
                #      batch_size=self.batch_size,
                #      lr=1e-3,
                #      patience=10,
                #      batch_norm=False,
                #      nonlin="relu",
                #      device="cuda:1"
                #  ),
                # "XLearner": cate_models.torch.XLearner(
                #      X_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=2,
                #      n_units_out=100,
                #      n_iter=self.n_iter,
                #      batch_size=self.batch_size,
                #      batch_norm=False,
                #      lr=1e-3,
                #      patience=10,
                #      nonlin="relu",
                #      device="cuda:1"
                #  ),
                #   "DRLearner": cate_models.torch.DRLearner(
                #       X_train.shape[1],
                #       device = "cuda:1",
                #       binary_y=(len(np.unique(Y_train)) == 2),
                #       n_layers_out=2,
                #       n_units_out=100,
                #       n_iter=self.n_iter,
                #       lr=1e-3,
                #       patience=10,
                #       batch_size=self.batch_size,
                #       batch_norm=False,
                #       nonlin="relu"
                #   ),
                #   "DRLearnerMask": pseudo_outcome_nets.DRLearnerMask(  
                #       X_train.shape[1],
                #       binary_y=(len(np.unique(Y_train)) == 2),
                #       device="cuda:1",
                #       n_layers_out=2,
                #       n_units_out=100,
                #       n_iter=self.n_iter,
                #       batch_size=self.batch_size,
                #       batch_norm=False,
                #       lr=1e-4,
                #       patience=10,
                #       nonlin="relu",
                #       mask_dis="Uniform"
                #       ),

                #   "RALearnerMask": pseudo_outcome_nets.RALearnerMask(
                #       X_train.shape[1],
                #       device = "cuda:0",
                #       binary_y=(len(np.unique(Y_train)) == 2),
                #       n_layers_out=2,
                #       n_units_out=100,
                #       n_iter=self.n_iter,
                #       lr=1e-3,
                #       patience=10,
                #       batch_size=self.batch_size,
                #       batch_norm=False,
                #       nonlin="relu"
                #   ),
                #   "RALearner": pseudo_outcome_nets.RALearner(
                #       X_train.shape[1],
                #       device = "cuda:0",
                #       binary_y=(len(np.unique(Y_train)) == 2),
                #       n_layers_out=2,
                #       n_units_out=100,
                #       n_iter=self.n_iter,
                #       lr=1e-3,
                #       patience=10,
                #       batch_size=self.batch_size,
                #       batch_norm=False,
                #       nonlin="relu"
                #   ),
            }

            learner_explainers = {}
            learner_explanations = {}
            learner_explaintion_lists = {}

            for learner_name in learners:                    
                if not  "mask" in learner_name.lower():
                    learner_explaintion_lists[learner_name] = [           
                        "integrated_gradients",
                        "shapley_value_sampling",
                        "naive_shap"
                    ]
                else:
                    learner_explaintion_lists[learner_name] = [
                        "explain_with_missingness"
                    ]

                log.info(f"Fitting {learner_name}.")
                if learner_name == "DRLearnerMask":
                    pretrained_te = deepcopy(learners["DRLearner"]._te_estimator)
                    learners[learner_name]._add_units(pretrained_te)

                learners[learner_name].fit(X=X_train, y=Y_train, w=W_train)

                # Obtaining explanation 

                learner_explainers[learner_name] = Explainer(
                    learners[learner_name],
                    feature_names=list(range(X_train.shape[1])),
                    explainer_list=learner_explaintion_lists[learner_name],
                )
                log.info(f"Explaining {learner_name}.")
                learner_explanations[learner_name] = learner_explainers[learner_name].explain(
                    X_test[:self.explainer_limit]
                )

            cate_test = sim.te(X_test)

            for learner_name in learners:
                for explainer_name in learner_explaintion_lists[learner_name]:

                    if  not "mask" in learner_name.lower():
                        cate_pred = learners[learner_name].predict(X=X_test)
                    else:
                        prediction_mask = torch.ones(X_test.shape)
                        cate_pred = learners[learner_name].predict(X=X_test, M=torch.ones((X_test.shape)))

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                    explainability_data.append(
                        [
                            predictive_scale,
                            learner_name,
                            explainer_name,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                        ]
                    )

            # Held out experiment - iterating x_s = D \ {i} for all i in features. 

            for feature_index in range(X_train.shape[1]):
                
                masks = np.ones((X_train.shape[1]))
                masks[feature_index] = 0

                X_train_subset = X_train[:, masks.astype(bool)]
                X_test_subset = X_test[:, masks.astype(bool)]

                X_train_mask = np.copy(X_train)
                X_train_mask[:, feature_index] = 0

                _ , W_train_pate , Y_train_pate,_ ,_ ,_ ,  = sim.simulate_dataset(
                    X_train_mask,
                    predictive_scale=predictive_scale,
                    binary_outcome=binary_outcome,
                )
                

                pate_learners = {
                    "TLearner": tlearner.TLearner(
                        X_train_subset.shape[1],
                        device="cuda:1",
                        binary_y=(len(np.unique(Y_train)) == 2),
                        n_layers_out=2,
                        n_units_out=100,
                        n_iter=self.n_iter,
                        batch_size=self.batch_size,
                        batch_norm=False,
                        lr=1e-3,
                        nonlin="relu"
                 ),
                #     "DRLearner": pseudo_outcome_nets.DRLearner(
                #         X_train_subset.shape[1],
                #         device="cuda:1",
                #         binary_y=(len(np.unique(Y_train)) == 2),
                #         n_layers_out=2,
                #         n_units_out=100,
                #         n_iter=self.n_iter,
                #         batch_size=self.batch_size,
                #         batch_norm=False,
                #         lr=1e-3,
                #         patience=10,
                #         nonlin="relu"
                #  )
                }

                subset_explainer_list = [
                    "integrated_gradients",
                    "shapley_value_sampling",
                    "naive_shap",
                    "explain_with_missingness",
                ]
                
                for learner_name in pate_learners:
                    # model_path = f"models/{dataset}_{num_important_features}/predictive_scale_{predictive_scale}_feature_index_{feature_index}_seed{self.seed}.pt"

                    # try:
                    #     log.info("checking if PATE models"+ learner_name + " exists.")
                    #     pate_learners[learner_name].load_state_dict(torch.load(model_path))
                    #     pate_learners[learner_name].eval()
                    
                    # except:
                    log.info("Fitting PATE models"+ learner_name)

                    pate_learners[learner_name].fit(
                        # nuisance function params
                        X=X_train_subset,
                        # X_subset=X_train_subset,
                        y=Y_train,
                        w=W_train,
                    )

                        # torch.save(pate_learners[learner_name].state_dict(), model_path)

                    # pate_learners[learner_name].fit(
                    #     # nuisance function params
                    #     X=X_train_subset,
                    #     y=Y_train,
                    #     w=W_train,
                    # )

                    cate_pred = learners[learner_name].predict(X=X_test[:self.explainer_limit]).detach().cpu().numpy()
                    

                    pate_pred = pate_learners[learner_name].predict( 
                        X=X_test_subset[:self.explainer_limit]
                    ).detach().cpu().numpy()

                    prediction_mask =  torch.ones((X_test[:self.explainer_limit ].shape))
                    prediction_mask[:, feature_index] = 0
                    pate_mask_pred = learners["TLearnerMask"].predict(X_test[:self.explainer_limit ], prediction_mask).detach().cpu().numpy()

                    pate_pehe = np.sqrt(np.square(pate_pred - pate_mask_pred))

                    # loading attribution score from learner_explanations

                    for explainer_name in subset_explainer_list:
                        if explainer_name == "explain_with_missingness":
                            learner_name += "Mask"
                        elif explainer_name == "shapley_value_sampling_half_mask":
                            learner_name = "DRLearnerHalf"
                            explainer_name = "shapley_value_sampling"

                        attribution = learner_explanations[learner_name][explainer_name][:, feature_index]

                        held_out_data.append(
                            [
                                predictive_scale,
                                feature_index,
                                learner_name,
                                explainer_name, 
                                attribution,
                                pate_pred,
                                pate_mask_pred,
                                cate_pred,
                                pate_pehe/np.sqrt(np.var(pate_pred)),
                            ]
                        )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Predictive Scale",
                "Learner",
                "Explainer",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],
        )
        
        results_path = self.save_path / "results/held_out/tlearner/predictive_sensitivity/model_performance"
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path / f"predictive_scale_{dataset}_{num_important_features}_"
            f"{self.synthetic_simulator_type}_random_{random_feature_selection}_"
            f"binary_{binary_outcome}_seed{self.seed}.csv"
        )
        
        results_path = self.save_path / "results/held_out/tlearner/predictive_sensitivity/held_out/"
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        with open( results_path / f"predictive_scale_{dataset}_{num_important_features}_"
            f"{self.synthetic_simulator_type}_random_{random_feature_selection}_"
            f"binary_{binary_outcome}_seed{self.seed}.pkl", 'wb') as handle:
            pkl.dump(held_out_data , handle)

class NonLinearitySensitivity:
    """
    Sensitivity analysis for nonlinearity in prognostic and predictive functions.
    """

    def __init__(
        self,
        n_units_hidden: int = 100,
        n_layers: int = 2,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 256,
        n_iter: int = 1000,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        nonlinearity_scales: list = [0.0, 0.2, 0.5, 0.7, 1.0],
        predictive_scale: float = 1,
        synthetic_simulator_type: str = "random",
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.nonlinearity_scales = nonlinearity_scales
        self.predictive_scale = predictive_scale
        self.synthetic_simulator_type = synthetic_simulator_type

    def run(
        self,
        dataset: str = "tcga_100",
        num_important_features: int = 15,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
            "naive_shap"
        ],
        train_ratio: float = 0.8,
        binary_outcome: bool = False,
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features}."
        )
        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        explainability_data = []
        insertion_deletion_data = []

        for nonlinearity_scale in self.nonlinearity_scales:
            log.info(f"Now working with a nonlinearity scale {nonlinearity_scale}...")
            sim = SyntheticSimulatorModulatedNonLinear(
                X_raw_train,
                num_important_features=num_important_features,
                non_linearity_scale=nonlinearity_scale,
                seed=self.seed,
                selection_type=self.synthetic_simulator_type,
            )
            (
                X_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=self.predictive_scale,
                binary_outcome=binary_outcome,
            )
            X_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=self.predictive_scale,
                binary_outcome=binary_outcome,
            )


            log.info("Fitting and explaining learners...")
            learners = {
                # "TLearner": cate_models.torch.TLearner(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                # "SLearner": cate_models.torch.SLearner(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     n_iter=self.n_iter,
                #     batch_size=1024,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                # "TARNet": cate_models.torch.TARNet(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_r=1,
                #     n_layers_out=1,
                #     n_units_out=100,
                #     n_units_r=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                #  "XLearnerMask": pseudo_outcome_nets.XLearnerMask(
                #      X_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=2,
                #      n_units_out=100,
                #      n_iter=10,
                #      batch_size=self.batch_size,
                #     lr=1e-3,
                #     patience=10,
                #      batch_norm=False,
                #      nonlin="relu",
                #      device="cuda:1"
                #  ),
                # "XLearner": pseudo_outcome_nets.XLearner(
                #      X_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=2,
                #      n_units_out=100,
                #      n_iter=self.n_iter,
                #      batch_size=self.batch_size,
                #      batch_norm=False,
                #      lr=1e-3,
                #      patience=10,
                #      nonlin="relu",
                #      device="cuda:1"
                #  ),

                #  "DRLearnerHalf": pseudo_outcome_nets.DRLearnerMaskHalf(
                #      X_train.shape[1],
                #      device = "cuda:0",
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=self.n_layers,
                #      n_units_out=self.n_units_hidden,
                #      n_iter=self.n_iter,
                #      lr=1e-3,
                #      patience=10,
                #      batch_size=self.batch_size,
                #      batch_norm=False,
                #      nonlin="relu"
                #  ),
                 "DRLearner": cate_models.torch.DRLearner(
                     X_train.shape[1],
                     device = "cuda:1",
                     binary_y=(len(np.unique(Y_train)) == 2),
                     n_layers_out=self.n_layers,
                     n_units_out=self.n_units_hidden,
                     n_iter=self.n_iter,
                     lr=1e-3,
                     patience=10,
                     batch_size=self.batch_size,
                     batch_norm=False,
                     nonlin="relu"
                 ),
                 "DRLearnerMask": pseudo_outcome_nets.DRLearnerMask(  
                     X_train.shape[1],
                     binary_y=(len(np.unique(Y_train)) == 2),
                     device="cuda:1",
                     n_layers_out=self.n_layers,
                     n_units_out=self.n_units_hidden,
                     n_iter=self.n_iter,
                     batch_size=self.batch_size,
                     batch_norm=False,
                     lr=1e-3,
                     patience=10,
                     nonlin="relu",
                     mask_dis="Uniform"
                     ),
            }

            learner_explainers = {}
            learner_explaintion_lists = {}
            learner_explanations = {}

            for learner_name in learners:                    
                if not  "mask" in learner_name.lower():
                    learner_explaintion_lists[learner_name] = [           
                                                        "integrated_gradients",
                                                        "shapley_value_sampling",
                                                        "naive_shap"
                                                    ]
                elif "half" in learner_name.lower():
                    learner_explaintion_lists[learner_name] = [
                        "shapley_value_sampling"
                        ]
                else:
                    learner_explaintion_lists[learner_name] = [
                        "explain_with_missingness"
                        ]

                log.info(f"Fitting {learner_name}.")

                if learner_name == "DRLearnerMask":
                    pretrained_te = deepcopy(learners["DRLearner"]._te_estimator)
                    learners[learner_name]._add_units(pretrained_te)
                
                learners[learner_name].fit(X=X_train, y=Y_train, w=W_train)
                learner_explainers[learner_name] = Explainer(
                    learners[learner_name],
                    feature_names=list(range(X_train.shape[1])),
                    explainer_list=learner_explaintion_lists[learner_name],
                )

                log.info(f"Explaining {learner_name}.")
                learner_explanations[learner_name] = learner_explainers[learner_name].explain(
                    X_test[:self.explainer_limit]
                )

            all_important_features = sim.get_all_important_features()
            pred_features = sim.get_predictive_features()
            prog_features = sim.get_prognostic_features()

            cate_test = sim.te(X_test)

            for learner_name in learners:
                for explainer_name in learner_explaintion_lists[learner_name]:

                    attribution_est = np.abs(
                        learner_explanations[learner_name][explainer_name]
                    )
                    acc_scores_all_features, acc_scores_all_features_score  = attribution_accuracy(
                        all_important_features, attribution_est
                    )
                    acc_scores_predictive_features, acc_scores_predictive_features_score = attribution_accuracy(
                        pred_features, attribution_est
                    )
                    acc_scores_prog_features, acc_scores_prog_features_score = attribution_accuracy(
                        prog_features, attribution_est
                    )
                    ### computing insertion/deletion results

                    if  not "mask" in learner_name.lower():
                        cate_pred = learners[learner_name].predict(X=X_test)
                        pate_model_name =  "DRLearnerMask"
                    else:
                        prediction_mask = torch.ones(X_test.shape)
                        cate_pred = learners[learner_name].predict(X=X_test, M=prediction_mask)
                        pate_model_name = learner_name

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                    # Obtain feature importance rank
                    rank_indices = attribution_ranking(learner_explanations[learner_name][explainer_name])

                    # Using PATE to predict CATE
                    insertion_results, deletion_results = attribution_insertion_deletion(
                            X_test[:self.explainer_limit, :],
                            rank_indices,
                            learners[pate_model_name]
                    )

                    insertion_deletion_data.append(
                        [
                            nonlinearity_scale,
                            learner_name,
                            explainer_name,
                            insertion_results,
                            deletion_results,
                            rank_indices
                        ]
                    )

                    explainability_data.append(
                        [
                            nonlinearity_scale,
                            learner_name,
                            explainer_name,
                            acc_scores_all_features,
                            acc_scores_all_features_score,
                            acc_scores_predictive_features,
                            acc_scores_predictive_features_score,
                            acc_scores_prog_features,
                            acc_scores_prog_features_score,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                        ]
                    )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Nonlinearity Scale",
                "Learner",
                "Explainer",
                "All features ACC",
                "All features ACC Score ",
                "Pred features ACC",
                "Pred features ACC Score ",
                "Prog features ACC",
                "Prog features ACC Score ",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],
        )
        
        results_path = (
            self.save_path
            / f"results/drlearner_missing=-1/nonlinearity_sensitivity/insertion_deletion/{self.synthetic_simulator_type}"
        )
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path
            / f"{dataset}_{num_important_features}_binary_{binary_outcome}_seed{self.seed}.csv"
        )

        results_path = self.save_path / "results/drlearner_missing=-1/nonlinearity_sensitivity/insertion_deletion/insertion_deletion"
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        with open( results_path
            / f"{dataset}_{num_important_features}_binary_{binary_outcome}_seed{self.seed}.pkl", 'wb') as handle:
            pkl.dump(insertion_deletion_data , handle)




class NonlinearitySensitivityLoss:
    """
    Sensitivity analysis for nonlinearity in prognostic and predictive functions.
    """

    def __init__(
        self,
        n_units_hidden: int = 50,
        n_layers: int = 2,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 256,
        n_iter: int = 1500,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        nonlinearity_scales: list = [0.0, 0.2, 0.5, 0.7, 1.0],
        predictive_scale: float = 1,
        synthetic_simulator_type: str = "random",
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.nonlinearity_scales = nonlinearity_scales
        self.predictive_scale = predictive_scale
        self.synthetic_simulator_type = synthetic_simulator_type

    def run(
        self,
        dataset: str = "tcga_100",
        num_important_features: int = 15,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
            "naive_shap"
        ],
        train_ratio: float = 0.8,
        binary_outcome: bool = False,
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features}."
        )
        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        explainability_data = []
        insertion_deletion_data = []

        for nonlinearity_scale in self.nonlinearity_scales:
            log.info(f"Now working with a nonlinearity scale {nonlinearity_scale}...")
            sim = SyntheticSimulatorModulatedNonLinear(
                X_raw_train,
                num_important_features=num_important_features,
                non_linearity_scale=nonlinearity_scale,
                seed=self.seed,
                selection_type=self.synthetic_simulator_type,
            )
            (
                X_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=self.predictive_scale,
                binary_outcome=binary_outcome,
            )
            X_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=self.predictive_scale,
                binary_outcome=binary_outcome,
            )


            log.info("Fitting and explaining learners...")
            learners = {
                # "TLearner": cate_models.torch.TLearner(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                # "SLearner": cate_models.torch.SLearner(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     n_iter=self.n_iter,
                #     batch_size=1024,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                # "TARNet": cate_models.torch.TARNet(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_r=1,
                #     n_layers_out=1,
                #     n_units_out=100,
                #     n_units_r=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                #  "XLearnerMask": pseudo_outcome_nets.XLearnerMask(
                #      X_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=2,
                #      n_units_out=100,
                #      n_iter=10,
                #      batch_size=self.batch_size,
                #     lr=1e-3,
                #     patience=10,
                #      batch_norm=False,
                #      nonlin="relu",
                #      device="cuda:1"
                #  ),
                # "XLearner": pseudo_outcome_nets.XLearner(
                #      X_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=2,
                #      n_units_out=100,
                #      n_iter=self.n_iter,
                #      batch_size=self.batch_size,
                #      batch_norm=False,
                #      lr=1e-3,
                #      patience=10,
                #      nonlin="relu",
                #      device="cuda:1"
                #  ),
                 "DRLearnerMask": pseudo_outcome_nets.DRLearnerMask(  
                     X_train.shape[1],
                     binary_y=(len(np.unique(Y_train)) == 2),
                     device="cuda:1",
                     n_layers_out=self.n_layers,
                     n_units_out=self.n_units_hidden,
                     n_iter=self.n_iter,
                     batch_size=256,
                     batch_norm=False,
                     lr=1e-3,
                     patience=10,
                     nonlin="relu",
                     mask_dis="Uniform"
                     ),
                 "DRLearnerMaskBeta": pseudo_outcome_nets.DRLearnerMask(  
                     X_train.shape[1],
                     binary_y=(len(np.unique(Y_train)) == 2),
                     device="cuda:1",
                     n_layers_out=self.n_layers,
                     n_units_out=self.n_units_hidden,
                     n_iter=self.n_iter,
                     batch_size=256,
                     batch_norm=False,
                     lr=1e-3,
                     patience=10,
                     nonlin="relu",
                     mask_dis="Beta"
                     ),
                 "DRLearnerMask1": pseudo_outcome_nets.DRLearnerMask1(  
                     X_train.shape[1],
                     binary_y=(len(np.unique(Y_train)) == 2),
                     device="cuda:1",
                     n_layers_out=self.n_layers,
                     n_units_out=self.n_units_hidden,
                     n_iter=self.n_iter,
                     batch_size=256,
                     batch_norm=False,
                     lr=1e-3,
                     patience=10,
                     nonlin="relu",
                     mask_dis="Uniform"
                     ),
                 "DRLearnerMask0": pseudo_outcome_nets.DRLearnerMask0(  
                     X_train.shape[1],
                     binary_y=(len(np.unique(Y_train)) == 2),
                     device="cuda:1",
                     n_layers_out=self.n_layers,
                     n_units_out=self.n_units_hidden,
                     n_iter=self.n_iter,
                     batch_size=256,
                     batch_norm=False,
                     lr=1e-3,
                     patience=10,
                     nonlin="relu",
                     mask_dis="Uniform"
                     ),
                 "DRLearnerHalfMask": pseudo_outcome_nets.DRLearnerMaskHalf(
                     X_train.shape[1],
                     device = "cuda:0",
                     binary_y=(len(np.unique(Y_train)) == 2),
                     n_layers_out=self.n_layers,
                     n_units_out=self.n_units_hidden,
                     n_iter=self.n_iter,
                     lr=1e-3,
                     patience=10,
                     batch_size=self.batch_size,
                     batch_norm=False,
                     nonlin="relu",
                     mask_dis="Uniform"
                 ),
                 "DRLearner": cate_models.torch.DRLearner(
                     X_train.shape[1],
                     device = "cuda:0",
                     binary_y=(len(np.unique(Y_train)) == 2),
                     n_layers_out=self.n_layers,
                     n_units_out=self.n_units_hidden,
                     n_iter=self.n_iter,
                     lr=1e-3,
                     patience=10,
                     batch_size=self.batch_size,
                     batch_norm=False,
                     nonlin="relu"
                 ),
            }

            for learner_name in learners:                    
                log.info(f"Fitting {learner_name}.")
                learners[learner_name].fit(X=X_train, y=Y_train, w=W_train)

            cate_test = sim.te(X_test)

            for learner_name in learners:
                ### computing insertion/deletion results

                if  not "mask" in learner_name.lower():
                    cate_pred = learners[learner_name].predict(X=X_test)
                    pate_model_name =  "DRLearnerMask"
                else:
                    prediction_mask = torch.ones(X_test.shape)
                    cate_pred = learners[learner_name].predict(X=X_test, M=prediction_mask)
                    pate_model_name = learner_name

                pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                explainability_data.append(
                    [
                        nonlinearity_scale,
                        learner_name,
                        pehe_test,
                        np.mean(cate_test),
                        np.var(cate_test),
                        pehe_test / np.sqrt(np.var(cate_test)),
                    ]
                )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Nonlinearity Scale",
                "Learner",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],
        )
        
        results_path = (
            self.save_path
            / f"results/losses/nonlinearity_sensitivity/insertion_deletion/{self.synthetic_simulator_type}"
        )
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path
            / f"{dataset}_{num_important_features}_binary_{binary_outcome}_seed{self.seed}.csv"
        )

class NonLinearityHeldOutOne:
    """
    Held out one analysis for predictive scale.
    """

    def __init__(
        self,
        n_units_hidden: int = 50,
        n_layers: int = 2,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 256,
        n_iter: int = 1000,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        nonlinearity_scales: list = [0.0, 0.2, 0.5, 0.7, 1.0],
        predictive_scale: float = 1.5,
        synthetic_simulator_type: str = "random",
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.nonlinearity_scales = nonlinearity_scales
        self.predictive_scale = predictive_scale
        self.synthetic_simulator_type = synthetic_simulator_type

    def run(
        self,
        dataset: str = "tcga_100",
        num_important_features: int = 15,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
            "naive_shap"
        ],
        train_ratio: float = 0.8,
        binary_outcome: bool = False,
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features}."
        )
        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        explainability_data = []
        held_out_data = []

        for nonlinearity_scale in self.nonlinearity_scales:
            log.info(f"Now working with a nonlinearity scale {nonlinearity_scale}...")
            sim = SyntheticSimulatorModulatedNonLinear(
                X_raw_train,
                num_important_features=num_important_features,
                non_linearity_scale=nonlinearity_scale,
                seed=self.seed,
                selection_type=self.synthetic_simulator_type,
            )
            (
                X_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=self.predictive_scale,
                binary_outcome=binary_outcome,
            )
            X_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=self.predictive_scale,
                binary_outcome=binary_outcome,
            )

            
            learners = {
                   # "TLearner": cate_models.torch.TLearner(
                   # X_train.shape[1],
                   # device = "cuda:1",
                   # binary_y=(len(np.unique(Y_train)) == 2),
                   # n_layers_out=2,
                   # n_units_out=100,
                   # batch_size=1024,
                   # n_iter=self.n_iter,
                   # batch_norm=False,
                   # nonlin="relu",
               # ),
               # "SLearner": cate_models.torch.SLearner(
               #     X_train.shape[1],
               #     device = "cuda:1",
               #     binary_y=(len(np.unique(Y_train)) == 2),
               #     n_layers_out=2,
               #     n_units_out=100,
               #     n_iter=self.n_iter,
               #     batch_size=1024,
               #     batch_norm=False,
               #     nonlin="relu",
               # ),
              #  "TARNet": cate_models.torch.TARNet(
              #      X_train.shape[1],
              #      device = "cuda:1",
              #      binary_y=(len(np.unique(Y_train)) == 2),
              #      n_layers_r=1,
              #      n_layers_out=1,
              #      n_units_out=100,
              #      n_units_r=100,
              #      batch_size=1024,
              #      n_iter=self.n_iter,
              #      batch_norm=False,
              #      nonlin="relu",
              #  ),
                #  "XLearnerMask": pseudo_outcome_nets.XLearnerMask(
                #      X_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=2,
                #      n_units_out=100,
                #      n_iter=self.n_iter,
                #      batch_size=self.batch_size,
                #      lr=1e-3,
                #      patience=10,
                #      batch_norm=False,
                #      nonlin="relu",
                #      device="cuda:1"
                #  ),
                # "XLearner": cate_models.torch.XLearner(
                #      X_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=2,
                #      n_units_out=100,
                #      n_iter=self.n_iter,
                #      batch_size=self.batch_size,
                #      batch_norm=False,
                #      lr=1e-3,
                #      patience=10,
                #      nonlin="relu",
                #      device="cuda:1"
                #  ),
                  "DRLearner": pseudo_outcome_nets.DRLearner(
                      X_train.shape[1],
                      device = "cuda:0",
                      binary_y=(len(np.unique(Y_train)) == 2),
                      n_layers_out=2,
                      n_units_out=100,
                      n_iter=self.n_iter,
                      lr=1e-3,
                      patience=10,
                      batch_size=self.batch_size,
                      batch_norm=False,
                      nonlin="relu"
                  ),
                  "DRLearnerMask": pseudo_outcome_nets.DRLearnerMask(  
                      X_train.shape[1],
                      binary_y=(len(np.unique(Y_train)) == 2),
                      device="cuda:1",
                      n_layers_out=2,
                      n_units_out=100,
                      n_iter=self.n_iter,
                      batch_size=self.batch_size,
                      batch_norm=False,
                      lr=1e-3,
                      patience=10,
                      nonlin="relu",
                      mask_dis="Uniform"
                      ),

                #   "RALearnerMask": pseudo_outcome_nets.RALearnerMask(
                #       X_train.shape[1],
                #       device = "cuda:0",
                #       binary_y=(len(np.unique(Y_train)) == 2),
                #       n_layers_out=2,
                #       n_units_out=100,
                #       n_iter=self.n_iter,
                #       lr=1e-3,
                #       patience=10,
                #       batch_size=self.batch_size,
                #       batch_norm=False,
                #       nonlin="relu"
                #   ),
                #   "RALearner": pseudo_outcome_nets.RALearner(
                #       X_train.shape[1],
                #       device = "cuda:0",
                #       binary_y=(len(np.unique(Y_train)) == 2),
                #       n_layers_out=2,
                #       n_units_out=100,
                #       n_iter=self.n_iter,
                #       lr=1e-3,
                #       patience=10,
                #       batch_size=self.batch_size,
                #       batch_norm=False,
                #       nonlin="relu"
                #   ),
            }

            learner_explainers = {}
            learner_explanations = {}
            learner_explaintion_lists = {}

            for name in learners:                    
                if not  "mask" in name.lower():
                    learner_explaintion_lists[name] = [           
                    "integrated_gradients",
                    "shapley_value_sampling",
                    "naive_shap"
                    ]
                else:
                    learner_explaintion_lists[name] = [
                        "explain_with_missingness"
                    ]

                log.info(f"Fitting {name}.")

                if name == "DRLearnerMask":
                    pretrained_te = deepcopy(learners["DRLearner"]._te_estimator)
                    learners[learner_name]._add_units(pretrained_te)

                learners[name].fit(X=X_train, y=Y_train, w=W_train)

                
                # Obtaining explanation 

                learner_explainers[name] = Explainer(
                    learners[name],
                    feature_names=list(range(X_train.shape[1])),
                    explainer_list=learner_explaintion_lists[name],
                )
                log.info(f"Explaining {name}.")
                learner_explanations[name] = learner_explainers[name].explain(
                    X_test[:self.explainer_limit]
                )

            cate_test = sim.te(X_test)

            for learner_name in learners:
                for explainer_name in learner_explaintion_lists[learner_name]:

                    if  not "mask" in learner_name.lower():
                        cate_pred = learners[learner_name].predict(X=X_test)
                    else:
                        prediction_mask = torch.ones(X_test.shape)
                        cate_pred = learners[learner_name].predict(X=X_test, M=torch.ones((X_test.shape)))

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                    explainability_data.append(
                        [
                            nonlinearity_scale,
                            learner_name,
                            explainer_name,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                        ]
                    )

            # Held out experiment - iterating x_s = D \ {i} for all i in features. 

            for feature_index in range(X_train.shape[1]):
                
                masks = np.ones((X_train.shape[1]))
                masks[feature_index] = 0

                X_train_subset = X_train[:, masks.astype(bool)]
                X_test_subset = X_test[:, masks.astype(bool)]

                pate_learners = {
                    "DRLearner": pseudo_outcome_nets.DRLearnerPate(
                        X_train.shape[1],
                        X_train_subset.shape[1],
                        device="cuda:1",
                        binary_y=(len(np.unique(Y_train)) == 2),
                        n_layers_out=2,
                        n_units_out=100,
                        n_iter=self.n_iter,
                        batch_size=self.batch_size,
                        batch_norm=False,
                        lr=1e-3,
                        patience=10,
                        nonlin="relu"
                 )
                }

                subset_explainer_list = [
                    "integrated_gradients",
                    "shapley_value_sampling",
                    "naive_shap",
                    "explain_with_missingness",
                    # "shapley_value_sampling_half_mask"
                ]
                
                for learner_name in pate_learners:
                    log.info("Fitting PATE models"+ learner_name)

                    pate_learners[learner_name].fit(
                        X=X_train,
                        X_subset=X_train_subset,
                        y=Y_train, 
                        w=W_train,
                    )

                    cate_pred = learners[learner_name].predict(X=X_test[:self.explainer_limit]).detach().cpu().numpy()

                    if learner_name == "XLearner":
                        pate_pred = pate_learners[learner_name].predict( 
                            X=X_test[:self.explainer_limit], 
                            X_subset=X_test_subset[:self.explainer_limit]
                        ).detach().cpu().numpy()
                    else:
                        pate_pred = pate_learners[learner_name].predict( 
                            X=X_test_subset[:self.explainer_limit]
                        ).detach().cpu().numpy()


                    prediction_mask =  torch.ones((X_test[:self.explainer_limit ].shape))
                    prediction_mask[:, feature_index] = 0
                    
                    pate_mask_pred = learners["DRLearnerMask"].predict(X_test[:self.explainer_limit ], prediction_mask).detach().cpu().numpy()

                    pate_pehe = np.sqrt(np.square(pate_pred - pate_mask_pred)/np.var(pate_pred))

                    # loading attribution score from learner_explanations

                    for explainer_name in subset_explainer_list:
                        if explainer_name == "explain_with_missingness":
                            learner_name += "Mask"
                        elif explainer_name == "shapley_value_sampling_half_mask":
                            learner_name = "DRLearnerHalf"
                            explainer_name = "shapley_value_sampling"

                        attribution = learner_explanations[learner_name][explainer_name][:, feature_index]

                        held_out_data.append(
                            [
                                nonlinearity_scale,
                                feature_index,
                                learner_name,
                                explainer_name, 
                                attribution,
                                pate_pred,
                                pate_mask_pred,
                                cate_pred,
                                pate_pehe
                            ]
                        )
        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Nonlinearity Scale",
                "Learner",
                "Explainer",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],
        )
        
        results_path = (
            self.save_path
            / f"results/held_out/drlearner_reweight_1/nonlinearity_sensitivity/model_preformance/{self.synthetic_simulator_type}"
        )

        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path
            / f"{dataset}_{num_important_features}_binary_{binary_outcome}_seed{self.seed}.csv"
        )

        results_path = self.save_path / "results/held_out/drlearner_reweight_1/nonlinearity_sensitivity/"
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        with open( results_path
            / f"{dataset}_{num_important_features}_binary_{binary_outcome}_seed{self.seed}.pkl", 'wb') as handle:
            pkl.dump(held_out_data , handle)

class NonLinearityAssignment:
    """
    Sensitivity analysis for nonlinearity in prognostic and predictive functions.
    """

    def __init__(
        self,
        n_units_hidden: int = 100,
        n_layers: int = 2,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 256,
        n_iter: int = 1000,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        nonlinearity_scales: list = [0.0, 0.2, 0.5, 0.7, 1.0],
        predictive_scale: float = 1,
        synthetic_simulator_type: str = "random",
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.nonlinearity_scales = nonlinearity_scales
        self.predictive_scale = predictive_scale
        self.synthetic_simulator_type = synthetic_simulator_type

    def run(
        self,
        dataset: str = "tcga_100",
        num_important_features: int = 15,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
            "naive_shap"
        ],
        train_ratio: float = 0.8,
        binary_outcome: bool = False,
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features}."
        )
        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        explainability_data = []
        insertion_deletion_data = []

        for nonlinearity_scale in self.nonlinearity_scales:
            log.info(f"Now working with a nonlinearity scale {nonlinearity_scale}...")
            sim = SyntheticSimulatorModulatedNonLinear(
                X_raw_train,
                num_important_features=num_important_features,
                non_linearity_scale=nonlinearity_scale,
                seed=self.seed,
                selection_type=self.synthetic_simulator_type,
            )
            (
                X_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=self.predictive_scale,
                binary_outcome=binary_outcome,
            )
            X_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=self.predictive_scale,
                binary_outcome=binary_outcome,
            )


            log.info("Fitting and explaining learners...")
            learners = {
                # "TLearner": cate_models.torch.TLearner(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                # "SLearner": cate_models.torch.SLearner(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     n_iter=self.n_iter,
                #     batch_size=1024,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                # "TARNet": cate_models.torch.TARNet(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_r=1,
                #     n_layers_out=1,
                #     n_units_out=100,
                #     n_units_r=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                #  "XLearnerMask": pseudo_outcome_nets.XLearnerMask(
                #      X_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=2,
                #      n_units_out=100,
                #      n_iter=10,
                #      batch_size=self.batch_size,
                #     lr=1e-3,
                #     patience=10,
                #      batch_norm=False,
                #      nonlin="relu",
                #      device="cuda:1"
                #  ),
                "XLearner": pseudo_outcome_nets.XLearner(
                     X_train.shape[1],
                     binary_y=(len(np.unique(Y_train)) == 2),
                     n_layers_out=2,
                     n_units_out=100,
                     n_iter=self.n_iter,
                     batch_size=self.batch_size,
                     batch_norm=False,
                     lr=1e-3,
                     patience=10,
                     nonlin="relu",
                     device="cuda:1"
                 ),

                #  "DRLearnerHalf": pseudo_outcome_nets.DRLearnerMaskHalf(
                #      X_train.shape[1],
                #      device = "cuda:0",
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=self.n_layers,
                #      n_units_out=self.n_units_hidden,
                #      n_iter=self.n_iter,
                #      lr=1e-3,
                #      patience=10,
                #      batch_size=self.batch_size,
                #      batch_norm=False,
                #      nonlin="relu"
                #  ),
                #  "DRLearner": cate_models.torch.DRLearner(
                #      X_train.shape[1],
                #      device = "cuda:1",
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=self.n_layers,
                #      n_units_out=self.n_units_hidden,
                #      n_iter=self.n_iter,
                #      lr=1e-3,
                #      patience=10,
                #      batch_size=self.batch_size,
                #      batch_norm=False,
                #      nonlin="relu"
                #  ),
                #  "DRLearnerMask": pseudo_outcome_nets.DRLearnerMask(  
                #      X_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      device="cuda:1",
                #      n_layers_out=self.n_layers,
                #      n_units_out=self.n_units_hidden,
                #      n_iter=self.n_iter,
                #      batch_size=self.batch_size,
                #      batch_norm=False,
                #      lr=1e-3,
                #      patience=10,
                #      nonlin="relu",
                #      mask_dis="Uniform"
                #      ),
            }

            learner_explainers = {}
            learner_explaintion_lists = {}
            learner_explanations = {}

            for learner_name in learners:                    
                if not  "mask" in learner_name.lower():
                    learner_explaintion_lists[learner_name] = [           
                                                        "integrated_gradients",
                                                        "shapley_value_sampling",
                                                        "naive_shap"
                                                    ]
                elif "half" in learner_name.lower():
                    learner_explaintion_lists[learner_name] = [
                        "shapley_value_sampling"
                        ]
                else:
                    learner_explaintion_lists[learner_name] = [
                        "explain_with_missingness"
                        ]

                log.info(f"Fitting {learner_name}.")

                if learner_name == "DRLearnerMask":
                    pretrained_te = deepcopy(learners["DRLearner"]._te_estimator)
                    learners[learner_name]._add_units(pretrained_te)
                
                learners[learner_name].fit(X=X_train, y=Y_train, w=W_train)
                learner_explainers[learner_name] = Explainer(
                    learners[learner_name],
                    feature_names=list(range(X_train.shape[1])),
                    explainer_list=learner_explaintion_lists[learner_name],
                )

                log.info(f"Explaining {learner_name}.")
                learner_explanations[learner_name] = learner_explainers[learner_name].explain(
                    X_test[:self.explainer_limit]
                )

            all_important_features = sim.get_all_important_features()
            pred_features = sim.get_predictive_features()
            prog_features = sim.get_prognostic_features()

            cate_test = sim.te(X_test)

            for learner_name in learners:
                for explainer_name in learner_explaintion_lists[learner_name]:

                    attribution_est = np.abs(
                        learner_explanations[learner_name][explainer_name]
                    )
                    acc_scores_all_features, acc_scores_all_features_score  = attribution_accuracy(
                        all_important_features, attribution_est
                    )
                    acc_scores_predictive_features, acc_scores_predictive_features_score = attribution_accuracy(
                        pred_features, attribution_est
                    )
                    acc_scores_prog_features, acc_scores_prog_features_score = attribution_accuracy(
                        prog_features, attribution_est
                    )
                    # computing insertion/deletion results

                    if  not "mask" in learner_name.lower():
                        cate_pred = learners[learner_name].predict(X=X_test)
                        pate_model_name =  "DRLearnerMask"
                    else:
                        prediction_mask = torch.ones(X_test.shape)
                        cate_pred = learners[learner_name].predict(X=X_test, M=prediction_mask)
                        pate_model_name = learner_name

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                    # Obtain feature importance rank
                    rank_indices = np.argsort(np.sum(attribution_est, axis=0))[::-1]
                    # Relabel positive outcome

                    y_cate_train = learners[learner_name].predict(X=X_train)
                    y_cate_test = learners[learner_name].predict(X=X_test)

                    cate_threshold = torch.mean(y_cate_train)
                    
                    y_subgroup_train = (y_cate_train > cate_threshold).float().detach().cpu().numpy()
                    y_subgroup_test = (y_cate_test > cate_threshold).float().detach().cpu().numpy()

                    # import ipdb; ipdb.set_trace()

                    xgb_model = xgb.XGBClassifier(objective="binary:logistic")
                    X_subset_train, X_subset_test = X_train[:,rank_indices[:5]],  X_test[:,rank_indices[:5]]

                    xgb_model.fit(X_subset_train, y_subgroup_train)
                    y_subgroup_pred = xgb_model.predict(X_subset_test)

                    auroc = metrics.roc_auc_score(y_subgroup_test, y_subgroup_pred)

                    # subgroup identification 
                    total_num = len(X_subset_test) + len(X_subset_train)

                    s_hat = X_test[np.where(y_subgroup_pred == 1), :]

                    utility_index = np.mean(learners[learner_name].predict(X=s_hat).detach().cpu().numpy())*len(s_hat)/total_num

                    # baseline with random features

                    random_xgb_model = xgb.XGBClassifier(objective="binary:logistic")
                    np.random.shuffle(rank_indices)
                    X_subset_train = X_train[:,rank_indices[:5]]
                    X_subset_test = X_test[:,rank_indices[:5]]

                    random_xgb_model.fit(X_subset_train, y_subgroup_train)
                    y_subgroup_pred = random_xgb_model.predict(X_subset_test)
                    
                    s_hat = X_test[np.where(y_subgroup_pred == 1), :]

                    if len(s_hat) <=1 :
                        random_utility_index = 0
                    else:
                        random_utility_index = np.mean(learners[learner_name].predict(X=s_hat).detach().cpu().numpy())*len(s_hat)/total_num

                    explainability_data.append(
                        [
                            nonlinearity_scale,
                            learner_name,
                            explainer_name,
                            acc_scores_all_features,
                            acc_scores_all_features_score,
                            acc_scores_predictive_features,
                            acc_scores_predictive_features_score,
                            acc_scores_prog_features,
                            acc_scores_prog_features_score,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                            auroc,
                            utility_index,
                            random_utility_index
                        ]
                    )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Nonlinearity Scale",
                "Learner",
                "Explainer",
                "All features ACC",
                "All features ACC Score",
                "Pred features ACC",
                "Pred features ACC Score",
                "Prog features ACC",
                "Prog features ACC Score",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
                "AUROC",
                "utility_score",
                "utility_score_random"
            ],
        )
        
        results_path = (
            self.save_path
            / f"results/nonlinearity_sensitivity/assignment/{self.synthetic_simulator_type}"
        )
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path
            / f"{dataset}_{num_important_features}_binary_{binary_outcome}_seed{self.seed}.csv"
        )


class PropensitySensitivity:
    """
    Sensitivity analysis for confounding.
    """

    def __init__(
        self,
        n_units_hidden: int = 50,
        n_layers: int = 1,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 256,
        n_iter: int = 2000,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        num_interactions: int = 1,
        synthetic_simulator_type: str = "linear",
        propensity_type: str = "pred",
        propensity_scales: list = [0, 0.5, 1, 2, 5, 10, 100],
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.num_interactions = num_interactions
        self.synthetic_simulator_type = synthetic_simulator_type
        self.propensity_type = propensity_type
        self.propensity_scales = propensity_scales

    def run(
        self,
        dataset: str = "tcga_10",
        train_ratio: float = 0.8,
        num_important_features: int = 2,
        binary_outcome: bool = False,
        random_feature_selection: bool = True,
        predictive_scale: float = 1,
        nonlinearity_scale: float = 0.5,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
            "naive_shap"
        ],
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features} and predictive scale {predictive_scale}."
        )

        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        if self.synthetic_simulator_type == "linear":
            sim = SyntheticSimulatorLinear(
                X_raw_train,
                num_important_features=num_important_features,
                random_feature_selection=random_feature_selection,
                seed=self.seed,
            )
        elif self.synthetic_simulator_type == "nonlinear":
            sim = SyntheticSimulatorModulatedNonLinear(
                X_raw_train,
                num_important_features=num_important_features,
                non_linearity_scale=nonlinearity_scale,
                seed=self.seed,
                selection_type="random",
            )
        else:
            raise Exception("Unknown simulator type.")

        explainability_data = []

        for propensity_scale in self.propensity_scales:
            log.info(f"Now working with propensity_scale = {propensity_scale}...")
            (
                X_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
                treatment_assign=self.propensity_type,
                prop_scale=propensity_scale,
            )

            X_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
                treatment_assign=self.propensity_type,
                prop_scale=propensity_scale,
            )

            log.info("Fitting and explaining learners...")
            learners = {
                # "TLearner": cate_models.torch.TLearner(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                # "SLearner": cate_models.torch.SLearner(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     n_iter=self.n_iter,
                #     batch_size=1024,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                # "TARNet": cate_models.torch.TARNet(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_r=1,
                #     n_layers_out=1,
                #     n_units_out=100,
                #     n_units_r=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                "DRLearner": pseudo_outcome_nets.DRLearner(
                    X_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    n_iter=self.n_iter,
                    batch_size=self.batch_size,
                    batch_norm=False,
                    lr=1e-3,
                    patience=10,
                    nonlin="relu",
                    device= "cuda:1"
                ),
                "XLearner": pseudo_outcome_nets.XLearner(
                     X_train.shape[1],
                     binary_y=(len(np.unique(Y_train)) == 2),
                     n_layers_out=2,
                     n_units_out=100,
                     n_iter=self.n_iter,
                    lr=1e-3,
                    patience=10,
                     batch_size=self.batch_size,
                     batch_norm=False,
                     nonlin="relu",
                     device="cuda:1"
                 ),
                # "CFRNet_0.01": cate_models.torch.TARNet(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_r=1,
                #     n_layers_out=1,
                #     n_units_out=100,
                #     n_units_r=100,
                #     batch_size=self.batch_size,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                #     penalty_disc=0.01,
                # ),
                # "CFRNet_0.001": cate_models.torch.TARNet(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_r=1,
                #     n_layers_out=1,
                #     n_units_out=100,
                #     n_units_r=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                #     penalty_disc=0.001,
                # ),
                # "CFRNet_0.0001": cate_models.torch.TARNet(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_r=1,
                #     n_layers_out=1,
                #     n_units_out=100,
                #     n_units_r=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                #     penalty_disc=0.0001,
                # ),
            }

            learner_explainers = {}
            learner_explanations = {}

            for name in learners:
                log.info(f"Fitting {name}.")
                learners[name].fit(X=X_train, y=Y_train, w=W_train)                    

                learner_explainers[name] = Explainer(
                    learners[name],
                    feature_names=list(range(X_train.shape[1])),
                    explainer_list=explainer_list,
                )
                log.info(f"Explaining {name}.")
                learner_explanations[name] = learner_explainers[name].explain(
                    X_test[: self.explainer_limit]
                )

            all_important_features = sim.get_all_important_features()
            pred_features = sim.get_predictive_features()
            prog_features = sim.get_prognostic_features()

            cate_test = sim.te(X_test)

            for explainer_name in explainer_list:
                for learner_name in learners:
                    attribution_est = np.abs(
                        learner_explanations[learner_name][explainer_name]
                    )
                    acc_scores_all_features, acc_scores_all_features_score = attribution_accuracy(
                        all_important_features, attribution_est
                    )
                    acc_scores_predictive_features,acc_scores_predictive_features_score = attribution_accuracy(
                        pred_features, attribution_est
                    )
                    acc_scores_prog_features, acc_scores_prog_features_score = attribution_accuracy(
                        prog_features, attribution_est
                    )
                    cate_pred = learners[learner_name].predict(X=X_test)

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                    treatment_ratio = W_train.sum()/W_train.shape[0]

                    explainability_data.append(
                        [
                            propensity_scale,
                            treatment_ratio,
                            learner_name,
                            explainer_name,
                            acc_scores_all_features,
                            acc_scores_all_features_score,
                            acc_scores_predictive_features,
                            acc_scores_predictive_features_score,
                            acc_scores_prog_features,
                            acc_scores_prog_features_score,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                        ]
                    )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Propensity Scale",
                "Treatment Ratio",
                "Learner",
                "Explainer",
                "All features ACC",
                "All features ACC Score",
                "Pred features ACC",
                "Pred features ACC Score",
                "Prog features ACC",
                "Prog features ACC Score",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],
        )

        results_path = (
            self.save_path
            / f"results/propensity_sensitivity/sample/{self.synthetic_simulator_type}/{self.propensity_type}"
        )
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path / f"propensity_scale_{dataset}_{num_important_features}_"
            f"proptype_{self.propensity_type}_"
            f"predscl_{predictive_scale}_"
            f"nonlinscl_{nonlinearity_scale}_"
            f"trainratio_{train_ratio}_"
            f"binary_{binary_outcome}-seed{self.seed}.csv"
        )


class PropensityAssignment:
    """
    Sensitivity analysis for confounding.
    """

    def __init__(
        self,
        n_units_hidden: int = 50,
        n_layers: int = 1,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 256,
        n_iter: int = 1000,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        num_interactions: int = 1,
        synthetic_simulator_type: str = "linear",
        propensity_type: str = "pred",
        propensity_scales: list = [0, 0.5, 1, 2, 5, 10],
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.num_interactions = num_interactions
        self.synthetic_simulator_type = synthetic_simulator_type
        self.propensity_type = propensity_type
        self.propensity_scales = propensity_scales

    def run(
        self,
        dataset: str = "tcga_10",
        train_ratio: float = 0.8,
        num_important_features: int = 2,
        binary_outcome: bool = False,
        random_feature_selection: bool = True,
        predictive_scale: float = 1,
        nonlinearity_scale: float = 0.5,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
            "naive_shap"
        ],
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features} and predictive scale {predictive_scale}."
        )

        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        if self.synthetic_simulator_type == "linear":
            sim = SyntheticSimulatorLinear(
                X_raw_train,
                num_important_features=num_important_features,
                random_feature_selection=random_feature_selection,
                seed=self.seed,
            )
        elif self.synthetic_simulator_type == "nonlinear":
            sim = SyntheticSimulatorModulatedNonLinear(
                X_raw_train,
                num_important_features=num_important_features,
                non_linearity_scale=nonlinearity_scale,
                seed=self.seed,
                selection_type="random",
            )
        else:
            raise Exception("Unknown simulator type.")

        explainability_data = []
        assignment_data = []

        for propensity_scale in self.propensity_scales:
            log.info(f"Now working with propensity_scale = {propensity_scale}...")
            (
                X_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
                treatment_assign=self.propensity_type,
                prop_scale=propensity_scale,
            )

            X_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
                treatment_assign=self.propensity_type,
                prop_scale=propensity_scale,
            )

            log.info("Fitting and explaining learners...")
            learners = {
                # "TLearner": cate_models.torch.TLearner(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     batch_size=self.batch_size,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                #     device= "cuda:1"                
                # ),
                # "SLearner": cate_models.torch.SLearner(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     n_iter=self.n_iter,
                #     batch_size=1024,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                # "TARNet": cate_models.torch.TARNet(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_r=1,
                #     n_layers_out=1,
                #     n_units_out=100,
                #     n_units_r=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                # ),
                # "DRLearner": pseudo_outcome_nets.DRLearner(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_out=2,
                #     n_units_out=100,
                #     n_iter=self.n_iter,
                #     batch_size=self.batch_size,
                #     batch_norm=False,
                #     lr=1e-3,
                #     patience=10,
                #     nonlin="relu",
                #     device= "cuda:1"
                # ),
                #  "XLearnerMask": pseudo_outcome_nets.XLearnerMask(
                #      X_train.shape[1],
                #      binary_y=(len(np.unique(Y_train)) == 2),
                #      n_layers_out=2,
                #      n_units_out=100,
                #      n_iter=self.n_iter,
                #      batch_size=self.batch_size,
                #      lr=1e-3,
                #      patience=10,
                #      batch_norm=False,
                #      nonlin="relu",
                #      device="cuda:1"
                #  ),
                "XLearner": pseudo_outcome_nets.XLearner(
                     X_train.shape[1],
                     binary_y=(len(np.unique(Y_train)) == 2),
                     n_layers_out=2,
                     n_units_out=100,
                     n_iter=self.n_iter,
                     lr=1e-3,
                     patience=10,
                     batch_size=self.batch_size,
                     batch_norm=False,
                     nonlin="relu",
                     device="cuda:1"
                 ),

                # "CFRNet_0.01": cate_models.torch.TARNet(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_r=1,
                #     n_layers_out=1,
                #     n_units_out=100,
                #     n_units_r=100,
                #     batch_size=self.batch_size,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                #     penalty_disc=0.01,
                # ),
                # "CFRNet_0.001": cate_models.torch.TARNet(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_r=1,
                #     n_layers_out=1,
                #     n_units_out=100,
                #     n_units_r=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                #     penalty_disc=0.001,
                # ),
                # "CFRNet_0.0001": cate_models.torch.TARNet(
                #     X_train.shape[1],
                #     binary_y=(len(np.unique(Y_train)) == 2),
                #     n_layers_r=1,
                #     n_layers_out=1,
                #     n_units_out=100,
                #     n_units_r=100,
                #     batch_size=1024,
                #     n_iter=self.n_iter,
                #     batch_norm=False,
                #     nonlin="relu",
                #     penalty_disc=0.0001,
                # ),
            }

            learner_explainers = {}
            learner_explanations = {}
            learner_explaintion_lists = {}

            for name in learners:
                if not  "mask" in name.lower():
                    learner_explaintion_lists[name] = [           
                    "integrated_gradients",
                    "shapley_value_sampling",
                    "naive_shap"
                    ]
                else:
                    learner_explaintion_lists[name] = [
                        "explain_with_missingness"
                    ]
                log.info(f"Fitting {name}.")
                learners[name].fit(X=X_train, y=Y_train, w=W_train)                    

                learner_explainers[name] = Explainer(
                    learners[name],
                    feature_names=list(range(X_train.shape[1])),
                    explainer_list=learner_explaintion_lists[name],
                )
                log.info(f"Explaining {name}.")
                learner_explanations[name] = learner_explainers[name].explain(
                    X_test[:self.explainer_limit]
                )

            all_important_features = sim.get_all_important_features()
            pred_features = sim.get_predictive_features()
            prog_features = sim.get_prognostic_features()

            cate_test = sim.te(X_test)

            for learner_name in learners:
                for explainer_name in learner_explaintion_lists[learner_name]:

                    result_auroc = []
                    cate_policy = []
                    cate_random = []
                    cate_original = []
                    
                    true_cate_policy = []
                    true_cate_random = []
                    true_cate_original = []

                    attribution_est = np.abs(
                        learner_explanations[learner_name][explainer_name]
                    )
                    acc_scores_all_features, acc_scores_all_features_score = attribution_accuracy(
                        all_important_features, attribution_est
                    )
                    acc_scores_predictive_features,acc_scores_predictive_features_score = attribution_accuracy(
                        pred_features, attribution_est
                    )
                    acc_scores_prog_features, acc_scores_prog_features_score = attribution_accuracy(
                        prog_features, attribution_est
                    )
                    if  not "mask" in learner_name.lower():
                        cate_pred = learners[learner_name].predict(X=X_test)
                    else:
                        prediction_mask = torch.ones(X_test.shape)
                        cate_pred = learners[learner_name].predict(X=X_test, M=prediction_mask)

                    cate_pred = learners[learner_name].predict(X=X_test)

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                    rank_indices = np.argsort(np.mean(attribution_est, axis=0))[::-1]
                    
                    num_feature = num_important_features
                    total_num = len(X_raw_test) + len(X_raw_train)

                    if  not "mask" in learner_name.lower():
                        new_y_train = learners[learner_name].predict(X=X_train)
                    else:
                        prediction_mask = torch.ones(X_train.shape)
                        new_y_train = learners[learner_name].predict(X=X_train, M=prediction_mask)

                    # Define threshold & relabel subgroup
                    
                    cate_threshold = torch.mean(new_y_train)
                    new_y_train = (new_y_train > cate_threshold ).float().detach().cpu().numpy()

                    if  not "mask" in learner_name.lower():
                        new_y_test = learners[learner_name].predict(X=X_test)
                    else:
                        prediction_mask = torch.ones(X_test.shape)
                        new_y_test = learners[learner_name].predict(X=X_test, M=prediction_mask)

                    new_y_test = (new_y_test > cate_threshold).float().detach().cpu().numpy()

                    # Oracle ATEs
                    
                    true_ites = po1_test - po0_test
                    true_ites_oracle = np.sum(true_ites[np.where(true_ites > np.mean(true_ites))])/total_num

                    # ATEs with identified important features

                    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=self.seed)
                    new_X_train = X_train[:,rank_indices[:num_feature]]
                    new_X_test = X_test[:,rank_indices[:num_feature]]

                    xgb_model.fit(new_X_train, new_y_train)

                    y_pred = xgb_model.predict(new_X_test)
                    s_hat = X_test[np.where(y_pred == 1), :]

                    result_auroc.append(metrics.roc_auc_score(new_y_test, y_pred))

                    if  not "mask" in learner_name.lower():
                        ites = learners[learner_name].predict(X=s_hat).detach().cpu().numpy()
                    else:
                        prediction_mask = torch.ones(s_hat.shape)
                        ites = learners[learner_name].predict(X=s_hat, M=prediction_mask).detach().cpu().numpy()

                    cate_policy.append(np.sum(ites)/total_num)
                    true_cate_policy.append(np.sum(true_ites[np.where(y_pred == 1)])/total_num)

                    # ATEs with random features assignment

                    random_xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=self.seed)
                    np.random.shuffle(rank_indices)
                    new_X_train = X_train[:,rank_indices[:num_feature]]
                    new_X_test = X_test[:,rank_indices[:num_feature]]

                    random_xgb_model.fit(new_X_train, new_y_train)
                    y_pred = random_xgb_model.predict(new_X_test)

                    s_hat = X_test[np.where(y_pred == 1),:]

                    if len(s_hat) != 0:
                        if  not "mask" in learner_name.lower():
                            random_ites = learners[learner_name].predict(X=s_hat).detach().cpu().numpy()
                        else:
                            prediction_mask = torch.ones(s_hat.shape)
                            random_ites = learners[learner_name].predict(X=s_hat, M=prediction_mask).detach().cpu().numpy()
                    else:
                        random_ites = 0
                    
                    cate_random.append(np.sum(random_ites)/total_num)
                    true_cate_random.append(np.sum(true_ites[np.where(y_pred == 1)])/total_num)

                    # ATEs with original assignment

                    s_original = X_test[np.where(W_test == 1), :]

                    if  not "mask" in learner_name.lower():
                        original_assignment_ites = np.sum(learners[learner_name].predict(X=s_original).detach().cpu().numpy())/total_num
                    else:
                        prediction_mask = torch.ones(s_original.shape)
                        original_assignment_ites = np.sum(learners[learner_name].predict(X=s_original, M=prediction_mask).detach().cpu().numpy())/total_num

                    cate_original.append(original_assignment_ites)
                    true_cate_original.append(np.sum(true_ites[np.where(W_test == 1)])/total_num)
                    
                    assignment_data.append(
                        [
                            propensity_scale,
                            learner_name,
                            explainer_name,
                            result_auroc,
                            cate_policy,
                            cate_random,
                            cate_original,
                            true_cate_policy,
                            true_cate_random,
                            true_cate_original,
                            true_ites_oracle
                        ]
                    )

                    explainability_data.append(
                        [
                            propensity_scale,
                            learner_name,
                            explainer_name,
                            acc_scores_all_features,
                            acc_scores_all_features_score,
                            acc_scores_predictive_features,
                            acc_scores_predictive_features_score,
                            acc_scores_prog_features,
                            acc_scores_prog_features_score,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test))
                        ]
                    )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Propensity Scale",
                "Learner",
                "Explainer",
                "All features ACC",
                "All features ACC Score",
                "Pred features ACC",
                "Pred features ACC Score",
                "Prog features ACC",
                "Prog features ACC Score",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],
        )

        results_path = (
            self.save_path
            / f"results/propensity_sensitivity/assignment/model_performance/{self.synthetic_simulator_type}/{self.propensity_type}"
        )
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path / f"propensity_scale_{dataset}_{num_important_features}_"
            f"proptype_{self.propensity_type}_"
            f"predscl_{predictive_scale}_"
            f"nonlinscl_{nonlinearity_scale}_"
            f"trainratio_{train_ratio}_"
            f"binary_{binary_outcome}-seed{self.seed}.csv"
        )
        results_path = (
            self.save_path
            / f"results/propensity_sensitivity/assignment/assignment/{self.synthetic_simulator_type}/{self.propensity_type}"
        )
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        with open( results_path / f"propensity_scale_{dataset}_{num_important_features}_"
            f"proptype_{self.propensity_type}_"
            f"predscl_{predictive_scale}_"
            f"nonlinscl_{nonlinearity_scale}_"
            f"trainratio_{train_ratio}_"
            f"binary_{binary_outcome}-seed{self.seed}.pkl", 'wb') as handle:
            pkl.dump(assignment_data , handle)
