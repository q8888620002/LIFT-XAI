"""Compute error for pseudo-surrogate for CATEs"""

import argparse
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklift.metrics import (  # uplift_at_k,; weighted_average_uplift,
    qini_auc_score,
    uplift_auc_score,
)

import src.CATENets.catenets.models as cate_models
from src.cate_utils import NuisanceFunctions, calculate_pehe
from src.CATENets.catenets.models.torch import pseudo_outcome_nets
from src.dataset import Dataset
from src.permucate.learners import CausalForest, DRLearner

# from src.utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("-d", "--dataset", help="Dataset", required=True, type=str)
    parser.add_argument("-t", "--num_trials", help="Dataset", required=True, type=int)
    parser.add_argument("-s", "--shuffle", help="shuffle", action="store_true")

    args = vars(parser.parse_args())

    cohort_name = args["dataset"]
    trials = args["num_trials"]
    shuffle = args["shuffle"]
    print(shuffle)
    DEVICE = "cuda:0"
    ensemble_num = 40
    data = Dataset(cohort_name)
    x_train, _, _ = data.get_data("train")

    learners = [
        "XLearner",
        "XLearner_ensemble",
        "DRLearner",
        "DRLearner_ensemble",
        "SLearner",
        "TLearner",
        "RLearner",
        "RALearner",
        "TARNet",
        "DragonNet",
        "CFRNet_0.01",
        "CFRNet_0.001",
        "CausalForest",
        "LinearDR",
    ]

    selection_types = ["if_pehe", "pseudo_outcome_r", "pseudo_outcome_dr"]

    results = {
        learner: {
            **{sec: np.zeros((trials)) for sec in selection_types},
            "prediction": np.zeros((trials, x_train.shape[0])),
            "qini_score": np.zeros((trials)),
            "uplift_score": np.zeros((trials)),
        }
        for learner in learners
    }

    for i in range(trials):

        np.random.seed(i)
        data = Dataset(cohort_name, i, shuffle)

        x_train, w_train, y_train = data.get_data("train")
        x_val, w_val, y_val = data.get_data("val")
        x_test, w_test, y_test = data.get_data("test")

        learners = {
            "XLearner": pseudo_outcome_nets.XLearner(
                x_train.shape[1],
                binary_y=(len(np.unique(y_train)) == 2),
                n_layers_out=2,
                n_units_out=100,
                batch_size=128,
                n_iter=1000,
                nonlin="relu",
                device=DEVICE,
            ),
            "XLearner_ensemble": pseudo_outcome_nets.XLearner(
                x_train.shape[1],
                binary_y=(len(np.unique(y_train)) == 2),
                n_layers_out=2,
                n_units_out=100,
                batch_size=128,
                n_iter=1000,
                nonlin="relu",
                device=DEVICE,
                seed=i,
            ),
            "DRLearner": pseudo_outcome_nets.DRLearner(
                x_train.shape[1],
                binary_y=(len(np.unique(y_train)) == 2),
                n_layers_out=2,
                n_units_out=100,
                batch_size=128,
                n_iter=1000,
                nonlin="relu",
                device=DEVICE,
            ),
            "DRLearner_ensemble": pseudo_outcome_nets.DRLearner(
                x_train.shape[1],
                binary_y=(len(np.unique(y_train)) == 2),
                n_layers_out=2,
                n_units_out=100,
                batch_size=128,
                n_iter=1000,
                nonlin="relu",
                device=DEVICE,
            ),
            "SLearner": cate_models.torch.SLearner(
                x_train.shape[1],
                binary_y=(len(np.unique(y_train)) == 2),
                n_layers_out=2,
                n_units_out=100,
                batch_size=128,
                n_iter=1000,
                nonlin="relu",
                device=DEVICE,
            ),
            "TLearner": cate_models.torch.TLearner(
                x_train.shape[1],
                binary_y=(len(np.unique(y_train)) == 2),
                n_layers_out=2,
                n_units_out=100,
                batch_size=128,
                n_iter=1000,
                nonlin="relu",
                device=DEVICE,
            ),
            "RALearner": pseudo_outcome_nets.RALearner(
                x_train.shape[1],
                binary_y=(len(np.unique(y_train)) == 2),
                n_layers_out=2,
                n_units_out=100,
                n_iter=1000,
                lr=1e-3,
                patience=10,
                batch_size=128,
                batch_norm=False,
                nonlin="relu",
                device=DEVICE,
            ),
            "RLearner": pseudo_outcome_nets.RLearner(
                x_train.shape[1],
                binary_y=(len(np.unique(y_train)) == 2),
                n_layers_out=2,
                n_units_out=100,
                n_iter=1000,
                lr=1e-3,
                patience=10,
                batch_size=128,
                batch_norm=False,
                nonlin="relu",
                device=DEVICE,
            ),
            "DragonNet": cate_models.torch.DragonNet(
                x_train.shape[1],
                binary_y=(len(np.unique(y_train)) == 2),
                batch_size=128,
                n_iter=1000,
                lr=1e-5,
                batch_norm=False,
                nonlin="relu",
            ),
            "TARNet": cate_models.torch.TARNet(
                x_train.shape[1],
                binary_y=(len(np.unique(y_train)) == 2),
                n_layers_r=2,
                n_layers_out=2,
                n_units_out=100,
                n_units_r=100,
                batch_size=128,
                n_iter=1000,
                lr=1e-5,
                batch_norm=False,
                early_stopping=True,
                nonlin="relu",
            ),
            "CFRNet_0.01": cate_models.torch.TARNet(
                x_train.shape[1],
                binary_y=(len(np.unique(y_train)) == 2),
                n_layers_r=2,
                n_layers_out=2,
                n_units_out=100,
                n_units_r=100,
                batch_size=128,
                n_iter=1000,
                lr=1e-5,
                batch_norm=False,
                nonlin="relu",
                penalty_disc=0.01,
            ),
            "CFRNet_0.001": cate_models.torch.TARNet(
                x_train.shape[1],
                binary_y=(len(np.unique(y_train)) == 2),
                n_layers_r=2,
                n_layers_out=2,
                n_units_out=100,
                n_units_r=100,
                lr=1e-5,
                batch_size=128,
                n_iter=1000,
                batch_norm=False,
                nonlin="relu",
                penalty_disc=0.001,
            ),
            "CausalForest": CausalForest(),
            "LinearDR": DRLearner(
                model_final=RidgeCV(alphas=np.logspace(-3, 3, 50)),
                model_propensity=LogisticRegressionCV(Cs=np.logspace(-3, 3, 50)),
                model_response=RidgeCV(alphas=np.logspace(-3, 3, 50)),
                cv=5,
                random_state=0,
            ),
        }

        if data.cohort_name in ["crash_2", "ist3", "sprint", "accord"]:
            nuisance_functions = NuisanceFunctions(rct=True)
        else:
            nuisance_functions = NuisanceFunctions(rct=False)

        nuisance_functions.fit(x_val, y_val, w_val)

        for learner_name, cate_model in learners.items():

            if learner_name == "XLearner_ensemble":
                print("training ensemble")
                results[learner_name]["prediction"][i] = 0

                for index in range(ensemble_num):

                    cate_model = pseudo_outcome_nets.XLearner(
                        x_train.shape[1],
                        binary_y=(len(np.unique(y_train)) == 2),
                        n_layers_out=2,
                        n_units_out=100,
                        batch_size=128,
                        n_iter=1000,
                        nonlin="relu",
                        device=DEVICE,
                        seed=index,
                    )

                    cate_model.fit(x_train, y_train, w_train)

                    prediction = (
                        cate_model.predict(x_train).detach().cpu().numpy().flatten()
                    )
                    results[learner_name]["prediction"][i] += prediction / ensemble_num

            elif learner_name == "DRLearner_ensemble":
                print("training ensemble")
                results[learner_name]["prediction"][i] = 0

                for index in range(ensemble_num):

                    cate_model = pseudo_outcome_nets.DRLearner(
                        x_train.shape[1],
                        binary_y=(len(np.unique(y_train)) == 2),
                        n_layers_out=2,
                        n_units_out=100,
                        batch_size=128,
                        n_iter=1000,
                        nonlin="relu",
                        device=DEVICE,
                        seed=index,
                    )

                    cate_model.fit(x_train, y_train, w_train)

                    prediction = (
                        cate_model.predict(x_train).detach().cpu().numpy().flatten()
                    )

                    results[learner_name]["prediction"][i] += prediction / ensemble_num
            else:

                cate_model.fit(x_train, y_train, w_train)

                if learner_name in ["CausalForest", "LinearDR"]:
                    prediction = cate_model.effect(x_train)
                else:
                    prediction = (
                        cate_model.predict(x_train).detach().cpu().numpy().flatten()
                    )

                results[learner_name]["prediction"][i] = prediction

            results[learner_name]["qini_score"][i] = qini_auc_score(
                y_train, results[learner_name]["prediction"][i], w_train
            )
            results[learner_name]["uplift_score"][i] = uplift_auc_score(
                y_train, results[learner_name]["prediction"][i], w_train
            )
            for sec in selection_types:
                results[learner_name][sec][i] = calculate_pehe(
                    prediction, data.get_data("train"), sec, nuisance_functions
                )

    with open(
        f"results/{cohort_name}/model_selection/model_selection_shuffle_{shuffle}.pkl",
        "wb",
    ) as output_file:
        pickle.dump(results, output_file)
