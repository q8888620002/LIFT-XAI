"""Compute error for pseudo-surrogate for CATEs"""

import argparse
import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklift.metrics import (  # uplift_at_k,; weighted_average_uplift,
    qini_auc_score,
    uplift_auc_score,
)

import src.CATENets.catenets.models as cate_models
from src.cate_utils import NuisanceFunctions, calculate_pehe
from src.CATENets.catenets.models.torch import pseudo_outcome_nets
from src.dataset import Dataset
from src.permucate.learners import CausalForest, DRLearner


def oof_predict(
    learner_name, x, y, w, device="cuda:0", n_splits=5, seed=0, ensemble_num=1
):
    """Make prediciotn in out-of-fold (OOF) set"""
    n = x.shape[0]
    oof = np.zeros(n, dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    binary_y = len(np.unique(y)) == 2
    do_ens = "ensemble" in learner_name.lower()

    ens_n = ensemble_num if do_ens else 1

    for tr_idx, oof_idx in skf.split(x, w.astype(int)):
        fold_pred = np.zeros(len(oof_idx))
        for m in range(ens_n):
            model = make_model_for(
                learner_name, x.shape[1], binary_y, device, seed=seed + m
            )

            model.fit(x[tr_idx], y[tr_idx], w[tr_idx])
            if hasattr(model, "effect"):
                p = model.effect(x[oof_idx])
            else:
                p = model.predict(x[oof_idx]).detach().cpu().numpy().ravel()
            fold_pred += p / ens_n
        oof[oof_idx] = fold_pred
    return oof


def make_model_for(name, x_dim, binary_y, device, seed=None):
    """Initialize a new CATE model"""

    if name == "XLearner" or name == "XLearner_ensemble":
        return pseudo_outcome_nets.XLearner(
            x_dim,
            binary_y=binary_y,
            n_layers_out=2,
            n_units_out=100,
            batch_size=128,
            n_iter=1000,
            nonlin="relu",
            device=device,
            seed=seed,
        )
    if name == "DRLearner" or name == "DRLearner_ensemble":
        return pseudo_outcome_nets.DRLearner(
            x_dim,
            binary_y=binary_y,
            n_layers_out=2,
            n_units_out=100,
            batch_size=128,
            n_iter=1000,
            nonlin="relu",
            device=device,
            seed=seed,
        )
    if name == "RALearner" or name == "RALearner_ensemble":
        return pseudo_outcome_nets.RALearner(
            x_dim,
            binary_y=binary_y,
            n_layers_out=2,
            n_units_out=100,
            n_iter=1000,
            lr=1e-3,
            patience=10,
            batch_size=128,
            batch_norm=False,
            nonlin="relu",
            device=device,
            seed=seed,
        )
    if name == "SLearner":
        return cate_models.torch.SLearner(
            x_dim,
            binary_y=binary_y,
            n_layers_out=2,
            n_units_out=100,
            batch_size=128,
            n_iter=1000,
            nonlin="relu",
            device=device,
        )
    if name == "TLearner":
        return cate_models.torch.TLearner(
            x_dim,
            binary_y=binary_y,
            n_layers_out=2,
            n_units_out=100,
            batch_size=128,
            n_iter=1000,
            nonlin="relu",
            device=device,
        )
    if name == "RLearner":
        return pseudo_outcome_nets.RLearner(
            x_dim,
            binary_y=binary_y,
            n_layers_out=2,
            n_units_out=100,
            n_iter=1000,
            lr=1e-3,
            patience=10,
            batch_size=128,
            batch_norm=False,
            nonlin="relu",
            device=device,
        )
    if name == "DragonNet":
        return cate_models.torch.DragonNet(
            x_dim,
            binary_y=binary_y,
            batch_size=128,
            n_iter=1000,
            lr=1e-5,
            batch_norm=False,
            nonlin="relu",
        )
    if name == "TARNet":
        return cate_models.torch.TARNet(
            x_dim,
            binary_y=binary_y,
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
        )
    if name == "CFRNet_0.01":
        return cate_models.torch.TARNet(
            x_dim,
            binary_y=binary_y,
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
        )
    if name == "CFRNet_0.001":
        return cate_models.torch.TARNet(
            x_dim,
            binary_y=binary_y,
            n_layers_r=2,
            n_layers_out=2,
            n_units_out=100,
            n_units_r=100,
            batch_size=128,
            n_iter=1000,
            lr=1e-5,
            batch_norm=False,
            nonlin="relu",
            penalty_disc=0.001,
        )
    if name == "CausalForest":
        return CausalForest()
    if name == "LinearDR":
        return DRLearner(
            model_final=RidgeCV(alphas=np.logspace(-3, 3, 50)),
            model_propensity=LogisticRegressionCV(Cs=np.logspace(-3, 3, 50)),
            model_response=RidgeCV(alphas=np.logspace(-3, 3, 50)),
            cv=5,
            random_state=0,
        )
    raise ValueError(f"Unknown learner {name}")


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
        "RALearner_ensemble",
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

        X_dev = np.vstack([x_train, x_val])
        W_dev = np.concatenate([w_train, w_val])
        Y_dev = np.concatenate([y_train, y_val])

        nuisance_functions = NuisanceFunctions(
            rct=(data.cohort_name in ["crash_2", "ist3", "sprint", "accord"])
        )
        nuisance_functions.fit(X_dev, Y_dev, W_dev)

        for learner_name in learners:

            oof_pred = oof_predict(
                learner_name,
                x_train,
                y_train,
                w_train,
                device=DEVICE,
                n_splits=5,
                seed=i,
                ensemble_num=ensemble_num,
            )
            results[learner_name]["prediction"][i] = oof_pred
            results[learner_name]["qini_score"][i] = qini_auc_score(
                y_true=y_train, uplift=oof_pred, treatment=w_train
            )
            results[learner_name]["uplift_score"][i] = uplift_auc_score(
                y_true=y_train, uplift=oof_pred, treatment=w_train
            )

            for sec in selection_types:
                results[learner_name][sec][i] = calculate_pehe(
                    oof_pred, (x_train, w_train, y_train), sec, nuisance_functions
                )

            X_dev = np.vstack([x_train, x_val])
            W_dev = np.concatenate([w_train, w_val])
            Y_dev = np.concatenate([y_train, y_val])
            binary_y_dev = len(np.unique(Y_dev)) == 2

            def fit_and_pred_test(seed_for_member=None):
                """Func to retrain with training + validation set"""
                m = make_model_for(
                    learner_name,
                    X_dev.shape[1],
                    binary_y_dev,
                    DEVICE,
                    seed=seed_for_member,
                )
                m.fit(X_dev, Y_dev, W_dev)
                if hasattr(m, "effect"):
                    return m.effect(x_test)
                else:
                    return m.predict(x_test).detach().cpu().numpy().ravel()

            if "ensemble" in learner_name.lower():
                preds = [
                    fit_and_pred_test(seed_for_member=s) for s in range(ensemble_num)
                ]
                pred_test = np.mean(np.column_stack(preds), axis=1)
            else:
                pred_test = fit_and_pred_test()

            test_qini = qini_auc_score(
                y_true=y_test, uplift=pred_test, treatment=w_test
            )
            test_uplift = uplift_auc_score(
                y_true=y_test, uplift=pred_test, treatment=w_test
            )

            results[learner_name].setdefault("test_qini", np.zeros(trials))
            results[learner_name].setdefault("test_uplift", np.zeros(trials))
            results[learner_name]["test_qini"][i] = test_qini
            results[learner_name]["test_uplift"][i] = test_uplift

            for sec in selection_types:
                key = f"{sec}_test"
                if key not in results[learner_name]:
                    results[learner_name][key] = np.zeros(trials)
                results[learner_name][key][i] = calculate_pehe(
                    pred_test, (x_test, w_test, y_test), sec, nuisance_functions
                )

    outdir = f"results/{cohort_name}/model_selection"
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/model_selection_shuffle_{shuffle}.pkl", "wb") as output_file:
        pickle.dump(results, output_file)
