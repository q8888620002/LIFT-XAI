"""
This is the script for calculating feature ranking for explanation methods
on clinical datasets, inlcuding ist3, responder, and massive transfusion
"""

import argparse
import os
import sys
import collections
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb

from scipy import stats
from sklearn.ensemble import RandomForestClassifier

from utilities import insertion_deletion_if_pehe, Dataset

module_path = os.path.abspath(os.path.join('CATENets/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from catenets.models.torch import pseudo_outcome_nets

import src.interpretability.logger as log
from src.interpretability.utils import attribution_ranking
from src.interpretability.explain import Explainer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-d','--dataset', help='Dataset', required=True)
    parser.add_argument('-t','--num_trials', help='Dataset', required=True)
    parser.add_argument('-n','--top_n_features', help='Dataset', required=True)
    parser.add_argument('-l','--learner', help='Dataset', required=True)

    args = vars(parser.parse_args())

    cohort_name = args["dataset"]
    trials = int(args["num_trials"])
    top_n_features = int(args["top_n_features"])
    learner = args["learner"]
    DEVICE = "cuda:1"

    data = Dataset(cohort_name, 10)
    names = data.get_feature_names()

    X_train, _, _ = data.get_training_data()
    X_test, _, _ = data.get_testing_data()
    feature_size = X_train.shape[1]

    explainers = [
        "integrated_gradients",
        "shapley_value_sampling",
        # "naive_shap"
    ]
    top_k_results = dict.fromkeys(
        explainers,
        []
    )
    result_sign = dict.fromkeys(
        explainers,
        np.zeros((trials,feature_size))
    )

    results_train = np.zeros((trials, len(X_train)))
    results_test = np.zeros((trials, len(X_test)))

    for i in range(trials):

        data = Dataset(cohort_name, i)
        X_train, w_train, y_train = data.get_training_data()
        X_test, w_test, y_test = data.get_testing_data()


        models = {
            "xlearner":
                pseudo_outcome_nets.XLearner(
                    X_train.shape[1],
                    binary_y=(len(np.unique(y_train)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    batch_size=128,
                    n_iter=1500,
                    nonlin="relu",
                    device=DEVICE,
                ),
            # "drlearner":
            #     pseudo_outcome_nets.DRLearner(
            #     X_train.shape[1],
            #     binary_y=(len(np.unique(y_train)) == 2),
            #     n_layers_out=2,
            #     n_units_out=100,
            #     batch_size=128,
            #     n_iter=1500,
            #     nonlin="relu",
            #     device=device,
            # )
        }

        learner_explanations = {}

        # Calculate plg-in error

        np.random.seed(i)

        xgb_plugin1 = xgb.XGBClassifier(max_depth=6, random_state=i, n_estimators=100)
        xgb_plugin0 = xgb.XGBClassifier(max_depth=6, random_state=i, n_estimators=100)

        rf = RandomForestClassifier(max_depth=6, random_state=i)

        x0 = X_train[w_train==0]
        x1 = X_train[w_train==1]

        y0 = y_train[w_train==0]
        y1 = y_train[w_train==1]

        xgb_plugin0.fit(x0, y0)
        xgb_plugin1.fit(x1, y1)

        rf.fit(X_train, w_train)

        y_pred0 = xgb_plugin0.predict(X_test)
        y_pred1 = xgb_plugin1.predict(X_test)

        t_plugin = y_pred1 - y_pred0

        ps = rf.predict_proba(X_test)[:, 1]
        a = w_test - ps

        ident = np.ones(len(ps))
        c = ps*(ident-ps)

        b = np.array([2]*len(w_test))*w_test*(w_test-ps) / c

        learner_explainers = {}
        insertion_deletion_data = []

        for model_name, model in models.items():

            model.fit(X_train, y_train, w_train)

            results_train[i] = model.predict(X=X_train).detach().cpu().numpy().flatten()
            results_test[i] = model.predict(X=X_test).detach().cpu().numpy().flatten()

            learner_explainers[learner] = Explainer(
                model,
                feature_names=list(range(X_train.shape[1])),
                explainer_list=explainers,
            )

            log.info(f"Explaining {learner}")
            learner_explanations[learner] = learner_explainers[learner].explain(
                X_test
            )
            # Calculate IF-PEHE for insertion and deletion for each explanation methods

            for explainer_name in explainers:

                rank_indices = attribution_ranking(learner_explanations[learner][explainer_name])

                insertion_results, deletion_results = insertion_deletion_if_pehe(
                    X_test,
                    rank_indices,
                    model,
                    data.get_replacement_value(),
                    a,
                    b,
                    c,
                    t_plugin,
                    y_test
                )

                insertion_deletion_data.append(
                    [
                        model_name,
                        explainer_name,
                        insertion_results,
                        deletion_results
                    ]
                )

        with open(
            f"results/{cohort_name}/"
            f"insertion_deletion_{learner}_{i}.pkl", "wb") as output_file:
            pickle.dump(insertion_deletion_data, output_file)

        #### Getting top n features

        for explainer_name in explainers:
            ind = np.argpartition(
                np.abs(
                    learner_explanations[learner][explainer_name]
                ).mean(0).round(2),
             -top_n_features)[-top_n_features:]

            top_k_results[explainer_name].extend(names[ind].tolist())

            for col in range(feature_size):
                result_sign[explainer_name][i, col] = stats.pearsonr(
                    X_test[:,col], learner_explanations[learner][explainer_name][:, col]
                )[0]


    for explainer_name in explainers:
        results = collections.Counter(top_k_results[explainer_name])
        summary = pd.DataFrame(
                results.items(),
                columns=['feature', 'count (%)']
            ).sort_values(
            by="count (%)",
            ascending=False
        )

        summary["count (%)"] = np.round(summary["count (%)"]/trials,2)*100

        indices = [names.tolist().index(i) for i in summary.feature.tolist()]
        summary["sign"] = np.sign(np.mean(result_sign[explainer_name], axis=0)[indices])

        summary.to_csv(
            f"results/{cohort_name}/"
            f"{explainer_name}_top_{top_n_features}_features_{learner}.csv"
        )

    with open( f"results/{cohort_name}/train_{learner}.pkl", "wb") as output_file:
        pickle.dump(results_train, output_file)

    with open( f"results/{cohort_name}/test_{learner}.pkl", "wb") as output_file:
        pickle.dump(results_test, output_file)
