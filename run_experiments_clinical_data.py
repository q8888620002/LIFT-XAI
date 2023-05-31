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

from scipy import stats

from utilities import *

module_path = os.path.abspath(os.path.join('CATENets/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from catenets.models.torch import pseudo_outcome_nets
from catenets.models.torch.base import BasicNet
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
    DEVICE = "cuda:6"
    explainer_limit = 1000

    selection_types = [
        "if_pehe",
        "pseudo_outcome_r",
        "pseudo_outcome_dr"
    ]

    data = Dataset(cohort_name, 10)
    names = data.get_feature_names()

    x_train, _, _ = data.get_training_data()
    x_test, _, _ = data.get_testing_data()


    feature_size = x_train.shape[1]

    explainers = [
        "integrated_gradients",
        "shapley_value_sampling",
        "naive_shap"
    ]
    top_n_results = {
        e:[] for e in explainers
    }

    result_sign = {
        e:np.zeros((trials,feature_size)) for e in explainers
    }

    results_train = np.zeros((trials, len(x_train)))
    results_test = np.zeros((trials, len(x_test)))

    for i in range(trials):

        data = Dataset(cohort_name, i)
        x_train, w_train, y_train = data.get_training_data()

        x_replacement = data.get_replacement_value()

        x_val, w_val, y_val = data.get_validation_data()
        x_test, w_test, y_test = data.get_testing_data()

        models = {
            "xlearner":
                pseudo_outcome_nets.XLearner(
                    x_train.shape[1],
                    binary_y=(len(np.unique(y_train)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    batch_size=128,
                    n_iter=1000,
                    nonlin="relu",
                    device=DEVICE,
                ),
            # "drlearner":
            #     pseudo_outcome_nets.DRLearner(
            #     x_train.shape[1],
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
        learner_explainers = {}
        insertion_deletion_data = []

        for model_name, model in models.items():

            model.fit(x_train, y_train, w_train)

            results_train[i] = model.predict(X=x_train).detach().cpu().numpy().flatten()
            results_test[i] = model.predict(X=x_test).detach().cpu().numpy().flatten()

            # Explain CATE
            learner_explainers[learner] = Explainer(
                model,
                feature_names=list(range(x_train.shape[1])),
                explainer_list=explainers,
            )

            log.info(f"Explaining {learner}")
            learner_explanations[learner] = learner_explainers[learner].explain(
                x_test[:explainer_limit]
            )

            # Calculate IF-PEHE for insertion and deletion for each explanation methods

            for explainer_name in explainers:
                rank_indices = attribution_ranking(learner_explanations[learner][explainer_name])

                top_5_indices = np.argpartition(
                    np.abs(
                        learner_explanations[learner][explainer_name]
                    ).mean(0).round(2),
                    -5
                )[-5:]

                ate, auroc = subgroup_identification(
                    top_5_indices,
                    x_train,
                    x_test,
                    model
                )

                insertion_results, deletion_results = insertion_deletion(
                    data,
                    rank_indices,
                    model,
                    x_replacement,
                    selection_types
                )

                insertion_deletion_data.append(
                    [
                        model_name,
                        explainer_name,
                        insertion_results,
                        deletion_results,
                        ate,
                        auroc
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
                -top_n_features
            )[-top_n_features:]

            top_n_results[explainer_name].extend(names[ind].tolist())

            for col in range(feature_size):
                result_sign[explainer_name][i, col] = stats.pearsonr(
                    x_test[:explainer_limit,col], learner_explanations[learner][explainer_name][:, col]
                )[0]



    for explainer_name in explainers:

        results = collections.Counter(top_n_results[explainer_name])
        summary = pd.DataFrame(
                results.items(),
                columns=[
                    'feature',
                    'count (%)'
                ]
            ).sort_values(
            by="count (%)",
            ascending=False
        )

        summary["count (%)"] = np.round(summary["count (%)"]/(trials), 2)*100

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
