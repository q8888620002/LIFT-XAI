"""
This is the script for calculating feature ranking for explanation methods
on clinical datasets, inlcuding ist3, responder, and massive transfusion
"""

import argparse
import os
import sys
import collections
import pickle
import random

import numpy as np
import pandas as pd

from scipy import stats

from utilities import *
from dataset import Dataset

module_path = os.path.abspath(os.path.join('CATENets/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from catenets.models.torch import pseudo_outcome_nets
from catenets.models.torch.base import BasicNet
import catenets.models as cate_models

import src.interpretability.logger as log
from src.interpretability.explain import Explainer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-d','--dataset', help='Dataset', required=True)
    parser.add_argument('-s','--shuffle', help='shuffle',  default=True, action='store_false')
    parser.add_argument('-t','--num_trials', help='number of runs ', required=True, type=int)
    parser.add_argument('-n','--top_n_features', help='how many features to extract', required=True, type=int)
    parser.add_argument('-l','--learner', help='learner', required=True)

    args = vars(parser.parse_args())

    cohort_name = args["dataset"]
    trials = args["num_trials"]
    top_n_features = args["top_n_features"]
    shuffle = args["shuffle"]
    learner = args["learner"]
    DEVICE = "cuda:1"
    
    print("shuffle dataset: ", shuffle)

    explainer_limit = 1000

    selection_types = [
        "if_pehe",
        "pseudo_outcome_r",
        "pseudo_outcome_dr"
    ]

    data = Dataset(cohort_name, 10)
    names = data.get_feature_names()

    x_train, _, _ = data.get_data("train")
    x_test, _, _ = data.get_data("test")


    feature_size = x_train.shape[1]

    explainers = [
        "saliency",
        "integrated_gradients",
        "baseline_shapley_value_sampling",
        "marginal_shapley_value_sampling"
        # "kernel_shap"
        # "marginal_shap"
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

        data = Dataset(cohort_name, i, shuffle)
        
        x_train, w_train, y_train = data.get_data("train")
        x_val, w_val, y_val = data.get_data("val")
        x_test, w_test, y_test = data.get_data("test")

        models = {
            "XLearner":
                pseudo_outcome_nets.XLearner(
                    x_train.shape[1],
                    binary_y=(len(np.unique(y_train)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    batch_size=128,
                    n_iter=1000,
                    nonlin="relu",
                    device=DEVICE,
                    seed=i
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
                        seed=i
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
                      device = DEVICE,
                      seed=i
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
                      device = DEVICE,
                      seed=i
                  ),
            "TLearner": cate_models.torch.TLearner(
                        x_train.shape[1],
                        binary_y=(len(np.unique(y_train)) == 2),
                        n_layers_out=2,
                        n_units_out=100,
                        batch_size=128,
                        n_iter=1000,
                        nonlin="relu",
                        device=DEVICE
                ),
            "TARNet": cate_models.torch.TARNet(
                    x_train.shape[1],
                    binary_y=True,
                    n_layers_r=1,
                    n_layers_out=1,
                    n_units_out=100,
                    n_units_r=100,
                    batch_size=128,
                    n_iter=1000,
                    batch_norm=False,
                    early_stopping = True,
                    nonlin="relu"
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
                lr=1e-3,
                batch_norm=False,
                nonlin="relu",
                penalty_disc=0.01,
            ),
            "DRLearner":
                pseudo_outcome_nets.DRLearner(
                x_train.shape[1],
                binary_y=(len(np.unique(y_train)) == 2),
                n_layers_out=2,
                n_units_out=100,
                batch_size=128,
                n_iter=1000,
                nonlin="relu",
                device=DEVICE,
            )
        }

        learner_explanations = {}
        learner_explainers = {}
        insertion_deletion_data = []

        ## Training nuisance function for pehe. 

        if data.cohort_name == "crash_2" or data.cohort_name =="ist3":
            nuisance_functions = NuisanceFunctions(rct=True)
        else:
            nuisance_functions = NuisanceFunctions(rct=False)

        nuisance_functions.fit(x_val, y_val, w_val)

        model = models[learner]

        # Create a matrix of zeros with the same shape as x_train        

        noise_matrix = np.zeros(x_train.shape)
        baseline = np.mean(x_train, axis=0)

        for _, idx_lst in data.discrete_indices.items():

            # Setting the one-hot variables within the same group with the same baseline/replacement value.

            baseline[idx_lst] = 1/len(idx_lst)

            # Fill in only the continuous columns with Gaussian noise
            # mask = np.zeros(feature_size, dtype=bool)
            # mask[idx_lst] = True
            # noise_matrix[:, mask] = np.random.normal(0, 0.1, (x_train.shape[0], len(idx_lst)))

        # x_train_noise = x_train + noise_matrix

        model.fit(x_train, y_train, w_train)

        results_train[i] = model.predict(X=x_train).detach().cpu().numpy().flatten()
        results_test[i] = model.predict(X=x_test).detach().cpu().numpy().flatten()

        # Explain CATE
        learner_explainers[learner] = Explainer(
            model,
            feature_names=list(range(x_train.shape[1])),
            explainer_list=explainers,
            perturbations_per_eval=1,
            baseline = baseline.reshape(1, -1)
        )

        log.info(f"Explaining dataset with: {learner}")

        learner_explanations[learner] = learner_explainers[learner].explain(
            x_test
        )

        # Calculate IF-PEHE for insertion and deletion for each explanation methods

        for explainer_name in explainers:
                        
            ate_results = []
            auroc_results = []
            rand_ate = []
            rand_auroc = []

            abs_explanation = np.abs(learner_explanations[learner][explainer_name])

            # Normalize explanation on row basis.

            norm_explanation = normalize(learner_explanations[learner][explainer_name])

            ## obtaining global & local ranking for insertion & deletion

            local_rank = attribution_ranking(abs_explanation)            
            global_rank = np.flip(np.argsort(abs_explanation.mean(0)))

            insertion_results, deletion_results = insertion_deletion(
                data.get_data("test"),
                local_rank,
                model,
                baseline,
                selection_types,
                nuisance_functions
            )

            perturb_resource_results, perturb_spurious_results = np.zeros((40, 4)), np.zeros((40, 4))
            n_steps = 20

            discrete_indices = [index for _, v in data.discrete_indices.items() for index in v]

            ## obtain oracle & threshold for supurious/resource experiment.
            
            perturbated_var = generate_perturbed_var(
                data, 
                x_test, 
                feature_size, 
                discrete_indices, 
                n_steps, 
                model
            )

            sprious_qualtile = np.quantile(perturbated_var, 0.8)
            threshold_vals = np.linspace(start=-1.0,stop=1.0,num=40)

            for feature_idx in range(feature_size):

                print("obtaining subgroup results for %s, feature_num: %s."%(explainer_name ,feature_idx+1),end='\r')
                
                ## Starting from 1 features 
                ate, auroc = subgroup_identification(
                    global_rank[:feature_idx+1],
                    x_train,
                    x_test,
                    model,
                    # True
                )
                auroc_results.append(auroc)
                ate_results.append(ate)
                
                random_ate, random_auroc = subgroup_identification(
                    random.sample(range(0, x_train.shape[1]), feature_idx+1),
                    x_train,
                    x_test,
                    model
                )

                rand_auroc.append(random_auroc)
                rand_ate.append(random_ate)

                ## Perturbation experiment with only using continuous variables 
                
                if feature_idx < np.min(discrete_indices):   

                    # Generating perturbed samples with (n*n_steps, d)
                    print("generating resource/spurious results for %s, feature: %s."%(explainer_name , feature_idx),end='\r')

                    perturbated_samples = generate_perturbations(
                        data, 
                        x_test, 
                        feature_idx,
                        n_steps
                    )

                    perturbed_output = model.predict(X=perturbated_samples).detach().cpu().numpy()
                    
                    resource_results, spurious_results  = perturbation(
                        perturbed_output,
                        norm_explanation[:, feature_idx],
                        threshold_vals,
                        n_steps,
                        sprious_qualtile                    
                    )
                    
                    perturb_resource_results += resource_results
                    perturb_spurious_results += spurious_results

            ## results with all features. 

            full_ate, full_auroc = subgroup_identification(
                [i for i in range(x_train.shape[1])],
                x_train,
                x_test,
                model
            )

            insertion_deletion_data.append(
                [
                    learner,
                    explainer_name,
                    insertion_results,
                    deletion_results,
                    ate_results,
                    auroc_results,
                    full_ate, 
                    full_auroc,
                    random_ate,
                    rand_auroc,
                    perturb_resource_results,
                    perturb_spurious_results
                ]
            )

        with open(
            f"results/{cohort_name}/"
            f"insertion_deletion_shuffle_{shuffle}_{learner}_seed_{i}.pkl", "wb") as output_file:
            pickle.dump(insertion_deletion_data, output_file)

        #### Getting top n features

        for explainer_name in explainers:

            ind = np.argpartition(
                np.abs(
                    learner_explanations[learner][explainer_name]
                ).mean(0),
                -top_n_features
            )[-top_n_features:]

            top_n_results[explainer_name].extend(names[ind].tolist())

            for col in range(feature_size):
                result_sign[explainer_name][i, col] = stats.pearsonr(
                    x_test[:,col], learner_explanations[learner][explainer_name][:, col]
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
            f"{explainer_name}_top_{top_n_features}_features_shuffle_{shuffle}_{learner}.csv"
        )

    with open( f"results/{cohort_name}/train_shuffle_{shuffle}_{learner}.pkl", "wb") as output_file:
        pickle.dump(results_train, output_file)

    with open( f"results/{cohort_name}/test_shuffle_{shuffle}_{learner}.pkl", "wb") as output_file:
        pickle.dump(results_test, output_file)
