import os
import sys
import pickle
import argparse
import numpy as np
import collections
import torch

from shapreg import shapley, games, removal, shapley_sampling
from utilities import *
from dataset import *
from captum.attr import ShapleyValueSampling

module_path = os.path.abspath(os.path.join('CATENets/'))
if module_path not in sys.path:
    sys.path.append(module_path)

import catenets.models.torch.pseudo_outcome_nets as pseudo_outcome_nets
from catenets.models.torch.base import BasicNet

DEVICE = "cuda:0"

def compute_shap_values(model, data_sample, data_baseline):
    shapley_model = ShapleyValueSampling(model)

    shap_values = shapley_model.attribute(
                        torch.from_numpy(data_sample).to(DEVICE).float(),
                        n_samples=1000,
                        baselines=torch.from_numpy(data_baseline.reshape(1, -1)).to(DEVICE).float(),
                        perturbations_per_eval=10,
                        show_progress=True
                    ).detach().cpu().numpy()
    return shap_values


def main(args):
    cohort_name = args["dataset"]
    trials = args["num_trials"]
    subgroup_col = args["subgroup_column"]
    bshap = args["baseline"]

    print("Baselines shapley:", bshap)
    data = Dataset(cohort_name, 0)
    x, y, w = data.get_data()

    if bshap == True:
        baseline = x.mean(0)

        for _, idx_lst in data.discrete_indices.items():
            if len(idx_lst) == 1:

                # setting binary vars to 0.5

                baseline[idx_lst] = 0.5
            else:
                # setting categorical baseline to 1/n 
                # category_counts = data[:, idx_lst].sum(axis=0)
                # baseline[idx_lst] = category_counts / category_counts.sum()

                baseline[idx_lst] = 1/len(idx_lst)
    
    subgroup_index = data.get_feature_names().tolist().index(subgroup_col)

    predict_results = np.zeros((trials, len(x)))
    ensemble_shap = np.zeros((trials, x.shape[0], x.shape[1]))
    single_shap = np.zeros((trials//10, x.shape[0], x.shape[1]))
    # ensemble_shap = np.zeros(x.shape)

    unique_subgroup_values = np.unique(x[:, subgroup_index])
    subgroup_shap_values = {f"{subgroup_col}={value}": np.zeros((trials, x[x[:, subgroup_index] == value].shape[0], x.shape[1])) 
                            for value in unique_subgroup_values}

    # ensemble = BasicNet(
    #     "EnsembleNet",
    #     n_unit_in = x.shape[1],
    #     binary_y=False,
    #     n_layers_out=2,
    #     n_units_out=100,
    #     batch_size=128,
    #     n_iter=1000,
    #     nonlin="relu",
    #     device=DEVICE,
    #     # prob_diff=True
    # )
    
    for i in range(trials):

        x, w, y = data.get_data()

        sampled_indices = np.random.choice(len(x), size=len(x), replace=True)

        x_sampled = x[sampled_indices]
        y_sampled = y[sampled_indices]
        w_sampled = w[sampled_indices]
        
        # Model training

        model = pseudo_outcome_nets.XLearner(
                x_sampled.shape[1],
                binary_y=(len(np.unique(y_sampled)) == 2),
                n_layers_out=2,
                n_units_out=100,
                batch_size=128,
                n_iter=1000,
                nonlin="relu",
                device=DEVICE,
                seed=i
        )

        model.fit(x_sampled, y_sampled, w_sampled)

        y_hat = model.predict(X=x).detach().cpu().numpy().flatten()
        predict_results[i] = y_hat

        # ensemble.fit(x_sampled, y_hat)

        if not bshap:
            baseline_index = np.random.choice(len(x_sampled) ,1)
            baseline = x_sampled[baseline_index]

        ensemble_shap[i] = compute_shap_values(model, x, baseline)

        for unique_value in unique_subgroup_values:
            subgroup_sample = x[x[:, subgroup_index] == unique_value]
            subgroup_baseline = subgroup_sample.mean(0)
            shap_value = compute_shap_values(model, subgroup_sample, subgroup_baseline)
            subgroup_shap_values[f"{subgroup_col}={unique_value}"][i] = shap_value

        # explaining single model without bootstrapped

        if i % 10 == 0:
            model = pseudo_outcome_nets.XLearner(
                    x_sampled.shape[1],
                    binary_y=(len(np.unique(y_sampled)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    batch_size=128,
                    n_iter=1000,
                    nonlin="relu",
                    device=DEVICE,
                    seed=i
            )

            model.fit(x, y, w)
            single_shap[i//10] = compute_shap_values(model, x, x.mean(0))

    # ensemble_shap = compute_shap_values(ensemble, x, baseline)

    save_path = os.path.join("results", cohort_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, f"predict_results_ensemble_{bshap}.pkl"), "wb") as output_file:
        pickle.dump(predict_results, output_file)

    with open(os.path.join(save_path, f"shap_bootstrapped_ensemble_{bshap}.pkl"), "wb") as output_file:
        pickle.dump(ensemble_shap, output_file)

    # with open(os.path.join(save_path, f"shap_distilled_{bshap}.pkl"), "wb") as output_file:
    #     pickle.dump(ensemble_shap, output_file)

    with open(os.path.join(save_path, f"single_model_{bshap}.pkl"), "wb") as output_file:
        pickle.dump(single_shap, output_file)

    for subgroup, shap_values in subgroup_shap_values.items():
        file_name = f"{subgroup}_shap_ensemble_{bshap}.pkl"
        with open(os.path.join(save_path, file_name), "wb") as output_file:
            pickle.dump(shap_values, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-d', '--dataset', help='Dataset', required=True, type=str)
    parser.add_argument('-t', '--num_trials', help='number of runs ', required=True, type=int)
    parser.add_argument('-g', '--subgroup_column', help='Dataset', required=True, type=str)
    parser.add_argument('-b', '--baseline', help='whether using baseline', default=True, action='store_false')

    args = vars(parser.parse_args())

    main(args)
