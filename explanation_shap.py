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


def compute_shap_values(model, data_sample, data_baseline):
    device = "cuda:1"
    shapley_model = ShapleyValueSampling(model)
    shap_values = shapley_model.attribute(
                        torch.tensor(data_sample).to(device),
                        n_samples=1000,
                        baselines=torch.tensor(data_baseline.reshape(1, -1)).to(device),
                        perturbations_per_eval=10,
                        show_progress=True
                    ).detach().cpu().numpy()
    return shap_values


def main(args):
    cohort_name = args["dataset"]
    trials = args["num_trials"]
    subgroup_col = args["subgroup_column"]

    data = Dataset(cohort_name, 0)
    x, y, w = data.get_data()

    subgroup_index = data.get_feature_names().tolist().index(subgroup_col)

    predict_results = np.zeros((trials, len(x)))
    average_shap = np.zeros((trials, x.shape[0], x.shape[1]))

    unique_subgroup_values = np.unique(x[:, subgroup_index])
    subgroup_shap_values = {f"{subgroup_col}={value}": np.zeros((trials, x[x[:, subgroup_index] == value].shape[0], x.shape[1])) 
                            for value in unique_subgroup_values}

    for i in range(trials):
        x, w, y = data.get_data()


        sampled_indices = np.random.choice(len(x), size=len(x), replace=True)
        x_sampled = x[sampled_indices]
        y_sampled = y[sampled_indices]
        w_sampled = w[sampled_indices]

        # Model training
        device = "cuda:1"

        model = pseudo_outcome_nets.XLearner(
                x_sampled.shape[1],
                binary_y=(len(np.unique(y_sampled)) == 2),
                n_layers_out=2,
                n_units_out=100,
                batch_size=128,
                n_iter=1000,
                nonlin="relu",
                device=device,
                seed=i
        )

        model.fit(x_sampled, y_sampled, w_sampled)

        predict_results[i] = model.predict(X=x_sampled).detach().cpu().numpy().flatten()

        mean_baseline = x.mean(0)

        mean_baseline = obtain_baselines()
        average_shap[i] = compute_shap_values(model, x, mean_baseline)

        for unique_value in unique_subgroup_values:
            subgroup_sample = x[x[:, subgroup_index] == unique_value]
            subgroup_baseline = subgroup_sample.mean(0)
            shap_value = compute_shap_values(model, subgroup_sample, subgroup_baseline)
            subgroup_shap_values[f"{subgroup_col}={unique_value}"][i] = shap_value

    save_path = os.path.join("results", cohort_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, "predict_results_accord_baseline.pkl"), "wb") as output_file:
        pickle.dump(predict_results, output_file)

    with open(os.path.join(save_path, "shap_bootstrapped_accord_baseline.pkl"), "wb") as output_file:
        pickle.dump(average_shap, output_file)

    for subgroup, shap_values in subgroup_shap_values.items():
        file_name = f"{subgroup}_shap_accord_baseline.pkl"
        with open(os.path.join(save_path, file_name), "wb") as output_file:
            pickle.dump(shap_values, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-d', '--dataset', help='Dataset', required=True, type=str)
    parser.add_argument('-t', '--num_trials', help='number of runs ', required=True, type=int)
    parser.add_argument('-g', '--subgroup_column', help='Dataset', required=True, type=str)
    args = vars(parser.parse_args())

    main(args)
