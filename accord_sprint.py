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

DEVICE = "cuda:1"

def compute_shap_values(model, data_sample, data_baseline):
    shapley_model = ShapleyValueSampling(model)
    shap_values = shapley_model.attribute(
                        torch.tensor(data_sample).to(DEVICE),
                        n_samples=1000,
                        baselines=torch.tensor(data_baseline.reshape(1, -1)).to(DEVICE),
                        perturbations_per_eval=10,
                        show_progress=True
                    ).detach().cpu().numpy()
    return shap_values


def main(args):
    trials = args["num_trials"]
    bshap = args["baseline"]

    print("Baselines shapley:", bshap)

    sprint = Dataset("sprint_filter", 0)
    accord =  Dataset("accord_filter", 0)
    
    x, _, _ = sprint.get_data()

    sprint_predict_results = np.zeros((trials, len(x)))
    sprint_average_shap = np.zeros((trials, x.shape[0], x.shape[1]))

    x, _, _ = accord.get_data()

    accord_predict_results = np.zeros((trials, len(x)))
    accord_average_shap = np.zeros((trials, x.shape[0], x.shape[1]))

    for i in range(trials):
        # Model training

        sprint_x, sprint_w, sprint_y, accord_x, accord_w, accord_y = obtain_accord_baselines()

        sampled_indices = np.random.choice(len(sprint_x), size=len(sprint_x), replace=True)
        
        x_sampled = sprint_x[sampled_indices]
        y_sampled = sprint_y[sampled_indices]
        w_sampled = sprint_w[sampled_indices]

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

        sprint_predict_results[i] = model.predict(X=sprint_x).detach().cpu().numpy().flatten()
        accord_predict_results[i] = model.predict(X=accord_x).detach().cpu().numpy().flatten()

        if bshap:
            baseline = sprint_x.mean(0)

            for _, idx_lst in sprint.discrete_indices.items():
                if len(idx_lst) == 1:

                    # setting binary vars to 0.5

                    baseline[idx_lst] = 0.5
                else:
                    # setting categorical baseline to 1/n 
                    # category_counts = data[:, idx_lst].sum(axis=0)
                    # baseline[idx_lst] = category_counts / category_counts.sum()

                    baseline[idx_lst] = 1/len(idx_lst)
        else:
            baseline_index = np.random.choice(len(sprint_x) ,1)
            baseline = sprint_x[baseline_index]

        sprint_average_shap[i] = compute_shap_values(model, sprint_x, baseline)

        if bshap:
            baseline = accord_x.mean(0)

            for _, idx_lst in accord.discrete_indices.items():
                if len(idx_lst) == 1:

                    # setting binary vars to 0.5

                    baseline[idx_lst] = 0.5
                else:
                    # setting categorical baseline to 1/n 
                    # category_counts = data[:, idx_lst].sum(axis=0)
                    # baseline[idx_lst] = category_counts / category_counts.sum()

                    baseline[idx_lst] = 1/len(idx_lst)
        else:
            baseline_index = np.random.choice(len(accord_x) ,1)
            baseline = accord_x[baseline_index]

        accord_average_shap[i] = compute_shap_values(model, accord_x, baseline)

    save_path = os.path.join("results", "accord_sprint")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, f"sprint_predict_results_{bshap}.pkl"), "wb") as output_file:
        pickle.dump(sprint_predict_results, output_file)

    with open(os.path.join(save_path, f"accord_predict_results_{bshap}.pkl"), "wb") as output_file:
        pickle.dump(accord_predict_results, output_file)

    with open(os.path.join(save_path, f"sprint_shap_bootstrapped_{bshap}.pkl"), "wb") as output_file:
        pickle.dump(sprint_average_shap, output_file)

    with open(os.path.join(save_path, f"accord_shap_bootstrapped_{bshap}.pkl"), "wb") as output_file:
        pickle.dump(accord_average_shap, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-t', '--num_trials', help='number of runs ', required=True, type=int)
    parser.add_argument('-b', '--baseline', help='whether using baseline', default=True, action='store_false')

    args = vars(parser.parse_args())

    main(args)