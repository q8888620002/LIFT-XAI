"""Script that perform cross cohort analysis"""
import argparse
import os
import pickle

import numpy as np
import torch
from captum.attr import ShapleyValueSampling

import src.CATENets.catenets.models.torch.pseudo_outcome_nets as pseudo_outcome_nets
from src.dataset import Dataset
from src.utils import *

DEVICE = "cuda:1"


def compute_shap_values(model, data_sample, data_baseline):
    """Function for shapley value sampling"""
    shapley_model = ShapleyValueSampling(model)
    shap_values = (
        shapley_model.attribute(
            torch.tensor(data_sample).to(DEVICE),
            n_samples=1000,
            baselines=torch.tensor(data_baseline.reshape(1, -1)).to(DEVICE),
            perturbations_per_eval=10,
            show_progress=True,
        )
        .detach()
        .cpu()
        .numpy()
    )
    return shap_values



def main(args):
    """Main function for computing shapley value"""

    trials = args["num_trials"]
    bshap = args["baseline"]
    cohort_name = args["cohort_name"]

    print("Baselines shapley:", bshap)

    dataset1 = Dataset(args.dataset, 0)

    cohort1_predict_results = np.zeros((trials, len(cohort1_x)))
    cohort1_average_shap = np.zeros((trials, cohort1_x.shape[0], cohort1_x.shape[1]))

    for i in range(trials):
        # Model training

        sampled_indices = np.random.choice(
            len(cohort1_x), size=len(cohort1_x), replace=True
        )

        x_sampled = cohort1_x[sampled_indices]
        y_sampled = cohort1_y[sampled_indices]
        w_sampled = cohort1_w[sampled_indices]

        model = pseudo_outcome_nets.XLearner(
            x_sampled.shape[1],
            binary_y=(len(np.unique(y_sampled)) == 2),
            n_layers_out=2,
            n_units_out=100,
            batch_size=128,
            n_iter=1000,
            nonlin="relu",
            device=DEVICE,
            seed=i,
        )

        model.fit(x_sampled, y_sampled, w_sampled)

        cohort1_predict_results[i] = (
            model.predict(X=cohort1_x).detach().cpu().numpy().flatten()
        )
        cohort2_predict_results[i] = (
            model.predict(X=cohort2_x).detach().cpu().numpy().flatten()
        )

        if bshap:
            baseline = cohort1_x.mean(0)

            for _, idx_lst in dataset1.discrete_indices.items():
                if len(idx_lst) == 1:

                    # setting binary vars to 0.5

                    baseline[idx_lst] = 0.5
                else:
                    # setting categorical baseline to 1/n
                    # category_counts = data[:, idx_lst].sum(axis=0)
                    # baseline[idx_lst] = category_counts / category_counts.sum()

                    baseline[idx_lst] = 1 / len(idx_lst)
        else:
            baseline_index = np.random.choice(len(cohort1_x), 1)
            baseline = cohort1_x[baseline_index]

        cohort1_average_shap[i] = compute_shap_values(model, cohort1_x, baseline)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(
        os.path.join(save_path, f"{cohort1}_predict_results_{bshap}.pkl"), "wb"
    ) as output_file:
        pickle.dump(cohort1_predict_results, output_file)

    with open(
        os.path.join(save_path, f"{cohort1}_shap_bootstrapped_{bshap}.pkl"), "wb"
    ) as output_file:
        pickle.dump(cohort1_average_shap, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument(
        "-t", "--num_trials", help="number of runs ", required=True, type=int
    )
    parser.add_argument(
        "-c",
        "--cohort_name",
        help="name of cross cohort analysis ",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-b",
        "--baseline",
        help="whether using baseline",
        default=True,
        action="store_false",
    )

    args = vars(parser.parse_args())

    main(args)
