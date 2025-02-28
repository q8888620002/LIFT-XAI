"""Script that perform cross cohort analysis"""
import argparse
import os
import pickle

import numpy as np
import torch
from captum.attr import ShapleyValueSampling

import src.CATENets.catenets.models.torch.pseudo_outcome_nets as pseudo_outcome_nets
from src.dataset import Dataset, obtain_accord_baselines, obtain_txa_baselines
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


def setup_datasets(cohort_name):
    """Configure and return datasets based on cohort name."""
    if cohort_name == "accord_sprint":
        return {
            "accord": Dataset("accord_filter", 0),
            "sprint": Dataset("sprint_filter", 0),
        }, obtain_accord_baselines()
    elif cohort_name == "crash2_txa":
        return {
            "crash2": Dataset("crash_2", 0),
            "txa": Dataset("txa", 0),
        }, obtain_txa_baselines()
    else:
        raise ValueError(f"Unsupported cohort name: {cohort_name}")


def initialize_results(trials, datasets):
    """Initialize result structures for predictions and SHAP values."""
    results = {}
    for name, data in datasets.items():
        results[name] = {
            "predict_results": np.zeros((trials, len(data["x"]))),
            "average_shap": np.zeros((trials, data["x"].shape[0], data["x"].shape[1])),
        }
    return results


def main(args):
    """Main function for computing shapley value"""

    trials = args["num_trials"]
    bshap = args["baseline"]
    cohort_name = args["cohort_name"]

    print("Baselines shapley:", bshap)

    if cohort_name == "accord_sprint":

        cohort1 = "accord"
        cohort2 = "sprint"

        dataset1 = Dataset("sprint_filter", 0)
        dataset2 = Dataset("accord_filter", 0)

        (
            cohort1_x,
            cohort1_w,
            cohort1_y,
            cohort2_x,
            _,
            _,
        ) = obtain_accord_baselines()

    elif cohort_name == "crash2_txa":
        cohort1 = "crash2"
        cohort2 = "txa"

        dataset1 = Dataset("crash_2", 0)
        dataset2 = Dataset("txa", 0)

        (
            cohort1_x,
            cohort1_w,
            cohort1_y,
            cohort2_x,
            _,
            _,
        ) = obtain_txa_baselines()

    cohort1_predict_results = np.zeros((trials, len(cohort1_x)))
    cohort1_average_shap = np.zeros((trials, cohort1_x.shape[0], cohort1_x.shape[1]))

    cohort2_predict_results = np.zeros((trials, len(cohort2_x)))
    cohort2_average_shap = np.zeros((trials, cohort2_x.shape[0], cohort2_x.shape[1]))

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

        if bshap:
            baseline = cohort2_x.mean(0)

            for _, idx_lst in dataset2.discrete_indices.items():
                if len(idx_lst) == 1:

                    # setting binary vars to 0.5

                    baseline[idx_lst] = 0.5
                else:
                    # setting categorical baseline to 1/n
                    # category_counts = data[:, idx_lst].sum(axis=0)
                    # baseline[idx_lst] = category_counts / category_counts.sum()

                    baseline[idx_lst] = 1 / len(idx_lst)
        else:
            baseline_index = np.random.choice(len(cohort2_x), 1)
            baseline = cohort2_x[baseline_index]

        cohort2_average_shap[i] = compute_shap_values(model, cohort2_x, baseline)

    save_path = os.path.join("results", f"{cohort1}_{cohort2}")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(
        os.path.join(save_path, f"{cohort1}_predict_results_{bshap}.pkl"), "wb"
    ) as output_file:
        pickle.dump(cohort1_predict_results, output_file)

    with open(
        os.path.join(save_path, f"{cohort2}_predict_results_{bshap}.pkl"), "wb"
    ) as output_file:
        pickle.dump(cohort2_predict_results, output_file)

    with open(
        os.path.join(save_path, f"{cohort1}_shap_bootstrapped_{bshap}.pkl"), "wb"
    ) as output_file:
        pickle.dump(cohort1_average_shap, output_file)

    with open(
        os.path.join(save_path, f"{cohort2}_shap_bootstrapped_{bshap}.pkl"), "wb"
    ) as output_file:
        pickle.dump(cohort2_average_shap, output_file)


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
