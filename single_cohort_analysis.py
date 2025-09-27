"""Script that perform cross cohort analysis"""
import argparse
import os
import pickle

import numpy as np
import torch
import wandb
from captum.attr import ShapleyValueSampling

import src.CATENets.catenets.models.torch.pseudo_outcome_nets as pseudo_outcome_nets
from src.dataset import Dataset

DEVICE = "cuda:1"
os.environ["WANDB_API_KEY"] = "a010d8a84d6d1f4afed42df8d3e37058369030c4"


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


def compute_shap_similarity(shap_values_1, shap_values_2):
    """Compute multiple similarity metrics for SHAP values."""

    shap_values_1 = shap_values_1.flatten()
    shap_values_2 = shap_values_2.flatten()

    # Cosine Similarity
    cosine_sim = np.dot(shap_values_1, shap_values_2) / (
        np.linalg.norm(shap_values_1) * np.linalg.norm(shap_values_2) + 1e-8
    )

    return cosine_sim


def parse_args():
    """Parser for arguments"""
    parser = argparse.ArgumentParser(description="Single Cohort SHAP Analysis")
    parser.add_argument(
        "--num_trials",
        help="number of runs ",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--cohort_name",
        help="name of cross cohort analysis ",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--baseline",
        help="whether using baseline",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "--wandb",
        help="whether using baseline",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "--relative_change_threshold",
        help="Threshold for stopping based on local SHAP relative change",
        default=0.05,
        type=float,
    )
    return parser.parse_args()


def main(args):
    """Main function for computing shapley value"""

    print(args)

    if args.wandb:

        wandb.init(
            project=f"Convergence for Shapley value {args.cohort_name}",
            notes=f"Experiment for {args.cohort_name};{args.num_trials}",
            dir="/data/mingyulu/wandb",
            config={
                "num_trials": args.num_trials,
                "dataset": args.cohort_name,
                "relative_change_threshold": args.relative_change_threshold,
                "model": "XLearner",
                "baseline": args.baseline,
            },
        )

    save_path = f"results/{args.cohort_name}/shapley"  # Define the save directory

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataset = Dataset(args.cohort_name, 0)
    x_train, w_train, y_train = dataset.get_data()

    cohort_predict_results = []
    cohort_shap_values = []

    for i in range(args.num_trials):
        # Model training

        sampled_indices = np.random.choice(
            len(x_train), size=int(0.9 * len(x_train)), replace=False
        )

        x_sampled = x_train[sampled_indices]
        y_sampled = y_train[sampled_indices]
        w_sampled = w_train[sampled_indices]

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

        cohort_predict_results.append(
            model.predict(X=x_train).detach().cpu().numpy().flatten()
        )

        if not args.baseline:
            baseline = np.median(x_sampled, 0)

            for _, idx_lst in dataset.discrete_indices.items():
                if len(idx_lst) == 1:
                    # setting binary vars to 0.5
                    baseline[idx_lst] = 0.5
                else:
                    # setting categorical baseline to 1/n
                    # category_counts = x_sampled[:, idx_lst].sum(axis=0)
                    # baseline[idx_lst] = category_counts / category_counts.sum()
                    baseline[idx_lst] = 1 / len(idx_lst)
        else:
            baseline_index = np.random.choice(len(x_train), 1)
            baseline = x_train[baseline_index]

        print(f"Trial {i+1}/{args.num_trials} - Computing SHAP values")

        # Compute SHAP values first
        shap_values = compute_shap_values(model, x_train, baseline)
        cohort_shap_values.append(shap_values)

        shap_values_array = np.array(
            cohort_shap_values
        )  # Shape: (num_trials, num_samples, num_features)
        mean_shap_values = np.mean(shap_values_array, axis=0)

        # Compute relative change in mean local SHAP explanations
        if i > 5:
            prev_mean_shap_values = np.mean(np.array(cohort_shap_values[:-1]), axis=0)
            relative_change = np.abs(mean_shap_values - prev_mean_shap_values) / (
                np.abs(prev_mean_shap_values) + 1e-8
            )
            avg_relative_change = np.mean(relative_change)

            cosine_sim = compute_shap_similarity(
                mean_shap_values, prev_mean_shap_values
            )

            if args.wandb:
                wandb.log(
                    {
                        "Trials": i + 1,
                        "Relative Change": avg_relative_change,
                        "cosine sim": cosine_sim,
                    }
                )

            print(
                f"Trial {i+1}: Average Relative Change in Mean Local SHAP Explanations"
                f" = {avg_relative_change:.6f}"
                f" cosine sim: {cosine_sim}"
            )

            if avg_relative_change < args.relative_change_threshold:
                print(
                    f"Mean local SHAP explanations stabilized at trial {i}"
                    f". Stopping early."
                )
                break

    with open(
        os.path.join(
            save_path, f"{args.cohort_name}_predict_results_{args.baseline}.pkl"
        ),
        "wb",
    ) as output_file:
        pickle.dump(np.stack(cohort_predict_results), output_file)

    with open(
        os.path.join(
            save_path, f"{args.cohort_name}_shap_bootstrapped_{args.baseline}.pkl"
        ),
        "wb",
    ) as output_file:
        pickle.dump(np.stack(cohort_shap_values), output_file)

    print("SHAP computation completed. Results saved to:", save_path)


if __name__ == "__main__":

    args = parse_args()
    main(args)
