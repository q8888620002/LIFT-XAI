"""Script that perform cross cohort analysis"""
import argparse
import collections
import json
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


def _to_py(x):
    """Convert numpy types to plain Python for JSON serialization."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    return x


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
    parser.add_argument(
        "--top_n_features",
        help="Number of top features to extract for summary",
        default=10,
        type=int,
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

    # Export JSON summary with detailed SHAP statistics
    shap_values_array = np.stack(cohort_shap_values)  # (num_trials, num_samples, num_features)
    feature_names = dataset.get_feature_names()
    num_trials_completed = len(cohort_shap_values)
    num_features = shap_values_array.shape[2]

    # Compute aggregated statistics across trials and samples
    # Mean absolute SHAP value per feature (averaged across samples, then across trials)
    abs_mean_per_trial = np.abs(shap_values_array).mean(axis=1)  # (trials, features)
    abs_mean = abs_mean_per_trial.mean(axis=0)  # (features,)
    abs_std = abs_mean_per_trial.std(axis=0)  # (features,)

    # Mean SHAP value per feature (signed)
    mean_per_trial = shap_values_array.mean(axis=1)  # (trials, features)
    mean_val = mean_per_trial.mean(axis=0)  # (features,)
    mean_std = mean_per_trial.std(axis=0)  # (features,)

    # Build per-feature records
    feature_records = []
    for j, fname in enumerate(feature_names):
        feature_records.append(
            {
                "feature_index": int(j),
                "feature": str(fname),
                "shap_mean_abs": float(abs_mean[j]),
                "shap_mean_abs_std": float(abs_std[j]),
                "shap_mean": float(mean_val[j]),
                "shap_mean_std": float(mean_std[j]),
            }
        )

    # Sort by mean absolute SHAP value
    feature_records.sort(key=lambda x: x["shap_mean_abs"], reverse=True)

    # Get top N features
    top_n = args.top_n_features
    top_features = [rec["feature"] for rec in feature_records[:top_n]]

    out = {
        "metadata": {
            "dataset": args.cohort_name,
            "model": "XLearner",
            "trials_completed": num_trials_completed,
            "total_trials_requested": args.num_trials,
            "baseline_mode": "random_sample" if args.baseline else "median",
            "relative_change_threshold": args.relative_change_threshold,
            "device": DEVICE,
        },
        "summary": {
            "num_features": int(num_features),
            "top_n_features": top_n,
            "top_features_by_mean_abs": feature_records[:min(top_n, num_features)],
        },
        "features": feature_records,  # full list for programmatic use
        "per_trial": {
            "shap_abs_mean_per_trial": _to_py(abs_mean_per_trial),
            "shap_mean_per_trial": _to_py(mean_per_trial),
        },
    }

    json_path = os.path.join(
        save_path, f"{args.cohort_name}_shap_summary_{args.baseline}.json"
    )
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"SHAP computation completed. Results saved to: {save_path}")
    print(f"JSON summary written to: {json_path}")
    print(f"\nTop {top_n} features by mean absolute SHAP value:")
    for i, feat in enumerate(top_features, 1):
        print(f"  {i}. {feat}")


if __name__ == "__main__":

    args = parse_args()
    main(args)
