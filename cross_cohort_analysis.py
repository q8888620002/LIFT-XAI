"""Script that perform cross cohort analysis"""
import argparse
import json
import os
import pickle

import numpy as np
import torch
from captum.attr import ShapleyValueSampling

import src.CATENets.catenets.models.torch.pseudo_outcome_nets as pseudo_outcome_nets
from src.dataset import Dataset, obtain_accord_baselines, obtain_txa_baselines
from src.utils import *

DEVICE = "cuda:1"


def _to_py(x):
    """Convert numpy types to plain Python for JSON serialization."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    return x


def get_feature_name_mappings():
    """Return human-readable names for encoded categorical features."""
    return {
        # IST3 stroke types
        "stroketype_1.0": "Stroke Type: TACI (Total Anterior Circulation Infarct)",
        "stroketype_2.0": "Stroke Type: PACI (Partial Anterior Circulation Infarct)",
        "stroketype_3.0": "Stroke Type: LACI (Lacunar Infarct)",
        "stroketype_4.0": "Stroke Type: POCI (Posterior Circulation Infarct)",
        "stroketype_5.0": "Stroke Type: Other",

        # IST3 infarct visible on CT
        "infarct_0": "Infarct Visible on CT: No",
        "infarct_1.0": "Infarct Visible on CT: Possibly Yes",
        "infarct_2.0": "Infarct Visible on CT: Definitely Yes",

        # CRASH-2 injury type
        "iinjurytype_1": "Injury Type: Blunt",
        "iinjurytype_2": "Injury Type: Penetrating",
    }


def translate_feature_name(feature_name):
    """Translate encoded feature name to human-readable version if mapping exists."""
    mappings = get_feature_name_mappings()
    return mappings.get(feature_name, feature_name)


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


def export_json_summary(cohort_name, cohort_x, cohort_shap, cohort_pred, feature_meta, baseline_mode, trials, save_path):
    """Export JSON summary compatible with clinical agent."""
    # cohort_shap and cohort_pred are already numpy arrays (trials, samples, features) and (trials, samples)
    shap_values_array = cohort_shap if isinstance(cohort_shap, np.ndarray) else np.stack(cohort_shap)
    pred_array = cohort_pred if isinstance(cohort_pred, np.ndarray) else np.stack(cohort_pred)
    feature_names = feature_meta["feature_names"]
    categorical_indices_map = feature_meta["categorical_indices"]
    num_trials = trials
    num_features = shap_values_array.shape[2]

    # CATE prediction statistics
    pred_pooled = pred_array.flatten()
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    pred_overall = {
        "mean": float(np.mean(pred_pooled)),
        "std": float(np.std(pred_pooled)),
        "quantiles": {str(q): float(np.quantile(pred_pooled, q)) for q in qs},
        "positive_rate": float((pred_pooled > 0).mean()),
        "negative_rate": float((pred_pooled < 0).mean()),
    }

    # Aggregate SHAP statistics
    abs_mean_per_trial = np.abs(shap_values_array).mean(axis=1)  # (trials, features)
    abs_mean = abs_mean_per_trial.mean(axis=0)
    abs_std = abs_mean_per_trial.std(axis=0)
    mean_per_trial = shap_values_array.mean(axis=1)
    mean_val = mean_per_trial.mean(axis=0)
    mean_std = mean_per_trial.std(axis=0)

    # Build per-feature records
    feature_records = []
    for j, fname in enumerate(feature_names):
        feature_records.append({
            "feature_index": int(j),
            "feature": translate_feature_name(str(fname)),
            "feature_original": str(fname),
            "shap_mean_abs": float(abs_mean[j]),
            "shap_mean_abs_std": float(abs_std[j]),
            "shap_mean": float(mean_val[j]),
            "shap_mean_std": float(mean_std[j]),
        })

    feature_records.sort(key=lambda x: x["shap_mean_abs"], reverse=True)

    # Aggregate categorical features
    categorical_aggregates = []
    categorical_feature_indices = set()

    for cat_name, cat_indices in categorical_indices_map.items():
        cat_shap_abs = abs_mean_per_trial[:, cat_indices].sum(axis=1)
        cat_shap_signed = mean_per_trial[:, cat_indices].sum(axis=1)

        categorical_aggregates.append({
            "feature": cat_name,
            "feature_original": cat_name,
            "is_categorical": True,
            "num_categories": len(cat_indices),
            "category_names": [str(feature_names[idx]) for idx in cat_indices],
            "category_indices": cat_indices,
            "shap_mean_abs": float(cat_shap_abs.mean()),
            "shap_mean_abs_std": float(cat_shap_abs.std()),
            "shap_mean": float(cat_shap_signed.mean()),
            "shap_mean_std": float(cat_shap_signed.std()),
        })
        categorical_feature_indices.update(cat_indices)

    # Filter non-categorical features
    non_categorical_features = [
        rec for rec in feature_records
        if rec["feature_index"] not in categorical_feature_indices
    ]

    # Combine and sort
    combined_features = non_categorical_features + categorical_aggregates
    combined_features.sort(key=lambda x: x["shap_mean_abs"], reverse=True)

    out = {
        "metadata": {
            "dataset": cohort_name,
            "learner": "DRLearner",
            "explainer": "ShapleyValueSampling",
            "trials_completed": num_trials,
            "baseline_mode": baseline_mode,
            "analysis_type": "cross_cohort",
        },
        "summary": {
            "num_features": int(num_features),
            "num_categorical_features": len(categorical_aggregates),
            "num_non_categorical_features": len(non_categorical_features),
            "cate_prediction_overall": pred_overall,
        },
        "features": combined_features,
        "features_all_original": feature_records,
    }

    json_path = os.path.join(save_path, f"{cohort_name}_shap_summary_{baseline_mode}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"JSON summary written to: {json_path}")
    return json_path


def main(args):
    """Main function for computing shapley value"""

    trials = args["num_trials"]
    bshap = args["baseline"]
    cohort_name = args["cohort_name"]

    print("Baselines shapley:", bshap)
    baseline_mode = "baseline_swapped" if bshap else "random_sample"

    if cohort_name == "accord_sprint":

        cohort1 = "sprint"
        cohort2 = "accord"

        dataset1 = Dataset("sprint_filter", 0)
        dataset2 = Dataset("accord_filter", 0)
        feature_meta1 = {
            "feature_names": list(dataset1.get_feature_names()),
            "categorical_indices": dataset1.categorical_indices,
            "discrete_indices": dataset1.discrete_indices,
        }
        feature_meta2 = {
            "feature_names": list(dataset2.get_feature_names()),
            "categorical_indices": dataset2.categorical_indices,
            "discrete_indices": dataset2.discrete_indices,
        }

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

        # obtain_txa_baselines() returns harmonized 9-feature arrays for both crash2 and txa:
        # [iage, isbp, irr, ihr, igcs, ninjurytime, iinjurytype_1, iinjurytype_2, isex]
        shared_feature_names = [
            "iage",
            "isbp",
            "irr",
            "ihr",
            "igcs",
            "ninjurytime",
            "iinjurytype_1",
            "iinjurytype_2",
            "isex",
        ]
        shared_categorical_indices = {"iinjurytype": [6, 7]}
        shared_discrete_indices = {"iinjurytype": [6, 7], "isex": [8]}

        feature_meta1 = {
            "feature_names": shared_feature_names,
            "categorical_indices": shared_categorical_indices,
            "discrete_indices": shared_discrete_indices,
        }
        feature_meta2 = {
            "feature_names": shared_feature_names,
            "categorical_indices": shared_categorical_indices,
            "discrete_indices": shared_discrete_indices,
        }

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

        model = pseudo_outcome_nets.DRLearner(
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

        # Cross-cohort baselines: each cohort uses the other cohort's baseline.
        if bshap:
            baseline = np.median(cohort2_x, axis=0)

            for _, idx_lst in feature_meta2["discrete_indices"].items():
                if len(idx_lst) == 1:
                    # setting binary vars to population proportion
                    baseline[idx_lst] = cohort2_x[:, idx_lst].mean()
                else:
                    # setting categorical baseline to population proportions
                    baseline[idx_lst] = cohort2_x[:, idx_lst].mean(axis=0)
        else:
            baseline_index = np.random.choice(len(cohort1_x), 1)
            baseline = cohort1_x[baseline_index]

        cohort1_average_shap[i] = compute_shap_values(model, cohort1_x, baseline)

        if bshap:
            baseline = np.median(cohort1_x, axis=0)

            for _, idx_lst in feature_meta1["discrete_indices"].items():
                if len(idx_lst) == 1:
                    # setting binary vars to population proportion
                    baseline[idx_lst] = cohort1_x[:, idx_lst].mean()
                else:
                    # setting categorical baseline to population proportions
                    baseline[idx_lst] = cohort1_x[:, idx_lst].mean(axis=0)
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

    # Export JSON summaries for clinical agent compatibility
    print("\nExporting JSON summaries for clinical agent...")

    json1 = export_json_summary(
        cohort1, cohort1_x, cohort1_average_shap, cohort1_predict_results,
        feature_meta1, baseline_mode, trials, save_path
    )

    json2 = export_json_summary(
        cohort2, cohort2_x, cohort2_average_shap, cohort2_predict_results,
        feature_meta2, baseline_mode, trials, save_path
    )

    print(f"\nCross-cohort analysis complete for {cohort1} ↔ {cohort2}")
    print(f"  {cohort1} uses {cohort2} baseline (swapped)")
    print(f"  {cohort2} uses {cohort1} baseline (swapped)")
    print(f"\nJSON summaries:")
    print(f"  - {json1}")
    print(f"  - {json2}")


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
