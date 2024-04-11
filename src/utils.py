"""Explainability related utility function."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch import nn

from dataset import Dataset
from src.cate_utils import calculate_pehe
from src.model_utils import NuisanceFunctions


def normalize(values: np.ndarray) -> np.ndarray:
    """Used to normalize the outputs of interpretability methods."""

    min_val = np.min(values, axis=1, keepdims=True)
    max_val = np.max(values, axis=1, keepdims=True)

    return 2 * (values - min_val) / np.maximum(max_val - min_val, 1e-10) - 1


def attribution_ranking(feature_attributions: np.ndarray) -> list:
    """ "
    Compute the ranking of features according to atribution score

    Args:
        feature_attributions: an n x d array of feature attribution scores
    Return:
        a d x n list of indices starting from the highest attribution score
    """

    rank_indices = np.argsort(feature_attributions, axis=1)[:, ::-1]
    rank_indices = list(map(list, zip(*rank_indices)))

    return rank_indices


def ablate(
    test_data: tuple,
    attr: np.ndarray,
    model: nn.module,
    refer: np.ndarray = None,
    impute: str = "pos",
    nuisancefunction: NuisanceFunctions = None,
    y_explic=None,
    is_loss=False,
):
    """
    Arguments:
    pred_fn  - prediction function for model being explained
    attr     - attribution corresponding to model and explicand
    x_explic - sample being explained
    impute   - the order in which we impute features
               "pos" imputes positive attributions by refer
               "neg" imputes negative attributions by refer
    refer    - reference to impute with (same shape as x_explic)
    y_explic - labels - only necessary for loss explanation
    is_loss  - if true, the prediction function should accept
               parameters x_explic and y_explic

    Returns:
    ablated_preds - predictions based on ablating
    """

    x_explic, _, _ = test_data

    # Set reference if None
    if refer is None:
        refer = np.zeros(x_explic.shape)

    if len(refer) != len(x_explic):
        refer = np.tile(refer, (len(x_explic), 1))

    # Get feature rank
    if "pos" in impute:
        feat_rank = np.argsort(-attr)
        condition = lambda x: x > 0

    if "neg" in impute:
        feat_rank = np.argsort(attr)
        condition = lambda x: x < 0

    # Explicand to modify
    explicand = np.copy(x_explic)

    # # Initialize ablated predictions
    ablated_preds = [model.predict(explicand).detach().cpu().numpy().mean()]

    # Ablate features one by one
    for i in range(explicand.shape[1]):
        samples = np.arange(x_explic.shape[0])  # Sample indices
        top_feat = feat_rank[:, i]  # Get top i features
        expl_val = explicand[samples, top_feat]  # Get values of top features
        mask_val = expl_val  # Initialize mask to explicand
        cond_true = condition(
            attr[samples, top_feat]
        )  # Ensure attributions satisfy positive/negative
        mask_val[cond_true] = refer[samples, top_feat][
            cond_true
        ]  # Update mask based on condition
        explicand[samples, top_feat] = mask_val  # Mask top features

        avg_pred = model.predict(explicand).detach().cpu().numpy().mean()
        ablated_preds.append(avg_pred)  # Get prediction on ablated explicand

    # Return ablated predictions
    return ablated_preds


def insertion_deletion(
    test_data: tuple,
    rank_indices: list,
    cate_model: torch.nn.Module,
    baseline: np.ndarray,
    selection_types: List[str],
    nuisance_functions: NuisanceFunctions,
    model_type: str = "CATENets",
) -> tuple:
    """
    Compute partial average treatment effect (PATE) with with proxy pehe and feature subsets by insertion and deletion

    Args:
        test_data: testing data to calculate pehe
        rank_indices: local ranking from a given explanation method
        cate_model: trained cate model
        baseline: replacement values for insertion & deletion
        selection_types: approximation methods for estimating ground truth pehe
        nuisance function: Nuisance functions to estimate proxy pehe.
        model_type: model types for cate package
    Returns:
        results of insertion and deletion of PATE.
    """
    ## training plugin estimator on

    x_test, _, _ = test_data

    n, d = x_test.shape
    x_test_del = x_test.copy()
    x_test_ins = np.tile(baseline, (n, 1))
    baseline = np.tile(baseline, (n, 1))

    original_cate_model = cate_model

    if model_type == "CATENets":
        cate_model = lambda x: original_cate_model.predict(X=x)
    else:
        cate_model = lambda x: original_cate_model.forward(X=x)

    deletion_results = {
        selection_type: np.zeros(d + 1) for selection_type in selection_types
    }
    insertion_results = {
        selection_type: np.zeros(d + 1) for selection_type in selection_types
    }

    for rank_index in range(len(rank_indices) + 1):
        # Skip this on the first iteration

        if rank_index > 0:
            col_indices = rank_indices[rank_index - 1]

            for i in range(n):
                x_test_ins[i, col_indices[i]] = x_test[i, col_indices[i]]
                x_test_del[i, col_indices[i]] = baseline[i, col_indices[i]]

        for selection_type in selection_types:
            # For the insertion process
            cate_pred_subset_ins = (
                cate_model(x_test_ins).detach().cpu().numpy().flatten()
            )

            insertion_results[selection_type][rank_index] = calculate_pehe(
                cate_pred_subset_ins, test_data, selection_type, nuisance_functions
            )

            # For the deletion process
            cate_pred_subset_del = (
                cate_model(x_test_del).detach().cpu().numpy().flatten()
            )
            deletion_results[selection_type][rank_index] = calculate_pehe(
                cate_pred_subset_del, test_data, selection_type, nuisance_functions
            )

    return insertion_results, deletion_results


def generate_perturbations(
    data: Dataset, examples: np.ndarray, feature: int, n_steps: int
) -> np.ndarray:
    """
    Generating perturbed sample

    Args:

        data: background dataset of perturbed samples
        example: sample to be perturbed
        feature: feature index

    Return:

        Perturbed samples
    """
    percent_perturb = 0.1
    value_range = data.get_feature_range(feature)

    perturbations = np.linspace(
        start=-percent_perturb * value_range,
        stop=percent_perturb * value_range,
        num=n_steps,
    )

    all_perturbed_examples = []

    # Create perturbed samples

    for example in examples:

        example = example.reshape(1, -1)
        perturbed_example = np.tile(data.get_unnorm_value(example), (n_steps, 1))
        perturbed_example[:, feature] += perturbations

        # Normalize continuous date and concatenate categorical data

        perturbed_example = data.get_norm(perturbed_example)
        perturbed_example = np.concatenate(
            [
                perturbed_example,
                np.tile(example[:, perturbed_example.shape[1] :], [n_steps, 1]),
            ],
            axis=1,
        )

        all_perturbed_examples.append(perturbed_example)

    return np.vstack(all_perturbed_examples)


def generate_perturbed_var(
    data: Dataset,
    x_test: np.ndarray,
    feature_size: int,
    categorical_indices: List,
    n_steps: int,
    model: nn.module,
) -> np.ndarray:

    perturbated_var = []

    for i in range(feature_size - len(categorical_indices)):
        perturbated_samples = generate_perturbations(data, x_test, i, n_steps)

        perturbed_output = model.predict(X=perturbated_samples).detach().cpu().numpy()
        perturbed_output = perturbed_output.reshape(len(x_test), -1)
        perturbated_var.append(np.var(perturbed_output, axis=1))

    return perturbated_var


from typing import List, Tuple


def perturbation(
    perturbed_output: np.ndarray,
    norm_explanation: np.ndarray,
    threshold_vals: np.ndarray,
    n_steps: int,
    spurious_quantile: np.ndarray,
) -> Tuple[List[float], List[float]]:
    """
    Function that conducts perturbation experiments for both "resource" and "spurious"
    and calculates true positive, true negative, false positive and false negative for each.

    Args:
        perturbed_output: Perturbed model outputs.
        norm_explanation: Normalized explanations.
        threshold_vals: Threshold values for explanations.
        n_steps: Number of steps in perturbation.
        spurious_quantile: Quantile threshold for spurious explanations.

    Returns:
        Two tuples containing counts of TP, TN, FP, and FN for "resource" and "spurious" respectively.
    """

    sample_size = len(norm_explanation)
    threshold_size = len(threshold_vals)

    perturbed_results_resource = np.zeros((threshold_size, 4))
    perturbed_results_spurious = np.zeros((threshold_size, 4))

    oracle_values_resource = np.zeros(sample_size)
    oracle_values_spurious = np.zeros(sample_size)
    explained_values = np.zeros((threshold_size, sample_size))

    for k in range(0, len(perturbed_output), n_steps):
        # Calculating oracle values for "resource"
        pred_first = np.mean(perturbed_output[k : k + int(n_steps / 2)])
        pred_snd = np.mean(perturbed_output[k + int(n_steps / 2) : k + n_steps])
        oracle_values_resource[int(k / n_steps)] = 1.0 * (pred_snd > pred_first)

        # Calculating oracle variance for "spurious"
        pred_var = np.var(perturbed_output[k : k + n_steps])
        oracle_values_spurious[int(k / n_steps)] = 1.0 * (pred_var > spurious_quantile)

    for threshold_idx, threshold in enumerate(threshold_vals):
        explained_values[threshold_idx, :] = 1.0 * (norm_explanation > threshold)

        perturbed_results_resource[threshold_idx, 0] = np.sum(
            (explained_values[threshold_idx] == 1) & (oracle_values_resource == 1)
        )
        perturbed_results_resource[threshold_idx, 1] = np.sum(
            (explained_values[threshold_idx] == 0) & (oracle_values_resource == 0)
        )
        perturbed_results_resource[threshold_idx, 2] = np.sum(
            (explained_values[threshold_idx] == 1) & (oracle_values_resource == 0)
        )
        perturbed_results_resource[threshold_idx, 3] = np.sum(
            (explained_values[threshold_idx] == 0) & (oracle_values_resource == 1)
        )

        explained_values[threshold_idx, :] = 1.0 * (
            np.abs(norm_explanation) > threshold
        )

        perturbed_results_spurious[threshold_idx, 0] = np.sum(
            (explained_values[threshold_idx] == 1) & (oracle_values_spurious == 1)
        )
        perturbed_results_spurious[threshold_idx, 1] = np.sum(
            (explained_values[threshold_idx] == 0) & (oracle_values_spurious == 0)
        )
        perturbed_results_spurious[threshold_idx, 2] = np.sum(
            (explained_values[threshold_idx] == 1) & (oracle_values_spurious == 0)
        )
        perturbed_results_spurious[threshold_idx, 3] = np.sum(
            (explained_values[threshold_idx] == 0) & (oracle_values_spurious == 1)
        )

    return perturbed_results_resource, perturbed_results_spurious
