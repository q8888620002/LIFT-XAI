"""Plotting function for model selection"""
import argparse
import os
import pickle

import numpy as np


def main():
    """Main function"""

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, type=str)
    parser.add_argument("-s", "--shuffle", action="store_true")
    parser.add_argument(
        "--results_root", default="results", help="Directory where results are saved"
    )
    args = parser.parse_args()

    # Path to results
    res_path = os.path.join(
        args.results_root,
        args.dataset,
        "model_selection",
        f"model_selection_shuffle_{args.shuffle}.pkl",
    )

    if not os.path.exists(res_path):
        raise FileNotFoundError(f"Results file not found at {res_path}")

    with open(res_path, "rb") as f:
        results = pickle.load(f)

    metrics = [
        "qini_score",
        "uplift_score",
        "if_pehe",
        "pseudo_outcome_r",
        "pseudo_outcome_dr",
    ]

    for learner, res in results.items():
        print(f"\n=== {learner} ===")
        for metric in metrics:
            vals = res[metric]
            mean = np.mean(vals)
            std = np.std(vals)
            print(f"{metric:20s}: {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
