import argparse
import glob
import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np

# indices inside each entry saved to insertion_deletion_data
# [learner, method_name, insertion_results, deletion_results,
#  train_score_results, train_mse_results, test_score_results, test_mse_results, ...]
IDX_TRAIN_MSE = 5
IDX_TEST_MSE = 7


def load_series_from_legacy(files, method, split):
    """Files WITHOUT method in filename; pick method from entries."""
    series = []
    for fp in files:
        with open(fp, "rb") as f:
            entries = pkl.load(f)
        picked = None
        for e in entries:
            if e[1] == method:
                picked = np.asarray(
                    e[IDX_TRAIN_MSE if split == "train" else IDX_TEST_MSE], float
                )
                break
        if picked is None:
            # fallback: if file has only one entry, use it
            if len(entries) == 1:
                e = entries[0]
                picked = np.asarray(
                    e[IDX_TRAIN_MSE if split == "train" else IDX_TEST_MSE], float
                )
            else:
                continue
        series.append(picked)
    return series


def load_series_loco(files, split):
    """Files WITH 'loco' in filename; just take the first (or only) entry."""
    series = []
    for fp in files:
        with open(fp, "rb") as f:
            entries = pkl.load(f)
        e = entries[0] if len(entries) else None
        if e is None:
            continue
        picked = np.asarray(
            e[IDX_TRAIN_MSE if split == "train" else IDX_TEST_MSE], float
        )
        series.append(picked)
    return series


def main():
    ap = argparse.ArgumentParser(
        description="Plot distillation loss (MSE) vs #features (mix legacy + LOCO files)."
    )
    ap.add_argument("--results_root", default="results/", help="e.g., results")
    ap.add_argument("--dataset", required=True, help="e.g., ist3")
    ap.add_argument("--learner", required=True, help="e.g., CausalForest")
    ap.add_argument("--shuffle", default="True")
    ap.add_argument("--zero_baseline", default="True")
    ap.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help="Methods to plot; include 'loco' to read LOCO files by filename",
    )
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--out", default="plots/", help="Output directory")
    args = ap.parse_args()

    per_method = {}

    # legacy pattern (no method in filename)
    legacy_pat = os.path.join(
        args.results_root,
        args.dataset,
        f"insertion_deletion_shuffle_{args.shuffle}_{args.learner}_zero_baseline_{args.zero_baseline}_seed_*.pkl",
    )
    legacy_files = sorted(glob.glob(legacy_pat))

    for m in args.methods:
        series_list = load_series_from_legacy(legacy_files, m, args.split)

        if series_list:
            per_method[m] = np.vstack(series_list)

    if not per_method:
        raise SystemExit("No series collected. Check filenames and --methods.")

    plt.figure(figsize=(7.5, 4.6))
    for mname, Y in sorted(per_method.items()):
        x = np.arange(1, Y.shape[1] + 1)  # 1..K features

        mean, std = Y.mean(0), Y.std(0)/(3*len(Y)**0.5)
        plt.plot(x, mean, lw=2, label=mname)
        plt.fill_between(x, mean - std, mean + std, alpha=0.15)


    plt.title(f"Distillation loss vs #features · {args.learner} · {args.split}")
    plt.xlabel("Number of features")
    plt.ylabel("Distillation loss (MSE)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    file_name = f"distillation_mse_{args.dataset}_{args.learner}_shuffle_{args.shuffle}_zero_baseline_{args.zero_baseline}_{args.split}.png"
    plt.savefig(
        os.path.join(
            args.out, file_name
        ), dpi=150
    )


if __name__ == "__main__":
    main()
