import argparse
import glob
import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np

# insertion_deletion_data layout per entry:
# [learner, method_name, insertion_results, deletion_results,
#  train_qini, train_mse, test_qini, test_mse, ...]
IDX = {
    ("train", "mse"): 5,
    ("test", "mse"): 7,
    ("train", "qini"): 4,
    ("test", "qini"): 6,
}


def load_series(files, method, split, metric):
    """Pick the requested series for a given method from each file."""
    series = []
    key = (split, metric)
    idx = IDX[key]
    for fp in files:
        with open(fp, "rb") as f:
            entries = pkl.load(f)
        picked = None
        for e in entries:
            if e[1] == method:
                picked = np.asarray(e[idx], float)
                break
        # fallback: if file has only one entry, use it
        if picked is None and len(entries) == 1:
            picked = np.asarray(entries[0][idx], float)
        if picked is not None:
            series.append(picked)
    return series


def main():
    ap = argparse.ArgumentParser(
        description="Plot metric vs #features from insertion_deletion results."
    )
    ap.add_argument("--results_root", default="results/", help="e.g., results")
    ap.add_argument("--dataset", required=True, help="e.g., ist3")
    ap.add_argument("--learner", required=True, help="e.g., XLearner")
    ap.add_argument("--shuffle", default="True")
    ap.add_argument("--zero_baseline", default="True")
    ap.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help="e.g., loco marginal_shapley_value_sampling",
    )
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--metric", choices=["mse", "qini"], default="qini")
    ap.add_argument("--seeds", default="*", help="glob for seeds, e.g., 0 or *")
    ap.add_argument("--out", default="plots/", help="Output directory")
    args = ap.parse_args()

    # collect all seed files
    pat = os.path.join(
        args.results_root,
        args.dataset,
        f"insertion_deletion_shuffle_{args.shuffle}_{args.learner}_zero_baseline_{args.zero_baseline}_seed_{args.seeds}.pkl",
    )
    files = sorted(glob.glob(pat))
    if not files:
        raise SystemExit(f"No files found for pattern: {pat}")

    per_method = {}
    for m in args.methods:
        series_list = load_series(files, m, args.split, args.metric)
        if series_list:
            per_method[m] = np.vstack(series_list)

    if not per_method:
        raise SystemExit("No series collected. Check --methods and files.")

    plt.figure(figsize=(7.5, 4.6))
    for mname, Y in sorted(per_method.items()):
        x = np.arange(1, Y.shape[1] + 1)  # 1..K features
        mean, se = Y.mean(0), Y.std(0) / np.sqrt(Y.shape[0])
        plt.plot(x, mean, lw=2, label=mname)
        plt.fill_between(x, mean - se, mean + se, alpha=0.15)

    title_metric = "Distillation loss (MSE)" if args.metric == "mse" else "Qini (AUUC)"
    plt.title(f"{title_metric} vs #features · {args.learner} · {args.split}")
    plt.xlabel("Number of features")
    plt.ylabel(title_metric)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    file_name = f"distillation_qini_{args.dataset}_{args.learner}_shuffle_{args.shuffle}_zero_baseline_{args.zero_baseline}_{args.split}.png"
    plt.savefig(
        os.path.join(
            args.out, file_name
        ), dpi=150
    )

if __name__ == "__main__":
    main()
