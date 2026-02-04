"""Plotting for ensemble distillation"""
import argparse
import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np

# Indices in your ensemble pickle structure
METHOD_IDX = 1
TRAIN_MSE_IDX = 3
TEST_MSE_IDX = 5


DEFAULT_EXPLAINERS = [
    "saliency",
    "smooth_grad",
    # "gradient_shap",
    # "lime",
    "baseline_lime",
    "baseline_shapley_value_sampling",
    # "marginal_shapley_value_sampling",
    "integrated_gradients",
    # "baseline_integrated_gradients",
    # "kernel_shap"
    # "marginal_shap"
]

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf"
]

# explainer-specific style (label, color, linestyle)
STYLE = {
    "baseline_shapley_value_sampling": ("Shapley value",            PALETTE[0], "--"),
    "marginal_shapley_value_sampling": ("Shapley",                  PALETTE[0], "-"),
    "smooth_grad":                    ("SmoothGrad",                 PALETTE[1], "--"),
    "lime":                           ("Lime",                       PALETTE[2], "--"),
    "baseline_lime":                  ("Lime",                       PALETTE[2], "-"),
    "integrated_gradients":           ("IG",                         PALETTE[3], "-"),
    "baseline_integrated_gradients":  ("IG - mean",                  PALETTE[3], "--"),
    "kernel_shap":                    ("Kernel Shap",                PALETTE[4], "--"),
    "loco":                             ("LOCO",                  PALETTE[6], "--"),
    "permucate":                       ("PermuCATE",                  PALETTE[7], "--")
}
FALLBACK = ("", PALETTE[5], "-")

def main():
    """Main function"""

    ap = argparse.ArgumentParser(
        description="Plot ensemble-teacher distillation loss (MSE) vs #features from a single ensemble pickle."
    )
    ap.add_argument(
        "--results_root",
        default="results/",
        help="Base results dir (default: results/)",
    )
    ap.add_argument("--dataset", required=True, help="e.g., ist3")
    ap.add_argument("--learner", required=True, help="e.g., RLearner")
    ap.add_argument("--shuffle", default="True")
    ap.add_argument("--zero_baseline", default="True")
    ap.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help="Explanation methods to plot (e.g., loco marginal_shapley_value_sampling)",
    )
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--out", default="plots/", help="Output directory for the figure")
    args = ap.parse_args()

    # Single ensemble file (no seed suffix)
    ens_path = os.path.join(
        args.results_root,
        args.dataset,
        f"ensemble_shuffle_{args.shuffle}_{args.learner}_zero_baseline_{args.zero_baseline}.pkl",
    )
    # ens_path_loco = os.path.join(
    #     args.results_root,
    #     args.dataset,
    #     f"ensemble_shuffle_{args.shuffle}_{args.learner}_zero_baseline_False.pkl",
    # )
    if not os.path.exists(ens_path):
        raise SystemExit(f"Ensemble file not found:\n  {ens_path}")

    with open(ens_path, "rb") as f:
        entries = pkl.load(f)

    # with open(ens_path_loco, "rb") as f:
    #     entries_loco = pkl.load(f)
    # Build a dict: method_name (without "|ensemble_teacher") -> vector of MSE values
    per_method = {}

    wanted = set(args.methods) | {f"{m}|ensemble_teacher" for m in args.methods}

    # --- Load methods from the main ensemble ---
    for e in entries:
        if not isinstance(e, (list, tuple)) or len(e) < 6:
            continue
        mname = str(e[METHOD_IDX])
        if mname not in wanted:
            continue  # skip loco here, we'll load it from ens_path_loco

        label = mname.replace("|ensemble_teacher", "")
        vec = e[TRAIN_MSE_IDX if args.split == "train" else TEST_MSE_IDX]
        y = np.asarray(vec, dtype=float).ravel()
        per_method[label] = y

    # # --- Load LOCO methods from the separate file ---
    # for e in entries_loco:
    #     if not isinstance(e, (list, tuple)) or len(e) < 6:
    #         continue
    #     mname = str(e[METHOD_IDX])
    #     if mname not in wanted or "loco" not in mname:
    #         continue  # only take loco methods from this file

    #     label = mname.replace("|ensemble_teacher", "")
    #     vec = e[TRAIN_MSE_IDX if args.split == "train" else TEST_MSE_IDX]

    #     y = np.asarray(vec, dtype=float).ravel()
    #     per_method[label] = y

    if not per_method:
        raise SystemExit(
            "No matching methods found in the ensemble file. "
            "Check --methods and file contents."
        )

    os.makedirs(args.out, exist_ok=True)

    # Plot
    plt.figure(figsize=(8, 6))

    for mname in sorted(per_method.keys()):
        print(mname)
        label, color, ls = STYLE.get(mname, (mname or FALLBACK[0], FALLBACK[1], FALLBACK[2]))

        y = per_method[mname]
        x = np.arange(1, len(y) + 1)  # 1..K features
        plt.plot(x, y, lw=2, label=label, color=color)

    # plt.title(
    #     f"Ensemble distillation loss vs #features · {args.learner} · {args.split}"
    # )
    plt.xlabel("Number of features", size=18)
    plt.ylabel("Distillation loss (MSE)", size=18)
    plt.grid(True, alpha=0.3)
    plt.legend(  bbox_to_anchor=(0.8, -0.15))
    plt.tight_layout()

    out_path = os.path.join(
        args.out,
        f"ensemble_distillation_mse_{args.dataset}_{args.learner}"
        f"_shuffle_{args.shuffle}_zero_baseline_{args.zero_baseline}_{args.split}.png",
    )
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
