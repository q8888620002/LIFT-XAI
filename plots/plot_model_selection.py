"""Plotting for model selection (with OOF + test metrics)."""

import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def _safe(results, learner, key, trials_len=None):
    """Return array for key or zeros if missing (length trials_len if provided)."""
    arr = results[learner].get(key, None)
    if arr is None:
        if trials_len is None:
            # try to infer length from any existing metric
            some_key = next(iter(results[learner]))
            trials_len = len(results[learner][some_key])
        return np.zeros(trials_len, dtype=float)
    return np.asarray(arr, dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, type=str)
    parser.add_argument("-s", "--shuffle", action="store_true")
    parser.add_argument("--results_root", default="results", help="Directory with results")
    parser.add_argument("--outdir", default=None, help="Where to save plots (defaults to results/.../model_selection)")
    parser.add_argument("--topk", type=int, default=0, help="If >0, only show top-k learners by median test_qini")
    args = parser.parse_args()

    res_path = os.path.join(
        args.results_root, args.dataset, "model_selection",
        f"model_selection_shuffle_{args.shuffle}.pkl"
    )
    if not os.path.exists(res_path):
        raise FileNotFoundError(f"Results file not found at {res_path}")

    with open(res_path, "rb") as f:
        results = pickle.load(f)

    # figure out number of trials from any metric
    learners = list(results.keys())
    any_arr = results[learners[0]][next(iter(results[learners[0]]))]
    trials = len(any_arr)

    # ---- Summarize like original ----
    base_metrics = ["qini_score", "uplift_score", "if_pehe", "pseudo_outcome_r", "pseudo_outcome_dr"]
    print("\n=== TRAIN (OOF) METRICS ===")
    for learner in learners:
        print(f"\n--- {learner} ---")
        for m in base_metrics:
            vals = _safe(results, learner, m, trials)
            print(f"{m:20s}: {np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}")

    print("\n=== TEST METRICS ===")
    test_metrics = ["test_qini", "test_uplift", "if_pehe_test", "pseudo_outcome_r_test", "pseudo_outcome_dr_test"]
    for learner in learners:
        print(f"\n--- {learner} ---")
        for m in test_metrics:
            vals = _safe(results, learner, m, trials)
            print(f"{m:20s}: {np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}")

    # ---- Build tidy arrays for plotting ----
    train_qini = {lrn: _safe(results, lrn, "qini_score", trials) for lrn in learners}
    test_qini  = {lrn: _safe(results, lrn, "test_qini", trials)  for lrn in learners}
    test_uplift = {lrn: _safe(results, lrn, "test_uplift", trials) for lrn in learners}

    pehe_test_keys = ["if_pehe_test", "pseudo_outcome_r_test", "pseudo_outcome_dr_test"]
    pehe_test = {k: {lrn: _safe(results, lrn, k, trials) for lrn in learners} for k in pehe_test_keys}

    # optional top-k filtering by median test_qini
    order_by_median = sorted(learners, key=lambda l: np.nanmedian(test_qini[l]), reverse=True)
    if args.topk and args.topk > 0:
        order = order_by_median[:args.topk]
    else:
        order = order_by_median

    outdir = args.outdir or os.path.dirname(res_path)
    os.makedirs(outdir, exist_ok=True)

    # ---- Plot 1: Boxplot of TEST Qini by learner ----
    plt.figure()
    data = [test_qini[l] for l in order]
    plt.boxplot(data, labels=order)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Qini (test)")
    plt.title(f"Test Qini by learner — {args.dataset} (shuffle={args.shuffle})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "boxplot_test_qini.png"), dpi=160)
    plt.close()

    # ---- Plot 2: Train (OOF) vs Test Qini scatter (generalization) ----
    plt.figure()
    all_vals = np.concatenate([np.concatenate([train_qini[l], test_qini[l]]) for l in order])
    mn, mx = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    pad = 0.05 * (mx - mn + 1e-12)
    lims = (mn - pad, mx + pad)
    for l in order:
        plt.scatter(train_qini[l], test_qini[l], alpha=0.6, label=l)
    plt.plot(lims, lims)
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("Qini (train OOF)")
    plt.ylabel("Qini (test)")
    plt.title(f"Generalization: OOF vs Test Qini — {args.dataset} (shuffle={args.shuffle})")
    plt.legend(bbox_to_anchor=(1.02,1), borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "scatter_oof_vs_test_qini.png"), dpi=160)
    plt.close()

    # ---- Plot 3: Boxplot of TEST Uplift AUC (optional but handy) ----
    plt.figure()
    data_u = [test_uplift[l] for l in order]
    plt.boxplot(data_u, labels=order)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Uplift AUC (test)")
    plt.title(f"Test Uplift AUC by learner — {args.dataset} (shuffle={args.shuffle})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "boxplot_test_uplift_auc.png"), dpi=160)
    plt.close()

    # ---- Plot 4: PEHE-test proxies (one plot per proxy) ----
    for key in pehe_test_keys:
        plt.figure()
        data_pehe = [pehe_test[key][l] for l in order]
        plt.boxplot(data_pehe, labels=order)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(key)
        plt.title(f"{key} by learner (test) — {args.dataset} (shuffle={args.shuffle})")
        plt.tight_layout()
        fname = f"boxplot_{key}.png"
        plt.savefig(os.path.join(outdir, fname), dpi=160)
        plt.close()

    print(f"\nSaved plots to: {outdir}")


if __name__ == "__main__":
    main()
