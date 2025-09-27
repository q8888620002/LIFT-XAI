"""Train CATE with CRASH-2 and validate in TXA cohort."""
import argparse
import os
import pickle

import numpy as np
import torch
from captum.attr import ShapleyValueSampling

from src.cate_utils import qini_score
from src.CATENets.catenets.models.torch import pseudo_outcome_nets
from src.dataset import Dataset, obtain_txa_baselines

DEVICE = "cuda:1"


def compute_shap_values(model, data_sample, data_baseline=None):
    shapley_model = ShapleyValueSampling(model)

    if data_baseline is not None:

        shap_values = (
            shapley_model.attribute(
                torch.tensor(data_sample).to(DEVICE),
                n_samples=5000,
                baselines=torch.tensor(data_baseline.reshape(1, -1)).to(DEVICE),
                perturbations_per_eval=100,
                show_progress=True,
            )
            .detach()
            .cpu()
            .numpy()
        )
    else:
        shap_values = (
            shapley_model.attribute(
                torch.tensor(data_sample).to(DEVICE),
                n_samples=5000,
                perturbations_per_eval=25,
                show_progress=True,
            )
            .detach()
            .cpu()
            .numpy()
        )

    return shap_values.reshape(len(data_sample), -1)


def main(args):
    trials = args["num_trials"]
    bshap = args["baseline"]

    print("Baselines shapley:", bshap)

    crash2_x, crash2_w, crash2_y, txa_x, txa_w, txa_y = obtain_txa_baselines()

    crash2_predict_results = np.zeros((trials, len(crash2_x)))
    crash2_average_shap = np.zeros((trials, len(crash2_x), crash2_x.shape[-1]))

    txa_predict_results = np.zeros((trials, len(txa_x)))
    txa_average_shap = np.zeros((trials, len(txa_x), txa_x.shape[-1]))

    for i in range(trials):
        # Model training
        crash2_x, crash2_w, crash2_y, txa_x, txa_w, txa_y = obtain_txa_baselines()
        # sampled_indices = np.random.choice(
        #     len(crash2_x), size=len(crash2_x), replace=True
        # )

        # x_sampled = crash2_x[sampled_indices]
        # y_sampled = crash2_y[sampled_indices]
        # w_sampled = crash2_w[sampled_indices]

        model = pseudo_outcome_nets.XLearner(
            crash2_x.shape[1],
            binary_y=(len(np.unique(crash2_y)) == 2),
            n_layers_out=2,
            n_units_out=100,
            batch_size=128,
            n_iter=1000,
            nonlin="relu",
            device=DEVICE,
        )

        model.fit(crash2_x, crash2_y, crash2_w)

        crash2_predict_results[i] = (
            model.predict(X=crash2_x).detach().cpu().numpy().flatten()
        )
        txa_predict_results[i] = model.predict(X=txa_x).detach().cpu().numpy().flatten()

        if bshap:
            baseline = crash2_x.mean(0)
            # baseline[5] = 0.5
            # baseline[6] = 0.5
            # baseline[7] = 0.5

        else:
            baseline_index = np.random.choice(len(crash2_x), 1)
            baseline = crash2_x[baseline_index]

        crash2_average_shap[i] = compute_shap_values(model, crash2_x, baseline)

        if bshap:
            txa_baseline = txa_x.mean(0)
            # txa_baseline[5] = 0.5
            # txa_baseline[6] = 0.5
            # txa_baseline[7] = 0.5

        else:
            baseline_index = np.random.choice(len(txa_x), 1)
            txa_baseline = txa_x[baseline_index]

        txa_average_shap[i] = compute_shap_values(model, txa_x, baseline)

    # global_rank_txa = np.flip(np.argsort(np.abs(txa_average_shap).mean(0).mean(0)))
    # global_rank_crash2= np.flip(np.argsort(np.abs(crash2_average_shap).mean(0).mean(0)))

    # txa_qini_score = [[] for _ in range(5)]
    # crash2_qini_score = [[] for _ in range(5)]

    # for index in range(20):
    #     for feature_idx in range(1, global_rank_txa.shape[0] + 1):
    #         txa_score, _, _, _ = qini_score(
    #             global_rank_txa[:feature_idx],
    #             (txa_x, txa_w, txa_y),
    #             (txa_x, txa_w, txa_y),
    #             model,
    #             "XLearner",
    #         )
    #         rand_txa, _, _, _ = qini_score(
    #             np.random.choice(txa_x.shape[1], feature_idx, replace=False),
    #             (txa_x, txa_w, txa_y),
    #             (txa_x, txa_w, txa_y),
    #             model,
    #             "XLearner",
    #         )

    #         txa_qini_score[index].append([txa_score, rand_txa])

    #     for feature_idx in range(1, global_rank_crash2.shape[0] + 1):
    #         crash2_score, _, _, _ = qini_score(
    #             global_rank_crash2[:feature_idx],
    #             (crash2_x, crash2_w, crash2_y),
    #             (crash2_x, crash2_w, crash2_y),
    #             model,
    #             "XLearner",
    #         )
    #         rand_crash2, _, _, _ = qini_score(
    #             np.random.choice(crash2_x.shape[1], feature_idx, replace=False),
    #             (crash2_x, crash2_w, crash2_y),
    #             (crash2_x, crash2_w, crash2_y),
    #             model,
    #             "XLearner",
    #         )
    #         crash2_qini_score[index].append([crash2_score, rand_crash2])

    save_path = os.path.join("results", "txa_crash2")
    os.makedirs(save_path, exist_ok=True)

    # with open(
    #     os.path.join(save_path, f"crash2_qini_scores_{bshap}.pkl"), "wb"
    # ) as output_file:
    #     pickle.dump(crash2_qini_score, output_file)

    # with open(
    #     os.path.join(save_path, f"txa_qini_scores_{bshap}.pkl"), "wb"
    # ) as output_file:
    #     pickle.dump(txa_qini_score, output_file)

    with open(
        os.path.join(save_path, f"crash2_predict_results_{bshap}.pkl"), "wb"
    ) as output_file:
        pickle.dump(crash2_predict_results, output_file)

    with open(
        os.path.join(save_path, f"txa_predict_results_{bshap}.pkl"), "wb"
    ) as output_file:
        pickle.dump(txa_predict_results, output_file)

    with open(
        os.path.join(save_path, f"crash2_shap_bootstrapped_{bshap}.pkl"), "wb"
    ) as output_file:
        pickle.dump(crash2_average_shap, output_file)

    with open(
        os.path.join(save_path, f"txa_shap_bootstrapped_{bshap}.pkl"), "wb"
    ) as output_file:
        pickle.dump(txa_average_shap, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument(
        "-t", "--num_trials", help="number of runs ", required=True, type=int
    )
    parser.add_argument(
        "-b",
        "--baseline",
        help="whether using baseline",
        default=False,
        action="store_true",
    )

    args = vars(parser.parse_args())

    main(args)
