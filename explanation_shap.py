import os
import pickle
import argparse
import numpy as np
import torch

from src.cate_utils import *
from src.utils import *
from src.dataset import *
from captum.attr import ShapleyValueSampling
from src.interpretability.explain import Explainer
from src.model_utils import TwoLayerMLP

from src.CATENets.catenets.models.torch import pseudo_outcome_nets
from sklift.metrics import (
    uplift_at_k, uplift_auc_score, qini_auc_score, weighted_average_uplift
)

def compute_shap_values(model, data_sample, data_baseline):
    shapley_model = ShapleyValueSampling(model)

    shap_values = shapley_model.attribute(
                        torch.from_numpy(data_sample).to(DEVICE).float(),
                        n_samples=5000,
                        baselines=torch.from_numpy(data_baseline.reshape(1, -1)).to(DEVICE).float(),
                        perturbations_per_eval=25,
                        show_progress=True
                    ).detach().cpu().numpy()
    return shap_values


def main(args):
    cohort_name = args["dataset"]
    trials = args["num_trials"]
    subgroup_col = args["subgroup_column"]
    bshap = args["baseline"]
    device = args["device"]
    num_seeds = 10
    print(args)

    data = Dataset(cohort_name, 0)
    x, y, w = data.get_data()

    if bshap == True:
        baseline = x.mean(0)
        # for _, idx_lst in data.discrete_indices.items():
        #     if len(idx_lst) == 1:

        #         # setting binary vars to 0.5

        #         baseline[idx_lst] = 0.5
        #     else:
        #         # setting categorical baseline to 1/n
        #         # category_counts = data[:, idx_lst].sum(axis=0)
        #         # baseline[idx_lst] = category_counts / category_counts.sum()

        #         baseline[idx_lst] = 1/len(idx_lst)

    # subgroup_index = data.get_feature_names().tolist().index(subgroup_col)

    # ensemble_shap = np.zeros(x.shape)

    # unique_subgroup_values = np.unique(x[:, subgroup_index])
    # subgroup_shap_values = {f"{subgroup_col}={value}": np.zeros((trials, x[x[:, subgroup_index] == value].shape[0], x.shape[1]))
    #                         for value in unique_subgroup_values}

    # ensemble = BasicNet(
    #     "EnsembleNet",
    #     n_unit_in = x.shape[1],
    #     binary_y=False,
    #     n_layers_out=2,
    #     n_units_out=100,
    #     batch_size=128,
    #     n_iter=1000,
    #     nonlin="relu",
    #     device=DEVICE,
    #     # prob_diff=True
    # )
    predict_results = np.zeros((trials, len(x)))

    explainers = [
        "saliency",
        "smooth_grad",
        # "lime",
        # "baseline_lime",
        "baseline_shapley_value_sampling",
        "marginal_shapley_value_sampling",
        "integrated_gradients",
        "baseline_integrated_gradients",
        # "kernel_shap"
        # "gradient_shap",
        # "marginal_shap"
    ]
    qini_score_results = {
        exp: np.zeros(( num_seeds, x.shape[-1], 3)) for exp in explainers
    }
    qini_score_results['rand'] = np.zeros((num_seeds, x.shape[-1],  3))

    learner_explanations = {
        exp: np.zeros((trials, x.shape[0], x.shape[1])) for exp in explainers
    }

    learner_explanations["rand"] = np.zeros((trials, x.shape[0], x.shape[1]))

    for i in range(trials):
        x, w, y = data.get_data()
        model = pseudo_outcome_nets.XLearner(
                x.shape[1],
                binary_y=(len(np.unique(y)) == 2),
                n_layers_out=2,
                n_units_out=100,
                batch_size=128,
                n_iter=1000,
                nonlin="relu",
                device=device,
                seed=i
        )

        model.fit(x, y, w)

        y_hat = model.predict(X=x).detach().cpu().numpy().flatten()
        predict_results[i] = y_hat

        # ensemble.fit(x_sampled, y_hat)

        # if not bshap:
        #     baseline_index = np.random.choice(len(x_sampled) ,1)
        #     baseline = x_sampled[baseline_index]
        learner_explainers = Explainer(
            model,
            feature_names=list(range(x.shape[1])),
            explainer_list=explainers,
            n_samples=5000,
            perturbations_per_eval=25,
            baseline=baseline.reshape(1, -1)
        )
        explanations = learner_explainers.explain(x, w, y)

        for explainer in explanations:
            learner_explanations[explainer][i] = explanations[explainer]

        # for unique_value in unique_subgroup_values:
        #     subgroup_sample = x[x[:, subgroup_index] == unique_value]
        #     subgroup_baseline = subgroup_sample.mean(0)
        #     shap_value = compute_shap_values(model, subgroup_sample, subgroup_baseline)
        #     subgroup_shap_values[f"{subgroup_col}={unique_value}"][i] = shap_value

        # explaining single model without bootstrapped

        # if i % 5 == 0:
        #     model = pseudo_outcome_nets.XLearner(
        #             x.shape[1],
        #             binary_y=(len(np.unique(y)) == 2),
        #             n_layers_out=2,
        #             n_units_out=100,
        #             batch_size=128,
        #             n_iter=1000,
        #             nonlin="relu",
        #             device=DEVICE,
        #             seed=i
        #     )

        #     model.fit(x, y, w)
        #     single_shap[i//5] = compute_shap_values(model, x, x.mean(0))

    train_effects = predict_results.mean(0).reshape(predict_results.shape[1],-1)
    for explainer in learner_explanations:
        # compute average explanation from ensemble.
        for seed in range(num_seeds):

            if explainer == "rand":
                global_rank = np.random.choice(x.shape[1], x.shape[1], replace=False)
            else:
                abs_explanation = np.abs(learner_explanations[explainer]).mean(0)
                global_rank = np.flip(np.argsort(abs_explanation.mean(0)))

            for feature_idx in range(1, x.shape[-1] + 1):
                print(
                    f"Training student model with {feature_idx} features in {explainer}."
                )
                # Starting from 1 features
                # model = pseudo_outcome_nets.XLearner(
                #         feature_idx,
                #         binary_y=(len(np.unique(y)) == 2),
                #         n_layers_out=2,
                #         n_units_out=100,
                #         batch_size=128,
                #         n_iter=1000,
                #         nonlin="relu",
                #         device=device,
                #         seed=seed
                # )
                # ## Subgroup identification with global feature ranking
                # model.fit(x[:, global_rank[:feature_idx]], y, w)
                # pred_train_cate = model.predict(X=x[:, global_rank[:feature_idx]]).detach().cpu().numpy().flatten()

                mlp = TwoLayerMLP(feature_idx, 32, 1)
                mlp.train_model(
                    x[:, global_rank[:feature_idx]], train_effects, epochs=100, batch_size=32
                )
                pred_train_cate = (
                    mlp(torch.from_numpy(x[:, global_rank[:feature_idx]]).float()).detach().cpu().numpy()
                )
                mse = np.mean((pred_train_cate - train_effects) ** 2)

                qini_score = qini_auc_score(
                    y,
                    pred_train_cate,
                    w
                )
                uplift_score = uplift_auc_score(
                    y,
                    pred_train_cate,
                    w
                )
                qini_score_results[explainer][seed][feature_idx-1] = np.asarray([qini_score, uplift_score, mse])

    save_path = os.path.join("results/ensemble", cohort_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, f"predict_results_ensemble_{bshap}_{cohort_name}.pkl"), "wb") as output_file:
        pickle.dump(predict_results, output_file)

    with open(os.path.join(save_path, f"ensemble_exp_{bshap}_{cohort_name}.pkl"), "wb") as output_file:
        pickle.dump(learner_explanations, output_file)

    with open(os.path.join(save_path, f"qini_score_{bshap}_{cohort_name}.pkl"), "wb") as output_file:
        pickle.dump(qini_score_results, output_file)

    # with open(os.path.join(save_path, f"shap_distilled_{bshap}.pkl"), "wb") as output_file:
    #     pickle.dump(ensemble_shap, output_file)

    # with open(os.path.join(save_path, f"single_model_{bshap}.pkl"), "wb") as output_file:
    #     pickle.dump(single_shap, output_file)

    # for subgroup, shap_values in subgroup_shap_values.items():
    #     file_name = f"{subgroup}_shap_ensemble_{bshap}.pkl"
    #     with open(os.path.join(save_path, file_name), "wb") as output_file:
    #         pickle.dump(shap_values, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-d', '--dataset', help='Dataset', required=True, type=str)
    parser.add_argument('-t', '--num_trials', help='number of runs ', required=True, type=int)

    parser.add_argument('-g', '--subgroup_column', help='Dataset', required=False, type=str)
    parser.add_argument('-device', '--device', required=False, type=str, default="cuda:0")

    parser.add_argument('-b', '--baseline', help='whether using baseline', default=False, action='store_true')

    args = vars(parser.parse_args())

    main(args)
