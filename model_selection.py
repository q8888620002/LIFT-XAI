
import argparse
import os
import sys
import pickle
import numpy as np

from utilities import *

module_path = os.path.abspath(os.path.join('CATENets/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from catenets.models.torch import pseudo_outcome_nets
import catenets.models as cate_models
import catenets.models.torch.tlearner as tlearner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-d','--dataset', help='Dataset', required=True, type =str)
    parser.add_argument('-t','--num_trials', help='Dataset', required=True, type =int)
    parser.add_argument('-s','--shuffle', help='shuffle',  default=True, action='store_false')

    args = vars(parser.parse_args())

    cohort_name = args["dataset"]
    trials = args["num_trials"]
    shuffle = args["shuffle"]
    print(shuffle)
    DEVICE = "cuda:1"
    ensemble_num = 20
    data = Dataset(cohort_name)
    x_test, _, _ = data.get_data("test")

    learners = [
        "XLearner",
        "XLearner_ensemble",
        "DRLearner",
        "SLearner",
        "TLearner",
        "RLearner",
        "RALearner",
        # "TARNet",
        # "DragonNet",
        # "CFRNet_0.01",
        # "CFRNet_0.001"
    ]

    selection_types = [
        "if_pehe",
        "pseudo_outcome_r",
        "pseudo_outcome_dr"
    ]

    results = {
         learner: {
             **{sec: np.zeros((trials)) for sec in selection_types},
             "prediction": np.zeros((trials, x_test.shape[0]))
            } for learner in learners
    }

    for i in range(trials):

        np.random.seed(i)
        data = Dataset(cohort_name, i, shuffle)

        x_train, w_train, y_train = data.get_data("train")
        x_val, w_val, y_val = data.get_data("val")
        x_test, w_test, y_test = data.get_data("test")

        learners = {
            "XLearner":pseudo_outcome_nets.XLearner(
                        x_train.shape[1],
                        binary_y=(len(np.unique(y_train)) == 2),
                        n_layers_out=2,
                        n_units_out=100,
                        batch_size=128,
                        n_iter=1000,
                        nonlin="relu",
                        device=DEVICE,
                        seed=i
                    ),
            "XLearner_ensemble": pseudo_outcome_nets.XLearner(
                        x_train.shape[1],
                        binary_y=(len(np.unique(y_train)) == 2),
                        n_layers_out=2,
                        n_units_out=100,
                        batch_size=128,
                        n_iter=1000,
                        nonlin="relu",
                        device=DEVICE,
                        seed=i
                    ),
            "DRLearner": pseudo_outcome_nets.DRLearner(
                        x_train.shape[1],
                        binary_y=(len(np.unique(y_train)) == 2),
                        n_layers_out=2,
                        n_units_out=100,
                        batch_size=128,
                        n_iter=1000,
                        nonlin="relu",
                        device=DEVICE,
                        seed=i
                ),
            "SLearner": cate_models.torch.SLearner(
                        x_train.shape[1],
                        binary_y=(len(np.unique(y_train)) == 2),
                        n_layers_out=2,
                        n_units_out=100,
                        batch_size=128,
                        n_iter=1000,
                        nonlin="relu",
                        device=DEVICE,
                        seed=i
                ),
            "TLearner": cate_models.torch.TLearner(
                        x_train.shape[1],
                        binary_y=(len(np.unique(y_train)) == 2),
                        n_layers_out=2,
                        n_units_out=100,
                        batch_size=128,
                        n_iter=1000,
                        nonlin="relu",
                        device=DEVICE,
                        seed=i
                ),
            "RALearner": pseudo_outcome_nets.RALearner(
                      x_train.shape[1],
                      binary_y=(len(np.unique(y_train)) == 2),
                      n_layers_out=2,
                      n_units_out=100,
                      n_iter=1000,
                      lr=1e-3,
                      patience=10,
                      batch_size=128,
                      batch_norm=False,
                      nonlin="relu",
                      device = DEVICE,
                    seed=i
                  ),
            "RLearner": pseudo_outcome_nets.RLearner(
                      x_train.shape[1],
                      binary_y=(len(np.unique(y_train)) == 2),
                      n_layers_out=2,
                      n_units_out=100,
                      n_iter=1000,
                      lr=1e-3,
                      patience=10,
                      batch_size=128,
                      batch_norm=False,
                      nonlin="relu",
                      device = DEVICE,
                      seed=i
            ),
            # "DragonNet": cate_models.torch.DragonNet(
            #         x_train.shape[1],
            #         binary_y=(len(np.unique(y_train)) == 2),
            #         batch_size=128,
            #         n_iter=1000,
            #         lr=1e-5,
            #         batch_norm=False,
            #         nonlin="relu",
            #         seed=i
            #     ),
            # "TARNet": cate_models.torch.TARNet(
            #         x_train.shape[1],
            #         binary_y=(len(np.unique(y_train)) == 2),
            #         n_layers_r=2,
            #         n_layers_out=2,
            #         n_units_out=100,
            #         n_units_r=100,
            #         batch_size=128,
            #         n_iter=1000,
            #         lr=1e-5,
            #         batch_norm=False,
            #         early_stopping = True,
            #         nonlin="relu",
            #         seed=i
            #     ),
            # "CFRNet_0.01": cate_models.torch.TARNet(
            #     x_train.shape[1],
            #     binary_y=(len(np.unique(y_train)) == 2),
            #     n_layers_r=2,
            #     n_layers_out=2,
            #     n_units_out=100,
            #     n_units_r=100,
            #     batch_size=128,
            #     n_iter=1000,
            #     lr=1e-5,
            #     batch_norm=False,
            #     nonlin="relu",
            #     penalty_disc=0.01,
            #     seed=i
            # ),
            # "CFRNet_0.001": cate_models.torch.TARNet(
            #         x_train.shape[1],
            #         binary_y=(len(np.unique(y_train)) == 2),
            #         n_layers_r=2,
            #         n_layers_out=2,
            #         n_units_out=100,
            #         n_units_r=100,
            #         lr=1e-5,
            #         batch_size=128,
            #         n_iter=1000,
            #         batch_norm=False,
            #         nonlin="relu",
            #         penalty_disc=0.001,
            #         seed=i
            #     ),
        }

        if data.cohort_name in ["crash_2","ist3","sprint","accord"]:
            nuisance_functions = NuisanceFunctions(rct=True)
        else:
            nuisance_functions = NuisanceFunctions(rct=False)

        nuisance_functions.fit(x_val, y_val, w_val)

        for learner_name, cate_model in learners.items():

            if learner_name == "XLearner_ensemble":
                print("training ensemble")
                results[learner_name]["prediction"][i] = 0

                for index in range(ensemble_num):

                    cate_model = pseudo_outcome_nets.XLearner(
                            x_train.shape[1],
                            binary_y=(len(np.unique(y_train)) == 2),
                            n_layers_out=2,
                            n_units_out=100,
                            batch_size=128,
                            n_iter=1000,
                            nonlin="relu",
                            device=DEVICE,
                            seed=i
                        )

                    sampled_indices = np.random.choice(len(x_train), size=len(x_train), replace=True)

                    x_sampled = x_train[sampled_indices]
                    y_sampled = y_train[sampled_indices]
                    w_sampled = w_train[sampled_indices]

                    cate_model.fit(x_sampled, y_sampled, w_sampled)

                    prediction = cate_model.predict(x_test).detach().cpu().numpy().flatten()

                    results[learner_name]["prediction"][i] += prediction/ensemble_num

                prediction = results[learner_name]["prediction"][i]
            else:

                cate_model.fit(x_train, y_train, w_train)

                prediction = cate_model.predict(x_test).detach().cpu().numpy().flatten()

                results[learner_name]["prediction"][i] = prediction

            for sec in selection_types:
                results[learner_name][sec][i] = calculate_pehe(
                    prediction,
                    data.get_data("test"),
                    sec,
                    nuisance_functions
                )

    with open(f"results/{cohort_name}/model_selection_shuffle_{shuffle}.pkl", "wb") as output_file:
        pickle.dump(results, output_file)

