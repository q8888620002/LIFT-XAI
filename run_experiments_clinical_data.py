import numpy as np
import pandas as pd
import argparse

import os
import sys
import collections
import pickle
import torch

from  utilities import normalize_data, Dataset
from scipy import stats
from shapreg import shapley, games, removal, shapley_sampling

from captum.attr import (
    IntegratedGradients,
    ShapleyValueSampling,
)

module_path = os.path.abspath(os.path.join('CATENets/'))
if module_path not in sys.path:
    sys.path.append(module_path)

import catenets.models.torch.pseudo_outcome_nets as pseudo_outcome_nets



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-d','--dataset', help='Dataset', required=True)
    parser.add_argument('-t','--num_trials', help='Dataset', required=True)
    parser.add_argument('-n','--top_n_features', help='Dataset', required=True)
    parser.add_argument('-l','--learner', help='Dataset', required=True)

    args = vars(parser.parse_args())

    cohort_name = args["dataset"]
    trials = int(args["num_trials"])
    top_n_features = int(args["top_n_features"])
    learner = args["learner"]

    data = Dataset(cohort_name)
    X_train, w_train, y_train = data.get_training_data()
    X_test, w_test, y_test = data.get_training_data()
    names = data.get_feature_names()

    feature_size = X_train.shape[1]

    device = "cuda:5"

    learner_explanations = {}

    models = {
        "xlearner":
            pseudo_outcome_nets.XLearner(  
            X_train.shape[1],
            binary_y=(len(np.unique(y_train)) == 2),
            n_layers_out=2,
            n_units_out=100,
            batch_size=128,
            n_iter=1500,
            nonlin="relu",
            device=device,
        ),
        "drlearner":
            pseudo_outcome_nets.DRLearner(  
            X_train.shape[1],
            binary_y=(len(np.unique(y_train)) == 2),
            n_layers_out=2,
            n_units_out=100,
            batch_size=128,
            n_iter=1500,
            nonlin="relu",
            device=device,
        )
    }
    explainers = [
        "ig", 
        "shapley_sampling_0", 
        "shap"
    ]

    top_k_results = {
        "ig":[],
        "shapley_sampling_0":[],
        "shap":[]
        }
    
    result_sign = {
        "ig": np.zeros((trials,feature_size)),
        "shapley_sampling_0": np.zeros((trials,feature_size)),
        "shap": np.zeros((trials,feature_size))
        }

    results_train = np.zeros((trials, len(X_train)))
    results_test = np.zeros((trials, len(X_test)))

    #### Getting top n features from multiple runs. 

    for i in range(trials):

        model = models[learner]
        model.fit(X_train, y_train, w_train)
        
        results_train[i] = model(X_train).detach().cpu().numpy().flatten()
        results_test[i] = model(X_test).detach().cpu().numpy().flatten()

        ig = IntegratedGradients(model)

        print("==================================================")
        print("explaining with IG")

        learner_explanations["ig"] = ig.attribute(
                                            torch.from_numpy(X_test).to(device).requires_grad_(),
                                            n_steps=500,
                                    ).detach().cpu().numpy()
        
        print("==================================================")
        print("explaining with shapley sampling -0")

        shapley_value_sampling_model = ShapleyValueSampling(model)
        learner_explanations["shapley_sampling_0"] = shapley_value_sampling_model.attribute(
                                                        torch.from_numpy(X_test).to(device).requires_grad_(),
                                                        n_samples=500,
                                                        perturbations_per_eval=10,
                                                    ).detach().cpu().numpy()
        
        print("==================================================")
        print("explaining with shapley sampling - marginal distribution")

        marginal_extension = removal.MarginalExtension(X_test, model)
        shap_values = np.zeros((X_test.shape))

        for test_ind in range(len(X_test)):
            instance = X_test[test_ind]
            game = games.PredictionGame(marginal_extension, instance)
            explanation = shapley_sampling.ShapleySampling(game, thresh=0.01, batch_size=128)
            shap_values[test_ind] = explanation.values.reshape(-1, X_test.shape[1])
        
        learner_explanations["shap"] = shap_values

        #### Getting top 5 features from multiple runs. 

        for e in explainers:
            ind = np.argpartition(np.abs(learner_explanations[e]).mean(0).round(2), -top_n_features)[-top_n_features:]
            top_k_results[e].extend(names[ind].tolist())

            for col in range(feature_size):
                result_sign[e][i, col] = stats.pearsonr(X_test[:,col], learner_explanations[e][:, col])[0]

    for e in explainers:
        results = collections.Counter(top_k_results[e])
        summary = pd.DataFrame(results.items(), columns=['feature', 'count (%)']).sort_values(by="count (%)", ascending=False)
        summary["count (%)"] = np.round(summary["count (%)"]/trials,2)*100

        indices = [names.tolist().index(i) for i in summary.feature.tolist()]
        summary["sign"] = np.sign(np.mean(result_sign[e], axis=(0))[indices])

        summary.to_csv(f"results/{cohort_name}/{e}_top_{top_n_features}_features_{learner}.csv")
    
    with open( f"results/{cohort_name}/train_{learner}.pkl", "wb") as output_file:    
        pickle.dump(results_train, output_file)

    with open( f"results/{cohort_name}/test_{learner}.pkl", "wb") as output_file:    
        pickle.dump(results_test, output_file)
