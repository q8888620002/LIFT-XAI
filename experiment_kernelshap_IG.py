import os
import sys
import xgboost
import shap
import argparse
import pickle as pkl

import matplotlib.pyplot as plt
from scipy import stats
from utilities import *
from sklearn import preprocessing
from captum.attr import IntegratedGradients

#### import CATE model
module_path = os.path.abspath(os.path.join('CATENets/'))

if module_path not in sys.path:
    sys.path.append(module_path)

from catenets.models.jax import  DRNet, RANet, PWNet, RNet, XNet, TNet, SNet,SNet1, SNet2, SNet3
from catenets.experiment_utils.simulation_utils import simulate_treatment_setup
import catenets.models.torch as cate_models


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-n','--n', help='sample size',required=True)
    parser.add_argument('-d','--d', help='feature dimension', required=True)
    parser.add_argument('-r','--r', help='random state',required=True)

    args = vars(parser.parse_args())
    
    n = int(args["n"])
    feature_size = int(args["d"])
    random_state = int(args["r"])
    oracle_device = "cuda:0"
    path = ("results/igs/results_d=%s_n=%s_r=%s/"%(feature_size, n, random_state))
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    X, y_po, w, p, KL = simulation(feature_size, n, 0, 0, random_state, False)
    
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)
    
    #print(KL)
    rng = np.random.default_rng(random_state)
    inds = np.arange(n)
    rng.shuffle(inds)
    
    n_train = int(0.8 * n)

    train_inds = inds[:n_train]
    test_inds = inds[n_train:]

    x_oracle_train = torch.from_numpy(X_scaled[train_inds,:]).to(oracle_device)
    x_oracle_test = torch.from_numpy(X_scaled[test_inds,:]).to(oracle_device)
    w_oracle_train = torch.from_numpy(w[train_inds,:]).to(oracle_device)
    y_oracle_train = torch.from_numpy(np.take_along_axis(y_po,w, 1)[train_inds, :]).to(oracle_device)
    y_test_cate = y_po[test_inds, 1] - y_po[test_inds, 0]
    
    # init DRNet
    torch_DRNet = cate_models.DRLearner(
                                        feature_size,
                                        binary_y=False,
                                        nonlin="relu",
                                        device=oracle_device
                                       )
    torch_DRNet.fit(x_oracle_train, y_oracle_train,w_oracle_train )

    ig = IntegratedGradients(torch_DRNet)

    attr, delta = ig.attribute( 
                                x_oracle_test.requires_grad_(), 
                                n_steps= 500, 
                                return_convergence_delta=True
                                )

    attr = attr.detach().cpu().numpy()
    dr_unbiased = np.mean(attr, axis=0)
    dr_unbiased_abs = np.mean(np.abs(attr), axis=0)

    models = [ 
                cate_models.RLearner(    
                                        feature_size,
                                        binary_y=False,
                                        nonlin="relu",
                                        device=oracle_device
                                        ),
                cate_models.XLearner(    
                                        feature_size,
                                        binary_y=False,
                                        nonlin="relu",
                                        device=oracle_device
                                        ),
                cate_models.DRLearner(
                                        feature_size,
                                        binary_y=False,
                                        nonlin="relu",
                                        device=oracle_device
                                     ), 
                cate_models.RALearner( feature_size,
                                        binary_y=False,
                                        nonlin="relu",
                                        device=oracle_device),

                cate_models.PWLearner( feature_size,
                                        binary_y=False,
                                        nonlin="relu",
                                        device=oracle_device)
            ]

            
    ### treated group mu1
    
    mu_t = np.arange(0, -6.5, -0.25, dtype=float)
    
    spearman_results = np.zeros((len(models), len(mu_t), 2))
    pehes = np.zeros((len(models), len(mu_t)))
    KL_results = np.zeros(len(mu_t))

    ### sample training index
    
    for i, value in enumerate(mu_t):
        
        #### Generate data with selection bias
        mu1_low, mu1_high = value - 0.2,  value
        X, y_po, w, p, KL = simulation(feature_size, n, mu1_low, mu1_high, random_state, False) 
        
        print("mu1 %s; %s KL is %s" %( mu1_low, mu1_high, KL))
        
        KL_results[i] = KL

        X_scaled =  torch.from_numpy(min_max_scaler.transform(X)).to(oracle_device)
        w_train = torch.from_numpy(w[train_inds,:]).to(oracle_device)
        y_train = torch.from_numpy(np.take_along_axis(y_po,w, 1)[train_inds, :]).to(oracle_device)

        y_test_cate = y_po[test_inds, 1] - y_po[test_inds, 0]

        #### train CATEs

        for model_index, model in enumerate(models):
            cate_net = model
            
            print("train model %s"%str(cate_net))
            
            X_train, X_test = X_scaled[train_inds,:], X_scaled[test_inds,:]
            cate_net.fit(X_train, y_train, w_train)   

            #### predict potential outcomes

            pred_cate = cate_net.predict(X_test).detach().cpu().numpy()

            pehe = mse(y_test_cate, pred_cate)

            pehes[model_index, i] = pehe

            print("PHE %s for model %s"%(pehe,str(cate_net)))

            #### explaining CATE on testing sets.

            ig = IntegratedGradients(cate_net)

            attr, delta = ig.attribute( 
                                        X_test.requires_grad_(), 
                                        n_steps= 500, 
                                        return_convergence_delta=True
                                        )

            attr = attr.detach().cpu().numpy()
            ig_imp = np.mean(attr, axis=0)
            ig_imp_abs = np.mean(np.abs(attr), axis=0)

            spearman_results[model_index, i, 0] = stats.spearmanr(dr_unbiased , ig_imp).correlation
            spearman_results[model_index, i, 1] = stats.spearmanr(dr_unbiased_abs , ig_imp_abs).correlation
            print(spearman_results[model_index, i, 0], spearman_results[model_index, i, 1])

    
    np.save(path+'correlations.npy', spearman_results)
    np.save(path+'pehes.npy', pehes)
    np.save(path+'kls.npy', KL_results)

    plt.figure(figsize=(10,4))

    for i in range(len(models)):
        plt.plot(KL_results, spearman_results[i,:, 0], label=str(models[i]))

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel("Spearman's Rank")
    plt.xlabel("KL divergence")
    plt.title("KL - shap")
    plt.savefig(path+"KL_spearman.png")

    plt.figure(figsize=(10,4))
    for i in range(len(models)):
        plt.plot(KL_results, spearman_results[i,:, 1], label=str(models[i]))

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel("Spearman's Rank")
    plt.xlabel("KL divergence")
    plt.title("KL - abs(shap)")
    plt.savefig(path+"KL_spearman_abs.png")

    plt.figure(figsize=(10,4))

    for i in range(len(models)):
        plt.plot(KL_results, pehes[i,:], label=str(models[i]))

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("KL divergence")
    plt.ylabel("PEHE")
    plt.savefig(path+"KL_pehe.png")

    plt.figure(figsize=(10,4))

    for i in range(len(models)):
        plt.plot(pehes[i,:], spearman_results[i,:, 0],label=str(models[i]))

    plt.legend()
    plt.xlabel("Pehe")
    plt.ylabel("Spearman's Rank Cor")
    plt.savefig(path+"spearman_pehe.png")