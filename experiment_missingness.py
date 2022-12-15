import os
import sys
import xgboost
import shap
import argparse
import pickle as pkl

import matplotlib.pyplot as plt

from scipy import stats
from utilities import *
from models import *
from sklearn import preprocessing
from shapreg import shapley, games
from captum.attr import IntegratedGradients


#### import CATE model
module_path = os.path.abspath(os.path.join('CATENets/'))

if module_path not in sys.path:
    sys.path.append(module_path)

from catenets.models.jax import TNet, SNet,SNet1, SNet2, SNet3, DRNet, RANet, PWNet, RNet, XNet
import catenets.models.torch.pseudo_outcome_nets_mask as cate_models_mask
import catenets.models.torch as cate_models


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-d','--d', help='feature dimension', required=True)
    parser.add_argument('-n','--n', help='sample size',required=True)
    parser.add_argument('-r','--r', help='random state',required=True)

    args = vars(parser.parse_args())
    
    n = int(args["n"])
    feature_size = int(args["d"])
    random_state = int(args["r"])
    device = "cuda:0"
    oracle_device = "cpu"

    #path = ("results_d=%s_n=%s_r=%s/"%(feature_size, n, random_state))
    
    #if not os.path.exists(path):
    #    os.makedirs(path)
    
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

    ### obtain oracle with DRNet()
    print("Training Baseline.")
                                                 
    torch_DRNet = cate_models.DRLearner(
                                        feature_size,
                                        binary_y=False,
                                        nonlin="relu",
                                        n_iter=2000,
                                        n_iter_print=250,
                                        early_stopping= False,
                                        device=oracle_device,
                                        )
    
    dr_unbiased, dr_unbiased_abs, torch_DRNet = oracle( 
                                                        x_oracle_train, 
                                                        w_oracle_train,
                                                        y_oracle_train, 
                                                        x_oracle_test, 
                                                        torch_DRNet
                                                        )

    ### IntGrad
    print("compute Intgradient")
    ig = IntegratedGradients(torch_DRNet)

    attr, delta = ig.attribute( 
                                x_oracle_test.requires_grad_(), 
                                n_steps= 500, 
                                return_convergence_delta=True
                                )

    attr = attr.detach().cpu().numpy()
    attr_mean = np.mean(attr, axis=0)
    attr_abs = np.mean(np.abs(attr), axis=0)

    print("training masking explainer.")

    ### Init Cate model 
    torch_DRNet_Mask = cate_models_mask.DRLearner(  
                                                    feature_size,
                                                    binary_y=False,
                                                    n_iter=2000,
                                                    n_iter_print=250,
                                                    nonlin="relu",
                                                    early_stopping= False,
                                                    device=device
                                                    )
    
    # Train model with masks
    
    torch_DRNet_Mask.fit(x_oracle_train, y_oracle_train, w_oracle_train)

    test_mask = torch.ones(x_oracle_test.size()).to(device)
    test_phe = torch_DRNet_Mask.predict(x_oracle_test,test_mask).cpu().detach().numpy()

    print("phe is %s" %mse(test_phe, y_test_cate))

    # Explanation method
    # init test result
    test_values = np.zeros((len(test_inds) ,feature_size ))
    
    print("======explanation starts.=======")

    for test_ind in range(x_oracle_test.size()[0]):
        instance = torch.from_numpy(X_scaled[test_ind, :])[None,:].to(device)
        game  = games.CateGame(instance, torch_DRNet_Mask)
        explanation = shapley.ShapleyRegression(game, batch_size=64)

        test_values[test_ind] = explanation.values
    
    mask_shap = np.mean(test_values, axis=0)
    mask_shap_abs = np.mean(np.abs(test_values), axis=0)

    print("Final results.")
    print("== Oracle, torchDRNet, MSE ==")
    print("phe is %s" %mse(torch_DRNet.predict(x_oracle_test).cpu().detach().numpy(), y_test_cate))

    print("== Kernel SHAP (torchDRNet) vs IntGrad (torchDRNet) ==")
    print(stats.spearmanr(dr_unbiased, attr_mean).correlation)
    print(stats.spearmanr(dr_unbiased_abs, attr_abs).correlation)

    print("== Kernel SHAP (torchDRNet) vs Shapley Regression (torch_mask) ==")
    print("phe is %s" %mse(test_phe, y_test_cate))
    print(stats.spearmanr(dr_unbiased , mask_shap).correlation)
    print(stats.spearmanr(dr_unbiased , mask_shap_abs).correlation)

    print("== IntGrad (torch_mask) vs Shapley Regression (torch_mask) ==")
    print(stats.spearmanr(attr_mean, mask_shap).correlation)
    print(stats.spearmanr(attr_abs, mask_shap_abs).correlation)