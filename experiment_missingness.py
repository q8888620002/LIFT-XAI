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


#### import CATE model
module_path = os.path.abspath(os.path.join('CATENets/'))

if module_path not in sys.path:
    sys.path.append(module_path)

from catenets.models.jax import TNet, SNet,SNet1, SNet2, SNet3, DRNet, RANet, PWNet, RNet, XNet
import catenets.models.pseudo_outcome_nets_mask as cate_models_mask
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
    print("training oracle.")


    dr_unbiased_jax, dr_unbiased_abs_jax = oracle(X_scaled[train_inds,:], 
                                                  w[train_inds,:],
                                                  np.take_along_axis(y_po,w, 1)[train_inds, :], 
                                                  X_scaled[test_inds,:], 
                                                  DRNet(nonlin="relu")
                                                 )

    torch_DRNet = cate_models.DRLearner(
                                        feature_size,
                                        binary_y=(len(np.unique(y_po)) == 2),
                                        nonlin="relu",
                                        device=oracle_device
                                        )
    
    dr_unbiased, dr_unbiased_abs = oracle(  x_oracle_train, 
                                            w_oracle_train,
                                            y_oracle_train, 
                                            x_oracle_test, 
                                            torch_DRNet)
    ### Create Cate model 
    print(stats.spearmanr(dr_unbiased , dr_unbiased_jax).correlation)
    print(stats.spearmanr(dr_unbiased_abs , dr_unbiased_abs_jax).correlation)

    print("training masking explainer.")

    torch_DRNet_Mask = cate_models_mask.DRLearner(
                                                    feature_size,
                                                    binary_y=(len(np.unique(y_po)) == 2),
                                                    nonlin="relu",
                                                    device=device
                                                    )
    
    ### Train model with maskes
    
    torch_DRNet_Mask.fit(x_oracle_train, y_oracle_train, w_oracle_train)
    #cate_model.fit(x_oracle_train, y_oracle_train, w_oracle_train, epoches)

    test_mask = torch.ones(x_oracle_test.size()).to(device)
    test_phe = torch_DRNet_Mask.predict(x_oracle_test,test_mask).cpu().detach().numpy()

    print("phe is %s" %mse(test_phe, y_test_cate))

    # Explanation method
    # init test result
    test_values = np.zeros((len(test_inds) ,feature_size ))
    
    print("======explanation starts.=======")

    for test_ind in range(x_oracle_test.size()[0]):
        instance = torch.reshape(torch.from_numpy(X_scaled[test_ind, :]), (1,-1)).to(device)
        game  = games.CateGame(instance, torch_DRNet_Mask)
        explanation = shapley.ShapleyRegression(game, batch_size=64)

        test_values[test_ind] = explanation.values
    
    mask_shap = np.mean(test_values, axis=0)
    mask_shap_abs = np.mean(np.abs(test_values), axis=0)

    print("Final results.")
    print(stats.spearmanr(dr_unbiased , mask_shap).correlation)
    print(stats.spearmanr(dr_unbiased , mask_shap_abs).correlation)
