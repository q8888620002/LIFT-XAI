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
import catenets.models as cate_models


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-d','--d', help='feature dimension', required=True)
    parser.add_argument('-n','--n', help='sample size',required=True)
    parser.add_argument('-e','--e', help='training epoches',required=True)
    parser.add_argument('-r','--r', help='random state',required=True)

    args = vars(parser.parse_args())
    
    n = int(args["n"])
    feature_size = int(args["d"])
    random_state = int(args["r"])
    epoches = int(args["e"])
    device = "cuda:0"
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

    x_oracle_train = torch.from_numpy(X_scaled[train_inds,:]).to(device)
    w_oracle_train = torch.from_numpy(w[train_inds,:]).to(device)
    y_oracle_train = torch.from_numpy(np.take_along_axis(y_po,w, 1)[train_inds, :]).to(device)
    y_test_cate = y_po[test_inds, 1] - y_po[test_inds, 0]

    ### obtain oracle with DRNet()
    print("training oracle.")
    dr_unbiased, dr_unbiased_abs = oracle(X_scaled[train_inds,:], 
                                        w[train_inds,:],
                                        np.take_along_axis(y_po,w, 1)[train_inds, :], 
                                        X_scaled[test_inds,:], 
                                        DRNet(nonlin="relu"))
    ### Create Cate model 

    torch_DR = cate_models.torch.DRLearner(
                2*feature_size,
                binary_y=(len(np.unique(y_po)) == 2),
                n_layers_out=2,
                n_units_out=100,
                n_iter=1,
                batch_size=128,
                batch_norm=False,
                nonlin="relu",
                ).to(device)
                
    mask_layer = MaskLayer().to(device)
    cate_model = Cate(torch_DR, mask_layer, device)

    ### Train model with maskes

    cate_model.fit(x_oracle_train, y_oracle_train, w_oracle_train, epoches)

    x_oracle_test = torch.from_numpy(X_scaled[test_inds,:]).to(device)
    w_oracle_test = torch.from_numpy(w[test_inds,:]).to(device)
    test_mask = torch.ones(x_oracle_test.size()[0],x_oracle_test.size()[1]).to(device)
    test_phe = cate_model.predict(x_oracle_test, test_mask).cpu().detach().numpy()

    print("phe is %s" %mse(test_phe, y_test_cate))

    #### Explanation method

    test_values = np.zeros((len(test_inds) ,feature_size ))
    
    print("======explanation starts.=======")

    for test_ind in range(x_oracle_test.size()[0]):
        instance = torch.reshape(torch.from_numpy(X_scaled[test_ind, :]), (1,-1)).to(device)
        game  = games.CateGame(instance, cate_model)
        explanation = shapley.ShapleyRegression(game, batch_size=32)

        test_values[test_ind] = explanation.values
    
    mask_shap = np.mean(test_values, axis=0)
    mask_shap_abs = np.mean(np.abs(test_values), axis=0)

    print("Final results.")
    print(stats.spearmanr(dr_unbiased , mask_shap).correlation)
    print(stats.spearmanr(dr_unbiased , mask_shap_abs).correlation)
