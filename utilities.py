import numpy as np
import math
import shap
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

def kl_mvn(m0, S0, m1, S1):
    """
    https://stackoverflow.com/questions/44549369/kullback-leibler-divergence-from-gaussian-pm-pv-to-gaussian-qm-qv
    The following function computes the KL-Divergence between any two 
    multivariate normal distributions 
    (no need for the covariance matrices to be diagonal)
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    - accepts stacks of means, but only one S0 and S1
    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    # 'diagonal' is [1, 2, 3, 4]
    tf.diag(diagonal) ==> [[1, 0, 0, 0]
                          [0, 2, 0, 0]
                          [0, 0, 3, 0]
                          [0, 0, 0, 4]]
    # See wikipedia on KL divergence special case.              
    #KL = 0.5 * tf.reduce_sum(1 + t_log_var - K.square(t_mean) - K.exp(t_log_var), axis=1)   
                if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))                               
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N) 


def KL_divergence(a, b):
    epsilon = 1e-35 
    a += epsilon
    b += epsilon

    return np.mean(a * np.log(a/b))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mse(x1, x2):
    return (np.mean((x1 - x2)**2))

def rmse(x1, x2):
    return np.sqrt(np.mean((x1 - x2)**2))

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def simulation(var_size, n, low , high, random_state, oracle=bool):
    
    np.random.seed(random_state)
    cov = np.random.uniform(low=-1, high=1, size=((var_size,var_size)))
    
    ### control 
    mu0 = np.zeros(var_size)
    x0 = np.random.multivariate_normal(mu0, 0.5*cov, size=int(n/2))
    
    ### treated
    mu1 = np.random.uniform(low = low, high=high, size=(var_size))
    x1 = np.random.multivariate_normal(mu1, 0.5*cov, size=int(n/2))
    
    X = np.concatenate((x0, x1))
    
    #### Treatment assignment
    w_t = np.random.uniform(low=-0.1, high=0.1, size=((var_size,1)))
    n_t = np.random.normal(0, 0.1)
    
    p = sigmoid(np.dot(X, w_t) + n_t)

    w = np.concatenate((np.zeros((int(n/2),1)), np.ones((int(n/2),1 ))), 0).astype(int)
    #### Potential outcome
    if oracle:
        p = 0.5*np.ones((n,1))
        w = np.random.binomial(1, p=p)
    
    w_ty = np.random.uniform(low=-1, high=1, size=((var_size,2)))
    n_ty = np.random.multivariate_normal(np.zeros((2)), 0.1*np.eye(2))    
    y_po = np.dot(X, w_ty) + n_ty

    return X, y_po, w, p, kl_mvn(mu0, 0.5*cov, mu1, 0.5*cov)


def oracle(X_train, w_train, y_train, X_test, model):

    model.fit(X_train, y_train, w_train)
    ### TODOs implement prediction function for different cate
    prediction = model.predict(X_test)

    if torch.is_tensor(X_train):
        X_train = X_train.detach().numpy()
        X_test = X_test.detach().numpy()
        model_lam = lambda x: model.predict(x).detach().numpy()
    else:
        model_lam = lambda x: model.predict(x)


    explainer = shap.Explainer(model_lam, X_train)

    #### showing explanation on oracle cate
    shap_values = explainer(X_test)
    shap_mean = (shap_values.values).mean(0)
    shap_abs_mean = np.abs(shap_values.values).mean(0)
    
    return (shap_mean, shap_abs_mean, model)
    
    
def generate_masks(X):
    
    batch_size = X.shape[0]
    num_features = X.shape[1]
    
    unif = torch.rand(batch_size, num_features)
    ref = torch.rand(batch_size, 1)
    
    return (unif > ref).float()
    
class CooperativeGame:
    '''Base class for cooperative games.'''

    def __init__(self):
        raise NotImplementedError

    def __call__(self, S):
        '''Evaluate cooperative game.'''
        raise NotImplementedError

    def grand(self):
        '''Get grand coalition value.'''
        return self.__call__(np.ones((1, self.players), dtype=bool))[0]

    def null(self):
        '''Get null coalition value.'''
        return self.__call__(np.zeros((1, self.players), dtype=bool))[0]

class CateGame(CooperativeGame):
    '''
    Cooperative game for CATE mdoels 
    '''

    def __init__(self, sample, cate_model):
        '''
        Cooperative game for an individual example's prediction.

        Args:
        sample: numpy array representing a single model input.
        cate_morel: treatment effect model that takes masking. 
        '''
        self.sample = sample
        self.model = cate_model
        self.players = self.sample.size()[1]

    def __call__(self, S):
        '''
        Evaluate cooperative game.

        Args:
          S: array of player coalitions with size (batch, players).
        '''
        device = next(self.model.parameters()).device
        S = torch.from_numpy(S).to(device)
        if len(S.shape) == 1:
            S = S.reshape(1, -1)

        if len(S) != len(self.sample):
            input_data = self.sample.repeat((len(S), 1))
        else:
            input_data = self.sample

        values = self.model.predict(input_data, S).detach().cpu().numpy()
        return values[:, 0]