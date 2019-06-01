import gc, argparse, sys, os, errno
import numpy as np
import pandas as pd
import os
from tqdm import tqdm as tqdm
import scipy
import sklearn
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')
from numba import jit

eps = 10e-3

@jit
def sigmoid(x):
    return 1/(1+np.exp(-x))

@jit
def J_cost(W,beta,X,Y,lambda0, lambda1, lambda2, lambda3, lambda5):
    return lambda0*sum((W*W)*(np.log(1+np.exp(X@beta))-Y*(X@beta))) \
         +lambda1*sum(balance_cost(W,X)) \
         +lambda2*((W*W).T@(W*W)) \
         +lambda3*sum(beta**2) \
         +lambda5*(sum(W*W)-1)**2
@jit
def balance_cost(W=None,X=None,*args,**kwargs):
    m = X.shape[1]
    f_x=np.zeros([m,1])
    for i in np.arange(0,m):
        X_sub=np.copy(X)
        X_sub[:,i]=0
        I=(X[:,i] > 0).astype('double')+eps
        loss=( np.dot( X_sub.T, np.multiply( np.multiply(W,W),I.reshape(-1,1) ) ) ) / (np.dot((np.multiply(W,W)).T,I.reshape(-1,1)))\
            -(np.dot(X_sub.T,(np.multiply((np.multiply(W,W)),(1 - I.reshape(-1,1)))))) / (np.dot((np.multiply(W,W)).T,(1 - I.reshape(-1,1))))
        #print (loss.shape)
        f_x[i]=np.dot(loss.T,loss)
    return f_x
@jit
def balance_grad(W=None,X=None,*args,**kwargs):
    n,m=X.shape

    g_w=np.zeros([n,m])
    for i in range(0,m):
        X_sub = np.copy(X)
        X_sub[:,i] = 0 # the ith column is treatment
        I = (X[:,i]>0).reshape(-1,1).astype('double')+eps
        J1 = (X_sub.T@((W*W)*I.reshape(-1,1)))/((W*W).T@(I.reshape(-1,1))) \
            -(X_sub.T@((W*W)*(1-I).reshape(-1,1)))/((W*W).T@(1-I).reshape(-1,1))
        dJ1W = 2*(X_sub.T*((W*I)@np.ones([1,m])).T*((W*W).T@I) \
                  -(X_sub.T@(((W*W)*I)@(W*I).T)))/((W*W).T@I)**2 \
                  -2*(X_sub.T*((W*(1-I))@np.ones([1,m])).T*((W*W).T@(1-I)) \
                  -((X_sub.T@( (W*W) * (1-I) )) @  (W*(1-I) ).T ))/((W*W).T@(1-I))**2
        g_w[:,i] = (2 * dJ1W.T @ J1).ravel()

    return g_w

@jit
def prox_l1(v=None,lambda_=None,*args,**kwargs):
    x=np.fmax(0,v - lambda_) - np.fmax(0,- v - lambda_)
    return x
@jit
def mainFunc(X, Y, \
    lambda0, lambda1, lambda2, lambda3, lambda4, lambda5,\
    MAXITER, ABSTOL, W_init, beta_init,paras_save_path=None):

    n,m = X.shape
    W = W_init
    W_prev = np.copy(W)
    beta = beta_init
    beta_prev = np.copy(beta)

    parameter_iter = 0.5
    J_loss = np.ones([MAXITER, 1])*(-1)

    lambda_W = 1
    lambda_beta = 1

    W_All = np.zeros([n, MAXITER])
    beta_All = np.zeros([m, MAXITER])


    # Optimization with gradient descent
    for iter in tqdm(range(1,MAXITER+1)):
        # Update beta
        y = np.copy(beta)
        beta = beta + (iter/(iter+3))*(beta-beta_prev) # fast proximal gradient
        f_base = J_cost(W, beta, X, Y, lambda0, lambda1, lambda2, lambda3, lambda5)
        grad_beta = lambda0*(((sigmoid(X@beta)-Y)*(W*W)).T@X).T \
                   +2*lambda3*beta

        while 1:
            z = prox_l1(beta - lambda_beta*grad_beta, lambda_beta*lambda4)
            if J_cost(W, z, X, Y, lambda0, lambda1, lambda2, lambda3, lambda5)\
               <= f_base + grad_beta.T@(z-beta)\
               + (1/(2*lambda_beta))*sum((z-beta)**2):
                break
            lambda_beta = parameter_iter*lambda_beta
            if lambda_beta<eps*eps*eps:
                break
        beta_prev = y
        beta = z

        # Update W
        y = np.copy(W)
        W = W+(iter/(iter+3))*(W-W_prev)
        f_base = J_cost(W, beta, X, Y, lambda0, lambda1, lambda2, lambda3, lambda5)

        grad_W = 2*lambda0*(np.log(1+np.exp(X@beta))-Y*(X@beta))*W \
                +lambda1*balance_grad(W, X)@np.ones([m,1]) \
                +4*lambda2*W*W*W \
                +4*lambda5*(sum(W*W)-1)*W

        while 1:
            z = prox_l1(W-lambda_W*grad_W, 0)
            #print (J_cost(z, beta, X, Y, lambda0, lambda1, lambda2, lambda3, lambda5),
            #f_base , grad_W.T@(z-W),(1/(2*lambda_W))*sum((z-W)**2))
            if J_cost(z, beta, X, Y, lambda0, lambda1, lambda2, lambda3, lambda5)\
                    <= f_base + grad_W.T@(z-W)\
                    + (1/(2*lambda_W+10e-100))*sum((z-W)**2):
                break
            lambda_W = parameter_iter*lambda_W
            if lambda_W<eps*eps*eps:
                break

        W_prev = y
        W = z

        W_All[:,iter-1] = W.ravel()
        beta_All[:,iter-1] = beta.ravel()

        J_loss[iter-1] = J_cost(W, beta, X, Y,\
                              lambda0, lambda1, lambda2, lambda3, lambda5)\
                     + lambda4*sum(abs(beta))
        print (lambda0*sum((W*W)*(np.log(1+np.exp(X@beta))-Y*(X@beta))) ,
         lambda1*sum(balance_cost(W,X)), \
         lambda2*((W*W).T@(W*W)), \
         lambda3*sum(beta**2), \
         lambda5*(sum(W*W)-1)**2 , J_loss[iter-1])            
        #print (J_loss[iter-1] , J_loss[iter-2])
        if (paras_save_path is not None) & (iter%10==1):
            #'output/models/crlr/somedir'
            if not os.path.exists(paras_save_path):
                os.makedirs(paras_save_path)
            np.savetxt(paras_save_path+'/beta.txt',beta)
            np.savetxt(paras_save_path+'/W.txt',W)
            np.savetxt(paras_save_path+'/J_loss.txt',J_loss)
        if (iter > 1) & ( abs(J_loss[iter-1] - J_loss[iter-2])[0]  < ABSTOL) or (iter == MAXITER):
            break
        #if (iter > 11) & ( ((J_loss[1:]-J_loss[:-1])[-10:]>0).sum()>0  ) or (iter == MAXITER):
         #   break
    W = W*W

    return W, beta, J_loss



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, \
    roc_curve, precision_recall_curve, average_precision_score, matthews_corrcoef, confusion_matrix

def report_metrics(y_test, y_pred):
    scorers = {'accuracy': accuracy_score,
           'recall': recall_score,
           'precision': precision_score,
           'f1': f1_score,
           'mcc': matthews_corrcoef
    }
    for metric in scorers.keys():
        print('{} = {}'.format(metric, scorers[metric](y_test, y_pred)))


if __name__ == '__main__':
    '''
    sample_size = 6000
    feature_size = 50
    X = 2*np.round(np.random.rand(sample_size, feature_size))-1 # 1000 samples and 20 features
    beta_true = np.ones([feature_size, 1])
    Y = (sigmoid(np.dot(X,beta_true))>=0.5).astype('double')
    lambda0 = 1 #Logistic loss
    lambda1 = 0.1 #Balancing loss
    lambda2 = 1 #L_2 norm of sample weight
    lambda3 = 0 #L_2 norm of beta
    lambda4 = 0.001 #L_1 norm of bata
    lambda5 = 1 #Normalization of sample weight
    ABSTOL = 1e-3

    print ('***********classic logistic***********')
    X_train, X_test, y_train, y_test = train_test_split(X,Y.astype('int'))
    print('number of training samples: {}, test samples: {}'.format(X_train.shape[0], X_test.shape[0]))
    model = LogisticRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    report_metrics(y_test, y_pred)
    '''

    print ('***********CRLR***********')
    sample_size = 6000
    feature_size = 50
    X = 2*np.round(np.random.rand(sample_size, feature_size))-1 # 1000 samples and 20 features
    beta_true = np.ones([feature_size, 1])
    Y = (sigmoid(np.dot(X,beta_true))>=0.5).astype('double')
    X_train, X_test, y_train, y_test = train_test_split(X,Y.astype('int'))
    print('number of training samples: {}, test samples: {}'.format(X_train.shape[0], X_test.shape[0]))
    lambda0 = 1 #Logistic loss
    lambda1 = 0.1 #Balancing loss
    lambda2 = 1 #L_2 norm of sample weight
    lambda3 = 0 #L_2 norm of beta
    lambda4 = 0.001 #L_1 norm of bata
    lambda5 = 0.01 #Normalization of sample weight
    ABSTOL = 1e-3
    import argparse
    parser = argparse.ArgumentParser(description='save and reload parameters')
    parser.add_argument('--save', dest='paras_save_path',  default=None, help='specify dir to save parameters')
    parser.add_argument('--load', dest='paras_load_path',default=None, help='specify dir to load parameters')
    parser.add_argument('--max_iter', dest='max_iter',default=1000,type=int, help='max iteration')
    parser.add_argument('--eps', dest='eps',default=10e-10,type=float, help='epsilon to avoid zero division')
    args = parser.parse_args()



    MAXITER = args.max_iter
    eps = args.eps
    paras_load_path = args.paras_load_path
    paras_save_path = args.paras_save_path

    if paras_load_path is not None:  #'output/models/crlr/somedir'
        W_init = np.loadtxt(paras_load_path+'/W.txt').reshape(-1,1)
        beta_init = np.loadtxt(paras_load_path+'/beta.txt').reshape(-1,1)
    else:
        W_init = np.random.rand(X_train.shape[0], 1)
        beta_init = 0.5*np.ones([feature_size, 1])

    W, beta, J_loss = mainFunc(X_train, y_train,\
            lambda0, lambda1, lambda2, lambda3, lambda4, lambda5,\
            1000, ABSTOL, W_init, beta_init, paras_save_path)
    y_pred = (sigmoid(np.dot(X_test,beta))>=0.5).astype('int')
    report_metrics(y_test, y_pred)

    '''
    python3 bin/CRLR_binary.py --save 'output/models/crlr/0' --max_iter 100 --eps 10e-10 
    python3 bin/CRLR_binary.py --save 'output/models/crlr/0' --load 'output/models/crlr/0' --max_iter 100 --eps 10e-10 
    '''