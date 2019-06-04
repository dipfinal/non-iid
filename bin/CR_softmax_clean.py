#!/usr/bin/env python
import gc, argparse, sys, os, errno
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#import h5py
import os
from tqdm import tqdm as tqdm
import scipy
import sklearn
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')




def sigmoid(x):
    #z = np.dot(X,W)
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x,axis=1)
    sum_exp_x= sum_exp_x.reshape((x.shape[0],1))
    sigmoid = exp_x / sum_exp_x
    return sigmoid



def cross_entropy_loss(W,X,Y,beta):
    '''
    part of J_loss first term
    '''
    n = X.shape[0]
    Y_cap = sigmoid(X@beta)
    #weight_term = (W*W).reshape(-1,1)
    #print (weight_term.shape,(Y.reshape(-1,1)*np.log(Y_cap)).shape)
    loss = -np.sum((W*W)*(Y*np.log(Y_cap)))
    return loss




def balance_cost(W=None,X=None,*args,**kwargs):
    m = X.shape[1]
    f_x=np.zeros([m,1])
    for i in np.arange(0,m):
        X_sub=np.copy(X)
        X_sub[:,i]=0
        I=(X[:,i] > 0).astype('double')+eps
        #print ('balance',( np.dot( X_sub.T, np.multiply( np.multiply(W,W),I.reshape(-1,1) ) ) ).shape,\
         #      (np.dot((np.multiply(W,W)).T,I.reshape(-1,1))).shape)
         
        
        loss=( np.dot( X_sub.T, np.multiply( np.multiply(W,W),I.reshape(-1,1) ) ) ) / (np.dot((np.multiply(W,W)).T,I.reshape(-1,1)).T)            -(np.dot(X_sub.T,(np.multiply((np.multiply(W,W)),(1 - I.reshape(-1,1)))))) / (np.dot((np.multiply(W,W)).T,(1 - I.reshape(-1,1))).T)
        #print (loss.shape)
        f_x[i]=np.sum(np.dot(loss.T,loss))
    return f_x




def balance_grad(W=None,X=None,*args,**kwargs):
    n,m=X.shape

    g_w=np.zeros([n,m])
    for i in range(0,m):
        X_sub = np.copy(X)
        X_sub[:,i] = 0 # the ith column is treatment
        I = (X[:,i]>0).reshape(-1,1).astype('double')+eps
        J1 = (X_sub.T@((W*W)*I.reshape(-1,1)))/((W*W).T@(I.reshape(-1,1)))             -(X_sub.T@((W*W)*(1-I).reshape(-1,1)))/((W*W).T@(1-I).reshape(-1,1))
        dJ1W = 2*(X_sub.T*((W*I)@np.ones([1,m])).T*((W*W).T@I)                   -(X_sub.T@(((W*W)*I)@(W*I).T)))/((W*W).T@I)**2                   -2*(X_sub.T*((W*(1-I))@np.ones([1,m])).T*((W*W).T@(1-I))                   -((X_sub.T@( (W*W) * (1-I) )) @  (W*(1-I) ).T ))/((W*W).T@(1-I))**2
        g_w[:,i] = (2 * dJ1W.T @ J1).ravel()

    return g_w




from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score,     roc_curve, precision_recall_curve, average_precision_score, matthews_corrcoef, confusion_matrix

def report_metrics(y_test, y_pred):
    scorers = {'accuracy': accuracy_score(y_test, y_pred),
           'recall': recall_score(y_test, y_pred,average=None),
           'precision': precision_score(y_test, y_pred,average=None),
           'f1': f1_score(y_test, y_pred,average=None),
           'mcc': matthews_corrcoef(y_test, y_pred)
    }
    for metric in scorers.keys():
        print('{} = {}'.format(metric, scorers[metric]))



def cross_entropy_loss(W,X,Y,beta):
    '''
    part of J_loss first term
    '''
    n = X.shape[0]
    Y_cap = sigmoid(X@beta)
    #weight_term = (W*W).reshape(-1,1)
    #print (weight_term.shape,(Y.reshape(-1,1)*np.log(Y_cap)).shape)
    loss = np.sum((W*W)*(Y*np.log(Y_cap)))
    return loss




def mainFunc(X, Y,     lambda0, lambda1, lambda2, lambda3, lambda4, lambda5,    MAXITER, ABSTOL, W_init, beta_init,paras_save_path=None,l1_use=False):

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
    beta_All = np.zeros([m,class_num, MAXITER])

    
    # Optimization with gradient descent
    for iter in tqdm(range(1,MAXITER+1)):
        
        # Update beta
        print (beta.shape)
        y = np.copy(beta)
        beta = beta + (iter/(iter+3))*(beta-beta_prev) # fast proximal gradient
        f_base = J_cost(W, beta, X, Y, lambda0, lambda1, lambda2, lambda3, lambda5)
        grad_beta = grad_CE(lambda0,W,X,Y,beta)                    +2*lambda3*beta
        while 1:
            if l1_use==False:
                z = beta - lambda_beta*grad_beta
            else:
                z = prox_l1(beta - lambda_beta*grad_beta, lambda_beta*lambda4)
            print ('beta',J_cost(W, z, X, Y, lambda0, lambda1, lambda2, lambda3, lambda5),f_base , np.sum(grad_beta.T@(z-beta)),            (1/(2*lambda_beta))*sum((z-beta)**2))
            #grad_beta.T@(z-beta)
            if J_cost(W, z, X, Y, lambda0, lambda1, lambda2, lambda3, lambda5)                <= f_base + np.sum(grad_beta.T@(z-beta))                + (1/(2*lambda_beta))*sum((z-beta)**2):
                #print ('test1')
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
        #print ('test1')
        grad_W = (2*np.sum(lambda0*Y*np.log(sigmoid(X@beta))*W,axis=1)).reshape(-1,1)                 +lambda1*balance_grad(W, X)@np.ones([m,1])                 +4*lambda2*W*W*W                 +4*lambda5*(sum(W*W)-1)*W
        #print ('test2')
        while 1:
            if l1_use==False:
                z = W-lambda_W*grad_W
            else:
                z = prox_l1(W-lambda_W*grad_W, 0)
            
            #print (z.shape)
            print ('W',J_cost(z, beta, X, Y, lambda0, lambda1, lambda2, lambda3, lambda5),                   f_base , np.sum(grad_W.T@(z-W)),                   (1/(2*lambda_W))*sum((z-W)**2))
            #grad_W.T@(z-W)
            if J_cost(z, beta, X, Y, lambda0, lambda1, lambda2, lambda3, lambda5)                    <= f_base + np.sum(grad_W.T@(z-W))+(1/(2*lambda_W+10e-100))*sum((z-W)**2):
                #print ('test2')
                break 
            lambda_W = parameter_iter*lambda_W
            if lambda_W<eps*eps*eps:
                break

        W_prev = y
        W = z
        
        W_All[:,iter-1] = W.ravel()
        beta_All[:,:,iter-1] = beta

        J_loss[iter-1] = J_cost(W, beta, X, Y,                              lambda0, lambda1, lambda2, lambda3, lambda5)                     + lambda4*sum(abs(beta))
        #print (J_loss[iter-1] , J_loss[iter-2])
        print (lambda0*cross_entropy_loss(W,X,Y,beta),            lambda1*sum(balance_cost(W,X)) ,            lambda2*np.sum((W*W).T@(W*W)) ,            lambda3*sum(beta**2) ,            lambda5*(sum(W*W)-1)**2, J_loss[iter-1]) 
        if (paras_save_path is not None) & (iter%10==1):
            #'output/models/crlr/somedir'
            if not os.path.exists(paras_save_path):
                os.makedirs(paras_save_path)
            np.savetxt(paras_save_path+'/beta.txt',beta)
            np.savetxt(paras_save_path+'/W.txt',W)
            np.savetxt(paras_save_path+'/J_loss.txt',J_loss)
        if (iter > 1) &(J_loss[iter-1] < J_loss[iter-2]) &( abs(J_loss[iter-1] - J_loss[iter-2])[0]  < ABSTOL) or (iter == MAXITER):
            break
    W = W*W

    return W, beta, J_loss





def prox_l1(v=None,lambda_=None):
    x=np.fmax(0,v - lambda_) - np.fmax(0,- v - lambda_)
    return x
#prox_l1输出的最终会成为小循环中的新beta




def J_cost(W,beta,X,Y,lambda0, lambda1, lambda2, lambda3, lambda5):
    term1 = lambda0*cross_entropy_loss(W,X,Y,beta)
    term2 = lambda1*sum(balance_cost(W,X)) 
    term3 = lambda2*np.sum((W*W).T@(W*W)) 
    term4 = lambda3*np.sum(beta**2) 
    term5 = lambda5*(sum(W*W)-1)**2
    print ('terms',term1 , term2 , term3 , term4 , term5 )
    return  term1 + term2 + term3 + term4 + term5 


def grad_CE(lambda0,W,X,Y,beta):
    '''
    part of J_loss's first term grad
    '''
    n = X.shape[0]
    Y_cap = sigmoid(X@beta)
    #original sigmoid(X@beta)-Y)*(W*W)).T@X
    grad = (lambda0*((Y_cap-Y)*(W*W)).T@X).T
    return grad #应该是正的还是负的？



feature_num = 20
sample_num = 500
class_num = 10
eps=10e-3
X = 2*np.round(np.random.rand(sample_num, feature_num))-1; # 1000 samples and 20 features
beta_true = np.random.random([feature_num, class_num]);
Y = np.random.randint(0,9,500).reshape(-1,1) #sigmoid(np.dot(X,beta_true))
lambda0 = 3; #Logistic loss
lambda1 = 0.1; #Balancing loss
lambda2 = 1; #L_2 norm of sample weight
lambda3 = 0; #L_2 norm of beta
lambda4 = 0.001; #L_1 norm of bata
lambda5 = 5; #Normalization of sample weight
MAXITER = 1000;
ABSTOL = 1e-3;


X_train, X_test, y_train, y_test = train_test_split(X,Y.astype('int'))
print('number of training samples: {}, test samples: {}'.format(X_train.shape[0], X_test.shape[0]))

W_init = np.random.rand(X_train.shape[0], 1);
beta_init = 0.5*np.ones([feature_num, class_num]);

W, beta, J_loss = mainFunc(X_train, y_train,        lambda0, lambda1, lambda2, lambda3, lambda4, lambda5,        1000, ABSTOL, W_init, beta_init,l1_use=False)






report_metrics(y_test, np.argmax(sigmoid(np.dot(X_test,beta)),axis=1))

