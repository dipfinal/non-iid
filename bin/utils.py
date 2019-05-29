#! /usr/bin/env python
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable # storing data while learning
from torch import optim
from torch.utils import data as utilsdata
import torch.nn.functional as F
from torch.distributions import Categorical
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from scipy.signal import argrelmax
import scipy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def mdn_logp(x, logpi, logsigma, mu):
    '''Loss function of a mixture density network is the negative log likelihood of a Gaussian mixture
        Args:
        x: Tensor of shape [batch_size, n_dim]
        logpi: Tensor of shape [batch_size, n_components]
        logsigma: Tensor of shape [batch_size, n_components, n_dim]
        mu: Tensor of shape [batch_size, n_components, n_dim]
        Returns:
        Log likelihoods of input samples. Tensor of shape [batch_size]
        '''
    batch_size, n_components, n_dim = logsigma.size()
    x = x.view(batch_size, -1, n_dim)
    logpi = logpi.view(batch_size, n_components, -1)
    var = torch.pow(torch.exp(logsigma), 2)
    #print(x.size(), logpi.size(), logsigma.size(), mu.size())
    ll_gaussian = -float(0.5*np.log(2*np.pi)) - logsigma - 0.5/var*torch.pow(x - mu, 2)
    ll = torch.logsumexp(ll_gaussian + logpi, 1)
    return ll

def mdn_loss(x, logpi, logsigma, mu):
    '''Same as mdn_logp except that the log likelihoods are negated and averaged across samples
        Returns:
        Negative log likelihood of input samples averaged over samples. A scalar.
        '''
    return torch.mean(-mdn_logp(x, logpi, logsigma, mu))






def preprocess(data,method='minmax'):
    if method =='minmax':
        scaler = MinMaxScaler()
        scaler.fit(data)
    elif method =='zscore':
        scaler = StandardScaler()
        scaler.fit(data)
    elif method =='robust':
        scaler = RobustScaler()
        scaler.fit(data)
    return scaler.transform(data),scaler

def preprocessing_pipeline(features,imputes=None,scales=None,feature_selection=None,
                           feature_remain_ratio=0.4):
    '''
        imputes: median, mean, most_frequent
        scales: minmax zscore robuts
        '''
    if imputes is not None:
        features_ = Imputer(axis=1,strategy=imputes).fit_transform(features)
    if scales is not None:
        features_ = preprocess(features_ ,method=scales)[0]
    if feature_selection is not None:
        if feature_selection=='var':
            selector = VarianceThreshold()
            selector.fit(features_)
            select_ind = np.where(selector.variances_>=np.percentile(selector.variances_,100*(1-feature_remain_ratio)))[0]
            features_ = features_[:,select_ind]
        if feature_selection=='missing':
            select_ind = np.argsort(np.isnan(features).sum(axis=0))[:int(features.shape[1]*feature_remain_ratio)]
            #print (select_ind.shape)
            features_  = features_[:,select_ind]
    #print (features_.shape)
    return features_, select_ind

def preprocess_zscore(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data),scaler

def get_original_parameters(logpi, logsigma, mu):
    '''
        input scaled and logged
        output exped, not reversed yet
        '''
    pi = np.exp(logpi.detach().numpy())
    sigma = np.exp(logsigma.detach().numpy())
    mu = mu.detach().numpy()
    return pi, sigma, mu

def report_metrics_regression(y_pred,y_test):
    rmse = np.mean(np.sum((y_pred - y_test)**2,axis=1)**0.5)
    pcc = scipy.stats.pearsonr(y_pred.ravel(),y_test.ravel())
    return rmse,pcc

def report_metrics_classification(y_pred,y_test):
    scorers = {'accuracy': accuracy_score,
           'recall': recall_score,
           'precision': precision_score,
           'f1': f1_score,
           'mcc': matthews_corrcoef
            }
    dicts = {}
    for metric in scorers.keys():
        print('{} = {}'.format(metric, scorers[metric](y_test, y_pred)))
        dicts[metric] = scorers[metric](y_test, y_pred)
    return rmse,pcc

def prepare_dataset(datafile='data/BoW_Training.mat',valid_method = 'classic',train_context_num=5):
    import scipy.io as sio
    BoW_Training = sio.loadmat(datafile)['data']
    print (BoW_Training.shape)
    BoW_Training_x = BoW_Training[:,:50]
    BoW_Training_y = BoW_Training[:,-2:]
    if valid_method == 'classic':
        return train_test_split(BoW_Training_x, BoW_Training_y, test_size=0.2, random_state=42)
    elif valid_method =='non-iid':
        '''
        split the dataset so for each class, split 7 contexts into 5:2, 
        predict 2 contexts corresponding class
        '''
        tmp_select_ind = np.array([]).astype('int')
        for i in np.unique(BoW_Training_y[:,0]):
            #print (np.unique(BoW_Training_y[:,1][BoW_Training_y[:,0] ==i]))
            tmp_context_ind = np.random.choice(np.unique(BoW_Training_y[:,1][BoW_Training_y[:,0] ==i]),
                             train_context_num,replace=False,)
            tmp_select_ind = np.concatenate((tmp_select_ind,np.where( (BoW_Training_y[:,0]==i)& 
                                (np.isin(BoW_Training_y[:,1],tmp_context_ind )==1) )[0]  ))

        train_ind = tmp_select_ind
        test_ind = np.setdiff1d(np.arange(0,BoW_Training_y.shape[0]),tmp_select_ind)
        return BoW_Training_x[train_ind],BoW_Training_x[test_ind],BoW_Training_y[train_ind],\
                BoW_Training_y[test_ind],train_ind,test_ind

def oneHotEncoding(y, numOfClasses):
    """
    Convert a vector into one-hot encoding matrix where that particular column value is 1 and rest 0 for that row.
    :param y: Label vector
    :param numOfClasses: Number of unique labels
    :return: one-hot encoding matrix
    """
    y = np.asarray(y, dtype='int32')
    if len(y) > 1:
        y = y.reshape(-1)
    if not numOfClasses:
        numOfClasses = np.max(y) + 1
    yMatrix = np.zeros((len(y), numOfClasses))
    yMatrix[np.arange(len(y)), y] = 1
    return yMatrix
    