#! /usr/bin/env python
import sys,argparse,os,time
sys.path.append('bin')

parser = argparse.ArgumentParser(description='test some parameters')
parser.add_argument('--n_components', dest='n_components',default=6,type=int, help='number of gaussians')
parser.add_argument('--batch', dest='batchsize',default=10,type=int, help='number of samples in one batch')
parser.add_argument('--early_epoch', dest='earlystopping_epoch_nums',default=20,type=int, help='earlystopping_epoch_nums')

parser.add_argument('--logsigmamin', dest='logsigmamin',default=-3,type=int, help='lower bound of logsigma')
parser.add_argument('--logsigmamax', dest='logsigmamax',default=3,type=int, help='upper bound of logsigma')

parser.add_argument('--weight_cut_threshold', dest='weight_cut_threshold',default=0.1,type=float, help='threshold for attention weight')
parser.add_argument('--feature_remain_ratio_RSS', dest='feature_remain_ratio_RSS',default=0.4,type=float, help='feature_remain_ratio_RSS')
parser.add_argument('--feature_remain_ratio_TOA', dest='feature_remain_ratio_TOA',default=0.4,type=float, help='feature_remain_ratio_TOA')
parser.add_argument('--feature_remain_ratio_DOA', dest='feature_remain_ratio_DOA',default=0.1,type=float, help='feature_remain_ratio_DOA')
parser.add_argument('--imputes_method', dest='imputes_method',default='median',type=str, help='imputes_method')
parser.add_argument('--feature_scales_method', dest='feature_scales_method',default='minmax',type=str, help='feature_scales_method')
parser.add_argument('--position_scales_method', dest='position_scales_method',default='mimax',type=str, help='position_scales_method')
parser.add_argument('--feature_selection_method', dest='feature_selection_method',default='missing',type=str, help='feature_selection_method')

#parser.add_argument('--nhidden1', dest='nhidden1',default=22,type=int, help='hidden1 units')
#parser.add_argument('--nhidden2', dest='nhidden2',default=20,type=int, help='hidden2 units')
#parser.add_argument('--nhidden3', dest='nhidden3',default=18,type=int, help='hidden3 units')
#parser.add_argument('--nhidden4', dest='nhidden2',default=20,type=int, help='hidden2 units')
#parser.add_argument('--nhidden5', dest='nhidden3',default=18,type=int, help='hidden3 units')
args = parser.parse_args()

nhidden1 = 64
nhidden2 = 48
nhidden3 = 36
nhidden4 = 28
nhidden5 = 24

imputes_method = args.imputes_method
feature_scales_method = args.feature_scales_method
position_scales_method = args.position_scales_method
feature_selection_method = args.feature_selection_method
components_num = args.n_components
weight_cut_threshold = args.weight_cut_threshold
feature_remain_ratio_RSS = args.feature_remain_ratio_RSS
feature_remain_ratio_TOA = args.feature_remain_ratio_TOA
feature_remain_ratio_DOA = args.feature_remain_ratio_DOA
logsigmamin = args.logsigmamin
logsigmamax = args.logsigmamax
batchsize = args.batchsize   #10, 20
earlystopping_epoch_nums = args.earlystopping_epoch_nums

dir_name = ('impute_'+imputes_method+
            '_feature_scales_'+feature_scales_method+
            '_position_scales_'+position_scales_method+
            '_feature_selection_'+feature_selection_method+
            '_weight_cut_'+str(weight_cut_threshold)+
            '_RSS_'+str(feature_remain_ratio_RSS)+
            '_TOA_'+str(feature_remain_ratio_TOA)+
            '_DOA_'+str(feature_remain_ratio_DOA)+
            '_batch_'+str(args.batchsize)+
            '_gaussian_'+str(args.n_components)+
            '_batch_'+str(args.batchsize)+
            '_earlyepoch_'+str(args.earlystopping_epoch_nums)+
            '_logsigma_min_max_'+str(args.logsigmamin)+
            '_'+str(args.logsigmamax))
path_model = os.path.join('models_attention/', dir_name)

if os.path.exists(path_model+'/paras'):
    print ('parameter combinations tested, skip')
    sys.exit()




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
from utils import mdn_loss, mdn_logp
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler,Imputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from utils import prepare_dataset, preprocess,preprocess_zscore, get_original_parameters, report_metrics, mdn_logp, mdn_loss
from model import MixtureDensityNetwork_attention, IsotropicGaussianMixture

with h5py.File('data/citydata.h5') as f:
    TOA = f['TOA'][:]
    DOA = f['DOA'][:]
    RSS = f['RSS'][:]
    RXloc_shifted_  = f['RXloc_shifted'][:]
    TXloc_shifted_ = f['TXloc_shifted'][:]
    weight = f['weight'][:]

feature_select, select_ind = preprocessing_pipeline(RSS,imputes=imputes_method,scales=feature_scales_method,
                                                    feature_selection=feature_selection_method,
                                                    feature_remain_ratio=feature_remain_ratio_RSS)
feature_select_, select_ind_ = preprocessing_pipeline(TOA,imputes=imputes_method,scales=feature_scales_method,
                                                      feature_selection=feature_selection_method,
                                                      feature_remain_ratio=feature_remain_ratio_TOA)
feature_select__, select_ind__ = preprocessing_pipeline(DOA,imputes=imputes_method,scales=feature_scales_method,
                                                        feature_selection=feature_selection_method,
                                                        feature_remain_ratio=feature_remain_ratio_DOA)
feature_combined = np.concatenate((feature_select,feature_select__,feature_select_),axis=1)

# 保持TOA的是最后的feature

indices = np.arange(1500)
X_train, X_test, y_train, y_test, indices_train,indices_test = train_test_split(feature_combined,
                                                                                RXloc_shifted_, indices,random_state=42)

scalers = {}
datas = [X_train, X_test, y_train, y_test]
for i in range(4):
    datas[i],scalers[i]  = preprocess(datas[i],position_scales_method) #minmax  zscore robust
X_train_, X_test_, y_train_, y_test_ = datas
#X_train_, X_test_, y_train_, y_test_ = add_channel(X_train_), add_channel(X_test_), \
#                                    add_channel(y_train_), add_channel(y_test_)
batch_size = batchsize
train_ = utilsdata.TensorDataset(torch.from_numpy(X_train_.astype('float32')),
                                 torch.from_numpy(y_train_.astype('float32')))
test_ = utilsdata.TensorDataset(torch.from_numpy(X_test_.astype('float32')),
                                torch.from_numpy(y_test_.astype('float32')))
train_loader_ = torch.utils.data.DataLoader(
                                            dataset=train_,
                                            batch_size=batch_size,
                                            shuffle=True)
test_loader_ = torch.utils.data.DataLoader(
                                           dataset=test_,
                                           batch_size=batch_size,
                                           shuffle=False)

#print('X_train.shape =', X_train_.shape, 'X_test.shape =', X_test_.shape,
#    'y_train.shape =', y_train_.shape, 'y_test.shape =', y_test_.shape)


model = MixtureDensityNetwork_attention(n_input=X_train.shape[1],
                    feature_thres_ind = -int(feature_remain_ratio_TOA*100),
                    thres_cut = weight_cut_threshold,
                    n_output=2,  n_components = components_num,
                    n_hiddens=[nhidden1,nhidden2,nhidden3,nhidden4,nhidden5],
                    logsigma_min=logsigmamin,logsigma_max=logsigmamax)
optimizer = optim.Adam(model.parameters())





trainlosses, testlosses = {},{}
for epoch in tqdm(range(2000)):
    train_loss = []
    for i_batch, batch_data in enumerate(train_loader_):
        x, y = batch_data
        #print (x.size())
        model.zero_grad()
        logpi, logsigma, mu, _ = model(x)
        loss = mdn_loss(y, logpi, logsigma, mu)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item()*x.size()[0])
    train_loss = np.sum(train_loss)/len(train_loader_.dataset)

test_loss = []
with torch.no_grad():
    for i_batch, batch_data in enumerate(test_loader_):
        x, y = batch_data
            logpi, logsigma, mu, _ = model(x)
            loss = mdn_loss(y, logpi, logsigma, mu)
            test_loss.append(loss.item()*x.size()[0])
        test_loss = np.sum(test_loss)/len(test_loader_.dataset)
    trainlosses[epoch] = train_loss
    testlosses[epoch] = test_loss
    if epoch%10 == 0:
        print('[Epoch {:d}] train loss: {}, test loss: {}'.format(epoch, train_loss, test_loss))
    ###### early stop to avoid unnecessary training######
    if epoch >300:
        if epoch%10 == 0:
            recentlossmin = np.min(np.array([testlosses[i] for i in np.arange(epoch-earlystopping_epoch_nums,epoch)]))
            otherlossmin = np.min(np.array([testlosses[i] for i in np.arange(0,epoch-earlystopping_epoch_nums)]))
            print (recentlossmin,otherlossmin)
            if recentlossmin > otherlossmin: # no longer decrease
                print ('exist at epoch:' +str(epoch))
                break



if not os.path.exists(path_model):
    os.makedirs(path_model)
torch.save(model.state_dict(), path_model+'/model')


logpi_pred, logsigma_pred, mu_pred, attention = model(torch.Tensor(X_test_))
pi_reversed, sigma_reversed, mu_reversed = get_original_parameters(logpi_pred, logsigma_pred, mu_pred)

def get_prediction(pi,mu,sigma,n_components,n_dim=3):
    model = IsotropicGaussianMixture(n_components, n_dim=n_dim)
    model.set_params(pi,mu,sigma)
    modes = model.find_modes(n_init=10)
    p_modes = model.pdf(modes)
    #print p_modes,modes
    index=np.where(p_modes==np.max(p_modes))
    return p_modes[index[0]], modes[index[0]]

prediction_xy = np.ndarray([pi_reversed.shape[0],2])
probabes = np.ndarray([pi_reversed.shape[0]])
for i in tqdm(range(pi_reversed.shape[0])):
    probabes[i], prediction_xy[i] = get_prediction(pi_reversed[i], mu_reversed[i],sigma_reversed[i],components_num,2)
prediction_xy_reverse = scalers[3].inverse_transform(prediction_xy)
rmse = np.array([report_metrics(prediction_xy_reverse,y_test)[0]]))
pcc = np.array([report_metrics(prediction_xy_reverse,y_test)[1][0]])
print ('RMSE: ', rmse)
print ('PCC: ', pcc)

adaptive_weight  = np.concatenate((weight[:100][select_ind],weight[200:][select_ind__],weight[100:200][select_ind_]))

parametersinfo = np.array(['impute_'+imputes_method+
                           '_feature_scales_'+feature_scales_method+
                           '_position_scales_'+position_scales_method+
                           '_feature_selection_'+feature_selection_method+
                           '_weight_cut_'+str(weight_cut_threshold)+
                           '_RSS_'+str(feature_remain_ratio_RSS)+
                           '_TOA_'+str(feature_remain_ratio_TOA)+
                           '_DOA_'+str(feature_remain_ratio_DOA)+
                           '_batch_'+str(args.batchsize)+
                           '_gaussian_'+str(args.n_components)+
                           '_batch_'+str(args.batchsize)+
                           '_earlyepoch_'+str(args.earlystopping_epoch_nums)+
                           '_logsigma_min_max_'+str(args.logsigmamin)+
                           '_'+str(args.logsigmamax)])

if not os.path.exists(path_model+'/paras'):
    os.makedirs(path_model+'/paras')

with h5py.File(path_model+'/paras/summary.h5') as f:
    f.create_dataset('rmse',data=rmse)
    f.create_dataset('pcc',data=pcc)
    f.create_dataset('probabes',data=probabes)
    f.create_dataset('parametersinfo',data=parametersinfo)
    f.create_dataset('attention',data=attention.detach().numpy())
    f.create_dataset('adaptive_weight',data=adaptive_weight)
    f.create_dataset('prediction_xy_reverse',data=prediction_xy_reverse)
    f.create_dataset('y_test',data=y_test)
