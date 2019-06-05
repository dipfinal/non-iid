DEBUG = False
from classifier import Classifier
import numpy as np
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from CRLR import mainFunc
import CR_softmax
from math import fabs
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
class MyKNNClassifier(Classifier):
    def __init__(self,n_feature=50,n_label=10):
        super(MyKNNClassifier,self).__init__()
        self.n_feature = n_feature
        self.n_label = n_label

    def train(self,X,Y,*args,**kwargs):
        X = X.astype(np.float)
        X = normalize(X,norm='l1')
        self.Xs = [np.empty(shape=[0,self.n_feature]) for i in range(self.n_label)]
        for x,y in zip(X,Y):
            self.Xs[y[0]] = np.concatenate((self.Xs[y[0]],x[np.newaxis,:]))


    def predict(self,X,*args,**kwargs):
        X = X.astype(np.float)
        X = normalize(X,norm='l1')
        Y = np.empty(shape=[0]).astype(np.int)
        def cost(sampleA,sampleB):
            return np.sum(np.square(sampleA - sampleB))
        def Loss(xs,x):
            loss = 0.0
            for _x in xs:
                loss+=cost(_x,x)
            return loss
        for x in X:
            loss = np.zeros(shape=(self.n_label),dtype=np.float)
            for i in range(self.n_label):
                loss[i]+=Loss(self.Xs[i],x) / self.Xs[i].shape[0]
            label = np.argmin(loss)
            Y = np.concatenate((Y,np.array([label])))

        return Y

class KNNClassifier(Classifier):
    def __init__(self,n_feature=50,n_label=10):
        super(KNNClassifier,self).__init__()
        self.n_feature = n_feature
        self.n_label = n_label

    def train(self,X,Y,*args,**kwargs):
        X = X.astype(np.float)
        X = normalize(X, norm='l1',axis=1)
        self.clf = KNeighborsClassifier(n_neighbors=140)
        self.clf.fit(X,Y[:,0])

    def predict(self,X,*args,**kwargs):
        X = X.astype(np.float)
        X = normalize(X, norm='l1',axis=1)
        return self.clf.predict(X)

class DTClassifier(Classifier):
    def __init__(self,n_feature=50,n_label=10):
        super(DTClassifier,self).__init__()
        self.n_feature = n_feature
        self.n_label = n_label
    def train(self,X,Y,*args,**kwargs):
        X = X.astype(np.float)
        X = normalize(X, norm='l1',axis=1)
        self.clf = tree.DecisionTreeClassifier(min_samples_split=35)
        self.clf.fit(X,Y[:,0])
    def predict(self,X,*args,**kwargs):
        X = X.astype(np.float)
        X = normalize(X, norm='l1', axis=1)
        return self.clf.predict(X)

class SVMClassifier(Classifier):
    def __init__(self,n_feature=50,n_label=10):
        super(SVMClassifier,self).__init__()
        self.n_feature = n_feature
        self.n_label = n_label
    def train(self,X,Y,*args,**kwargs):
        X = X.astype(np.float)
        X = normalize(X, norm='l1',axis=1)
        self.clfs = [SVC(gamma="auto",kernel='rbf',class_weight='balanced',probability=True) for i in range(self.n_label)]
        for i in range(self.n_label):
            Y_ = Y[:,0].copy()
            for j in range(Y_.shape[0]):
                if(Y_[j] == i):
                    Y_[j] = 1
                else:
                    Y_[j] = 0
            #print(Y_.shape)
            self.clfs[i].fit(X,Y_)
    def predict(self,X,*args,**kwargs):
        X = X.astype(np.float)
        X = normalize(X, norm='l1', axis=1)
        Y = np.empty(shape=(X.shape[0],0))
        for clf in self.clfs:
            Y = np.concatenate((Y,clf.predict_proba(X)[:,1][:,np.newaxis]),axis=1)

        Y = np.argmax(Y,axis=1)
        return Y

class CRLR(Classifier):
    lambda0 = 1  # Logistic loss
    lambda1 = 0.1  # Balancing loss
    lambda2 = 1  # L_2 norm of sample weight
    lambda3 = 0  # L_2 norm of beta
    lambda4 = 0.001  # L_1 norm of bata
    lambda5 = 1  # Normalization of sample weight
    # MAXITER = 1000
    if DEBUG:
        MAXITER = 10
    else:
        MAXITER = 1000
    ABSTOL = 1e-3

    def __init__(self, n_feature=50, n_label=10):
        super(CRLR, self).__init__()
        self.n_feature = n_feature
        self.n_label = n_label
    def train(self,X,Y,load=False,*args,**kwargs):
        self.betas = []
        X = X.astype(np.float)
        X = normalize(X, norm='l1', axis=1)
        if load:
            for i in range(self.n_label):
                beta = np.load("./CRLR_betas/beta{}.npy".format(i))
                self.betas.append(beta)
            return
        n_sample,n_feature = X.shape
        for i in range(self.n_label):
            Y_ = Y[:, 0].copy()
            for j in range(Y_.shape[0]):
                if (Y_[j] == i):
                    Y_[j] = 1
                else:
                    Y_[j] = 0
            Y_ = Y_[:,np.newaxis]
            W_init = np.random.rand(n_sample, 1)
            beta_init = 0.5 * np.ones([n_feature, 1])
            W, beta, J_loss = mainFunc(X, Y_, \
                                       self.lambda0, self.lambda1, self.lambda2, self.lambda3, self.lambda4, self.lambda5, \
                                       self.MAXITER, self.ABSTOL, W_init, beta_init)
            np.save("./CRLR_betas/beta{}.npy".format(i),beta)
            np.save("./CRLR_ws/w{}.npy".format(i),W)
            self.betas.append(beta)
    def predict(self, X,*args,**kwargs):
        X = X.astype(np.float)
        X = normalize(X, norm='l1', axis=1)
        Y = np.empty(shape=(X.shape[0], 0)).astype(np.int)
        def sigmoid(x):
            return 1 / (np.exp(-x) + 1)
        for beta in self.betas:
            Y = np.concatenate((Y, sigmoid(np.dot(X,beta))), axis=1)
        Y = np.argmax(Y, axis=1)
        return Y

class CRLR_SVM(Classifier):
    lambda0 = 1  # Logistic loss
    lambda1 = 0.1  # Balancing loss
    lambda2 = 1  # L_2 norm of sample weight
    lambda3 = 0  # L_2 norm of beta
    lambda4 = 0.001  # L_1 norm of bata
    lambda5 = 1  # Normalization of sample weight
    if DEBUG:
        MAXITER = 10
    else:
        MAXITER = 1000
    threshold = 0.01
    ABSTOL = 1e-3

    def __init__(self, n_feature=50, n_label=10):
        super(CRLR_SVM, self).__init__()
        self.n_feature = n_feature
        self.n_label = n_label
    def train(self,X,Y,load=False):
        self.betas = []
        self.ws = []
        X = X.astype(np.float)
        X = normalize(X, norm='l1', axis=1)
        _X = X.copy()
        _Y = Y.copy()
        if load:
            for i in range(self.n_label):
                beta = np.load("./CRLR_betas/beta{}.npy".format(i))
                self.betas.append(beta)
            for i in range(self.n_label):
                w = np.load("./CRLR_ws/w{}.npy".format(i))
                self.ws.append(w)
        else:
            for i in range(self.n_label):
                Y_ = Y[:, 0].copy()
                for j in range(Y_.shape[0]):
                    if (Y_[j] == i):
                        Y_[j] = 1
                    else:
                        Y_[j] = 0
                #print(Y_.shape)
                Y_ = Y_[:,np.newaxis]
                n_sample = X.shape[0]
                n_feature = X.shape[1]
                W_init = np.random.rand(n_sample, 1)
                beta_init = 0.5 * np.ones([n_feature, 1])
                W, beta, J_loss = mainFunc(X, Y_, \
                                       self.lambda0, self.lambda1, self.lambda2, self.lambda3, self.lambda4, self.lambda5, \
                                       self.MAXITER, self.ABSTOL, W_init, beta_init)
                np.save("./CRLR_betas/beta{}.npy".format(i),beta)
                np.save("./CRLR_ws/w{}.npy".format(i),W)
                self.betas.append(beta)
                self.ws.append(W)
        X = _X
        Y = _Y
        self.clfs = [SVC(gamma="auto", kernel='rbf', class_weight='balanced', probability=True) for i in range(self.n_label)]
        n_sample,n_feature = X.shape
        for i in range(self.n_label):
            Y_ = Y[:, 0].copy()
            X_ = X.copy()
            for j in range(Y_.shape[0]):
                if (Y_[j] == i):
                    Y_[j] = 1
                else:
                    Y_[j] = 0
            beta = self.betas[i]
            w = self.ws[i]
            for j in range(n_feature):
                if fabs(beta[j][0]) < self.threshold:
                    X_[:,j] = 0
            self.clfs[i].fit(X_, Y_,sample_weight=w[:,0])

    def predict(self, X):
        X = X.astype(np.float)
        X = normalize(X, norm='l1', axis=1)
        Y = np.empty(shape=(X.shape[0], 0)).astype(np.int)
        for clf in self.clfs:
            Y = np.concatenate((Y, clf.predict_proba(X)), axis=1)
        Y = np.argmax(Y, axis=1)
        return Y

class CRLR_softmax(Classifier):
    lambda0 = 1  # Logistic loss
    lambda1 = 0.1  # Balancing loss
    lambda2 = 1  # L_2 norm of sample weight
    lambda3 = 0  # L_2 norm of beta
    lambda4 = 1e-5  # L_1 norm of bata
    lambda5 = 1  # Normalization of sample weight
    # MAXITER = 1000
    if DEBUG:
        MAXITER = 10
    else:
        MAXITER = 1000
    ABSTOL = 0

    def __init__(self, n_feature=50, n_label=10):
        super(CRLR_softmax, self).__init__()
        self.n_feature = n_feature
        self.n_label = n_label
    def train(self,X,Y,load=False,*args,**kwargs):
        X = X.astype(np.float)
        X = normalize(X, norm='l1', axis=1)
        if load:
            self.beta = np.load("./CRLR_betas/beta_softmax.npy")
            self.W = np.load("./CRLR_ws/w.npy")
            return
        self.n_sample = X.shape[0]
        W_init = np.random.rand(self.n_sample,1)
        beta_init = 0.5 * np.ones([self.n_feature,1])
        print(X.shape)
        print(Y.shape)
        self.W, self.beta, J_loss = CR_softmax.mainFunc(X, Y[:,:1], \
                                       self.lambda0, self.lambda1, self.lambda2, self.lambda3, self.lambda4, self.lambda5, \
                                       self.MAXITER, self.ABSTOL, W_init, beta_init)
        np.save("./CRLR_betas/beta_softmax.npy",self.beta)
        np.save("./CRLR_ws/w.npy",self.W)
    def predict(self, X,*args,**kwargs):
        X = X.astype(np.float)
        X = normalize(X, norm='l1', axis=1)
        Y = np.empty(shape=(X.shape[0], 0)).astype(np.int)
        def sigmoid(x):
            return 1 / (np.exp(-x) + 1)
        
        Y = (sigmoid(np.dot(X,self.beta)) >= 0.5).astype('int')
        return np.squeeze(Y)

class MyNN(torch.nn.Module):

    def __init__(self,n_feature,n_hidden,n_label):
        super(MyNN,self).__init__()
        self.net1 = torch.nn.Sequential(
            nn.Linear(n_feature,n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden,n_label)            
        )
        self.net2 = torch.nn.Sequential(
            nn.Linear(n_feature,n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden,n_label)            
        )
        self.c = Variable(torch.ones(n_feature).type(torch.FloatTensor),requires_grad=False)
        self.w = Variable(torch.randn(n_feature).type(torch.FloatTensor),requires_grad=True)
        self.half = Variable(torch.ones(n_feature).type(torch.FloatTensor)/2,requires_grad=False)
        self.beta1 = torch.clamp(self.w,float(0.0),float(1.0)).type(torch.FloatTensor)
        self.beta2 = torch.clamp(self.c-self.w,float(0.0),float(1.0)).type(torch.FloatTensor)
    def forward(self,x):    # (-1,n_feature)
        x1 = x.type(torch.FloatTensor) * self.w#self.beta1     # (-1,n_feature)
        x2 = x.type(torch.FloatTensor) * (self.c-self.w)#self.beta2
        y1 = self.net1(x1)               # (-1,n_label)
        y2 = self.net2(x2)
        return y1,y2
        
class NN(Classifier):
    if DEBUG:
        epoches = 1
    else:
        epoches = 1000
    batch_size =128
    learning_rate = 0.1
    lambda1 = 1.0
    lambda2 = 1.0
    lambda3 = 1.0
    #gpu = torch.cuda.is_available()
    def __init__(self,n_feature=50,n_label=10):
        super(NN,self).__init__()
        self.n_feature=n_feature
        self.n_label = n_label
        self.net = MyNN(n_feature,10,n_label)
        #if gpu:
        #    self.net = self.net.cuda()
    def train(self,X,Y,*args,**kwargs): #X: (n_sample,n_feature) Y: (n_sample,2)
        X = X.astype(np.float)
        X = normalize(X, norm='l1', axis=1)
        

        optimizer = torch.optim.Adam([
                {'params':self.net.net1.parameters(),'weight_decay':self.lambda1},
                {'params':self.net.net2.parameters(),'weight_decay':self.lambda2},
                {'params':self.net.w}
            ],lr=self.learning_rate)
        MSE = nn.MSELoss()
        CrossEntropy = nn.CrossEntropyLoss()
        #if gpu:
        #    MSE = MSE.cuda()
        #    CrossEntropy = CrossEntropy.cuda()
        dataset = Data.TensorDataset(torch.from_numpy(X),torch.from_numpy(Y))
        train_loader = Data.DataLoader(dataset=dataset,batch_size=self.batch_size,shuffle=True,num_workers=4)
            
        for epoch in range(self.epoches):
            print("epoch = ",epoch)
            Flag = False
            for batch_x,batch_y in train_loader:
                #if gpu:
                #    batch_x = batch_x.cuda()
                y1,y2 = self.net(batch_x)
                y1_,y2_ = batch_y[:,0],batch_y[:,1]
                #if gpu:
                #    y1_ = y1_.cuda()
                #    y2_ = y2_.cuda()
                #loss = CrossEntropy(y1,y1_)
                #loss = -self.lambda3 * MSE(self.net.half,self.net.w)
                loss = CrossEntropy(y1,y1_)+CrossEntropy(y2,y2_)
                if not Flag:
                    print(loss)
                    Flag = True
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
        
    def predict(self,X,*args,**kwargs):
        y1,y2 = self.net(torch.from_numpy(X).type(torch.FloatTensor))     # -1,2*n_feature
        #print(Y)
        y1 = y1.data.numpy()
        pred = np.argmax(y1,axis=1)
        return pred
        

            
        
           
