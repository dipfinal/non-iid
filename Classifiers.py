from classifier import Classifier
import numpy as np
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from CRLR import mainFunc
class KNNClassifier(Classifier):
    def __init__(self):
        pass

    def train(self,X,Y):
        self.Xs = [np.empty(shape=[0,50]) for i in range(10)]
        for x,y in zip(X,Y):

            x = x.astype(float)
            x/= np.sum(x)

            self.Xs[y[0]] = np.concatenate((self.Xs[y[0]],x[np.newaxis,:]))


    def predict(self,X):
        Y = np.empty(shape=[0])
        def cost(sampleA,sampleB):
            return np.sum(np.square(sampleA - sampleB))
        def Loss(xs,x):
            loss = 0.0
            for _x in xs:
                loss+=cost(_x,x)
            return loss
        for x in X:
            assert x.shape == (50,)
            loss = np.zeros(shape=(10),dtype=np.float)
            for i in range(10):
                loss[i]+=Loss(self.Xs[i],x)/self.Xs[i].shape[0]
            label = np.argmin(loss)
            #print(label)
            Y = np.concatenate((Y,np.array([label])))

        return Y

class knnClassifier(Classifier):
    def __init__(self):
        pass

    def train(self,X,Y):
        self.clf = KNeighborsClassifier(n_neighbors=200)
        self.clf.fit(X,Y[:,0])

    def predict(self,X):
        return self.clf.predict(X)

class DTClassifier(Classifier):
    def __init__(self):
        pass
    def train(self,X,Y):
        self.clf = tree.DecisionTreeClassifier(min_samples_split=20)
        self.clf.fit(X,Y[:,0])
    def predict(self,X):
        return self.clf.predict(X)

class SVMClassifier(Classifier):
    def __init__(self):
        pass
    def train(self,X,Y):
        self.clfs = [SVC(gamma="auto",kernel='rbf',class_weight='balanced',probability=True) for i in range(10)]
        for i in range(10):
            Y_ = Y[:,0].copy()
            for j in range(Y_.shape[0]):
                if(Y_[j]==i):
                    Y_[j]==1
                else:
                    Y_[j]==0
            print(Y_.shape)
            self.clfs[i].fit(X,Y_)
    def predict(self,X):
        Y = np.empty(shape=(X.shape[0],0))
        for clf in self.clfs:
            Y = np.concatenate((Y,clf.predict_proba(X)[:,1][:,np.newaxis]),axis=1)

        Y = np.argmax(Y,axis=1)
        return Y

class CRLR_SVM(Classifier):
    lambda0 = 1  # Logistic loss
    lambda1 = 0.1  # Balancing loss
    lambda2 = 1  # L_2 norm of sample weight
    lambda3 = 0  # L_2 norm of beta
    lambda4 = 0.001  # L_1 norm of bata
    lambda5 = 1  # Normalization of sample weight
    MAXITER = 1000
    ABSTOL = 1e-3

    def __init(self):
        pass
    def train(self,X,Y):
        self.clfs = [SVC(gamma="auto", kernel='rbf', class_weight='balanced', probability=True) for i in range(10)]
        for i in range(10):
            Y_ = Y[:, 0].copy()
            for j in range(Y_.shape[0]):
                if (Y_[j] == i):
                    Y_[j] == 1
                else:
                    Y_[j] == 0
            #print(Y_.shape)
            Y_ = Y_[:,np.newaxis]
            n_sample = X.shape[0]
            n_feature = X.shape[1]
            W_init = np.random.rand(n_sample, 1)
            beta_init = 0.5 * np.ones([n_feature, 1])
            W, beta, J_loss = mainFunc(X, Y_, \
                                       self.lambda0, self.lambda1, self.lambda2, self.lambda3, self.lambda4, self.lambda5, \
                                       1000, self.ABSTOL, W_init, beta_init)

            for i in range(n_sample):
                for j in range(n_feature):
                    X[i][j] *= beta[j]
            self.clfs[i].fit(X, Y_,sample_weight=np.squeeze(beta))

    def predict(self, X):
        Y = np.empty(shape=(X.shape[0], 0))
        for clf in self.clfs:
            Y = np.concatenate((Y, clf.predict_proba(X)[:, 1][:, np.newaxis]), axis=1)

        Y = np.argmax(Y, axis=1)
        return Y

