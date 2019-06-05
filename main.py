import numpy as np
from Classifiers import *
from sklearn.model_selection import train_test_split
from CR_softmax import mainFunc
def accuracy(Y,Y_):
    return 1.0*np.sum(Y==Y_)/Y.shape[0]

if __name__ =="__main__":

    trainX = np.load("./trainX.npy")
    trainY = np.load("./trainY.npy")
    valX = np.load("./valX.npy")
    valY = np.load("./valY.npy")

    """
    n_sample = 6000
    n_feature = 50
    X = 2 * np.round(np.random.rand(n_sample, n_feature)) - 1
    beta_true = np.ones([n_feature, 1])
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    Y = (sigmoid(np.dot(X, beta_true)) >= 0.5).astype('double')
    #print(Y.shape)
    trainX, valX, trainY, valY = train_test_split(X, Y.astype('int'))
    print('number of training samples: {}, test samples: {}'.format(trainX.shape[0], valX.shape[0]))
    """
    clf = NN()
    clf.train(trainX,trainY)

    valY_ = clf.predict(valX)
    print(valY_)
    print(accuracy(valY_,valY[:,0]))

    trainY_ = clf.predict(trainX)
    print(accuracy(trainY_,trainY[:,0]))
