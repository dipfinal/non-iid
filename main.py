import numpy as np
from Classifiers import *
from CRLR import mainFunc
def accuracy(Y,Y_):
    return 1.0*np.sum(Y==Y_)/Y.shape[0]

if __name__ =="__main__":
    trainX = np.load("./trainX.npy")
    trainY = np.load("./trainY.npy")
    valX = np.load("./valX.npy")
    valY = np.load("./valY.npy")
    clf = CRLR_SVM()
    clf.train(trainX,trainY)
    Y = clf.predict(valX)
    print(Y)

    #clf = SVMClassifier()
    #clf.train(trainX,trainY)
    #Y = clf.predict(valX)
    print(accuracy(Y,valY[:,0]))