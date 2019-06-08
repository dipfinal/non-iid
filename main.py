import numpy as np
from sklearn import tree
from sklearn.preprocessing import normalize

min_samples_leaf = 10
max_depth = 11

def acc(Y,Y_):
    return 1.0*np.sum(Y==Y_)/Y.shape[0]
def err(Y,Y_):
    return 1.0*np.sum(Y!=Y_)/Y.shape[0]

if __name__ =="__main__":

    trainX = np.load("./trainX.npy")
    trainY = np.load("./trainY.npy")
    valX = np.load("./valX.npy")
    valY = np.load("./valY.npy")
    trainX = trainX.astype(np.float)
    trainX = normalize(trainX, norm='l1',axis=1)
    valX = valX.astype(np.float)
    valX = normalize(valX, norm='l1',axis=1)
    trainY = trainY[:,0]
    valY = valY[:,0]
    
    delfeature = []
    with open("./best_beta.txt")as file:
        for id,num in enumerate(file):
            num = num.strip()
            num = int(float(num))
            if num==1:
                delfeature.append(id)
    trainX[:,delfeature] = 0
    valX[:,delfeature] = 0
    clf = tree.DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf)
    clf.fit(trainX,trainY)
    print(acc(clf.predict(valX),valY))
