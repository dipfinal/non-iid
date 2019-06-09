import numpy as np
from sklearn import tree
from sklearn.preprocessing import normalize

min_samples_leaf = 10
max_depth = 11

def acc(Y,Y_):
    return 1.0*np.sum(Y==Y_)/Y.shape[0]
def err(Y,Y_):
    return 1.0*np.sum(Y!=Y_)/Y.shape[0]

def Classifier():
    return tree.DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf)


def Bagging(trainX,trainY,valX,*classifiers):
    if len(classifiers)==0:
        return
    num = len(classifiers)
    n = trainX.shape[0]
    w = 1.0/num
    df = pds.DataFrame(np.concatenate((trainX,trainY[:,np.newaxis]),axis=1))
    trainset = [df.sample(n,replace=True).to_numpy() for i in range(num)]
    for traindata,classifier in zip(trainset,classifiers):
        trainX = traindata[:,:-1]
        trainY = traindata[:,-1].squeeze().astype(np.int)
        classifier.train(trainX,trainY)
    valY_proba = np.zeros((valX.shape[0],2)).astype(np.float)
    for classifier in classifiers:
        valY_proba += classifier.pred_proba(valX)
    valY_proba*=w
    return valY_proba

def AdaBoostM1(trainX,trainY,valX,classifier,num=10):
    n = trainX.shape[0]
    ws = np.ones(n).astype(np.float)
    ws/=n
    df = pds.DataFrame(np.concatenate((trainX,trainY[:,np.newaxis],np.array(list(range(n)))[:,np.newaxis]),axis=1))
    classifiers = []
    betas = []
    for t in range(num):
        Data = df.sample(n,replace=True,weights=ws).to_numpy()
        trainX = Data[:,:-2]
        trainY = Data[:,-2].squeeze().astype(np.int)
        ID = Data[:,-1].squeeze().astype(np.int)
        clf = classifier.copy("{} {}".format(classifier.name,t))
        clf.train(trainX,trainY)
        Y_=clf.pred(trainX)
        sum_w = np.sum(ws[trainY!=Y_])
        if sum_w>0.5:
            break
        if sum_w == 0.0:
            classifiers = [clf]
            betas = [0.1]
            break
        betas.append(sum_w/(1-sum_w))
        classifiers.append(clf)
        changed = set()
        for i in range(n):
            if trainY[i]==Y_[i]:
                if ID[i] in changed:
                    continue
                else:
                    changed.add(ID[i])
                    ws[ID[i]]*=betas[-1]
        ws/=np.sum(ws)
    valY_proba = np.zeros((valX.shape[0],2))
    for classifier,beta in zip(classifiers,betas):
        valY_proba += math.log(1/max(beta,0.0000001))*classifier.pred_proba(valX)

    valsum= np.sum(valY_proba,axis=1)
    for i in range(valsum.shape[0]):
        valY_proba[i]/=valsum[i]
    return valY_proba



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
    
    clf.fit(trainX,trainY)
    print(acc(clf.predict(valX),valY))
