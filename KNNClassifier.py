from classifier import Classifier
import numpy as np
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

