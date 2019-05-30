import scipy.io as scio
import numpy as np


data = scio.loadmat("./BoW_Training.mat")
data = data['data']
np.random.shuffle(data)
delbgds = [3,2,9,4,0,6,8,1,5,7]
n_feature = data.shape[1]-2
trainX = np.empty(shape=(0,n_feature))
valX = np.empty(shape=(0,n_feature))
trainY = np.empty(shape=(0,2))
valY = np.empty(shape=(0,2))
for sample in data:
    if delbgds[sample[50]]==sample[51]:
        valX = np.concatenate((valX,sample[np.newaxis,:50]))
        valY = np.concatenate((valY,sample[np.newaxis,-2:]))
    else:
        trainX = np.concatenate((trainX,sample[np.newaxis,:50]))
        trainY = np.concatenate((trainY,sample[np.newaxis,-2:]))

print("trainX.shape = ",trainX.shape)
print("trainY.shape = ",trainY.shape)
print("valX.shape = ",valX.shape)
print("valY.shape = ",valY.shape)
np.save("trainX.npy",trainX)
np.save("trainY.npy",trainY)
np.save("valX.npy",valX)
np.save("valY.npy",valY)