import scipy.io as scio
import numpy as np
import random 

data = scio.loadmat("./BoW_Training.mat")
data = data['data']
np.random.shuffle(data)

scenes = [list() for i in range(10)]

for sample in data:
    if sample[-1] not in scenes[sample[-2]]:
        scenes[sample[-2]].append(sample[-1])

for scene in scenes:
    print(scene)

delscene = list(range(10))

def legal_del_scene(delscenelist):
    global scenes
    for dels,scene in zip(delscenelist,scenes):
        if dels not in scene:
            return False
    return True

for group in range(5):
    print("group = ",group)
    random.shuffle(delscene)
    while not legal_del_scene(delscene):
        random.shuffle(delscene)
    print("del scene: ",delscene)
    n_feature = data.shape[1]-2
    trainX = np.empty(shape=(0,n_feature))
    valX = np.empty(shape=(0,n_feature))
    trainY = np.empty(shape=(0,2))
    valY = np.empty(shape=(0,2))
    for sample in data:
        if delscene[sample[50]]==sample[51]:
            valX = np.concatenate((valX,sample[np.newaxis,:50]))
            valY = np.concatenate((valY,sample[np.newaxis,-2:]))
        else:
            trainX = np.concatenate((trainX,sample[np.newaxis,:50]))
            trainY = np.concatenate((trainY,sample[np.newaxis,-2:]))

    trainX = trainX.astype(np.int)
    trainY = trainY.astype(np.int)
    valX = valX.astype(np.int)
    valY = valY.astype(np.int)

    print("trainX.shape = ",trainX.shape)
    print("trainY.shape = ",trainY.shape)
    print("valX.shape = ",valX.shape)
    print("valY.shape = ",valY.shape)
    np.save("trainX{}.npy".format(group),trainX)
    np.save("trainY{}.npy".format(group),trainY)
    np.save("valX{}.npy".format(group),valX)
    np.save("valY{}.npy".format(group),valY)