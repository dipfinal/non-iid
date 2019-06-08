import numpy as np
from sklearn import tree
from sklearn.preprocessing import normalize
import geatpy
import time
import sys

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

"""
min_feature_num : 最少使用min_feature个维度进行计算
max_depth_alpha,max_depth : 决策树最大深度为 min {len(feature)*max_depth_alpha ,max_depth }
N : 每一代有N个个体
"""
min_feature_num = 10
max_depth_alpha = 0.5
max_depth = 20
N = 5 
MaxGeneration = 10
Feature = 50
GGAP = 0.1
recopt = 0.5
pm = 0.1
def err(Y,Y_):
    return 1.0*np.sum(Y!=Y_)/Y.shape[0]

def aimfunc(Phen,LegV):
    # 值越小越适应
    f = np.empty(shape=(Phen.shape[0],))
    for id,gene in enumerate(Phen):
        feature = [i for i,x in enumerate(gene) if x==1]
        
        if len(feature)<min_feature_num:
            f[id]=1.0
            continue
        tX = trainX.copy()
        vX = valX.copy()
        tX[:,feature] = 0
        vX[:,feature] = 0
        clf = tree.DecisionTreeClassifier(max_depth=min(max_depth,int(max_depth_alpha*len(feature))),min_samples_leaf=10)
        clf.fit(tX,trainY)
        f[id] = err(clf.predict(vX),valY)
    return [f[:,np.newaxis],LegV]

if __name__ == "__main__":
    AIM_M = __import__("geat")
    FieldD = np.array([
        [0]*Feature,
        [1]*Feature
    ])
    [pop_trace,var_trace,times] = geatpy.sga_new_real_templet(
        AIM_M, 'aimfunc', None, None, FieldD, 'I', 1, MaxGeneration, N, 1, GGAP, 'rws', 'xovsp', recopt, pm, False, 1)
    best_gen = np.argmin(pop_trace[:, 0]) # 记录最优种群是在哪一代
    print('best target function = ', np.min(pop_trace[:, 0]))
    print('best variable = ')
    for i in range(var_trace.shape[1]):
        print(var_trace[best_gen, i])
    print('best generation = ',best_gen + 1)
    print('time=', times, 's')
    geatpy.trcplot(pop_trace, [['各代种群最优目标函数值'], ['各代种群个体平均适应度值', '各代种群最优个体适应度值']], ['demo_result1', 'demo_result2'])
    # 输出结果

    