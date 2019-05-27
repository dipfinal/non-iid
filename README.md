# non-iid
Final Project for Digit Image Processing

## classifier.py

分类器基类，尽可能继承该类，便于验证准确率

## 数据

trainX.npy,trainY.npy,valX.npy,valY.npy 从BoW_Training.mat中划分出来的

trainX.npy : shape = [3556,50]

trainY.npy : shape = [3556,2] (category,background)

valX.npy : shape = [600,50]

valY.npy : shape = [600,2]

BoW_Training 中每一类的场景分别缺失为[5,6,7] [7,8,9] [5,6,8] [6,8,9] [2,4,9] [2,7,9] [2,4,9] [2,5,6] [4,6,8] [0,6,8] 

valX为BoW_Training中每一类分别取走一种场景 3 2 9 4 0 6 8 1 5 7

