# -*- coding: utf-8 -*-

from numpy import *
import numpy as np
import copy
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

'''
实现逻辑回归
'''

def sigmod(x):
    return 1.0/(1+exp(-x))

def SGD(data,labels,theta):
    maxIter=10000
    epsilon=1e-6
    iter=0
    m,n=data.shape
    while iter<maxIter:
        oldTheta=copy.deepcopy(theta)
        # print(w,'111111')
        for i in range(m):
            # diff = np.dot(w, input_data[i]) - target_data[i]  # 训练集代入,计算误差值

            # 采用随机梯度下降算法,更新一次权值只使用一组训练数据
            # w = w - 0.001 * diff * input_data[i]
            index=np.random.randint(0,m)
            # print(theta,'222')
            # #当为数组时候，dot和*是有区别的
            # temp=data[index]
            # temp1=labels[index]
            diff=sigmod(np.dot(theta, data[index]))-labels[index]
            # print(theta,'3333333')
            # # print( theta, data[index],'uuu')
            theta=theta-0.001*diff*data[index]
            # print(theta,'4')
            #计算误差
        if sum((oldTheta-theta)**2)/2<epsilon:
            break
        iter+=1
    return theta

def predictOne(theta,dataNode):
    print(dataNode)
    fx=dot(theta,dataNode)
    res=sigmod(fx)
    if res>0:
        return 1
    else:
        return 0
def predictSet(theta,dataSet):
    re=[0]*len(dataSet)
    for i in range(len(dataSet)):
        re[i]=predictOne(theta,dataSet[i])
    return array(re)

if __name__=='__main__':
    dataMatrix,labels=make_classification(n_samples=300,n_features=2,n_clusters_per_class=1,n_classes=2,n_redundant=0)
    # print(dataMatrix)
    b=ones((len(dataMatrix),1))
    print(shape(dataMatrix),shape(b))
    dataMatrix=hstack([dataMatrix,b])
    train_data,test_data,train_label,test_label=train_test_split(dataMatrix,labels,test_size=0.3,random_state=0)
    theta=[1]*len(dataMatrix[0])
    # theta=SGD(train_data,train_label,theta)
    theta=SGD(dataMatrix,labels,theta)
    res=predictSet(theta,test_data)
    print(sum(res==test_label))
    print(theta)
    x=dataMatrix[:,0]
    y=-(theta[0]/theta[1])*x-theta[2]/theta[1]

    plt.scatter(dataMatrix[:,0],dataMatrix[:,1],c='green')
    #画出分类面
    plt.scatter(x,y,c='red')
    plt.show()
