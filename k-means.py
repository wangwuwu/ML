# -*- coding: utf-8 -*-

import numpy as np
import random
import copy
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
def eurDistance(x,y):
    if not isinstance(x,np.ndarray):
        x=np.array(x)
    if not isinstance(y,np.ndarray):
        y=np.array(y)
    return np.sqrt(np.sum((x-y)**2))

def closeCenter(data,centerlist):
    '''
    :param data:datanode
    :param centerlist: centerlist
    :return: the index od closecenter in centerlist
    '''
    dis=float('inf')
    index=0
    for i in range(len(centerlist)):
        dist=eurDistance(data,centerlist[i])
        if dist<dis:
            dis=dist
            index=i
    return index


def kMeans(dataMatrix,k):
    '''

    :param dataMatrix: 二维矩阵，每个元素为data,label
    :param k:
    :return:
    '''
    length=len(dataMatrix)
    centerIndex=np.random.choice(length,k)
    centerList=dataMatrix[centerIndex]
    closeList=np.array([-1]*length)
    maxIter=1000
    iter=0
    thresHold=1e-20#当数据量大时，此应足够小
    while iter<maxIter:
        iter+=1
        for i ,value in enumerate(dataMatrix):
            close=closeCenter(value,centerList)
            closeList[i]=close
        iter+=1

        #update centers
        prevCenterList=copy.deepcopy(centerList)
        for i in range(k):
            mask=closeList==i
            data=dataMatrix[mask]
            # print(len(data),'++++++++')
            centerList[i]=np.mean(data,axis=0)

        error=eurDistance(prevCenterList,centerList)
        if error<thresHold:
            print(iter,'0000000000')
            break
    return dataMatrix,closeList,centerList

if __name__ == '__main__':
    dataMatrix,target=make_blobs(n_samples=500,n_features=2,centers=3)
    result,closeList,centerList=kMeans(dataMatrix,3)
    # print(result)
    color=['red','blue','green']
    l=[]
    for i in range(3):
        mask=closeList==i
        temp=result[mask]
        l.append(temp)
        print(len(temp))
        print('----------')
    #     plt.scatter(temp[:,0],temp[:,1],c='red',marker='*')
    # # plt.scatter(result[:,0],result[:,1],marker='+',c='blue')
    plt.scatter(l[0][:,0],l[0][:,1],c='red',marker='*',alpha=1,label='go')
    plt.scatter(l[1][:,0],l[1][:,1],c='green',marker='*',alpha=1,label='g0o')

    plt.scatter(l[2][:,0],l[2][:,1],c='blue',marker='*',alpha=1,label='g00o')

    plt.show()







