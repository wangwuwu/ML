# -*- coding: utf-8 -*-

'''
实现PCA算法
'''

from numpy import *
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from mpl_toolkits.mplot3d import Axes3D




#计算方差
def Var(array):
    meann=mean(array)
    res=sum((array-meann)**2)/((len(array))-1)
    return res

def Cov(arr1,arr2):
    me1=mean(arr1)
    me2=mean(arr2)
    # print(sum(multiply((arr1-me1),(arr2-me2))))
    return sum(multiply((arr1-me1),(arr2-me2)))/(len(arr2)-1)

#区别于numpy中的cov其以行数为维度数
def covMatrix(dataMatrix):
    m,n=dataMatrix.shape
    covmatrix=zeros((n,n))
    for i in range(n):
        for j in range(n):
            covmatrix[i,j]=Cov(dataMatrix[:,i],dataMatrix[:,j])
    return covmatrix

#返回最大n个特征值及其对应的特征值向量,这里使用要保留的信息百分比代替具体的维度数
def maxN_eigVal_eigVec(dataMatrix,prop):
    eigval,eigvec=linalg.eig(dataMatrix)
    index_val=sorted(enumerate(eigval),key=lambda x:x[1],reverse=True)
    print(index_val)
    summ=sum(eigval)
    presice=summ*prop
    print(summ,presice,eigval)
    temp=0
    index=[]#要保留的列
    for i in index_val:
        temp+=i[1]
        index.append(i[0])
        if temp>presice:
            break
    print(index)
    print(eigvec[:,index])
    return eigvec[:,index]



def pPCA(dataMatrix,pro):
    #获得协方差矩阵
    convmatrix=covMatrix(dataMatrix)
    #获得协方差矩阵的千若干特征向量
    colsmatrix=maxN_eigVal_eigVec(convmatrix,pro)
    #将原始矩阵与映射矩阵相乘，达到降维
    res=dot(dataMatrix,colsmatrix)
    return res

if __name__=='__main__':
    # dataMatrix=random.randint(2,9,size=(5,3))
    # result=PCA(dataMatrix,0.95)
    # print(result)
    X, y = make_blobs(n_samples=10000, n_features=3, centers=[[3, 3, 3], [0, 0, 0], [1, 1, 1], [2, 2, 2]],
                      cluster_std=[0.2, 0.1, 0.2, 0.2],
                      random_state=9)

    # plt.scatter(X_new[:, 0], X_new[:, 1], marker='o')#原三维空间下
    fig = plt.figure()
    res=pPCA(X,0.99)
    plt.scatter(res[:, 0], res[:, 1], marker='o')
    plt.show()
