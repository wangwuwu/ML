# -*- coding: utf-8 -*-
'''
实现cart回归树,数据集为二维表，最后一列为target_data，

'''
from numpy import *
import numpy as np
import matplotlib.pyplot as ply
from sklearn.datasets import make_regression
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz

#产生叶子节点，其值为数据集target均值
def regLeaf(dataSet):
    return mean(dataSet[:,-1])

#计算数据集内的误差
def regErr(dataSet):
    return var(dataSet[:,-1])*len(dataSet)#返回总误差

#更具最佳切分属性和最佳切分点，将数据划分为两个部分
def binarySpiltDataSet(dataSet,feature,value):
    left=dataSet[dataSet[:,feature]>value,:]
    right=dataSet[dataSet[:,feature]<=value,:]
    return left,right

def choseBestSplit(dataSet,op=[1,4]):
    #当数据量小于阈值时不再进行划分
    if len(dataSet)<op[1]:
        return None,regLeaf(dataSet)
    m,n=dataSet.shape
    bestFeature=None
    bestValue=None
    #遍历所有属性
    Error=float('inf')
    for i in range(n-1):
        data=dataSet[:,i]
        for j in data:
            left,right=binarySpiltDataSet(dataSet,i,j)
            errNew=regErr(left)+regErr(right)
            if errNew<Error:
                bestFeature=i
                bestValue=j
                Error=errNew
    #判断最佳划分误差和总误差之间差值，若小于阈值则不再划分
    # print(regErr(dataSet),Error,abs(Error-regErr(dataSet)),'++')

    if abs(Error-regErr(dataSet))<op[0]:
        return None,regLeaf(dataSet)
    #如果分裂后子集元素数量小于指定阈值，则不再划分
    L,R=binarySpiltDataSet(dataSet,bestFeature,bestValue)
    # print(len(L),len(R),'++')
    if len(L)<op[1] or len(R)<op[1]:
        return None,regLeaf(dataSet)
    return bestFeature,bestValue

def createTree(dataSet,op):
    bestFeatures,bestValue=choseBestSplit(dataSet,op)
    # print bestFeatures,bestValue,'///'
    if bestFeatures==None:
        return bestValue#叶子节点
    retTree={}
    retTree['bestFeatures']=bestFeatures
    retTree['bestVal']=bestValue
    lSet,rSet=binarySpiltDataSet(dataSet,bestFeatures,bestValue)
    retTree['right']=createTree(rSet,op)
    retTree['left']=createTree(lSet,op)
    # print retTree
    return  retTree

def isTree(obj):#判断是否是叶子节点
    return isinstance(obj,dict)#(type(obj).__name__=='dict')

#预测单个节点
def treeForecast_oneData(tree,data):
    #树的深度优先遍历
    # squence=[]
    # node=tree
    # squence.append(node)
    # while True:
    #     node=squence.pop()
    #     if isinstance(node,dict):
    #         feature=node['bestFeature']
    #         value=node['bestValue']
    #         # print(dataNode)
    #         if dataNode[feature]>value:
    #             squence.append(node['left'])
    #         else:
    #             squence.append(node['right'])
    #     else:
    #         return node


    if not isTree(tree):
        return float(tree)#此时为叶子节点
    if data[tree['bestFeatures']]>tree['bestVal']:
        if isTree(tree['left']):
            return treeForecast_oneData(tree['left'], data)
        else:
            return float(tree['left'])
    else:
        if isTree(tree['right']):
            return treeForecast_oneData(tree['right'], data)
        else:
            return float(tree['right'])

#预测数据集
def predict_DataSet(tree,dataSet):
    m,n=shape(dataSet)
    res=[0]*m
    for i in range(m):
        temp=treeForecast_oneData(tree,dataSet[i,:])
        res[i]=temp
    return res

def creat_data(n):
    '''
    产生用于回归问题的数据集
    :param n:  数据集容量
    :return: 返回一个元组，元素依次为：训练样本集、测试样本集、训练样本集对应的值、测试样本集对应的值
    '''
    np.random.seed(0)
    X = 5 * np.random.rand(n, 1)
    y = np.sin(X).ravel()
    noise_num=(int)(n/5)
    y[::5] += 3 * (0.5 - np.random.rand(noise_num)) # 每第5个样本，就在该样本的值上添加噪音
    X_train, X_test, y_train, y_test=train_test_split(X, y,test_size=0.25,random_state=1)
    return X_train, X_test, y_train, y_test #

if __name__=='__main__':
    #一维效果不错，多维不好
    # X,Y=make_regression(n_samples=100,n_features=4)
    # x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25)
    x_train, x_test, y_train, y_test=creat_data(1000)
    y_test=y_test.reshape((len(y_test),1))

    y_train=y_train.reshape((len(y_train),1))#将y_train的形状由（600，）变为（600,1）
    print(shape(x_train))
    print(shape(y_train))
    dataSet=hstack([x_train,y_train])

    drt=DecisionTreeRegressor()
    drt.fit(x_train,y_train)
    res=drt.predict(x_test)
    # print(res-y_test)
    # print(sum(res-y_test))

    import pydotplus

    dot_data = export_graphviz(drt, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("C:\\Users\\168\\Downloads\\iris.pdf")


    # tree=createTree(dataSet,[2,4])
    # res=predict_DataSet(dataSet=x_test,tree=tree)
    # # print(res)
    # print(res-y_test)
    # print(sum(res-y_test))
    #










