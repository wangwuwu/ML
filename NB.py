# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from collections import Counter
from sklearn.cross_validation import train_test_split






















class NB:
    def __init__(self):
        self.C=[]
        self.m=0

# 存放类信息具体包括
# {'data':,[{i:nums,j:nums},{i:nums,j:nums},{i:nums,j:inums}]}
# data为类内的数据，[]个数为属性数，其中的字典{}表示属性内不同值对应的个数
#
    def fit(self,input_data,target_data):
    #
    # :param input_data:输入数据，数组或者矩阵形式
    # :param target_data: 标签值，一维数组或者矩阵形式
    # :return:
    #
        self.m=len(input_data)
        self.input_data=input_data
        m,n=input_data.shape
        for i in set(target_data):
            mask=target_data==i
            dataCi=input_data[mask]
            coldic={}
            for j in range(n):#取每一列数据
                tempData=dataCi[:,j]
                tempdic=Counter(tempData)
                coldic[j]=tempdic
            self.C.append({'ck':i,'data':dataCi,'coldic':coldic})
    def preictNode(self,dataNode):

        # :param data: 输入数据
        # :return: 所属类别

        G=-1000
        target=None
        for i in range(len(self.C)):
            data=self.C[i]['data']
            coldic=self.C[i]['coldic']
            Pci=len(data)/self.m#这个在实际应用中不使其发生
            p=1
            for j in range(len(dataNode)):
                if dataNode[j] in coldic[j]:#取dataNode 地i维数据
                   p*=(coldic[j][dataNode[j]]+1)/(len(data)+len(set(self.input_data[:,j])))#这个对分类影响很大，所以在主函数中对数据只保留了一位小数，使得其有意义或者也可以找到距离最近（距离在一定范围内）那个点，当做目前点来用、、、、括号中等价于属性中某一值占类中总数的比例
                else:
                    p*=1/(len(set(self.input_data[:,j]))+len(data))
                #else:暂时使用极大似然估计
            p*=Pci
            if p>G:
                G=p
                target=self.C[i]['ck']
        return target

    def predictSet(self,set):
        result=[]
        for i in set:
            result.append(self.preictNode(i))
        return result


if __name__=='__main__':


    input_data,target_data=make_classification(n_samples=500, n_features=2, n_redundant=0, n_clusters_per_class=1, n_classes=2)
    for i in range(len(input_data)):
        for j in range(len(input_data[0])):
            input_data[i,j]=round(input_data[i,j],1)
    train_data,test_data,train_label,test_label=train_test_split(input_data,target_data,test_size=0.3)
    nb=NB()
    print(len(train_label))
    nb.fit(train_data,train_label)
    result=nb.predictSet(test_data)
    mask=result==test_label
    sum=0
    for x in mask:
        if x :
            sum+=1
    print(sum)
    # plt.scatter(train_data[:,0],train_data[:,1],c='red')
    # plt.scatter(test_data[:,0],test_data[:,1],c='green')
    plt.scatter(input_data[:,0],input_data[:,1])
    plt.show()