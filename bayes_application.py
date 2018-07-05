# -*- coding: utf-8 -*-

from numpy import *
import numpy as np
import matplotlib.pyplot as plt

#项目一
def loadDataSet():
    data = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please','haha'],  # [0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage','fuck','you','yes'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how',  'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid','you','are']]
    labels=[0,1,0,1,0,1]
    return np.array(data),np.array(labels)

#得到单词表
def getVocabList(dataset):
    sett=set()
    for  i in dataset:
        # print(type(sett),type(i))
        sett=sett|set(i)
    return list(sett)

#判断dataset中的数据在单词表中是否出现，出现为1，没出现为0,这里dataset中的一行数据即为一个文档
def word2Vec(input_data,vocaList):
    print(shape(input_data))
    m,k=input_data.shape
    n=len(vocaList)
    data=ones((m,n))
    for i in range(m):
        for j in range(k):
            if input_data[i,j]  in vocaList:
                data[i,vocaList.index(input_data[i,j])]+=1
    return data

def trainBayes(vecMatrix,labels):
    numsDoc,numsVec=shape(vecMatrix)#文档数；变量数
    vec0P=[0]*numsVec#在这个问题中只有两个类别
    vec1P=[0]*numsVec
    abusiveP=sum(labels)/len(labels)
    for i in labels:
        mask=labels==i
        print(mask)
        #data为同一类内的数据
        data=vecMatrix[mask]
        for j in range(numsVec):
            if i==1:
                vec1P[j]=sum(data[:,j])/(len(data)+numsVec)
            if i==0:
                vec0P[j]=sum(data[:,j])/(len(data)+numsVec)
    return np.array(vec0P),np.array(vec1P),abusiveP

def predict(wordList,vocaList,*args):
    # from operator import mul
    vec0P,vec1P,abusive=args
    index=[vocaList.index(x) for x in wordList]
    p0=vec0P[index]
    p1=vec1P[index]
    re0=1
    for i in p0:
        re0*=i
    re1=1
    for i in p1:
        re1*=i
    print(p0,'--')
    print(p1,'==')
    if re1*abusive>re0*(1-abusive):
        # print()
        return 1
    else:
        return 0


if __name__=='__main__':
    dataSet,label=loadDataSet()
    vocaList=getVocabList(dataSet)
    vecMatrix=word2Vec(dataSet,vocaList)
    vec0P,vec1P,abusive=trainBayes(vecMatrix,label)
    #['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid']
    res=predict(['maybe', 'not', 'take', 'him', 'problems', 'help', 'please','haha'],vocaList,vec0P,vec1P,abusive)
    print(res)





