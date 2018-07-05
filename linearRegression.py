# # -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import copy

def SGD(data,labels,theta):
    maxIter=10000
    epsilon=1e-6
    iter=0
    m,n=input_data.shape
    while iter<maxIter:
        oldTheta=copy.deepcopy(w)
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
            diff=np.dot(theta, data[index])-labels[index]
            # print(theta,'3333333')
            # # print( theta, data[index],'uuu')
            theta=theta-0.001*diff*data[index]
            # print(theta,'4')
            #计算误差
        if sum((oldTheta-theta)**2)/2<epsilon:
            break
        iter+=1
    return theta

if __name__=='__main__':
    # x,y=make_regression(n_samples=200,n_features=1,noise=0.5)
    x = np.arange(0., 10., 0.2)
    m = len(x)  # 训练数据点数目
    x0 = np.full(m, 1.0)
    input_data = np.vstack([x0, x]).T  # 将偏置b作为权向量的第一个分量
    target_data = 2 * x + 5 + np.random.randn(m)
    np.random.seed(0)
    w = np.random.randn(2)
    theta=SGD(input_data,target_data,w)
    print(theta)
    plt.scatter(x,x*theta[1]+theta[0],c='red')
    plt.scatter(x,target_data,c='green')
    plt.show()

'''

# -*- coding: cp936 -*-
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 构造训练数据
x = np.arange(0., 10., 0.2)
m = len(x)  # 训练数据点数目
x0 = np.full(m, 1.0)
input_data = np.vstack([x0, x]).T  # 将偏置b作为权向量的第一个分量
target_data = 2 * x + 5 + np.random.randn(m)

# 两种终止条件
loop_max = 10000  # 最大迭代次数(防止死循环)
epsilon = 1e-3

# 初始化权值
np.random.seed(0)
w = np.random.randn(2)
# w = np.zeros(2)

alpha = 0.001  # 步长(注意取值过大会导致振荡,过小收敛速度变慢)
diff = 0.
error = np.zeros(2)
count = 0  # 循环次数
finish = 0  # 终止标志
# -------------------------------------------随机梯度下降算法----------------------------------------------------------

while count < loop_max:
    count += 1

    # 遍历训练数据集，不断更新权值
    for i in range(m):
        diff = np.dot(w, input_data[i]) - target_data[i]  # 训练集代入,计算误差值

        # 采用随机梯度下降算法,更新一次权值只使用一组训练数据
        w = w - alpha * diff * input_data[i]

        # ------------------------------终止条件判断-----------------------------------------
        # 若没终止，则继续读取样本进行处理，如果所有样本都读取完毕了,则循环重新从头开始读取样本进行处理。

    # ----------------------------------终止条件判断-----------------------------------------
    # 注意：有多种迭代终止条件，和判断语句的位置。终止判断可以放在权值向量更新一次后,也可以放在更新m次后。
    if np.linalg.norm(w - error) < epsilon:     # 终止条件：前后两次计算出的权向量的绝对误差充分小
        finish = 1
        break
    else:
        error = w
# check with scipy linear regression
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, target_data)
plt.plot(x, target_data, 'k+')
plt.plot(x, w[1] * x + w[0], 'r')
plt.show()
'''