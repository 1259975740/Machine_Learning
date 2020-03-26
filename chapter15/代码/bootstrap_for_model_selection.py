# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:27:17 2020

@author: Zhuo
"""

import numpy as np
def bootstrapping(model,X,y,scoring,T=10000):    #自变量model为待检验算法，score为拟合优度指标，T为子样本个数，默认值为10000
    Si = []    #定义一个空列表，用于放置拟合优度
    for _ in range(T):
        idx = np.random.choice(len(X),size=len(X),replace=True)    #有放回地随机抽样
        X_boot = X[idx,:]    #通过有放回地随机抽样得到新样本
        y_boot = y[idx]
        model.fit(X_boot,y_boot)    #用子样本训练模型
        idx_out = np.array([x not in idx for x
                            in np.arange(len(X))])    #跳出没被抽中的个体索引
        X_out = X[idx_out,:]    #将没被抽中的个体构成测试集
        y_out = y[idx_out]
        y_out_pred = model.predict(X_out)    #模型预测值
        Si.append(scoring(y_out,y_out_pred))
    Si = np.array(Si)
    S_bar,S_var = Si.mean(),Si.var()   #分别输出Si的均值与方差
    return S_bar,S_var

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

X,y = load_iris(return_X_y=True)
knn = KNeighborsClassifier(n_neighbors=5)    #KNN算法
lg = LogisticRegression(penalty='none')    #无正则化的逻辑回归算法，参数penalty为惩罚项，可选l1，l2，分别对应Lasso、Ridge正则化
from sklearn.metrics import accuracy_score    #导入精确度函数

S_bar_knn,_ = bootstrapping(knn, X, y, scoring=accuracy_score,T=10)    #迭代次数为10，使用精确度评价模型，评价kNN算法在鸢尾花数据集中的效果
S_bar_lg,_ = bootstrapping(lg, X, y, scoring=accuracy_score,T=10)    #迭代次数为10，使用精确度评价模型，评价逻辑算法在鸢尾花数据集中的效果
print(S_bar_knn)    #输出模型的总体效果S_bar
print(S_bar_lg)

        