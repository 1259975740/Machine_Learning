# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:49:45 2020

@author: Zhuo
"""

"""导入有关库"""
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score,make_scorer    #导入精确度函数
X,y = load_iris(return_X_y=True)    #导入数据
knn = KNeighborsClassifier(n_neighbors=5)    #KNN算法
lg = LogisticRegression(penalty='none')    #无正则化的逻辑回归算法，参数penalty为惩罚项，可选l1，l2，分别对应Lasso、Ridge正则化
acc_scorer = make_scorer(accuracy_score)    #用于设置评价指标，作为cross_val_score函数的参数
from sklearn.model_selection import LeaveOneOut    #导入LeaveOneOut类
S_knn = cross_val_score(knn,X,y,scoring=acc_scorer,cv=LeaveOneOut())   #留一法，通过scoring参数设置评价模型的拟合优度，cv设置为LeaveOneOut()。
S_lg = cross_val_score(lg,X,y,scoring=acc_scorer,cv=LeaveOneOut())   
print(S_knn.mean())    #输出模型的总体效果S_bar
print(S_lg.mean())