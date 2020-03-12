# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 21:27:04 2020

@author: Zhuo
"""

from sklearn.datasets import make_moons
X,y=make_moons(n_samples=100,noise=0.1)   #产生过数据集

"""参考答案"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
dtc = tree.DecisionTreeClassifier()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=4)    #拆分数据
dtc.fit(X,y)  #训练模型
"""画出分类边界"""
def plot_boundary(model,X,y):
    x_min,x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min,y_max = X[:,0].min()-1, X[:,0].max()+1
    h = 0.02
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),
                        np.arange(y_min,y_max,h))
    X_hypo = np.column_stack((xx.ravel().astype(np.float32),
                             yy.ravel().astype(np.float32)))
    zz = model.predict(X_hypo)
    zz = zz.reshape(xx.shape)
    plt.contourf(xx,yy,zz,cmap=plt.cm.binary,alpha=0.2)
    X0 = X[y.ravel()==0]
    plt.scatter(X0[:, 0], X0[:, 1], marker='o')  
    X1 = X[y.ravel()==1]
    plt.scatter(X1[:, 0], X1[:, 1], marker='x')  

plot_boundary(dtc,X,y)   #画出决策树分类边界