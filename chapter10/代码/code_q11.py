# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:36:29 2020

@author: Zhuo
"""
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
X = np.random.randn(200,2)
y_xor = np.logical_xor(X[:,0]>0,X[:,1]>0)
y = np.where(y_xor,0,1)

"""画图代码"""
X0 = X[y.ravel()==0]
plt.scatter(X0[:, 0], X0[:, 1], marker='o')  
X1 = X[y.ravel()==1]
plt.scatter(X1[:, 0], X1[:, 1], marker='x') 

"""参考答案"""
svc = SVC()
svc.fit(X,y)
y_pred = svc.predict(X)
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

plot_boundary(svc,X,y)   #画出划
