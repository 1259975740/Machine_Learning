# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 21:51:26 2020

@author: Zhuo
"""
from sklearn import datasets
X,y = datasets.make_blobs(100,2,centers=2,
                          random_state=1,cluster_std=2)    #产生数据集
import numpy as np
from sklearn import model_selection
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.3)

from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
NB = GaussianNB()
NB.fit(X_train,y_train)
def plot_proba(model,X,y):
    h = 0.02
    x_min,x_max = X[:,0].min()-1, X[:,0].max()+4
    y_min,y_max = X[:,0].min()+4, X[:,0].max()+12
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),
                        np.arange(y_min,y_max,h))
    X_hypo = np.column_stack((xx.ravel().astype(np.float32),
                             yy.ravel().astype(np.float32)))
    y_proba = model.predict_proba(X_hypo)
    zz = y_proba[:,1] - y_proba[:,0]
    zz = zz.reshape(xx.shape)
    plt.contourf(xx,yy,zz,cmap=plt.cm.coolwarm,alpha=0.8)
    plt.scatter(X[:,0],X[:,1],c=y,s=50)

plot_proba(NB,X,y)