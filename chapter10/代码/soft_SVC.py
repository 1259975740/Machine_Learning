# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 21:37:17 2020

@author: Zhuo
"""
from sklearn.datasets import make_classification
X,y = make_classification(n_samples=300,n_features=2,n_informative=1
                          ,n_redundant=0,n_clusters_per_class=1)    #生成数据集
import matplotlib.pyplot as plt
X0 = X[y.ravel()==0]
plt.scatter(X0[:, 0], X0[:, 1], marker='o')  
X1 = X[y.ravel()==1]
plt.scatter(X1[:, 0], X1[:, 1], marker='x')  
from sklearn.svm import LinearSVC    #导入线性软间隔SVC库

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)    #拆分数据
svc = LinearSVC(C=0.8)   #设置软间隔SVC的惩罚参数为0.8
svc.fit(X_train,y_train)    #进行模型训练

y_train_pred = svc.predict(X_train)
y_test_pred = svc.predict(X_test)    #用模型得出预测值
from sklearn.metrics import classification_report , accuracy_score
print('训练集结果报表')
print(classification_report(y_train,y_train_pred))
print('训练集中的精度为： ',accuracy_score(y_train,y_train_pred))
print('测试集结果报表')
print(classification_report(y_test,y_test_pred))
print('测试集集中的精度为： ',accuracy_score(y_test,y_test_pred))

"""画图代码"""
import numpy as np
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

plot_boundary(svc,X,y)   #画出划分超平面