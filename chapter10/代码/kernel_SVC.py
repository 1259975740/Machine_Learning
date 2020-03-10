# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 22:33:38 2020

@author: Zhuo
"""

from sklearn.datasets import make_gaussian_quantiles  
X, y = make_gaussian_quantiles(n_samples=300,n_features=2, n_classes=2)  #产生数据集
"""画图代码"""
import matplotlib.pyplot as plt 
X0 = X[y.ravel()==0]
plt.scatter(X0[:, 0], X0[:, 1], marker='o')  
X1 = X[y.ravel()==1]
plt.scatter(X1[:, 0], X1[:, 1], marker='x') 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)    #拆分数据
k_svc = SVC(C=1.0,kernel='rbf',gamma=1)    #使用RBF核的核SVC，设置惩罚参数为1
k_svc.fit(X_train,y_train)    #训练模型

y_train_pred = k_svc.predict(X_train)
y_test_pred = k_svc.predict(X_test)    #用模型得出预测值
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

plot_boundary(k_svc,X,y)   #画出划分超平面
