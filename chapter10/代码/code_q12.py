# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:47:56 2020

@author: Zhuo
"""

from sklearn.datasets import load_digits    #导入数据集MINIST
digits = load_digits()    #导入数据集
X = digits.data/255
y = digits.target

from sklearn.decomposition import PCA
pca = PCA(n_components=30)    #指定d=30的PCA类
X = pca.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.1,random_state=1)    #拆分数据
from sklearn.svm import SVC
svc = SVC(C=1.0,kernel='rbf',decision_function_shape='ovr')
svc.fit(X_train,y_train)
from sklearn.metrics import accuracy_score	#引入评价用包
y_train_pred = svc.predict(X_train)
y_test_pred = svc.predict(X_test)
print('训练集中的精确度为',accuracy_score(y_train,y_train_pred))
print('测试集中的精确度为',accuracy_score(y_test,y_test_pred))

