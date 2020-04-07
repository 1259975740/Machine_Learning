# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 21:43:05 2020

@author: Zhuo
"""
from sklearn import datasets
X,y = datasets.make_blobs(100,2,centers=2,
                          random_state=1,cluster_std=2)    #产生数据集

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X
        ,y,test_size=0.3,random_state=1)    #按7：3拆分数据集

from sklearn.naive_bayes import GaussianNB   
NB = GaussianNB ()    
NB.fit(X_train,y_train)    #训练模型
y_train_pred = NB.predict(X_train)		
y_test_pred = NB.predict(X_test)	

from sklearn.metrics import r2_score
print('模型在训练集的R方为',r2_score(y_train, y_train_pred))
print('模型在测试集的R方为',r2_score(y_test,y_test_pred))