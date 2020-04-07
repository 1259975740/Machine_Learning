# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:56:24 2020

@author: Zhuo
"""
import pandas as pd
df = pd.read_csv(r'D:\桌面\我的书\chapter10\数据集\airfoil_self_noise.dat'
                 ,sep='\t',header=None,engine='python')
X = df.iloc[:,0:5].values
y = df.iloc[:,5].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(
        data,labels,test_size=0.1,random_state=1)    #拆分数据

from sklearn.svm import SVR
svr = SVR(C=1.0,kernel='rbf',epsilon=1.0)
svr.fit(X_train,y_train)
from sklearn.metrics import r2_score	#引入评价用包
y_train_pred = svr.predict(X_train)
y_test_pred = svr.predict(X_test)
print('训练集中的R方为',r2_score(y_train,y_train_pred))
print('测试集中的R方为',r2_score(y_test,y_test_pred))