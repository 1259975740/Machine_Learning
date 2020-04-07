# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:55:44 2020

@author: Zhuo
"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error,mean_squared_error,median_absolute_error,r2_score,explained_variance_score
from sklearn.model_selection import train_test_split

X,y = load_boston(return_X_y=True)    #导入数据
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=4)    #按照7:3拆分数据
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train,y_train)

"""参考答案"""
y_test_pred = knn.predict(X_test)
print('测试集的绝对误差均值为: ',mean_absolute_error(y_test,y_test_pred))
print('测试集的均方误差为: ',mean_squared_error(y_test,y_test_pred))
print('测试集的绝对误差中值为: ',median_absolute_error(y_test,y_test_pred))
print('测试集的R方为: ',r2_score(y_test,y_test_pred))
print('测试集的解释方差得分为: ',explained_variance_score(y_test,y_test_pred))
