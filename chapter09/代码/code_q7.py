# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 17:59:10 2020

@author: Zhuo
"""
from sklearn.datasets import load_wine
wine = load_wine()

"""参考答案"""
X = wine.data
y = wine.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.3,random_state=1)    #拆分数据

from sklearn.linear_model import LogisticRegression
softmax = LogisticRegression(multi_class = 'multinomial')

softmax.fit(X_train,y_train)    #模型训练
y_train_pred = softmax.predict(X_train)
y_test_pred = softmax.predict(X_test)    #用回归模型得出预测值
#以下代码用于生成评价报表、算出精确率等
from sklearn.metrics import classification_report 
print('训练集结果报表')
print(classification_report(y_train,y_train_pred))
print('测试集结果报表')
print(classification_report(y_test,y_test_pred))
