# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:57:27 2020

@author: Administrator
"""

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
lg = LogisticRegression(multi_class='ovr',penalty='none')  #使用ovr方法生成多分类模型
iris = load_iris()
X = iris.data;
y = iris.target
lg.fit(X,y)    #训练模型
y_pred = model.predict(X)    #输出预测值
print(classification_report(y,y_pred))  #输出结果报表