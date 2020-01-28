# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:57:27 2020

@author: Administrator
"""

from sklearn import linear_model
import numpy as np
from sklearn import model_selection
from sklearn import datasets
from sklearn import metrics
model = linear_model.LogisticRegression(multi_class='ovr')  #使用ovr方法生成多分类模型
iris = datasets.load_iris()
X = iris.data;
y = iris.target
model.fit(X,y)    #训练模型
y_pred = model.predict(X)    #输出预测值
print(metrics.classification_report(y,y_pred))  #输出结果报表