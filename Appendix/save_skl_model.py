# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:18:43 2020

@author: Zhuo
"""

from sklearn.externals.joblib import dump,load
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X,y = load_iris(return_X_y=True)    #导入数据集
lg = LogisticRegression()    #定义模型
lg.fit(X,y)    #训练模型
dump(lg,'model.pkl')    #保存模型到代码所在的文件夹中
"""从pickle文件中导入模型"""
model = load('model.pkl')   #从model.pkl文件中导入模型