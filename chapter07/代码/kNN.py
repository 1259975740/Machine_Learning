# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:19:42 2020

@author: Zhuo
"""
from sklearn.neighbors import KNeighborsRegressor #导入kNN回归包
import numpy as np
#产生一个示例数据集
X = np.array([[158,1],[170,1],[183,1],[191,1],[155,0],[163,0],[180,0],[158,0],[170,0]])
y = [64,86,84,80,49,59,67,54,67]
kNN = KNeighborsRegressor(n_neighbors=4)    #定义kNN模型
kNN.fit(X,y)    #训练模型
x_new = np.array([[180,1]])    #定义一个新的特征向量
y_pred = kNN.predict(x_new)    #若新向量的y为NaN，可用kNN模型预测其值
