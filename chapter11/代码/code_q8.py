# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 19:04:51 2020

@author: Zhuo
"""

import numpy as np
rng = np.random.RandomState(2)
X = np.arange(1,2,0.01).reshape(-1,1)
y = np.tan(X).ravel()

"""参考答案"""

from sklearn import tree
dtr = tree.DecisionTreeRegressor()
dtr.fit(X,y)

y_pred = dtr.predict(X)
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.scatter(X,y,c='k',label='数据集')
plt.plot(X,y)
