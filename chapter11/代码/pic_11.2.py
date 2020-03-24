# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 18:28:11 2020

@author: Zhuo
"""
from sklearn import tree
dtr = tree.DecisionTreeRegressor(max_depth=2,criterion='mse')
import numpy as np
rng = np.random.RandomState(2)
X = np.sort(5*rng.rand(100,1),axis=0)
y = np.sin(X).ravel()
y[::2] += 0.5*(0.5-rng.rand(50))

dtr.fit(X,y)

import matplotlib.pyplot as plt
font1 = {'family' : 'SimHei',
'weight' : 'normal',
'size'   : 20,
}
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.scatter(X,y,c='k',label='数据集')
plt.plot(X,dtr.predict(X),label='拟合曲线',linewidth=5)
plt.xlabel('特征',fontsize=20)
plt.ylabel('因变量',fontsize=20)
plt.legend(prop=font1)
