# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:02:51 2020

@author: Zhuo
"""
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score,make_scorer
X,y = load_boston(return_X_y = True)    #导入数据集
grid = {'C':[0.1,0.2,0.3,0.7,0.8,0.9]
              ,'epsilon':[0.6,0.8,0.85,0.9,0.95,1.0]}    #生成一个图15.3所示的参数网格
r2_scorer = make_scorer(r2_score)    #以R方为拟合优度Si，生成一个scorer类，用于设置GridSearchCV类的scoring参数
grid_search = GridSearchCV(SVR(kernel='linear'),param_grid=grid,cv=10,scoring=r2_scorer)
grid_search.fit(X,y)   #进行网格寻优
print(grid_search.best_params_)    #输出最优的参数对
print(grid_search.best_score_)