# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:40:59 2020

@author: Zhuo
"""

from data_prepro import data_generate_final
X_pca,y_cod,y_cod_rate,y_vfa = data_generate_final()    #导入数据集
"""导入相关库"""
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor    #导入kNN算法
from sklearn.linear_model import LinearRegression   #线性回归
from sklearn.metrics import make_scorer,r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score

""""筛选kNN算法的最合适参数k"""
grid = {'n_neighbors':[50,55,60,65,70]}    #定义参数网格为近邻个数k
r2_scorer = make_scorer(r2_score)    #用R方作为评价模型的指标
grid_search = GridSearchCV(KNeighborsRegressor(),param_grid=grid,cv=5,scoring=r2_scorer)
grid_search.fit(X_pca,y_cod)   #进行网格寻优
print(grid_search.best_params_)    #输出最优的参数对

"""筛选最合适的SVR"""
grid = {'C':[0.4,0.5,0.6,0.7,0.8],
        'epsilon':[1.4,1.5,1.6,1.7,1.8],
        'kernel':['linear','rbf','poly']}    #定义参数网格惩罚参数C、epsilon和核函数类型
grid_search = GridSearchCV(SVR(),param_grid=grid,cv=5,scoring=r2_scorer)
grid_search.fit(X_pca,y_cod)   #进行网格寻优
print(grid_search.best_params_)    #输出最优的参数对

"""筛选最合适的决策树"""
grid = {'max_depth':[8,10,12,14,15],
        'ccp_alpha':[0.05,0.055,0.06,0.065]}    #定义参数网格为最大树深度和α_ccp
grid_search = GridSearchCV(DecisionTreeRegressor(),param_grid=grid,cv=5,scoring=r2_scorer)
grid_search.fit(X_pca,y_cod)   #进行网格寻优
print(grid_search.best_params_)    #输出最优的参数对

"""筛选最合适的随机森林"""
grid = {'n_estimators':[110,115,120,125,130]}    #定义参数网格为子模型数量
grid_search = GridSearchCV(RandomForestRegressor(max_samples=0.67,max_features=0.33,max_depth=5)
                            ,param_grid=grid,cv=5,scoring=r2_scorer)
grid_search.fit(X_pca,y_cod)   #进行网格寻优
print(grid_search.best_params_)    #输出最优的参数对

"""筛选最合适的AdaBoost"""
lr = LinearRegression()   #定义子模型为线性回归
grid = {'n_estimators':[15,20,25,30,35]}    #定义参数网格
grid_search = GridSearchCV(AdaBoostRegressor(base_estimator=lr)
                            ,param_grid=grid,cv=5,scoring=r2_scorer)
grid_search.fit(X_pca,y_cod)   #进行网格寻优
print(grid_search.best_params_)    #输出最优的参数对


"""5折交叉验证筛选模型"""
"""定义算法，算法中的参数是从网格搜索得出的最优参数"""
lr = LinearRegression()
knn = KNeighborsRegressor(n_neighbors=60)    #k=60的kNN算法
svr = SVR(C=0.4,epsilon=1.5,kernel='rbf')    #惩罚参数C=0.4的，epsilon=1.5,核函数为RBF的软间隔SVR
dtr = DecisionTreeRegressor(max_depth=15,ccp_alpha=0.055)   #剪枝条件为：最大深度=15，α_ccp=0.055的决策树
rf = RandomForestRegressor(n_estimators=125,max_depth=5,max_samples=0.67,max_features=0.33)    #子决策树的深度均为5，包含125颗子决策树的随机森林
adaboost = AdaBoostRegressor(base_estimator=lr,n_estimators=1000)    #将子模型设置为lr，子模型个数为25个，从而构成AdaBoost模型
""""用5折交叉验证，计算所有模型的Si,并计算其均值"""
S_lr_i = cross_val_score(lr,X_pca,y_cod,scoring=r2_scorer,cv=5)    #计算出线性回归模型的Si
S_knn_i = cross_val_score(knn,X_pca,y_cod,scoring=r2_scorer,cv=5)    #计算出kNN模型的Si
S_svr_i = cross_val_score(svr,X_pca,y_cod,scoring=r2_scorer,cv=5)    #计算出SVR模型的Si
S_dtr_i = cross_val_score(dtr, X_pca,y_cod,scoring=r2_scorer,cv=5)    #计算出决策树模型的Si
S_rf_i = cross_val_score(rf, X_pca,y_cod,scoring=r2_scorer,cv=5)    #计算随机森林模型的Si
S_ada_i = cross_val_score(adaboost,X_pca,y_cod,scoring=r2_scorer,cv=5)    #计算Adaboost模型的Si
print(np.mean(S_lr_i))    #输出均值
print(np.mean(S_knn_i))
print(np.mean(S_svr_i))
print(np.mean(S_dtr_i))
print(np.mean(S_rf_i))
print(np.mean(S_ada_i))

