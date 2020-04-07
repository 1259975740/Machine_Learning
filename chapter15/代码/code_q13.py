# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:06:54 2020

@author: Zhuo
"""


from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer,r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
X,y = load_boston(return_X_y=True)

r2_scorer = make_scorer(r2_score)
grid = {'max_depth':[5,6,7,8,9,10],
        'ccp_alpha':[0,0.001,0.01,0.03,0.05]}    #定义参数网格为最大树深度和α_ccp
grid_search = GridSearchCV(DecisionTreeRegressor(),param_grid=grid,cv=5,scoring=r2_scorer)
grid_search.fit(X,y)   #进行网格寻优
print(grid_search.best_params_)    #输出最优的参数对

grid = {'C':[0.05,0.07,0.1,0.12],
        'epsilon':[0.6,0.7,0.8],
        'kernel':['linear','rbf','poly']}    #定义参数网格惩罚参数C、epsilon和核函数类型
grid_search = GridSearchCV(SVR(),param_grid=grid,cv=5,scoring=r2_scorer)

grid_search.fit(X,y)   #进行网格寻优
print(grid_search.best_params_)    #输出最优的参数对

grid = {'n_neighbors':[1,3,5,7,9,11,13,15,17]}
grid_search = GridSearchCV(KNeighborsRegressor(),param_grid=grid,cv=5,scoring=r2_scorer)
grid_search.fit(X,y)   #进行网格寻优
print(grid_search.best_params_)    #输出最优的参数对

dtr = DecisionTreeRegressor(max_depth=5,ccp_alpha=0.03)  
svr = SVR(C=0.07,epsilon=0.8,kernel='linear')    
knn = KNeighborsRegressor(n_neighbors=9)   

S_svr_i = cross_val_score(svr,X,y,scoring=r2_scorer,cv=5)    
S_dtr_i = cross_val_score(dtr,X,y,scoring=r2_scorer,cv=5)
S_knn_i = cross_val_score(knn,X,y,scoring=r2_scorer,cv=5)  


X_train,X_test,y_train,y_test = train_test_split(X,y,
        test_size=0.3,random_state=1)
dtr.fit(X_train,y_train)

y_train_pred = dtr.predict(X_train)
y_test_pred = dtr.predict(X_test)
print(r2_score(y_train, y_train_pred))
print(r2_score(y_test,y_test_pred))