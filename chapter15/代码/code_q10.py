# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:49:36 2020

@author: Zhuo
"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer,mean_squared_error
from sklearn.model_selection import GridSearchCV
X,y = load_boston(return_X_y=True)
grid = {'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21,23]}
mse_scorer = make_scorer(mean_squared_error)

gridsearch = GridSearchCV(KNeighborsRegressor(),param_grid=grid,cv=5,scoring=mse_scorer)
gridsearch.fit(X,y)
print(gridsearch.best_params_)    #输出参数


X_train,X_test,y_train,y_test = train_test_split(X,y,
        test_size=0.3,random_state=1)
knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(X_train,y_train)

y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)
print(mean_squared_error(y_train, y_train_pred))
print(mean_squared_error(y_test,y_test_pred))