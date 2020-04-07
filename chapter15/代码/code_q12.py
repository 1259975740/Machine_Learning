# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:59:55 2020

@author: Zhuo
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
X,y = load_iris(return_X_y=True)

acc_scorer = make_scorer(accuracy_score)
grid = {'max_depth':[1,2,3,4,5],
        'ccp_alpha':[0,0.0001,0.001,0.005,0.01]}    #定义参数网格为最大树深度和α_ccp
grid_search = GridSearchCV(DecisionTreeClassifier(),param_grid=grid,cv=5,scoring=acc_scorer)
grid_search.fit(X,y)   #进行网格寻优
print(grid_search.best_params_)    #输出最优的参数对


grid = {'C':[0.4,0.5,0.6,0.7,0.8],
        'kernel':['linear','rbf','poly']}    
grid_search = GridSearchCV(SVC(),param_grid=grid,cv=5,scoring=acc_scorer)
grid_search.fit(X,y)   #进行网格寻优
print(grid_search.best_params_)    #输出最优的参数对


NB = GaussianNB()
svc = SVC(C=0.5,kernel='linear')
dtc = DecisionTreeClassifier(max_depth=3,ccp_alpha=0.005)
S_NB_i = cross_val_score(NB,X,y,scoring=acc_scorer,cv=5)    
S_svc_i = cross_val_score(svc,X,y,scoring=acc_scorer,cv=5)
S_dtc_i = cross_val_score(dtc,X,y,scoring=acc_scorer,cv=5)  

X_train,X_test,y_train,y_test = train_test_split(X,y,
        test_size=0.3,random_state=1)
svc.fit(X_train,y_train)

y_train_pred = svc.predict(X_train)
y_test_pred = svc.predict(X_test)
print(accuracy_score(y_train, y_train_pred))
print(accuracy_score(y_test,y_test_pred))