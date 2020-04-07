# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 23:03:56 2020

@author: Zhuo
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer,accuracy_score,classification_report
from sklearn.model_selection import GridSearchCV
X,y = load_breast_cancer(return_X_y=True)
grid = {'n_neighbors':[1,3,5,7,9,11,13,15]}
acc_scorer = make_scorer(accuracy_score)
gridsearch = GridSearchCV(KNeighborsClassifier(),param_grid=grid,cv=5,scoring=acc_scorer)
gridsearch.fit(X,y)
print(gridsearch.best_params_)    #输出参数

X_train,X_test,y_train,y_test = train_test_split(X,y,
        test_size=0.3,random_state=1)
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train,y_train)

y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)
print(classification_report(y_train, y_train_pred))
print(classification_report(y_test,y_test_pred))

