# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 22:28:31 2020

@author: Zhuo
"""
from sklearn.datasets import make_classification
X,y = make_classification(n_samples=1000,n_features=2,
     n_redundant=0,weights=(0.90,0.10),random_state=37)
from imblearn.under_sampling import (NearMiss, 
ClusterCentroids,EditedNearestNeighbours,TomekLinks)
from imblearn.over_sampling import (ADASYN,BorderlineSMOTE,
    SVMSMOTE,KMeansSMOTE)

method = ADASYN(n_neighbors=5)    #设置k=5的ADASYN
X_resample1,y_resample1 = method.fit_resample(X,y)
method = BorderlineSMOTE()
X_resample2,y_resample2 = method.fit_resample(X,y)
method = SVMSMOTE()
X_resample3,y_resample3 = method.fit_resample(X,y)
method = KMeansSMOTE()
X_resample4,y_resample4 = method.fit_resample(X,y)
method = NearMiss(n_neighbors=5,version=1)
X_resample5,y_resample5 = method.fit_resample(X,y)
method = NearMiss(n_neighbors=5,version=2)
X_resample6,y_resample6 = method.fit_resample(X,y)
method = NearMiss(n_neighbors=5,version=3)
X_resample7,y_resample7 = method.fit_resample(X,y)
method = TomekLinks()
X_resample8,y_resample8 = method.fit_resample(X,y)
method = ClusterCentroids()
X_resample9,y_resample9 = method.fit_resample(X,y)
method = EditedNearestNeighbours()
X_resample10,y_resample10 = method.fit_resample(X,y)

