# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:02:17 2020

@author: Zhuo
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score,jaccard_score,precision_score,recall_score,fbeta_score,matthews_corrcoef
from sklearn.model_selection import train_test_split

X,y = load_iris(return_X_y=True)    #导入数据
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=4)    #按照7:3拆分数据
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_test_pred = knn.predict(X_test)
print('测试集的精确度为: ',accuracy_score(y_test,y_test_pred))
print('测试集的马修斯系数为: ',matthews_corrcoef(y_test,y_test_pred))
print('测试集的Jaccard得分为: ',jaccard_score(y_test,y_test_pred,average='macro'))
print('测试集的准确率为: ',precision_score(y_test,y_test_pred,average='macro'))
print('测试集的召回率为: ',recall_score(y_test,y_test_pred,average='macro'))
print('测试集的F1值为: ',fbeta_score(y_test,y_test_pred,average='macro',beta=1))
