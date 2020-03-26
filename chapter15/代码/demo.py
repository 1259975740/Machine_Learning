# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:40:27 2020

@author: Zhuo
"""
from sklearn.datasets import load_wine   
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.linear_model import LogisticRegression    #线性回归
from sklearn.svm import SVC    #导入SVC库
from sklearn.neighbors import KNeighborsClassifier    #导入kNN算法
from sklearn.model_selection import GridSearchCV,cross_val_score
X,y = load_wine(return_X_y=True)    #导入数据集

""""筛选kNN算法的最合适参数k"""
grid = {'n_neighbors':[3,5,7,9,11]}    #定义参数网格
acc_scorer = make_scorer(accuracy_score)    #以精确度
grid_search = GridSearchCV(KNeighborsClassifier(),param_grid=grid,cv=5,scoring=acc_scorer)
grid_search.fit(X,y)   #进行网格寻优
print(grid_search.best_params_)    #输出最优的参数对

"""筛选最合适的SVC"""
grid = {'C':[0.80,0.85,0.90,0.95,1.00],
        'kernel':['linear','rbf','poly']}    #定义参数网格
acc_scorer = make_scorer(accuracy_score)    #以精确度
grid_search = GridSearchCV(SVC(),param_grid=grid,cv=5,scoring=acc_scorer)
grid_search.fit(X,y)   #进行网格寻优
print(grid_search.best_params_)    #输出最优的参数对

"""定义算法"""
lg = LogisticRegression(penalty='none')
knn = KNeighborsClassifier(n_neighbors=3)    #k=3的kNN算法
svc = SVC(C=0.9,kernel='linear')
""""用5折交叉验证，计算所有模型的Si,并计算其均值"""
S_lg_i = cross_val_score(lg,X,y,scoring=acc_scorer,cv=5)    #计算出逻辑回归模型的Si
S_knn_i = cross_val_score(knn,X,y,scoring=acc_scorer,cv=5)    #计算出kNN模型的Si
S_svc_i = cross_val_score(svc,X,y,scoring=acc_scorer,cv=5)    #计算出SVC模型的Si
print('逻辑回归模型的总体效果：',S_lg_i.mean())
print('kNN算法的总体效果：',S_knn_i.mean())
print('SVC模型的总体效果：',S_svc_i.mean())

"""T检验"""
from scipy.stats import ttest_ind    #导入相关库
ttest_ind(S_lg_i,S_svc_i)     #对S_lg_i,S_svc_i进行T检验

""""训练逻辑回归模型"""
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=4)    #按照6:4拆分数据
lg = LogisticRegression(penalty='none')
lg.fit(X_train,y_train)    #训练模型
y_train_pred = lg.predict(X_train)    #训练集的预测值
y_test_pred = lg.predict(X_test)    #测试集中的预测值
print('训练集的精确度为: ',accuracy_score(y_train,y_train_pred))
print('测试集的精确度为: ',accuracy_score(y_test,y_test_pred))
