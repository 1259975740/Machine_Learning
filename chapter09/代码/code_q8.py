# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 18:20:42 2020

@author: Administrator
"""
from sklearn.datasets import make_classification
X,y = make_classification(n_samples=200, n_features=20, n_informative=4, n_redundant=5,  
                    n_repeated=2, n_classes=4, n_clusters_per_class=2, weights=None,  
                    flip_y=0.01, class_sep=1.0, hypercube=True,shift=0.0, scale=1.0,   
                    shuffle=True, random_state=None)  #产生一个分类用数据集

"""参考答案"""
from sklearn.decomposition import PCA
pca = PCA(n_components=6)
X = pca.fit_transform(X)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.3,random_state=1)    #拆分数据

from sklearn.linear_model import LogisticRegression
softmax = LogisticRegression(multi_class = 'multinomial')

softmax.fit(X_train,y_train)    #模型训练
y_train_pred = softmax.predict(X_train)
y_test_pred = softmax.predict(X_test)    #用回归模型得出预测值
#以下代码用于生成评价报表、算出精确率等
from sklearn.metrics import classification_report 
print('训练集结果报表')
print(classification_report(y_train,y_train_pred))
print('测试集结果报表')
print(classification_report(y_test,y_test_pred))