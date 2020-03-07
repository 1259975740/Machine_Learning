# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 18:34:12 2020

@author:Zhuo
"""

import matplotlib.pyplot as plt  
from sklearn.datasets import make_gaussian_quantiles  
X, y = make_gaussian_quantiles(n_samples=400,n_features=2, n_classes=4)  

"""画图代码"""
X0 = X[y.ravel()==0]
plt.scatter(X0[:, 0], X0[:, 1], marker='o')  
X1 = X[y.ravel()==1]
plt.scatter(X1[:, 0], X1[:, 1], marker='x')  
X2 = X[y.ravel()==2]
plt.scatter(X2[:, 0], X2[:, 1], marker='*')  
X3 = X[y.ravel()==3]
plt.scatter(X3[:, 0], X3[:, 1], marker='^')  

#参考代码
#from sklearn import model_selection
#from sklearn.linear_model import LogisticRegression
#X_train,X_test,y_train,y_test = model_selection.train_test_split(
#        X,y,test_size=0.3,random_state=1)    #拆分数据
#softmax = LogisticRegression(multi_class = 'multinomial')   #使用softmax回归
#softmax.fit(X_train,y_train)    #模型训练
#y_train_pred = softmax.predict(X_train)
#y_test_pred = softmax.predict(X_test)    #用回归模型得出预测值
##以下代码用于生成评价报表、算出精确率等
#from sklearn.metrics import classification_report , accuracy_score	#引入评价用包
#print('训练集结果报表')
#print(classification_report(y_train,y_train_pred))
#print('训练集中的精度为： ',accuracy_score(y_train,y_train_pred))
#print('测试集结果报表')
#print(classification_report(y_test,y_test_pred))
#print('测试集集中的精度为： ',accuracy_score(y_test,y_test_pred))
