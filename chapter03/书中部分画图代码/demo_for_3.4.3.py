# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:46:18 2020

@author: Zhuo
"""
import numpy as np    #导入numpy包
from sklearn import datasets   #导入sklearn数据集包
from sklearn import metrics    #导入sklearn指标包
from sklearn import model_selection  #导入sklearn模型包
from sklearn import linear_model     #导入sklearn线性模型包
import matplotlib.pyplot as plt    #导入画图包pyplot，并更名为plt
plt.style.use('ggplot')   #使用ggplot画图格式，以便用numpy数据格式画图


#X1, Y1 = datasets.make_classification(n_samples=100,n_features=2, n_redundant=0, n_informative=2)
#"""画图"""
plt.figure()
blue = X1[Y1.ravel()==0]
red = X1[Y1.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],marker='o',s=80,label='class 1')
plt.scatter(red[:,0],red[:,1],marker='^',s=80,label='class 2')
plt.legend(loc='best')

X_train,X_test,y_train,y_test = model_selection.train_test_split(
        X1,Y1,test_size=0.3,random_state=42)
model = linear_model.LogisticRegression()
model.fit(X_train,y_train)
w0 = model.intercept_
w = model.coef_
x_1 = np.linspace(-8,8,20)
x_2 = (-w[0][0] * x_1 - w0[0])/w[0][1]
plt.plot(x_1, x_2)
plt.axis([-3.5,4.5,-2.5,3.5])

y_score = model.decision_function(X_test)
fpr,tpr,thresholds = metrics.roc_curve(y_test,y_score)
plt.figure()
plt.plot(fpr,tpr,'k',label='ROC Curve')
plt.plot([0,1],[0,1],'k--')
plt.axis([-0.05,1,0,1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc='best')

