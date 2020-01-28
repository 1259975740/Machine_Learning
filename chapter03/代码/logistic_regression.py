# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:35:40 2020

@author: Zhuo
"""
import numpy as np    #导入numpy包
from sklearn import datasets   #导入sklearn数据集包
from sklearn import metrics    #导入sklearn指标包
from sklearn import model_selection  #导入sklearn模型包
from sklearn import linear_model     #导入sklearn线性模型包
import matplotlib.pyplot as plt    #导入画图包pyplot，并更名为plt
iris = datasets.load_iris()
idx = iris.target!=2    #以下三行代码提出了类别2的鸢尾花
x = iris.data[idx]
y = iris.target[idx]
X_train,X_test,y_train,y_test = model_selection.train_test_split(x
        ,y,test_size=0.3,random_state=1)    #拆分数据集
model = linear_model.LogisticRegression()   #生成逻辑回归模型
model.fit(X_train,y_train)  #训练模型
"""模型评价"""
y_train_pred = model.predict(X_train)
print(metrics.classification_report(y_train,y_train_pred))
y_test_pred = model.predict(X_test)
print(metrics.classification_report(y_test,y_test_pred))
"""画出ROC曲线、计算AUC"""
y_score = model.decision_function(X_test)   #计算属于各类的概率
fpr,tpr,thresholds = metrics.roc_curve(y_test,y_score)    #找到阀值拐点，并相应的计算TPR和FPR
print('AUC的值:',metrics.auc(fpr,tpr))    #计算并输出AUC
plt.figure()    #以下代码画出ROC曲线
plt.plot(fpr,tpr,'k',label='ROC Curve')
plt.plot([0,1],[0,1],'k--')
plt.axis([-0.05,1,0,1.05])  #画中位虚线
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc='best')


