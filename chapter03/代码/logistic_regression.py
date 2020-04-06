# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:35:40 2020

@author: Zhuo
"""
import numpy as np    #导入numpy包
from sklearn.datasets import load_iris    #导入load_iris函数
from sklearn.metrics import classification_report,roc_curve,auc     #用于计算分类模型的各项拟合优度指标
from sklearn.model_selection import train_test_split  #用于拆分数据集
from sklearn.linear_model import LogisticRegression    #导入逻辑回归模块
import matplotlib.pyplot as plt    #导入画图模块pyplot，并更名为plt
iris = load_iris()    #导入数据集
idx = iris.target!=2    #以下三行代码剔除了类别2的鸢尾花
X = iris.data[idx]    #将特征与因变量赋值为X，y
y = iris.target[idx]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)    #按7：3拆分数据集
lg = LogisticRegression(penalty='none')   #生成逻辑回归模型
lg.fit(X_train,y_train)  #训练模型
"""模型评价"""
y_train_pred = lg.predict(X_train)    #计算模型的预测值
print(classification_report(y_train,y_train_pred))    #输出准确率、召回率、F1值、精确度等构成的报表
y_test_pred = lg.predict(X_test)
print(classification_report(y_test,y_test_pred))

"""画出ROC曲线、计算AUC"""
y_score = lg.decision_function(X_test)   #计算属于各类的概率
fpr,tpr,thresholds = roc_curve(y_test,y_score)    #找到阀值拐点，并相应的计算TPR和FPR
print('AUC的值:',auc(fpr,tpr))    #计算并输出AUC
plt.figure()    #以下代码画出ROC曲线
plt.plot(fpr,tpr,'k',label='ROC Curve')
plt.plot([0,1],[0,1],'k--')
plt.axis([-0.05,1,0,1.05])  #画中位虚线
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc='best')


