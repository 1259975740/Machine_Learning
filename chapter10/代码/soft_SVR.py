# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 23:11:30 2020

@author: Zhuo
"""
import numpy as np
"""产生一个实例数"""
x = np.arange(-2,2,0.05).reshape(-1,1)
l = len(x)
y = np.zeros(l)
for i in range(0,l):
    y[i] = 2*x[i]**3+3*x[i]**2+x[i]+1.5+np.random.uniform(-6,4)
    
import matplotlib.pyplot as plt    #画图代码
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.scatter(x,y,s=15)
# plt.xlabel(r'$x$',fontsize=24)
# plt.ylabel(r'$y$',fontsize=24)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)    #拆分数据
from sklearn.svm import SVR
svr = SVR(C=1.0,epsilon=0.5,kernel='poly',degree=3)  #使用C=1.0，e=0.5的软间隔SVR模型，并使用三次多项式核函数
svr.fit(x_train,y_train)    #模型训练
y_train_pred = svr.predict(x_train)
y_test_pred = svr.predict(x_test)    #用回归模型得出预测值

from sklearn.metrics import r2_score
print('模型训练集的R方为：',r2_score(y_train,y_train_pred))   #计算并输出训练集中的R方
print('模型测试集的R方为：',r2_score(y_test,y_test_pred))

plt.subplot(1,2,2)
y_pred = svr.predict(x);    #画出带超平面
plt.scatter(x,y,s=15)
plt.plot(x,y_pred)
# plt.xlabel(r'$x$',fontsize=24)
# plt.ylabel(r'$y$',fontsize=24)