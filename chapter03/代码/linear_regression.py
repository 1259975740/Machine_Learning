# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:09:02 2020

@author: Zhuo
"""
"""模型训练"""
import numpy as np    #导入numpy包
from sklearn import datasets   #导入sklearn数据集包
from sklearn import metrics    #导入sklearn指标包
from sklearn import model_selection  #导入sklearn模型包
from sklearn import linear_model     #导入sklearn线性模型包
import matplotlib.pyplot as plt    #导入画图包pyplot，并更名为plt
plt.style.use('ggplot')   #使用ggplot画图格式，以便用numpy数据格式画图
boston = datasets.load_boston()    #导入boston数据集
model = linear_model.LinearRegression()    #创建线性回归模型
X_train,X_test,y_train,y_test = model_selection.train_test_split(
        boston.data,boston.target,test_size=0.3,random_state=1)    
#将数据集按照7：3划分
model.fit(X_train,y_train)    #训练线性回归模型
print('模型的截距项：',model.intercept_)    #输出模型参数
for i in range(0,13):
    print('模型的ω%d项：%f'%(i+1,model.coef_[i]))

"""模型评价"""
y_train_pred = model.predict(X_train)   #模型在训练集上的预测值
mse_train = metrics.mean_squared_error(y_train,y_train_pred)    #在训练集上的MSE
r2_train = metrics.r2_score(y_train,y_train_pred)   #在训练集上的R方
print("训练集MSE：%.3f\n训练集R方：%.3f"%(mse_train,r2_train))   #输出R方与MSE
y_test_pred = model.predict(X_test)     #同上，数据集为测试集
mse_test = metrics.mean_squared_error(y_test,y_test_pred)
r2_test = metrics.r2_score(y_test,y_test_pred)
print("测试集集MSE：%.3f\n测试集集R方：%.3f"%(mse_test,r2_test))

"""可视化评价指标"""
plt.figure()    #创建画布
plt.subplot(211)    
plt.plot(y_test,"-",linewidth=1,label='real value') #画出测试集实际数据点
plt.plot(y_test_pred,"--",linewidth=1,label='fitted value') #画出测试集拟合数据点
plt.legend(loc='best')  #设置标签位置
plt.xlabel('test dataset points')   #设置X轴
plt.ylabel('target y value')    #设置X轴
plt.subplot(212)
plt.plot(y_test,y_test_pred,'o')    #以实际值为x轴，以预测值为y轴
plt.plot([-10,60],[-10,60],'k--')   #画出对角线
plt.axis([-10,60,-10,60])   #设置图形范围
plt.xlabel('fitted value')   
plt.ylabel('real value')
r2 = 'R$^2$ = %.3f' %r2_test
mse = 'MSE = %.3f' %mse_test
plt.text(-5,50,mse,fontsize=12) #画出左上角的文字
plt.text(-5,35,r2,fontsize=12)

"""Rigde回归"""
model_rgd = linear_model.Ridge(alpha = 0.5)     #使用Ridge回归模型，设置alpha为0.5
model_rgd.fit(X_train,y_train)  #模型训练
y_train_pred = model_rgd.predict(X_train)   #计算训练集中的拟合值
mse_train_2 = metrics.mean_squared_error(y_train,y_train_pred)  #计算训练集中的MSE与R方，下同
r2_train_2 = metrics.r2_score(y_train,y_train_pred)
y_test_pred = model_rgd.predict(X_test)
mse_test_2 = metrics.mean_squared_error(y_test,y_test_pred)
r2_test_2 = metrics.r2_score(y_test,y_test_pred)
print("训练集MSE：%.3f\n训练集R方：%.3f"%(mse_train_2,r2_train_2))   #输出评价指标
print("测试集集MSE：%.3f\n测试集集R方：%.3f"%(mse_test_2,r2_test_2))
