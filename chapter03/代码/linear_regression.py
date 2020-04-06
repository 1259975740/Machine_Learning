# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:09:02 2020

@author: Zhuo
"""
"""模型训练"""
from sklearn.datasets import load_boston  #导入sklearn自带数据集load_boston
from sklearn.metrics import r2_score,mean_squared_error     #导入函数r2_score、mean_squared_error，用于评价模型
from sklearn.model_selection import train_test_split  #该模块用于拆分数据集
from sklearn.linear_model import LinearRegression    #导入LinearRegression类，用于产生线性回归模型
import matplotlib.pyplot as plt    #导入画图模块pyplot，并更名为plt
boston = load_boston()    #导入boston数据集
X,y = boston.data,boston.target
lr = LinearRegression()    #创建线性回归模型
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)    
#将数据集按照7：3划分
lr.fit(X_train,y_train)    #训练线性回归模型
print('模型的截距项：',lr.intercept_)    #输出模型参数
for i in range(0,13):
    print('模型的ω%d项：%f'%(i+1,lr.coef_[i]))

"""模型评价"""
y_train_pred = lr.predict(X_train)   #模型在训练集上的预测值
mse_train =mean_squared_error(y_train,y_train_pred)    #在训练集上的MSE
r2_train = r2_score(y_train,y_train_pred)   #在训练集上的R方
print("训练集MSE：%.3f\n训练集R方：%.3f"%(mse_train,r2_train))   #输出R方与MSE
y_test_pred = lr.predict(X_test)     #同上，数据集为测试集
mse_test = mean_squared_error(y_test,y_test_pred)
r2_test = r2_score(y_test,y_test_pred)
print("测试集集MSE：%.3f\n测试集集R方：%.3f"%(mse_test,r2_test))

"""画图代码"""
"""由于matplotlib模块默认不显示中文，因此需要额外设置中文字体"""
font1 = {'family' : 'SimHei',
'weight' : 'normal',
'size'   : 16,
}
plt.rcParams['font.sans-serif']=['SimHei']    #画图时显示中文字体
plt.rcParams['axes.unicode_minus'] = False
"""画出图3.15"""
plt.figure(figsize=(12,4))    #创建画布
plt.subplot(121)    
plt.plot(y_test,"-",linewidth=1.5,label='实际值') #画出测试集实际数据点
plt.plot(y_test_pred,"--",linewidth=1.5,label='预测值') #画出测试集拟合数据点
plt.legend(loc='best',prop=font1)  #设置标签位置
plt.xlabel('个体',fontsize=20)   #设置X轴
plt.ylabel('输出值',fontsize=20)    #设置X轴
plt.subplot(122)
plt.plot(y_test,y_test_pred,'o')    #以实际值为x轴，以预测值为y轴
plt.plot([-10,60],[-10,60],'k--')   #画出对角线
plt.axis([-10,60,-10,60])   #设置图形范围
plt.xlabel('预测值',fontsize=20)   
plt.ylabel('实际值',fontsize=20)
r2 = 'R$^2$ = %.3f' %r2_test
mse = 'MSE = %.3f' %mse_test
plt.text(-5,50,mse,fontsize=20) #画出左上角的文字
plt.text(-5,35,r2,fontsize=20)

"""Rigde回归"""
from sklearn.linear_model import Ridge    #导入Ridge类，用于产生Ridge回归
rdg = Ridge(alpha = 0.5)     #使用Ridge回归模型，设置alpha为0.5
rdg.fit(X_train,y_train)   #训练Ridge回归模型
y_train_pred = rdg.predict(X_train)    #计算训练集中的拟合值
mse_train_2 = mean_squared_error(y_train,y_train_pred)    #计算训练集中的MSE与R方，下同
r2_train_2 = r2_score(y_train,y_train_pred)
y_test_pred =rdg.predict(X_test)
mse_test_2 = mean_squared_error(y_test,y_test_pred)
r2_test_2 = r2_score(y_test,y_test_pred)
print("训练集MSE：%.3f\n训练集R方：%.3f"%(mse_train_2,r2_train_2))   #输出评价指标
print("测试集集MSE：%.3f\n测试集集R方：%.3f"%(mse_test_2,r2_test_2))
