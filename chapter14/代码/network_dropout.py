# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:12:20 2020

@author: Zhuo
"""

from keras.models import Sequential
from keras.layers import Dense,Dropout    #导入神经层构造包
from keras.utils import to_categorical    #导入one-hot编码法
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
X,y = load_digits(return_X_y=True)   #导入数据集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=4)    #按照7：3拆分数据
scaler = MinMaxScaler()  #进行Minmax标准化
X = scaler.fit_transform(X)    #标准化数据
y = scaler.fit_transform(y.reshape(-1,1))


BP = Sequential()    #顺序地构建神经网络
BP.add(Dense(input_shape=(X.shape[1],),units=16,activation='relu'))    #第一层隐藏层
BP.add(Dense(units=32,activation='relu'))    #第二层隐藏层
BP.add(Dense(units=64,activation='relu'))   #第三层隐藏层
BP.add(Dense(units=128,activation='relu'))    #第四层隐藏层
BP.add(Dense(units=256,activation='relu'))    #第五层隐藏层
BP.add(Dense(units=128,activation='relu'))    #第六层隐藏层
BP.add(Dense(units=1,activation='linear'))    #输出层，激活函数为线性函数。
BP.compile(optimizer='adam',loss='mse')    #设置搜索算法为adam，风险函数为MSE
BP.fit(X_train,y_train,epochs=300,batch_size=100)    #设置最大迭代步骤为300， mini-batch=100，并训练模型
y_train_pred = BP.predict(X_train)    #使用BP神经网络模型
y_test_pred = BP.predict(X_test)
print('训练集R方： ',r2_score(y_train, y_train_pred))    #计算BP神经网络在训练集中的R方，下同
print('测试集R方： ',r2_score(y_test, y_test_pred))


# """正则化后"""
rBP = Sequential()    #顺序构造BP神经网络
rBP.add(Dense(input_shape=(X.shape[1],),units=16,activation='relu'))    #第一层隐藏层
rBP.add(Dropout(0.1))    #对第一层进行节点Dropout正则化，每一次Dropout的比例为10%，下同
rBP.add(Dense(units=32,activation='relu'))
rBP.add(Dropout(0.1))
rBP.add(Dense(units=64,activation='relu'))
rBP.add(Dropout(0.1))
rBP.add(Dense(units=128,activation='relu'))
rBP.add(Dropout(0.1))
rBP.add(Dense(units=256,activation='relu'))
rBP.add(Dropout(0.1))
rBP.add(Dense(units=128,activation='relu'))
rBP.add(Dropout(0.1))
rBP.add(Dense(units=1,activation='linear'))
rBP.compile(optimizer='adam',loss='mse')    #设置搜索算法为adam，风险函数为MSE
rBP.fit(X_train,y_train,epochs=300,batch_size=100)    #设置最大迭代步骤为300， mini-batch=100，并训练模型
y_train_pred = rBP.predict(X_train)    #使用BP神经网络模型
y_test_pred = rBP.predict(X_test)
print('训练集R方： ',r2_score(y_train, y_train_pred))    #计算BP神经网络在训练集中的R方，下同
print('测试集R方： ',r2_score(y_test, y_test_pred))
