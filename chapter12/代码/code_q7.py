# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 23:38:27 2020

@author: Zhuo
"""
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Activation    #导入神经层构造包
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
datasets = load_boston()
X = datasets.data
y = datasets.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=4)    #按照7：3拆分数据
ANN = Sequential()    #定义一个sequential类，以便构造神经网络
ANN.add(Dense(units=128,activation='relu',input_shape=(X.shape[1],)))    #构造第一层隐藏层，第一层隐藏层需要input_shape，用来设置输入层的神经元个数。
ANN.add(Dense(units=64,activation='relu'))    #第二次隐藏层，神经元个数为64
ANN.add(Dense(units=32,activation='relu'))    #第三层隐藏层
ANN.add(Dense(units=16,activation='relu'))    
ANN.add(Dense(units=8,activation='relu'))    
ANN.add(Dense(units=1,activation='linear'))   
ANN.compile(optimizer='adam',loss='mse')
ANN.fit(X_train,y_train,epochs=400,batch_size=256,validation_data=(X_test,y_test))


y_train_pred = ANN.predict(X_train)
y_test_pred = ANN.predict(X_test)
print('训练集中的R方为',r2_score(y_train,y_train_pred))
print('测试集中的R方为',r2_score(y_test,y_test_pred))