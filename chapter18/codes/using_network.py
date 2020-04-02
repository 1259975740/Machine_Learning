# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:45:49 2020

@author: Zhuo
"""
from data_prepro import data_generate_final
X_pca,y_cod,_,_ = data_generate_final()    #导入数据集
from keras.models import Sequential
import numpy as np
from keras.layers import Dense,Dropout    #导入神经层构造包
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
X_train,X_test,y_train,y_test = train_test_split(X_pca,y_cod,test_size=0.3,random_state=4)    #按照7：3拆分数据

ANN = Sequential()    #定义一个sequential类，以便构造神经网络
ANN.add(Dense(units=256,activation='relu',input_shape=(7,)))    #构造第一层隐藏层，第一层隐藏层需要input_shape，用来设置输入层的神经元个数。
ANN.add(Dropout(0.15))    #节点的Dropout正则化，Dropout比例为15%
ANN.add(Dense(units=128,activation='linear',input_shape=(7,)))    #第二次隐藏层，神经元个数为128，设置激活函数为线性函数
ANN.add(Dropout(0.15))
ANN.add(Dense(units=64,activation='relu'))    
ANN.add(Dropout(0.15))
ANN.add(Dense(units=32,activation='linear'))   
ANN.add(Dropout(0.15))
ANN.add(Dense(units=16,activation='relu'))    
ANN.add(Dropout(0.15))
ANN.add(Dense(units=8,activation='linear'))    
ANN.add(Dense(units=4,activation='relu'))    
ANN.add(Dense(units=1,activation='linear'))    #输出层，激活函数为线性函数
ANN.compile(optimizer='adam',loss='mse',metrics=['mse'])
#使用adam作为参数的搜索算法，设置MSE作为风险函数，同时用计算测试集的MSE
ANN.fit(X_train,y_train,epochs=500,batch_size=None,validation_data=(X_test,y_test))
#对模型进行训练，最大迭代步数为500，同时每一次迭代用测试集进行一次模型评价
y_train_pred = ANN.predict(X_train)
y_test_pred = ANN.predict(X_test)    #输出模型在训练集、测试集中的预测值
print('模型在训练集中的R方为',r2_score(y_train,y_train_pred))
print('模型在测试集中的R方为',r2_score(y_test,y_test_pred))    #分别计算模型的R方


