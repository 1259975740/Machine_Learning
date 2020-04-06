# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:43:08 2020

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
"""定义神经网络构造函数"""
def create_network():
    ANN = Sequential()    #定义一个sequential类，以便构造神经网络
    ANN.add(Dense(units=256,activation='relu',input_shape=(7,)))    #构造第一层隐藏层，第一层隐藏层需要input_shape，用来设置输入层的神经元个数。
    ANN.add(Dropout(0.15))    #节点的Dropout正则化，Dropout比例为15%
    ANN.add(Dense(units=128,activation='linear'))    #第二次隐藏层，神经元个数为128，设置激活函数为线性函数
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
    ANN.compile(optimizer='adam',loss='mse')
    return ANN
#使用adam作为参数的搜索算法，设置MSE作为风险函数，同时用计算测试集的MSE
"""集成神经网络模型的搭建、训练与评价"""
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import make_scorer,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingRegressor  #导入Bagging集成分类器库
ANN = KerasClassifier(build_fn=create_network,epochs=500,batch_size=None)  
bagging = BaggingRegressor(base_estimator=ANN,
                            n_estimators=50,max_samples=0.67)    #以ANN为子模型，设置子模型个数为50个，新样本占总样本的0~2/3
bagging.fit(X_train,y_train)    #训练集成模型
y_train_pred = bagging.predict(X_train)
y_test_pred = bagging.predict(X_test)    #计算模型预测值
print('模型在训练集中的R方为',r2_score(y_train,y_train_pred))
print('模型在测试集中的R方为',r2_score(y_test,y_test_pred))
