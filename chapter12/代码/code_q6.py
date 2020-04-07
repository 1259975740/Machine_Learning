# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:10:33 2020

@author: Zhuo
"""

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils import to_categorical    #导入one-hot编码法
from keras.layers import Dense,Activation    #导入神经层构造包
from sklearn.metrics import classification_report,accuracy_score

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
datasets = load_iris()
X = datasets.data
y = datasets.target
y1 = to_categorical(y)   #使用one-hot编码法对y进行编码，使y称为一个二维向量，向量的每一个元素代表一个类

X_train,X_test,y_train,y_test = train_test_split(X,y1,test_size=0.3,random_state=4)    #按照7：3拆分数据
ANN = Sequential()    #定义一个sequential类，以便构造神经网络
ANN.add(Dense(units=128,activation='relu',input_shape=(X.shape[1],)))    #构造第一层隐藏层，第一层隐藏层需要input_shape，用来设置输入层的神经元个数。
ANN.add(Dense(units=64,activation='relu'))    #第二次隐藏层，神经元个数为64
ANN.add(Dense(units=32,activation='relu'))    #第三层隐藏层
ANN.add(Dense(units=16,activation='relu'))    
ANN.add(Dense(units=8,activation='relu'))    
ANN.add(Dense(units=3,activation='sigmoid'))    
ANN.compile(optimizer='adam',loss='binary_crossentropy')
ANN.fit(X_train,y_train,epochs=400,batch_size=256,validation_data=(X_test,y_test))


