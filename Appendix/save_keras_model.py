# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:28:35 2020

@author: Zhuo
"""

from keras.models import Sequential
from keras.layers import Dense,Activation    #导入神经层构造包
from keras.utils import to_categorical    #导入one-hot编码法
from sklearn.datasets import load_boston
from keras.models import load_model
X,y = load_boston(return_X_y=True)
ANN = Sequential()    #定义一个sequential类，以便构造神经网络
ANN.add(Dense(units=64,activation='relu',input_shape=(len(X[1,:]),)))    #构造第一层隐藏层，第一层隐藏层需要input_shape，用来设置输入层的神经元个数。
ANN.add(Dense(units=1,activation='linear'))    #输出层，units即节点个数
ANN.compile(optimizer='adam',loss='mse')
#使用adam作为参数的搜索算法，设置交叉熵作为风险函数（极大似然法），同时使用精确度作为度量模型拟合优度的指标
ANN.fit(X,y,epochs=100,batch_size=50)
#对模型进行训练，最大迭代步数为100，随机搜索算法的mini-batch为50
ANN.save('model.h5')    #保存模型为model.h5文件
model = load_model("model.h5")    #读取model.h5文件，并导入模型

