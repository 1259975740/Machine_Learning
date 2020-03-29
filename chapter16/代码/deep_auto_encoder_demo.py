# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:59:19 2020

@author: Zhuo
"""

from keras.layers import Input,Dense
from keras.models import Sequential,Model
from sklearn.datasets import load_digits
digits = load_digits()    #导入digits数据集，该数据集有64个特征
X = digits.data/255    #导入数据集，进行最大绝对值标准化
Y = X    #生成因变量
autoencoder = Sequential()
autoencoder.add(Dense(32, input_shape=(64,), activation='relu'))    #定义输入层与隐藏层，通过input_shape设置输入层的神经元个数
autoencoder.add(Dense(16, activation='relu'))
autoencoder.add(Dense(2, activation='relu',name="encode_output_layer"))     #添加神经元为2的隐藏层，同时命名该层。
autoencoder.add(Dense(16, activation='relu'))
autoencoder.add(Dense(32, activation='relu'))
autoencoder.add(Dense(64, activation='linear'))
autoencoder.compile(optimizer="adam", loss="mse")    #设置寻优算法为Adam，风险函数为mse
autoencoder.fit(X, Y, epochs=100,batch_size=80,validation_split=0.3)    #训练模型,同时用30%的数据作为测试集评价模型
encoder = Model(autoencoder.input,autoencoder.get_layer("encode_output_layer").output)    #截断原神经网络，产生一个自动编码器
reduced_X = encoder.predict(X)    #对原数据进行编码（降维）
print(len(reduced_X[1,:]))    #输出降维后数据的特征个数。