# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:29:13 2020

@author: Zhuo
"""

import sys 
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D,AveragePooling2D,Flatten
import numpy as np
sys.path.append(r'D:\桌面\我的书\chapter10\代码')
import code_q14
X_train,y_train,X_test,y_test = code_q14.data_generate()   #使用code_q14产生数据集
width = height = X_train.shape[1]    #得到图像的尺寸
y_train = to_categorical(y_train,num_classes=10)
y_test = to_categorical(y_test,num_classes=10)   #one-hot编码将类别因变量转换成10个变两个
X_train = X_train.reshape(X_train.shape[0],28,28,1).astype(np.float32)/255
X_test = X_test.reshape(X_test.shape[0],28,28,1).astype(np.float32)/255    #将数据标准化
model = Sequential()   #创建神经网络类
model.add(Conv2D(16,kernel_size=(3,3),
                 input_shape=(width,height,1),activation='relu'))    #创建一个卷积层，核为3X3
model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))    #添加池化层，并根据平均值将矩阵缩放成2X2
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))    #添加池化层，缩放成2X2
model.add(Flatten())    #添加Flatten层
model.add(Dense(units=512,activation='relu'))    #BP神经网络的隐藏层，512个节点
model.add(Dense(units=10,activation='sigmoid'))    #输出层
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])    #使用Adam算法训练模型
history = model.fit(X_train,y_train,epochs=100,batch_size=256,validation_data=(X_test,y_test))    #定义随机搜索算法的mini-batch=256
import matplotlib.pyplot as plt
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
}
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(history.history['acc'],linewidth=3,label='Train')
plt.plot(history.history['val_acc'],linewidth=3,linestyle='dashed',label='Test')
plt.xlabel('Epoch',fontsize=20)
plt.ylabel('精确度',fontsize=20)
plt.legend(prop=font1)

