# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 19:22:00 2020

@author: Zhuo
"""

from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import demo_data   #导入文件demo_data，以产生数据集
from keras.layers import Dense,Activation    #导入神经层构造包
from keras.utils import to_categorical    #导入one-hot编码法
from sklearn.model_selection import train_test_split
def plot_boundary(model,X,y):
    x_min,x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min,y_max = X[:,0].min()-5, X[:,0].max()+5
    h = 0.1
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),
                        np.arange(y_min,y_max,h))
    X_hypo = np.column_stack((xx.ravel().astype(np.float32),
                             yy.ravel().astype(np.float32)))
    
    try:
        zz = np.argmax(model.predict(X_hypo),axis=1)

    except:
        zz = model.predict(X_hypo)
    zz = zz.reshape(xx.shape)
    plt.contourf(xx,yy,zz,cmap=plt.cm.binary,alpha=0.4)
    X0 = X[y.ravel()==0]
    plt.scatter(X0[:, 0], X0[:, 1], marker='o',c='k')  
    X1 = X[y.ravel()==1]
    plt.scatter(X1[:, 0], X1[:, 1], marker='x')  
X,y = demo_data.data_generate()    #产生示例数据集

lg = LogisticRegression()    #训练一个逻辑回归函数
lg.fit(X,y)    #训练数据（这里不拆分数据集）
"""画出分类界限"""
plt.figure(figsize=(12,4))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.3)    #设置制图之间的左右间距
plt.rcParams['font.sans-serif']=['SimHei']    #画图时显示中文字体
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(1,2,1)
plt.title('逻辑回归',fontsize=16)
plt.xlabel(r'$x_1$',fontsize=16)
plt.ylabel(r'$x_2$',fontsize=16)
plot_boundary(lg,X,y)   #画出分类界限。函数代码请参阅代码文件

y1 = to_categorical(y)   #使用one-hot编码法对y进行编码，使y称为一个二维向量，向量的每一个元素代表一个类
X_train,X_test,y_train,y_test = train_test_split(X,y1,test_size=0.3,random_state=4)    #按照7：3拆分数据
ANN = Sequential()    #定义一个sequential类，以便构造神经网络
ANN.add(Dense(units=64,activation='relu',input_shape=(2,)))    #构造第一层隐藏层，第一层隐藏层需要input_shape，用来设置输入层的神经元个数。
ANN.add(Dense(units=64,activation='relu'))    #第二次隐藏层，神经元个数为64
ANN.add(Dense(units=64,activation='relu'))    #第三层隐藏层
ANN.add(Dense(units=2,activation='sigmoid'))    #输出层，units即节点个数必须等于向量y的元素个数
ANN.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#使用adam作为参数的搜索算法，设置交叉熵作为风险函数（极大似然法），同时使用精确度作为度量模型拟合优度的指标
ANN.fit(X_train,y_train,epochs=500,batch_size=50,validation_data=(X_test,y_test))
#对模型进行训练，最大迭代步数为500，随机搜索算法的mini-batch为50，同时每一次迭代用测试集进行一次模型评价
    
"""画出分类界限"""
plt.subplot(1,2,2)
plt.title('神经网络',fontsize=16)
plt.xlabel(r'$x_1$',fontsize=16)
plt.ylabel(r'$x_2$',fontsize=16)
plot_boundary(ANN,X,y)

from keras.utils import plot_model
plot_model(ANN,to_file='12.6.png',show_shapes=True)    #可视化神经网络模型ANN
