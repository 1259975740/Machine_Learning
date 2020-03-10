# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:53:11 2020

@author: Zhuo
"""

import numpy as np
import cv2
import os
os.getcwd() #获取当前工作目录
os.chdir('D:\桌面\我的书\chapter06\数据集')    #修改工作路径
images = []
data = np.empty([110, 100], np.float32)
for idx in range(110):
    image = cv2.imread(u'Data/s' + str(idx + 1) + '.bmp',0)
    images.append(image)
    """可在此进行图像处理"""
    image = cv2.resize(image,(10,10))    #图片降维处理
    data[idx] = image.flatten()
file = open('../数据集/Data/labels.txt')
labels = np.array(file.readline().strip('\n').split(','), np.int32)


"""画图代码：读者可以运行该代码直观地查看数据集"""
import matplotlib.pyplot as plt
fig, axes = plt.subplots(5,5, figsize=(8, 8),subplot_kw={'xticks':[], 'yticks':[]},
                        gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i], cmap='gray', interpolation='nearest')
    ax.text(0.07, 0.07, str(label[i]),transform=ax.transAxes, color='white')



"""参考答案"""
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(
        data,labels,test_size=0.1,random_state=1)    #拆分数据
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train)
from sklearn.metrics import accuracy_score	#引入评价用包
y_train_pred = svc.predict(X_train)
y_test_pred = svc.predict(X_test)
print('训练集中的精确度为',accuracy_score(y_train,y_train_pred))
print('测试集中的精确度为',accuracy_score(y_test,y_test_pred))



