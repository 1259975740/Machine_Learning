# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 23:01:09 2020

@author: Zhuo
"""
import numpy as np
import cv2
import os
os.getcwd() #获取当前工作目录
os.chdir('D:\桌面\我的书\chapter06\数据集')    #注意修改工作路径
data = []
for i in range(110):
    image = cv2.imread(u'Data/s' + str(i + 1) + '.bmp',0)
    """可在此进行图像处理"""
    image = cv2.resize(image,(10,10))
    #可进行resize，如image = resize(image,(50,50))
    data.append(image.flatten())
file = open('../数据集/Data/labels.txt')
labels = np.array(file.readline().strip('\n').split(','), np.int32)

"""参考答案"""
from sklearn.model_selection  import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
X_train,X_test,y_train,y_test = train_test_split(data,labels,
        test_size=0.3,random_state=1)
lg = LogisticRegression(penalty='none')
lg.fit(X_train,y_train)
y_train_pred = lg.predict(X_train)
y_test_pred = lg.predict(X_test)
print(classification_report(y_train, y_train_pred))
print(classification_report(y_test,y_test_pred))