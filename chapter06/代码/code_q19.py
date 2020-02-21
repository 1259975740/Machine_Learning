# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 23:01:09 2020

@author: Zhuo
"""
import numpy as np
import cv2
import os
os.getcwd() #获取当前工作目录
os.chdir('D:\桌面\我的书\chapter06\数据集')    #修改工作路径
data = np.empty([110, 10000], np.float32)
for idx in range(110):
    image = cv2.imread(u'Data/s' + str(idx + 1) + '.bmp',0)
    """可在此进行图像处理"""
    #可进行resize，如image = resize(image,(50,50))
    data[idx] = image.flatten()
file = open('../数据集/Data/labels.txt')
label = np.array(file.readline().strip('\n').split(','), np.int32)