# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:09:02 2020

@author: Zhuo
"""

import numpy as np    #导入numpy包
from sklearn import datasets   #导入sklearn数据集包
from sklearn import metrics    #导入sklearn指标包
from sklearn import model_selection  #导入sklearn模型包
from sklearn import linear_model     #导入sklearn线性模型包
import matplotlib.pyplot as plt    #导入画图包pyplot，并更名为plt
#plt.style.use('ggplot')   #使用ggplot画图格式，以便用numpy数据格式画图

plt.figure()
X1, Y1 = datasets.make_blobs(n_samples=100,n_features=2, centers=2)
blue = X1[Y1.ravel()==0]
red = X1[Y1.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],marker='o',s=80,label='class 1')
plt.scatter(red[:,0],red[:,1],marker='^',s=80,label='class 2')
plt.legend(loc='best')

plt.figure()
x2,y2=datasets.make_moons(n_samples=100,noise=0.1)
blue = x2[y2.ravel()==0]
red = x2[y2.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],marker='o',s=80,label='class 1')
plt.scatter(red[:,0],red[:,1],marker='^',s=80,label='class 2')
plt.show()

