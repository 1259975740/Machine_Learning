# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:17:32 2020

@author: Administrator
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')
np.random.seed(42)

def generate_data(num,features=2):
    data_size=(num,features)
    data=np.random.randint(0,100,size=data_size)
    label_size = (num,1)
    labels=np.random.randint(0,2,size=label_size)
    return data.astype(np.float32),labels

train_data,labels=generate_data(11)

def plot_data(blue,red):
    plt.scatter(blue[:,0],blue[:,1],c='b',marker='s',s=180)
    plt.scatter(red[:,0],red[:,1],c='r',marker='^',s=180)
    plt.xlabel('坐标1',fontsize=20)
    plt.ylabel('坐标2',fontsize=20)
    
blue=train_data[labels.ravel()==0]
red=train_data[labels.ravel()==1]

   
knn = cv2.ml.KNearest_create()
knn.train(train_data,cv2.ml.ROW_SAMPLE,labels)
new,_ = generate_data(1)
plot_data(blue,red)
plt.plot(new[0,0],new[0,1],'go',markersize=14);
ret,results,neighbor,dist=knn.findNearest(new,3)
print('predicted label:\t',results)
print('Neighbor"s label:\t',neighbor)
print('Distance to neighbor:\t',dist)


data,labels=generate_data(10,3)
data1,_ = generate_data(1,3)