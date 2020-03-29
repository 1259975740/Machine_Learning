# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:37:03 2020

@author: Zhuo
"""

import cv2    #导入Opencv库
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

img = cv2.imread('test.jpg',1)    #用灰度通道读取图像

img_data = img/255.0
img_data = img_data.reshape((-1,3))

def plot_pixels(data,colors=None):
    pixel = data.T
    R,G,B = pixel[0],pixel[1],pixel[2] 
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(R,G,B)
    ax.set_zlabel('B', fontdict={'size': 18})
    ax.set_ylabel('G', fontdict={'size': 18})
    ax.set_xlabel('R', fontdict={'size': 18})
# plot_pixels(img_data)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=16)
km.fit(img_data)
centers = km.cluster_centers_
labels = km.labels_
new_colors = centers[labels].reshape((-1,3))
img_recolored = new_colors.reshape(img.shape)
plt.imshow(img_recolored,cmap="gray"),plt.axis('off')    #使用plt库打开
plt.show()
#cv2.imshow("test picture",img)   #用cv2库显示图像，窗口标签为“test picture"
#cv2.waitKey()
