# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 10:30:18 2020

@author: Zhuo
"""


import cv2    #导入Opencv库
import matplotlib.pyplot as plt


img = cv2.imread('test.jpg',1)    #读取彩色图像
img_data = img/255.0    #进行最大绝对值标准化
img_data = img_data.reshape((-1,3))    #将每个通道展开成向量
from sklearn.cluster import KMeans    #K均值聚类库
km = KMeans(n_clusters=36)    #设置k=24
km.fit(img_data)    #进行Kmeans聚类
centers = km.cluster_centers_    #输出聚类中心
labels = km.labels_    #输出每个个体的所属簇
new_colors = centers[labels].reshape((-1,3))    #用像素对应的聚类中心替代原有像素值
img_recolored = new_colors.reshape(img.shape)    #将向量还原成矩阵
print(len(set(list(new_colors[:,1]))))    #输出压缩后图像的颜色种类个数

plt.figure(figsize=(12,4))
plt.subplot(121)
plt.title('压缩调色板前',fontsize=20)
plt.imshow(img),plt.axis('off')    #展示图像

plt.subplot(122)
plt.title('压缩调色板后',fontsize=20)
plt.imshow(img_recolored),plt.axis('off')    #展示图像
plt.show()
