# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 09:34:06 2020

@author: Zhuo
"""
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
plt.rcParams['font.sans-serif']=['SimHei']    #画图时显示中文字体
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt   
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}

X,y_true = make_blobs(n_samples=300,centers=4,cluster_std=1.0)    #产生如图16.1所示的数据集

"""画图16.1（左）"""
# plt.figure(figsize=(12,4))
# plt.subplot(1,2,1)
# plt.scatter(X[:,0],X[:,1],s=50)
# plt.title('聚类前',fontsize=20)
# plt.subplot(1,2,2)
# plt.title('聚类后',fontsize=20)
km = KMeans(n_clusters=4)    #设置参数n_clusters为4，即聚类簇数为4
km.fit(X)    #进行Kmeans聚类
labels = km.labels_    #聚类后每个个体的所属簇
centers = km.cluster_centers_    #查看每个簇的聚类中心

"""画图16.1（右）"""
# plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap='viridis')
# plt.scatter(centers[:,0],centers[:,1],c='k',s=300,marker='d')

