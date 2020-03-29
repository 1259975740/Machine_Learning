# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:40:37 2020

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




X,y_true = make_blobs(n_samples=300,centers=4,cluster_std=1.0)
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.scatter(X[:,0],X[:,1],s=50)
plt.title('聚类前',fontsize=20)
plt.subplot(1,2,2)
plt.title('聚类后',fontsize=20)
km = KMeans(n_clusters=4)
km.fit(X)
labels = km.labels_
centers = km.cluster_centers_
plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap='viridis')
plt.scatter(centers[:,0],centers[:,1],c='k',s=300,marker='d')
