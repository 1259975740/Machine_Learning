# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 12:14:10 2020

@author: Zhuo
"""

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
plt.style.use('ggplot')
X,y = make_moons(n_samples=200,noise=0.05)   #产生数据集、
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.title('聚类前',fontsize=20)
plt.scatter(X[:,0],X[:,1],s=50)
plt.subplot(122)
plt.title('聚类后',fontsize=20)
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2)    #进行K-means聚类
km.fit(X)
labels = km.labels_
plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap='viridis')    #画图