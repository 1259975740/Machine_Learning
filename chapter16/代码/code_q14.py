# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:28:11 2020

@author: Zhuo
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

from sklearn.datasets import make_circles
X,y = make_circles(n_samples=500, factor=.5,noise=.05)

"""参考答案"""
dbscan = DBSCAN(eps=0.2,min_samples=4)
gmm = GaussianMixture(n_components=2, max_iter=1000)
km = KMeans(n_clusters=2)
sc = SpectralClustering(n_clusters=2,affinity='nearest_neighbors'
                        ,n_neighbors=10,gamma=2)


dbscan.fit(X)  
gmm.fit(X)
km.fit(X)
sc.fit(X)


plt.rcParams['font.sans-serif']=['SimHei']    #画图时显示中文字体
plt.rcParams['axes.unicode_minus'] = False
labels = dbscan.labels_    #输出个体的所属簇
plt.figure(figsize=(24,8))
plt.subplot(221)
labels = dbscan.labels_    #输出个体的所属簇
plt.title('DBSCAN',fontsize=20)
plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap='viridis')

plt.subplot(222)
y_proba = gmm.predict_proba(X)    #输出每个个体在某个簇中的概率
labels = np.argmax(y_proba,axis=1)
plt.title('GMM',fontsize=20)
plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap='viridis')    #画图


plt.subplot(223)
labels = km.labels_    #聚类后每个个体的所属簇
plt.title('K-means',fontsize=20)
plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap='viridis') 


plt.subplot(224)
labels = sc.labels_  
plt.title('K-means',fontsize=20)
plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap='viridis') 