# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:18:46 2020

@author: Zhuo
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

"""产生猫爪数据集"""
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=500,random_state=170)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)


"""参考答案"""
plt.style.use('ggplot')
dbscan = DBSCAN(eps=0.2,min_samples=4)
gmm = GaussianMixture(n_components=3, max_iter=1000)
km = KMeans(n_clusters=3)
sc = SpectralClustering(n_clusters=3,affinity='nearest_neighbors'
                        ,n_neighbors=10,gamma=2)

dbscan.fit(X_aniso)  
gmm.fit(X_aniso)
km.fit(X_aniso)
sc.fit(X_aniso)

"""画图"""
plt.rcParams['font.sans-serif']=['SimHei']    #画图时显示中文字体
plt.rcParams['axes.unicode_minus'] = False
labels = dbscan.labels_    #输出个体的所属簇
plt.figure(figsize=(24,8))
plt.subplot(221)
labels = dbscan.labels_    #输出个体的所属簇
plt.title('DBSCAN',fontsize=20)
plt.scatter(X_aniso[:,0],X_aniso[:,1],c=labels,s=50,cmap='viridis')

plt.subplot(222)
y_proba = gmm.predict_proba(X_aniso)    #输出每个个体在某个簇中的概率
labels = np.argmax(y_proba,axis=1)
plt.title('GMM',fontsize=20)
plt.scatter(X_aniso[:,0],X_aniso[:,1],c=labels,s=50,cmap='viridis')    #画图


plt.subplot(223)
labels = km.labels_    #聚类后每个个体的所属簇
plt.title('K-means',fontsize=20)
plt.scatter(X_aniso[:,0],X_aniso[:,1],c=labels,s=50,cmap='viridis') 


plt.subplot(224)
labels = sc.labels_  
plt.title('K-means',fontsize=20)
plt.scatter(X_aniso[:,0],X_aniso[:,1],c=labels,s=50,cmap='viridis') 
