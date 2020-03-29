# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 18:24:26 2020

@author: Zhuo
"""
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
plt.style.use('ggplot')
X,y = make_moons(n_samples=200,noise=0.05)   #产生数据集
sc = SpectralClustering(n_clusters=2,affinity='nearest_neighbors'
                        ,n_neighbors=10,gamma=2)    #设置聚类簇数（子图数）为2，构成子图的方法为近邻个数为10的k-近邻法，设置相似度的高斯距离参数为：gamma=2，
sc.fit(X)   #进行谱聚类
plt.rcParams['font.sans-serif']=['SimHei']    #画图时显示中文字体
plt.rcParams['axes.unicode_minus'] = False
labels = sc.labels_    #输出个体的所属簇
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.title('聚类前',fontsize=20)
plt.scatter(X[:,0],X[:,1],s=50)
plt.subplot(122)
plt.title('聚类后',fontsize=20)
plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap='viridis')    #画图