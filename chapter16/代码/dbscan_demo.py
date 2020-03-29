# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:33:31 2020

@author: Zhuo
"""
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
plt.style.use('ggplot')
X,y = make_moons(n_samples=200,noise=0.05)   #产生数据集
dbscan = DBSCAN(eps=0.2,min_samples=4)    #定义DBSCAN的参数为epsilon=1,m'=4   
dbscan.fit(X)   #进行DBSCAN聚类
plt.rcParams['font.sans-serif']=['SimHei']    #画图时显示中文字体
plt.rcParams['axes.unicode_minus'] = False
labels = dbscan.labels_    #输出个体的所属簇
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.title('聚类前',fontsize=20)
plt.scatter(X[:,0],X[:,1],s=50)
plt.subplot(122)
plt.title('聚类后',fontsize=20)
plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap='viridis')    #画图
