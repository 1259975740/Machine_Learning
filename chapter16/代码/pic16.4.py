# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 12:58:47 2020

@author: Zhuo
"""

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
plt.rcParams['font.sans-serif']=['SimHei']    #画图时显示中文字体
plt.rcParams['axes.unicode_minus'] = False

X,y_true = make_blobs(n_samples=300,centers=4,cluster_std=1.0)    #产生如图16.1所示的数据集
kvals = np.arange(2,10)    #遍历k=2：10
inertias = []
for k in kvals:
    km = KMeans(n_clusters=k)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(kvals,inertias,'o-',linewidth=3,markersize=12)
plt.xlabel('聚类簇数k',fontsize=16)
plt.ylabel('惯量',fontsize=16)