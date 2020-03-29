# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 13:52:33 2020

@author: Zhuo
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture


X,y_true = make_blobs(n_samples=300,centers=4,cluster_std=1.0)    #产生如图16.1所示的数据集
gmm = GaussianMixture(n_components=4, max_iter=1000)    #产生高斯混合模型,k=4
gmm.fit(X)    #进行GMM聚类
print(r'均值(聚类中心):',gmm.means_)    #输出GMM参数
print(r'协方差矩阵:',gmm.covariances_)
print(r'权重',gmm.weights_)

"""画出图16.5"""
plt.rcParams['font.sans-serif']=['SimHei']    #画图时显示中文字体
plt.rcParams['axes.unicode_minus'] = False

plt.style.use('ggplot')
fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(122)
plt.subplot(1,2,1) 
plt.title('聚类前',fontsize=20)
plt.scatter(X[:,0],X[:,1],s=50)
c = gmm.covariances_
m = gmm.means_
g1 = Ellipse(xy=m[0], width=4 * np.sqrt(c[0][0, 0]), height=4 * np.sqrt(c[0][1, 1]), fill=False, linestyle='dashed',
             linewidth=3,color='k')
g1_1 = Ellipse(xy=m[0], width=3 * np.sqrt(c[0][0, 0]), height=3 * np.sqrt(c[0][1, 1]), fill=False,
               linestyle='dashed', linewidth=4,color='k')
g1_2 = Ellipse(xy=m[0], width=1.5 * np.sqrt(c[0][0, 0]), height=1.5 * np.sqrt(c[0][1, 1]), fill=False,
               linestyle='dashed', linewidth=5,color='k')
g2 = Ellipse(xy=m[1], width=4 * np.sqrt(c[1][0, 0]), height=4 * np.sqrt(c[1][1, 1]), fill=False, linestyle='dashed',
             linewidth=3,color='k')
g2_1 = Ellipse(xy=m[1], width=3 * np.sqrt(c[1][0, 0]), height=3 * np.sqrt(c[1][1, 1]), fill=False,
               linestyle='dashed', linewidth=4,color='k')
g2_2 = Ellipse(xy=m[1], width=1.5 * np.sqrt(c[1][0, 0]), height=1.5 * np.sqrt(c[1][1, 1]), fill=False,
               linestyle='dashed', linewidth=5,color='k')
g3 = Ellipse(xy=m[2], width=4 * np.sqrt(c[2][0, 0]), height=4 * np.sqrt(c[2][1, 1]), fill=False, linestyle='dashed',
             linewidth=3,color='k')
g3_1 = Ellipse(xy=m[2], width=3 * np.sqrt(c[2][0, 0]), height=3 * np.sqrt(c[2][1, 1]), fill=False,
               linestyle='dashed', linewidth=4,color='k')
g3_2 = Ellipse(xy=m[2], width=1.5 * np.sqrt(c[2][0, 0]), height=1.5 * np.sqrt(c[2][1, 1]), fill=False,
               linestyle='dashed', linewidth=5,color='k')
g4 = Ellipse(xy=m[3], width=4 * np.sqrt(c[3][0, 0]), height=4 * np.sqrt(c[3][1, 1]), fill=False, linestyle='dashed',
             linewidth=3,color='k')
g4_1 = Ellipse(xy=m[3], width=3 * np.sqrt(c[3][0, 0]), height=3 * np.sqrt(c[3][1, 1]), fill=False,
               linestyle='dashed', linewidth=4,color='k')
g4_2 = Ellipse(xy=m[3], width=1.5 * np.sqrt(c[3][0, 0]), height=1.5 * np.sqrt(c[3][1, 1]), fill=False,
               linestyle='dashed', linewidth=5,color='k')
y_proba = gmm.predict_proba(X)    #输出每个个体在某个簇中的概率
labels = np.argmax(y_proba,axis=1)
plt.subplot(1,2,2)
plt.title('聚类后',fontsize=20)
plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap='viridis')
ax.add_artist(g1)
ax.add_artist(g1_1)
ax.add_artist(g1_2)
ax.add_artist(g2)
ax.add_artist(g2_1)
ax.add_artist(g2_2)
ax.add_artist(g3)
ax.add_artist(g3_1)
ax.add_artist(g3_2)
ax.add_artist(g4)
ax.add_artist(g4_1)
ax.add_artist(g4_2)




"""画出AIC和BIC与k的图像"""
kvals = np.arange(2,10)
aics = []
bics = []
for k in kvals:
    gmm = GaussianMixture(n_components=k, max_iter=1000)
    gmm.fit(X)
    aics.append(gmm.aic(X))
    bics.append(gmm.bic(X))
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.title("AIC-k曲线",fontsize=20)
plt.plot(kvals,aics,'o-',linewidth=3,markersize=12)
plt.xlabel('聚类簇数k',fontsize=20)
plt.ylabel('AIC',fontsize=20)

plt.subplot(122)
plt.title("BIC-k曲线",fontsize=20)
plt.plot(kvals,bics,'o-',linewidth=3,markersize=12)
plt.xlabel('聚类簇数k',fontsize=20)
plt.ylabel('BIC',fontsize=20)
