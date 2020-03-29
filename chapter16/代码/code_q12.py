# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 21:29:19 2020

@author: Zhuo
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture

X,y = make_moons(n_samples=200,noise=0.05)   #产生数据集
gmm = GaussianMixture(n_components=2, max_iter=1000)    #产生高斯混合模型,k=4
gmm.fit(X)    #进行GMM聚类
y_proba = gmm.predict_proba(X)    #输出每个个体在某个簇中的概率
labels = np.argmax(y_proba,axis=1)  #输出个体的所属簇
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.title('聚类前',fontsize=20)
plt.scatter(X[:,0],X[:,1],s=50)
plt.subplot(122)
plt.title('聚类后',fontsize=20)
plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap='viridis') 