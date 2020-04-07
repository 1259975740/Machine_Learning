# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 12:04:32 2020

@author: Zhuo
"""

import numpy as np
import matplotlib.pyplot as plt
#随机生成一个二特征数据集，范围在（0.5,1.5）之间
cluster1=np.random.uniform(0.5,1.5,(2,50))
#随机生成一个二特征数据集，范围在（2.5,3.5）之间
cluster2=np.random.uniform(2.5,3.5,(2,50))
#hstack拼接操作
X=np.hstack((cluster1,cluster2)).T
plt.figure()
plt.style.use('ggplot')
plt.axis([0,4,0,4])
plt.xlabel(u'feature 1')
plt.ylabel(u'feature 2')
plt.plot(X[:,0],X[:,1],'.',MarkerSize=10)

"""参考答案"""
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2)
km.fit(X)
labels = km.predict(X)    #返回各个个体的所属簇，详细内容将在第十六章讨论
