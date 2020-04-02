# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 19:05:35 2020

@author: Zhuo
"""
from load_dataset import data_generate
_,waste_df_before = data_generate()    #导入去除缺失值、异常值、没有经过标准化的数据
X = waste_df_before[['COD去除率 %','出水VFA','反应器的ph']]    #提取三个用于聚类的特征

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()    #进行绝对值最大标准化
X = scaler.fit_transform(X)    #进行绝对值最大标准化（式7.4）




"""画出折臂图18.6的代码"""
plt.rcParams['font.sans-serif']=['SimHei']    #画图时显示中文字体
plt.rcParams['axes.unicode_minus'] = False
kvals = np.arange(2,8)    #遍历k=2：7
inertias = []
for k in kvals:   #对每一个k，进行K-means聚类，同时计算惯量
    km = KMeans(n_clusters=k)
    km.fit(X)
    inertias.append(km.inertia_)    #计算本次K-means聚类的惯量
plt.plot(kvals,inertias,'o-',linewidth=3,markersize=12)    #画出折臂图
plt.xlabel('聚类簇数k',fontsize=16)    #设置坐标轴名称
plt.ylabel('惯量',fontsize=16)

km = KMeans(n_clusters=3)    #进行k=3的K-means聚类
km.fit(X)
print(km.cluster_centers_)    #输出K-means聚类的聚类中心
X_centers = km.cluster_centers_    #聚类中心
print('逆标准化后聚类中心的值',scaler.inverse_transform(X_centers))    #对聚类中心进行逆标准化

X_belong_2=scaler.inverse_transform(X[km.labels_==1])   #找到属于第二簇的个体
print('最低VFA',np.min(X_belong_2[:,1]))    #输出属于第二簇的个体中，VFA的最低值