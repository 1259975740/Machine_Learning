# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:36:21 2020

@author: Zhuo
"""
import pandas as pd
sale_tranc_df = pd.read_csv(r'D:\桌面\我的书\chapter16\数据集\Sales_Transactions_Dataset_Weekly.csv',
                      sep=',',engine='python') 
X = sale_tranc_df.iloc[:,1:53]    #提取数据集

"""参考答案"""
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
plt.rcParams['font.sans-serif']=['SimHei']    #画图时显示中文字体
plt.rcParams['axes.unicode_minus'] = False

kvals = np.arange(2,10)    #遍历k=2：10
inertias = []
for k in kvals:
    km = KMeans(n_clusters=k)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(kvals,inertias,'o-',linewidth=3,markersize=12)
plt.xlabel('聚类簇数k',fontsize=16)
plt.ylabel('惯量',fontsize=16)

"""选择k=3的K-mane聚类即可,当然也可以考虑进行标准化再聚类"""