# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:20:17 2020

@author: Zhuo
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')    #忽略warning信息

df = pd.DataFrame(columns=['身高','体重','肺活量'])
df['身高'] = [162,165,168,163,170,174,176,178,173,180,182,185]
df['体重'] = [600,45.2,50.2,60.3,59.6,62.4,64.5,72.7,70.0,75.4,76.1,74.8]
df['肺活量'] = [2500,3200,4240,4000,4400,3800,4120,4400,44111,3800,5200,6000]    
target = np.array([0,0,0,0,1,0,1,1,1,1,1,1])


"""LOF法"""
from sklearn.neighbors import LocalOutlierFactor    #导入LOF库
def detect_outliers_1(df):
    data = []
    for i in range(0,len(df.iloc[0,:])):
        x = np.array(df.iloc[:,i],dtype = np.float)
        detector = LocalOutlierFactor(n_neighbors=8)    #定义k=5
        labels = detector.fit_predict(x.reshape(-1,1))
        x[labels==-1] = np.nan
        data.append(x)
    col_name = df.columns.values.tolist()
    data = np.array(data).T
    df = pd.DataFrame(data,columns=col_name)
    return df
df_after_1 = detect_outliers_1(df)


"""Kmeans聚类法"""
from sklearn.cluster import KMeans
from collections import Counter
def detect_outliers_2(df):
    data = []
    for i in range(0,len(df.iloc[0,:])):
        x = np.array(df.iloc[:,i],dtype = np.float)
        detector = KMeans(n_clusters=4)
        detector.fit(x.reshape(-1,1))
        labels = detector.predict(x.reshape(-1,1))
        d = Counter(labels)
        k = np.array([k for k,v in d.items() if v==1])    #找到那些只有一个个体的聚类簇
        x[labels == k] = np.nan
        data.append(x)
    col_name = df.columns.values.tolist()
    data = np.array(data).T
    df = pd.DataFrame(data,columns=col_name)
    return df
df_after_2 = detect_outliers_2(df)

