# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:39:39 2020

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


"""四分差法"""
def detect_outliers_1(df):    #输入df
    for i in range(0,len(df.iloc[0,:])):
        x = df.iloc[:,i]
        q1,q3 = np.percentile(x, [25,75])
        iqr = q3 - q1
        upper = q3+iqr*1.5
        lower = q1-iqr*1.5
        x[(x[:]>upper)|(x[:]<lower)] = np.nan
    return df
df_copy = df.copy()
df_after_1  = detect_outliers_1(df_copy)

"""正态分布法"""
from sklearn.covariance import EllipticEnvelope
def detect_outliers_2(df):
    data = []
    for i in range(0,len(df.iloc[0,:])):
        x = np.array(df.iloc[:,i],dtype = np.float)
        detector = EllipticEnvelope(contamination=0.1)
        detector.fit(x.reshape(-1,1))
        labels = detector.predict(x.reshape(-1,1))
        x[labels==-1] = np.nan
        data.append(x)
    col_name = df.columns.values.tolist()
    data = np.array(data).T
    df = pd.DataFrame(data,columns=col_name)
    return df
df_after_2 = detect_outliers_2(df)
        

