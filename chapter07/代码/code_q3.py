# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:25:59 2020

@author: Zhuo
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')    #忽略warning信息

df = pd.DataFrame(columns=['身高','体重','肺活量'])
df['身高'] = [162,165,168,163,170,174,176,178,173,180,182,185]
df['体重'] = [600,45.2,50.2,60.3,59.6,62.4,64.5,72.7,70.0,75.4,76.1,74.8]
df['肺活量'] = [2500,3200,4240,4000,4400,3800,4120,4400,44111,3800,5200,6000]    
target = np.array([0,0,0,0,1,0,1,1,1,1,1,1])

def detect_outliers(df):    #输入df
    for i in range(0,len(df.iloc[0,:])):
        x = df.iloc[:,i]
        q1,q3 = np.percentile(x, [25,75])
        iqr = q3 - q1
        upper = q3+iqr*1.5
        lower = q1-iqr*1.5
        x[(x[:]>upper)|(x[:]<lower)] = np.nan
    return df
        

df_after  = detect_outliers(df)