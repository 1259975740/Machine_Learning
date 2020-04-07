# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:49:53 2020

@author: Zhuo
"""
import numpy as np
import pandas as pd 
df = pd.DataFrame(columns=['身高','体重','肺活量'])
df['身高'] = [162,165,168,163,170,174,176,178,173,180,182,185]
df['体重'] = [600,45.2,50.2,60.3,59.6,62.4,64.5,72.7,70.0,75.4,76.1,74.8]
df['肺活量'] = [2500,3200,4240,4000,4400,3800,4120,4400,44111,3800,5200,6000]    
target = np.array([0,0,0,0,1,0,1,1,1,1,1,1])
col_name = ['身高', '体重', '肺活量']
"""实现Zscore标准化"""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_zscore = scaler.fit_transform(df)    #进行Zscore标准化
df_zscore = pd.DataFrame(df_zscore, columns=col_name)

"""实现MinMax标准化"""
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df_minmax = scaler.fit_transform(df)    #进行最大最小值标准化
df_minmax = pd.DataFrame(df_minmax, columns=col_name)

"""实现Robust标准化::"""
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df_rob = scaler.fit_transform(df)    
df_rob = pd.DataFrame(df_rob, columns=col_name)
"""绝对值最大标准化"""
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df_max = scaler.fit_transform(df)    
df_max = pd.DataFrame(df_max, columns=col_name)

