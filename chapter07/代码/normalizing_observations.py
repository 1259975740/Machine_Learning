# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:00:29 2020

@author: Zhuo
"""
import numpy as np
import pandas as pd 
test_df = pd.DataFrame(columns=['身高','体重','肺活量'])
test_df['身高'] = [162,165,185]
test_df['体重'] = [60,45,75]
test_df['肺活量'] = [2500,3200,4240]    #创建一个dataframe
from sklearn.preprocessing import Normalizer
scaler_sample = Normalizer(norm='l2') #使用第二范数对个体进行标准化
df_obs = scaler_sample.fit_transform(test_df)