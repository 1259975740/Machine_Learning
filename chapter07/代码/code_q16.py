# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 12:50:53 2020

@author: Zhuo
"""

import numpy as np
import pandas as pd 
test_df = pd.DataFrame(columns=['身高','体重','肺活量'])
test_df['身高'] = [162,165,168,163,170,174,176,178,173,180,182,185]
test_df['体重'] = [600,45.2,50.2,60.3,59.6,62.4,64.5,72.7,70.0,75.4,76.1,74.8]
test_df['肺活量'] = [2500,3200,4240,4000,4400,3800,4120,4400,44111,3800,5200,6000]    
target = np.array([0,0,0,0,1,0,1,1,1,1,1,1])

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator=model,scoring='neg_mean_squared_error')
rfecv.fit(test_df,target)
col_name = df.columns.values.tolist()
new_data = rfecv.transform(test_df)    #过滤后
