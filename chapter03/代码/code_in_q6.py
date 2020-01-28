# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 10:32:59 2020

@author: Zhuo
"""
import pandas as pd
from sklearn import model_selection 
from sklearn import metrics
from sklearn import linear_model   
import matplotlib.pyplot as plt 
import numpy as np

fire_df = pd.read_csv(r'D:\桌面\我的书\chapter03\数据集\forestfires.csv'
                           ,sep=',',engine='python')
fire_df.rename(columns = lambda x:x.replace(' ','_'),inplace=True)
month = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,
         'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
week = {'mon':1,'tue':2,'wed':3,'thu':4,'fri':5,'sat':6,'sun':7}
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html
fire_df['month'] = fire_df['month'].replace(month)
fire_df['day'] = fire_df['day'].replace(week)
fire = np.array(fire_df)
X = fire[:,0:np.shape(fire)[1]-1];
y = fire[:,-1]
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,
        test_size=0.3,random_state=1)
model = linear_model.LinearRegression()
model.fit(X_train,y_train)
