# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:36:21 2020

@author: Zhuo
"""
import pandas as pd
sale_tranc_df = pd.read_csv(r'D:\桌面\我的书\chapter16\数据集\Sales_Transactions_Dataset_Weekly.csv',
                      sep=',',engine='python') 
sale_tranc_df = sale_tranc_df.iloc[:,0:53]    #提取数据集
