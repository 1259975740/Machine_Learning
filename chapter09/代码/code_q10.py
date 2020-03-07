# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 19:26:33 2020

@author: Zhuo
"""
import pandas as pd
soybean_df = pd.read_excel(r'D:\桌面\我的书\chapter09\数据集\soybean.xlsx')    #读取数据文件
import numpy as np
soybean_df = soybean_df.replace(['?'],np.nan)   #将缺失数据替换成NaN
X = soybean_df.iloc[:,1:36]   #提取出特征集
y = soybean_df.iloc[:,0]   #提取出因变量标签
