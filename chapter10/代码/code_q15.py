# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:56:24 2020

@author: Zhuo
"""
import pandas as pd
df = pd.read_csv(r'D:\桌面\我的书\chapter10\数据集\airfoil_self_noise.dat'
                 ,sep='\t',header=None,engine='python')
X = df.iloc[:,0:5].values
y = df.iloc[:,5].values
