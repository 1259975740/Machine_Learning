# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 19:47:47 2020

@author: Administrator
"""

import pandas as pd
datasets = pd.read_excel(r'D:\桌面\我的书\chapter11\数据集\Concrete_Data.xls')    #导入数据集
X = datasets.iloc[:,0:9]
y = datasets.iloc[:,-1]