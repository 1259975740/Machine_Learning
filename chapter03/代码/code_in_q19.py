# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 20:41:31 2020

@author: Administrator
"""
import pandas as pd
from sklearn import model_selection 
from sklearn import metrics
from sklearn import linear_model   
import matplotlib.pyplot as plt 
import numpy as np

dignosis_df = pd.read_excel(r'D:\桌面\我的书\chapter03\数据集\diagnosis.xlsx')
change = {'yes':1,'no':0}
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html
dignosis_df['恶心症状'] = dignosis_df['恶心症状'].replace(change)
dignosis_df['肌肉酸疼'] = dignosis_df['肌肉酸疼'].replace(change)
dignosis_df['咳嗽不止'] = dignosis_df['咳嗽不止'].replace(change)
dignosis_df['血尿'] = dignosis_df['血尿'].replace(change)
dignosis_df['肺门肿大'] = dignosis_df['肺门肿大'].replace(change)
dignosis_df['是否感染'] = dignosis_df['是否感染'].replace(change)
dignosis = np.array(dignosis_df)
