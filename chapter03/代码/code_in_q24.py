# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:05:11 2020

@author: Zhuo
"""

import pandas as pd
from sklearn import model_selection 
from sklearn import metrics
from sklearn import linear_model   
import matplotlib.pyplot as plt 
import numpy as np

cars_df = pd.read_excel(r'D:\桌面\我的书\chapter03\数据集\car.xlsx')
buying_maint_safety = {'low':0,'med':1,'high':2,'vhigh':3}
doors = {'5more':5}
persons = {'more':6}
boot = {'small':0,'med':1,'big':2}
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html
cars_df['总体价格'] = cars_df['总体价格'].replace(buying_maint_safety)
cars_df['保养要求'] = cars_df['保养要求'].replace(buying_maint_safety)
cars_df['安全性'] = cars_df['安全性'].replace(buying_maint_safety)
cars_df['车门数'] = cars_df['车门数'].replace(doors)
cars_df['核载人数'] = cars_df['核载人数'].replace(persons)
cars_df['车厢大小'] = cars_df['车厢大小'].replace(boot)
cars = np.array(cars_df)
