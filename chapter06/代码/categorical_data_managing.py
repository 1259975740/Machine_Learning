# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:14:40 2020

@author: Zhuo
"""
import numpy as np   #导入numpy库，并更名为np，用于产生随机数据
import pandas as pd    #导入pandas库，用于产生dataframe格式
test_data = np.random.choice(('low','med','high','vhigh'),size=(20))
#np.random.choice 能够根据元组随机产生数据，参数size定义数据长度
test_df= pd.DataFrame({"Price":test_data})
#pd.DataFrame将np格式的数据转换为dataframe，注意要定义键值：“Price”
price_change = {'low':1,'med':2,'high':3,'vhigh':4}
#定义字典，用于数据转换。
test_df['Price'] = test_df['Price'].replace(price_change)
"""LabelEncoder处理"""
test_data = np.random.choice(('low','med','high','vhigh'),size=(20))
test_df= pd.DataFrame({"Price":test_data})

from sklearn.preprocessing import LabelEncoder #导入LabelEncoder库
le = LabelEncoder()   #生成LabelEncoder模型
test_df = le.fit_transform(test_df)   #转换数据
span = list(set(test_df))    #将span赋值为test_df的取值范围（转换后）
le.inverse_transform(span)    #数据拟转换，可以查看各个数字的含义