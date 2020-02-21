# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:59:25 2020

@author: Zhuo
"""
import numpy as np   #导入numpy库，并更名为np，用于产生随机数据
import pandas as pd    #导入pandas库，用于产生dataframe格式
from sklearn.preprocessing import MultiLabelBinarizer
test_data = np.random.choice(('Beijing','Shanghai','Guangzhou','Shenzhen'),size=(20,2))
#产生20组2维数据
test_df= pd.DataFrame({"所在地":test_data[:,0],'出生地':test_data[:,1]})
#将数据转化为dataframe格式
one_hot = MultiLabelBinarizer()    #生成One—hot模型
test_trans = one_hot.fit_transform(test_data)    #使用one-hot算法转换数据
one_hot.classes_    #显示转换后各个特征代表的意义。
"""输出dataframe"""
test_df_trans = pd.DataFrame(test_trans,columns=one_hot.classes_)

