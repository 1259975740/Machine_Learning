# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:28:06 2020

@author: Zhuo
"""
import numpy as np   #导入numpy库，并更名为np，用于产生随机数据
import pandas as pd    #导入pandas库，用于产生dataframe格式
from sklearn.feature_extraction import DictVectorizer

doc_1 = {"Apple":3,"Banana":4,"Cherry":5,"Orange":5,'Author':'Zhang'}
doc_2 = {"Apple":1,"Banana":10,"Cherry":3,"Orange":1,'Author':'Wang'}
doc_3 = {"Apple":2,"Banana":8,"Cherry":5,"Orange":7,'Author':'Chen'}
doc_4 = {"Apple":0,"Banana":1,"Cherry":8,"Orange":2,'Author':'Zhuo'}
doc_counts = [doc_1,doc_2,doc_3,doc_4]
vec = DictVectorizer(sparse=False)    #生成字典向量化器
doc_trans = vec.fit_transform(doc_counts)    #使用向量化器将每一个字典向量化
columns_name = vec.feature_names_   #获取生成的特征名称
doc_df = pd.DataFrame(doc_trans,columns=columns_name)   #生成dataframe