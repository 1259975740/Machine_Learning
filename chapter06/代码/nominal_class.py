# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:27:07 2020

@author: Zhuo
"""
import numpy as np   #导入numpy库，并更名为np，用于产生随机数据
import pandas as pd    #导入pandas库，用于产生dataframe格式
from sklearn.preprocessing import LabelBinarizer
test_data = np.random.choice(('Beijing','Shanghai','Guangzhou','Shenzhen'),size=(20))
#np.random.choice 能够根据元组随机产生数据，参数size定义数据长度
test_df= pd.DataFrame({"City":test_data})
#pd.DataFrame将np格式的数据转换为dataframe，注意要定义键值：“City”
one_hot = LabelBinarizer()    #生成One—hot模型
test_df_trans = one_hot.fit_transform(test_df)    #使用one-hot算法转换数据
one_hot.classes_    #显示转换后各个特征代表的意义。

test_df_trans_2 = pd.get_dummies(test_df)