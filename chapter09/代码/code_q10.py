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

"""参考答案"""
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')    #忽略warning信息
soybean_df = soybean_df.dropna()    #按行删除缺失个体
X = soybean_df.iloc[:,1:36]   #提取出特征集
y = soybean_df.iloc[:,0]   #提取出因变量标签


X_train,X_test,y_train,y_test = model_selection.train_test_split(
        X,y,test_size=0.3,random_state=1)    #拆分数据
softmax = LogisticRegression(multi_class = 'multinomial')   #使用softmax回归
softmax.fit(X_train,y_train)    #模型训练
y_train_pred = softmax.predict(X_train)
y_test_pred = softmax.predict(X_test)    #用回归模型得出预测值
#以下代码用于生成评价报表、算出精确率等
from sklearn.metrics import accuracy_score	#引入评价用包
print('训练集中的精度为： ',accuracy_score(y_train,y_train_pred))
print('测试集集中的精度为： ',accuracy_score(y_test,y_test_pred))
