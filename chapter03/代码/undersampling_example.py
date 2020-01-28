# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 12:07:47 2020

@author: Zhuo
"""

from sklearn import datasets
from imblearn.under_sampling import OneSidedSelection    #载入单边选择包
X,y = datasets.make_classification(n_samples=1000,n_features=2,
                                     n_redundant=0,weights=(0.95,0.05),
                                     random_state=37)    #按照95：5的比例产生2分类数据集
print('多数类个数: ',X[y==0].shape[0])
print('少数类个数: ',X[y==1].shape[0])
OneSidedSelection_method = OneSidedSelection()  #建立单边选择算法类
X_resample,y_resample = OneSidedSelection_method.fit_resample(X,y)  #进行单边选择，并输出
print('欠采样后多数类个数: ',X_resample[y_resample==0].shape[0])
print('欠采样后少数类个数: ',X_resample[y_resample==1].shape[0])