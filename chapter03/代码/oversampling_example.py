# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:12:37 2020

@author: Zhuo
"""
from sklearn import datasets
from imblearn.over_sampling import KMeansSMOTE  #边界SMOTE包
X,y = datasets.make_classification(n_samples=1000,n_features=2,
                                     n_redundant=0,weights=(0.95,0.05),
                                     random_state=37)    #按照95：5的比例产生2分类数据集
print('多数类个数: ',X[y==0].shape[0])
print('少数类个数: ',X[y==1].shape[0])
BorderlineSMOTE_method = BorderlineSMOTE()  #建立边界SMOTE法类
X_resample,y_resample = BorderlineSMOTE_method.fit_resample(X,y)  #进行边界SMOTE过采样，并输出
print('欠采样后多数类个数: ',X_resample[y_resample==0].shape[0])
print('欠采样后少数类个数: ',X_resample[y_resample==1].shape[0])
