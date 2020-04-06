# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 20:41:31 2020

@author: Administrator
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import numpy as np

dignosis_df = pd.read_excel(r'D:\桌面\我的书\chapter03\数据集\diagnosis.xlsx')
change = {'yes':1,'no':0}
dignosis_df['恶心症状'] = dignosis_df['恶心症状'].replace(change)
dignosis_df['肌肉酸疼'] = dignosis_df['肌肉酸疼'].replace(change)
dignosis_df['咳嗽不止'] = dignosis_df['咳嗽不止'].replace(change)
dignosis_df['血尿'] = dignosis_df['血尿'].replace(change)
dignosis_df['肺门肿大'] = dignosis_df['肺门肿大'].replace(change)
dignosis_df['是否感染'] = dignosis_df['是否感染'].replace(change)
dignosis = np.array(dignosis_df)
X = dignosis_df.iloc[:,range(0,6)]
y = dignosis_df['是否感染']

X_train,X_test,y_train,y_test = train_test_split(X,y,
        test_size=0.3,random_state=1)
lg = LogisticRegression(penalty='none')
lg.fit(X_train,y_train)
y_train_pred = lg.predict(X_train)
y_test_pred = lg.predict(X_test)
print(classification_report(y_train, y_train_pred))
print(classification_report(y_test,y_test_pred))
