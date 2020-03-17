# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 21:20:37 2020

@author: Zhuo
"""
import pandas as pd
hr = pd.read_csv(r'D:\桌面\我的书\chapter13\数据集\HR.csv',engine='python')
attrition_change = {'Yes':1,'No':0}   
hr['Attrition'] = hr['Attrition'].replace(attrition_change)    #使用replace方法替代字符串

"""one-hot编码法编码无序离散特征"""
dummy_busnstrvl = pd.get_dummies(hr['BusinessTravel'], prefix='busns_trvl')
dummy_dept = pd.get_dummies(hr['Department'], prefix='dept')
dummy_edufield = pd.get_dummies(hr['EducationField'], prefix='edufield')
dummy_gender = pd.get_dummies(hr['Gender'], prefix='gend')
dummy_jobrole = pd.get_dummies(hr['JobRole'], prefix='jobrole')
dummy_maritstat = pd.get_dummies(hr['MaritalStatus'], prefix='maritalstat') 
dummy_overtime = pd.get_dummies(hr['OverTime'], prefix='overtime') 


"""保留连续特征"""
continuous_columns = ['Age','DailyRate','DistanceFromHome','Education','EnvironmentSatisfaction',
'HourlyRate', 'JobInvolvement', 'JobLevel','JobSatisfaction','MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 
'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction','StockOptionLevel', 'TotalWorkingYears', 
'TrainingTimesLastYear','WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
'YearsWithCurrManager']    #连续性特征，不需要one-hot编码
hr_continuous = hr[continuous_columns]


"""产生新的数据集"""
hr_new = pd.concat([dummy_busnstrvl,dummy_dept,dummy_edufield,dummy_gender,dummy_jobrole,
  dummy_maritstat,dummy_overtime,hr_continuous],axis=1)    #特征
y = hr['Attrition']    #因变量