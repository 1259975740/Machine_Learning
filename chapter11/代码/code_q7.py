
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 22:41:28 2020

@author: Zhuo
"""

import numpy as np
import pandas as pd
datasets = pd.read_csv(r'D:\桌面\我的书\chapter11\数据集\winequality-white.csv',sep=';',engine='python')    #导入数据集
X = datasets.iloc[:,0:12]
y = datasets.iloc[:,-1]

"""参考答案"""
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)    #拆分数据
from sklearn import tree
dtc = tree.DecisionTreeClassifier(random_state=1)   #创建决策树模型
import os    
os.environ["PATH"] += os.pathsep + r'E:\Anaconda\release\bin'  #Graphviz工具的路径
import graphviz 
dot_data = tree.export_graphviz(dtr,out_file=None
                                )
import pydotplus
graph = pydotplus.graph_from_dot_data(dot_data) 
with open('q7.pdf', 'wb') as file:
    file.write(graph.create_pdf())    #保存文件到当前路径中，并命名为1.png 


