# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 19:47:47 2020

@author: Administrator
"""

import pandas as pd
datasets = pd.read_excel(r'D:\桌面\我的书\chapter11\数据集\Concrete_Data.xls')    #导入数据集
X = datasets.iloc[:,0:9]
y = datasets.iloc[:,-1]


"""参考答案"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=4)    #拆分数据
from sklearn import tree

alphas = np.arange(0,0.2,0.005) #遍历alphas的取值
dtrs = []   #构建一个包含多个决策树模型的列表
for alpha in alphas:
    dtr = tree.DecisionTreeRegressor(ccp_alpha=alpha)   #对每一个alpha，训练一个决策树
    dtr.fit(X_train, y_train)   #训练决策树模型
    dtrs.append(dtr)
train_R2 = [r2_score(y_train,dtr.predict(X_train)) for dtr in dtrs]  #求出每一个决策树模型在训练集中的R方，并顺序存放在一个列表中，下同
test_R2 = [r2_score(y_test,dtr.predict(X_test)) for dtr in dtrs]    #测试集
"""画出图像"""
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt   
font1 = {'family' : 'SimHei',
'weight' : 'normal',
'size'   : 16,
}
plt.xlabel(r"$\alpha_{cpp}$",fontsize=20)
plt.ylabel(r"$R^2$",fontsize=16)
plt.plot(alphas, train_R2, label="训练集",
        linestyle = 'dashed',drawstyle="steps-post")
plt.plot(alphas, test_R2, label="测试集",
        drawstyle="steps-post")
plt.legend(prop=font1)   #显示图例
