# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:17:42 2020

@author: Zhuo
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
datasets = load_boston()    #导入boston房价数据
X,y = datasets.data,datasets.target  
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=4)    #拆分数据
from sklearn import tree
dtr = tree.DecisionTreeRegressor(criterion='mse')    #使用mse作为节点的不纯度度量方法
dtr.fit(X_train,y_train)    #训练决策树模型
"""计算模型的R方"""
from sklearn.metrics import r2_score
y_train_pred = dtr.predict(X_train)
y_test_pred = dtr.predict(X_test)
print('模型在训练集中的R方',r2_score(y_train,y_train_pred))
print('模型在测试集中的R方',r2_score(y_test,y_test_pred))

import numpy as np
"""防止过拟合:"""
alphas = np.arange(0,0.2,0.005) #遍历alphas的取值
dtrs = []   #构建一个包含多个决策树模型的列表
for alpha in alphas:
    dtr = tree.DecisionTreeRegressor(ccp_alpha=alpha)   #对每一个alpha，训练一个决策树
    dtr.fit(X_train, y_train)   #训练决策树模型
    dtrs.append(dtr)
train_R2 = [r2_score(y_train,dtr.predict(X_train)) for dtr in dtrs]  #求出每一个决策树模型在训练集中的R方，并顺序存放在一个列表中，下同
test_R2 = [r2_score(y_test,dtr.predict(X_test)) for dtr in dtrs]
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
"""决策树可视化----输出PDF文件"""
#import os    
#os.environ["PATH"] += os.pathsep + r'E:\Anaconda\release\bin'  #注意修改你的路径
#import graphviz 
#dot_data = tree.export_graphviz(dtr,out_file=None,
#                                feature_names=datasets.feature_names,
#                                )
#import pydotplus
#graph = pydotplus.graph_from_dot_data(dot_data) 
#with open('1.pdf', 'wb') as file:
#    file.write(graph.create_pdf())    #保存文件到当前路径中，并命名为1.png 
