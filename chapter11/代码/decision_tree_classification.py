# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 22:41:28 2020

@author: Zhuo
"""

from sklearn.datasets import load_breast_cancer
import numpy as np
import os    
os.environ["PATH"] += os.pathsep + r'E:\Anaconda\release\bin'  #注意修改你的路径

datasets = load_breast_cancer()
X,y = datasets.data,datasets.target    #导入数据
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=4)    #拆分数据
from sklearn import tree
dtc = tree.DecisionTreeClassifier(random_state=4)   #创建决策树模型
dtc.fit(X_train,y_train)    #模型训练
y_train_pred = dtc.predict(X_train)   #用模型求出预测值
y_test_pred = dtc.predict(X_test)
from sklearn.metrics import accuracy_score    #计算模型的精确度
print("模型在训练集中的精度为：",accuracy_score(y_train,y_train_pred))
print("模型在测试集中的精度为：",accuracy_score(y_test,y_test_pred))

#
"""画图"""

import matplotlib.pyplot as plt
#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
path = dtc.cost_complexity_pruning_path(X_train, y_train)   #得到不同的alpha_ccp与模型效果
alphas = path.ccp_alphas
node_num = []
dtcs = []
for alpha in alphas:
    dtc = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
    dtc.fit(X_train, y_train)
    dtcs.append(dtc)
    node_num.append(dtc.tree_.node_count)
train_scores = [dtc.score(X_train, y_train) for dtc in dtcs]
test_scores = [dtc.score(X_test, y_test) for dtc in dtcs]


plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=None, hspace=None)

plt.subplot(2,1,1)
plt.xlabel(r"$\alpha_{cpp}$",fontsize=20)
plt.ylabel("精确度",fontsize=16)
plt.plot(alphas[:-1], train_scores[:-1], label="训练集",linestyle='dashed',
        drawstyle="steps-post")
plt.plot(alphas[:-1], test_scores[:-1], label="测试集",
        drawstyle="steps-post")
plt.xlim(0,0.03)
plt.ylim(0.85,1.025)
plt.legend()

plt.subplot(2,1,2)
plt.plot(alphas[:-1], node_num[:-1], drawstyle="steps-post")
plt.xlabel(r"$\alpha_{ccp}$",fontsize=20)
plt.ylabel("树的节点个数",fontsize=16)
plt.xlim(0,0.03)
plt.ylim(0.38)
plt.show()

"""可视化决策树"""
import graphviz 
dot_data = tree.export_graphviz(dtc,out_file=None,
                             feature_names=datasets.feature_names
                             ,class_names=datasets.target_names)
#创建一个.dot文件，该文件是一个图像文件
import pydotplus
from IPython.display import Image
graph = pydotplus.graph_from_dot_data(dot_data)    #将.dot文件转换成png文件并展示
with open('1.png', 'wb') as file:
    file.write(graph.create_png())    #保存文件到当前路径中，并命名为1.png