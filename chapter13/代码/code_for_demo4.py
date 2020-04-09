# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:12:30 2020

@author: Zhuo
"""

from sklearn.ensemble import AdaBoostRegressor  #导入Bagging集成分类器库
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import demo_data   #导入文件demo_data，以产生数据集
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
"""产生数据集"""
X,y = demo_data.data_generate()    #产生示例数据集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=4)    #按照7：3拆分数据
"""深度为1的决策树模型"""
dtc = DecisionTreeRegressor(criterion='mse',max_depth=1)
adaboost = AdaBoostRegressor(base_estimator=dtc,n_estimators=10)    #用AdaBoost集成10个单深度决策树
stacking = BaggingRegressor(base_estimator=dtc,max_samples=1.0,n_estimators=10)    #将Ada集成，同时设置新样本占总样本的0~100%
stacking.fit(X_train,y_train)   #训练模型 
y_train_pred = stacking.predict(X_train)
y_test_pred = stacking.predict(X_test)
print('训练集的R方为：',r2_score(y_train,y_train_pred))
print('测试集的R方为：',r2_score(y_test,y_test_pred))


import matplotlib.pyplot as plt   
font1 = {'family' : 'SimHei',
'weight' : 'normal',
'size'   : 16,
}
plt.figure(figsize=(6,4))
plt.rcParams['font.sans-serif']=['SimHei']    #画图时显示中文字体
plt.rcParams['axes.unicode_minus'] = False
plt.plot(y,label='观测数据',linewidth=2)
plt.plot(stacking.predict(X),linestyle='dashed',label='拟合曲线',linewidth=4)
plt.xlabel(r'$x$',fontsize=20)
plt.ylabel(r'$y$',fontsize=24)
plt.legend(prop=font1)
