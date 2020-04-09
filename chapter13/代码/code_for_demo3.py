# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 09:23:02 2020

@author: Zhuo
"""

from sklearn.svm import SVR    #SVR模型
from sklearn.tree import DecisionTreeRegressor    #回归决策树
from sklearn.linear_model import LinearRegression    #线性回归
from sklearn.linear_model import Ridge    #Ridge回归
import pandas as pd
import matplotlib.pyplot as plt
import demo_data   #导入文件demo_data，以产生数据集
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
"""产生数据集"""
X,y = demo_data.data_generate()    #产生示例数据集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=4)    #按照6:4拆分数据
"""训练软间隔线性SVR"""
svr = SVR(kernel='linear',C=1.0,epsilon=0.1).fit(X_train,y_train)    #训练一个软间隔线性SVR，惩罚因素C=1
"""训练决策树"""
dtr = DecisionTreeRegressor(criterion='mse',max_depth=2).fit(X_train,y_train)   #训练一个深度为2的决策树
"""训练线性回归"""
lg = LinearRegression().fit(X_train,y_train)    #训练一个线性回归模型
"""Ridge回归"""
ridge = Ridge(alpha=1.0).fit(X_train,y_train)    #训练一个alpha=1的Rigde回归模型
"""产生新数据集"""
new_X = pd.DataFrame()  
new_X['svr'] = svr.predict(X)
new_X['dtr'] = dtr.predict(X)
new_X['lg'] = lg.predict(X)
new_X['ridge'] = ridge.predict(X)
"""训练次级模型"""
X_train,X_test,y_train,y_test = train_test_split(new_X,y,test_size=0.4,random_state=4)    #按照7：3拆分数据
#将6：4的比例对新数据集拆分
meta_svr = SVR(kernel='linear',C=1.0,epsilon=0.1)    #次级模型为线性软间隔SVR，C=1，epsilon=0.1
meta_svr.fit(X_train,y_train)
y_train_pred = meta_svr.predict(X_train)
y_test_pred = meta_svr.predict(X_test)
print('训练集的R方为：',r2_score(y_train,y_train_pred))
print('测试集的R方为：',r2_score(y_test,y_test_pred))
#
"""画出图13.7"""
import matplotlib.pyplot as plt   
font1 = {'family' : 'SimHei',
'weight' : 'normal',
'size'   : 16,
}
plt.figure(figsize=(6,4))
plt.rcParams['font.sans-serif']=['SimHei']    #画图时显示中文字体
plt.rcParams['axes.unicode_minus'] = False
plt.plot(y,label='观测数据',linewidth=2)
plt.plot(meta_svr.predict(new_X),linestyle='dashed',label='拟合曲线',linewidth=4)
plt.xlabel(r'$x$',fontsize=20)
plt.ylabel(r'$y$',fontsize=24)
plt.legend(prop=font1)
