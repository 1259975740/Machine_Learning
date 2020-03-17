# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 09:23:02 2020

@author: Zhuo
"""
from sklearn.ensemble import BaggingRegressor  #导入Bagging集成分类器库
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import demo_data   #导入文件demo_data，以产生数据集
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
"""产生数据集"""
X,y = demo_data.data_generate()    #产生示例数据集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=4)    #按照6:4拆分数据
"""用SVR进行Bagging集成"""
weak_learner = SVR(C=1.0,kernel='linear',epsilon=0.1)    #定义逻辑回归弱学习机类
bagging = BaggingRegressor(base_estimator=weak_learner,
                            n_estimators=50,max_samples=0.67)    #以weak_learner为子模型，设置子模型个数为50个，新样本占总样本的0~2/3
bagging.fit(X_train,y_train)    #训练模型

y_train_pred = bagging.predict(X_train)    #训练集的预测值
y_test_pred = bagging.predict(X_test)    #测试集中的预测值
print('训练集的R方为: ',r2_score(y_train,y_train_pred))
print('测试集的R方为: ',r2_score(y_test,y_test_pred))
"""使用随机森林模型"""
rf = RandomForestRegressor(n_estimators=50,max_samples=0.67,max_features=0.33,
                            criterion='mse',max_depth=6)  #定义一个由50个决策树构成的随机森林，剪枝条件为最大深度，用mse作为其不纯度度量
rf.fit(X_train,y_train)    #训练模型
y_train_pred = rf.predict(X_train)    #训练集的预测值
y_test_pred = rf.predict(X_test)    #测试集中的预测值
print('训练集的R方为: ',r2_score(y_train,y_train_pred))
print('测试集的R方为: ',r2_score(y_test,y_test_pred))


"""画出图13.4"""
import matplotlib.pyplot as plt   
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}
plt.figure(figsize=(12,4))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.3)    #设置制图之间的左右间距
plt.rcParams['font.sans-serif']=['SimHei']    #画图时显示中文字体
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(1,2,1)
plt.scatter(X_train,y_train,s=15,c='k',label='Train')
plt.scatter(X_test,y_test,s=55,c='k',marker='x',label='Test')
plt.xlabel('x',fontsize=20)
plt.title('Bagging集成',fontsize=16)
y_pred = bagging.predict(X);    
plt.legend(prop=font1)
plt.plot(X,y_pred)
#plot_boundary(bagging,X,y)   #画出分类界限。函数代码请参阅代码文件

plt.subplot(1,2,2)
plt.scatter(X_train,y_train,s=15,c='k',label='Train')
plt.scatter(X_test,y_test,s=55,c='k',marker='x',label='Test')
plt.xlabel('x',fontsize=20)
plt.title('随机森林',fontsize=16)
y_pred = rf.predict(X);    #画出带超平面
plt.legend(prop=font1)
plt.plot(X,y_pred)

"""决策树"""
from sklearn import tree
dtc = tree.DecisionTreeRegressor(criterion='mse',max_depth=6)
dtc.fit(X_train,y_train)
y_train_pred = dtc.predict(X_train)    #训练集的预测值
y_test_pred = dtc.predict(X_test)    #测试集中的预测值
print('训练集的精确度为: ',r2_score(y_train,y_train_pred))
print('测试集的精确度为: ',r2_score(y_test,y_test_pred))


