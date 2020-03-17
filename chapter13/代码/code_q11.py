# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:44:51 2020

@author: Zhuo
"""

from sklearn.ensemble import BaggingClassifier  #导入Bagging集成分类器库
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import demo_data2   #导入文件demo_data，以产生数据集
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
"""画出分类边界"""
def plot_boundary(model,X,y):
    x_min,x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min,y_max = X[:,0].min()-5, X[:,0].max()+5
    h = 0.1
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),
                        np.arange(y_min,y_max,h))
    X_hypo = np.column_stack((xx.ravel().astype(np.float32),
                             yy.ravel().astype(np.float32)))
    
    try:
        zz = np.argmax(model.predict(X_hypo),axis=1)

    except:
        zz = model.predict(X_hypo)
    zz = zz.reshape(xx.shape)
    plt.contourf(xx,yy,zz,cmap=plt.cm.binary,alpha=0.4)
    X0 = X[y.ravel()==0]
    plt.scatter(X0[:, 0], X0[:, 1], marker='o',c='k')  
    X1 = X[y.ravel()==1]
    plt.scatter(X1[:, 0], X1[:, 1], marker='x')  



"""产生数据集"""
X,y = demo_data2.data_generate()    #产生示例数据集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=4)    #按照6:4拆分数据
"""以逻辑回归为弱学习机的Bagging模型"""
weak_learner = LogisticRegression()   #定义逻辑回归弱学习机类
bagging = BaggingClassifier(base_estimator=weak_learner,,max_samples=0.67
                            n_estimators=20)    #以weak_learner为子模型，设置子模型个数 T=20 个
bagging.fit(X_train,y_train)    #训练模型

y_train_pred = bagging.predict(X_train)    #训练集的预测值
y_test_pred = bagging.predict(X_test)    #测试集中的预测值
print('训练集的精确度为: ',accuracy_score(y_train,y_train_pred))
print('测试集的精确度为: ',accuracy_score(y_test,y_test_pred))

"""逻辑回归模型"""
svc = SVC(kernel='linear',C=1.0)
bagging2 = BaggingClassifier(base_estimator=svc,max_samples=0.67
                            n_estimators=20)    
bagging2.fit(X_train,y_train)    #训练模型
y_train_pred = bagging2.predict(X_train)    #训练集的预测值
y_test_pred = bagging2.predict(X_test)    #测试集中的预测值
print('训练集的精确度为: ',accuracy_score(y_train,y_train_pred))
print('测试集的精确度为: ',accuracy_score(y_test,y_test_pred))



"""画图"""
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
plt.title('逻辑回归Bagging',fontsize=16)
plt.xlabel(r'$x_1$',fontsize=16)
plt.ylabel(r'$x_2$',fontsize=16)
plot_boundary(bagging,X,y)

plt.subplot(1,2,2)
plt.title('软间隔SVC-Bagging',fontsize=16)
plt.xlabel(r'$x_1$',fontsize=16)
plt.ylabel(r'$x_2$',fontsize=16)
plot_boundary(bagging2,X,y)


