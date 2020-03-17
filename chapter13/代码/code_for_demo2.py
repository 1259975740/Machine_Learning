# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:03:20 2020

@author: Zhuo
"""


from sklearn.ensemble import AdaBoostClassifier  #导入AdaBoost集成分类器库
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
import demo_data2   #导入文件demo_data，以产生数据集
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

"""深度为1的决策树模型"""
dtc = DecisionTreeClassifier(criterion='gini',max_depth=1)
dtc.fit(X_train,y_train)    #训练模型
y_train_pred = dtc.predict(X_train)    #训练集的预测值
y_test_pred = dtc.predict(X_test)    #测试集中的预测值
print('训练集的精确度为: ',accuracy_score(y_train,y_train_pred))
print('测试集的精确度为: ',accuracy_score(y_test,y_test_pred))

"""以dtc为弱学习机的AdaBoost模型"""
boost = AdaBoostClassifier(base_estimator=dtc,
                            n_estimators=30)    #以weak_learner为子模型，设置子模型个数 T=50 个
boost.fit(X_train,y_train)    #训练模型
y_train_pred = boost.predict(X_train)    #训练集的预测值
y_test_pred = boost.predict(X_test)    #测试集中的预测值
print('训练集的精确度为: ',accuracy_score(y_train,y_train_pred))
print('测试集的精确度为: ',accuracy_score(y_test,y_test_pred))


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
plt.subplot(1,2,2)
plt.title('Adaboost',fontsize=16)
plt.xlabel(r'$x_1$',fontsize=16)
plt.ylabel(r'$x_2$',fontsize=16)
plot_boundary(boost,X,y)

plt.subplot(1,2,1)
plt.title('单深度的决策树',fontsize=16)
plt.xlabel(r'$x_1$',fontsize=16)
plt.ylabel(r'$x_2$',fontsize=16)
plot_boundary(dtc,X,y)

