# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:20:11 2020

@author: Zhuo
"""

from data_create_prepro import data_generate_final
import numpy as np
X,y = data_generate_final()    #产生数据集
X = np.array(X,dtype=np.float32)[:,:,0]    #将list转换为np.narray

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier    #导入kNN算法
from sklearn.linear_model import LogisticRegression    #线性回归
from sklearn.metrics import make_scorer,recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

# """"筛选kNN算法的最合适参数k"""
# grid = {'n_neighbors':[3,5,7,9,11]}    #定义参数网格为近邻个数k
# recall_scorer = make_scorer(recall_score)    #用召回率作为评价模型的指标
# grid_search = GridSearchCV(KNeighborsClassifier(),param_grid=grid,cv=5,scoring=recall_scorer)
# grid_search.fit(X,y)   #进行网格寻优
# print(grid_search.best_params_)    #输出最优的参数对

# """筛选最合适的SVC"""
# grid = {'C':[0.80,0.85,0.90,0.95,1.00],
#         'kernel':['linear','rbf','poly']}    #定义参数网格惩罚参数C和核函数类型
# grid_search = GridSearchCV(SVC(),param_grid=grid,cv=5,scoring=recall_scorer)
# grid_search.fit(X,y)   #进行网格寻优
# print(grid_search.best_params_)    #输出最优的参数对

# """筛选最合适的决策树"""
# grid = {'max_depth':[15,21,27,33,39,42],
#         'ccp_alpha':[0.005,0.01,0.05,0.1,0.2]}    #定义参数网格为最大树深度和α_ccp
# grid_search = GridSearchCV(DecisionTreeClassifier(),param_grid=grid,cv=5,scoring=recall_scorer)
# grid_search.fit(X,y)   #进行网格寻优
# print(grid_search.best_params_)    #输出最优的参数对

# """筛选最合适的随机森林"""
# grid = {'n_estimators':[500,1000,1100,1200,1300,1400,1500]}    #定义参数网格为子模型数量
# grid_search = GridSearchCV(RandomForestClassifier(max_samples=0.67,max_features=0.33,max_depth=5)
#                             ,param_grid=grid,cv=5,scoring=recall_scorer)
# grid_search.fit(X,y)   #进行网格寻优
# print(grid_search.best_params_)    #输出最优的参数对

# """筛选最合适的AdaBoost"""
# dtc = DecisionTreeClassifier(criterion='gini',max_depth=5)
# grid = {'n_estimators':[200,300,400,500,1000]}    #定义参数网格
# grid_search = GridSearchCV(AdaBoostClassifier(base_estimator=dtc)
#                            ,param_grid=grid,cv=5,scoring=recall_scorer)
# grid_search.fit(X,y)   #进行网格寻优
# print(grid_search.best_params_)    #输出最优的参数对


# """5折交叉验证筛选模型"""
# """定义算法，算法中的参数是从网格搜索得出的最优参数"""
# lg = LogisticRegression(penalty='none')
# knn = KNeighborsClassifier(n_neighbors=3)    #k=3的kNN算法
# svc = SVC(C=0.8,kernel='rbf')    #惩罚参数C=0.8的，核函数为RBF的软间隔SVC
# dtc = DecisionTreeClassifier(max_depth=15,criterion='gini',ccp_alpha=0.05)   #剪枝条件为：最大深度——15，α_ccp——0.05的决策树
# rf = RandomForestClassifier(n_estimators=500,max_depth=5,max_samples=0.67,max_features=0.33)    #子决策树的深度均为5，包含500颗子决策树的随机森林
# base_dtc = DecisionTreeClassifier(criterion='gini',max_depth=5)    #用深度为5的决策树作为子模型
# adaboost = AdaBoostClassifier(base_estimator=base_dtc,n_estimators=1000)    #将子模型设置为base_dtc，子模型个数为500个，从而构成AdaBoost模型

# """"用5折交叉验证，计算所有模型的Si,并计算其均值"""
# S_lg_i = cross_val_score(lg,X,y,scoring=recall_scorer,cv=5)    #计算出逻辑回归模型的Si
# S_knn_i = cross_val_score(knn,X,y,scoring=recall_scorer,cv=5)    #计算出kNN模型的Si
# S_svc_i = cross_val_score(svc,X,y,scoring=recall_scorer,cv=5)    #计算出SVC模型的Si
# S_dtc_i = cross_val_score(dtc, X,y,scoring=recall_scorer,cv=5)    #计算出决策树模型的Si
# S_rf_i = cross_val_score(rf, X,y,scoring=recall_scorer,cv=5)    #计算随机森林模型的Si
# S_ada_i = cross_val_score(adaboost, X,y,scoring=recall_scorer,cv=5)    #计算Adaboost模型的Si
# print(np.mean(S_lg_i))    #输出均值
# print(np.mean(S_knn_i))
# print(np.mean(S_svc_i))
# print(np.mean(S_dtc_i))
# print(np.mean(S_rf_i))
# print(np.mean(S_ada_i))
# """T检验"""
# from scipy.stats import ttest_ind    #导入相关库
# AdaBoostClassifier()ttest_ind(S_svc_i,S_lg_i)     #对S_svc_i,S_lg_i进行T检验
# ttest_ind(S_svc_i,S_rf_i)     #对S_svc_i,S_rf_i进行T检验
# ttest_ind(S_rf_i,S_lg_i)     #对S_lg_i,S_rf_i进行T检验

"""拆分数据集，训练逻辑回归模型"""
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=4)    #将数据按7：3拆分成训练集和测试集
svc = SVC(C=0.8,kernel='rbf')    #惩罚参数C=0.8的，核函数为RBF的软间隔SVC
svc.fit(X_train,y_train)    #模型训练
y_train_pred = svc.predict(X_train)    #计算模型的预测值
y_test_pred = svc.predict(X_test)
print('模型在训练集中的召回率为',recall_score(y_train, y_train_pred))
print('模型在测试集中的召回率为',recall_score(y_test, y_test_pred))
"""计算模型的精确率、准确度、F1值，生成一个报表"""
from sklearn.metrics import classification_report
print(classification_report(y_train, y_train_pred))
print(classification_report(y_test, y_test_pred))


import cv2
def pedestrians_detect(img,stride=16):
    found = []    #定义一个空列表，用于存放识别带行人的窗口
    for y in np.arange(0,img.shape[0],stride):
        for x in np.arange(0,img.shape[1],stride):
            if y + 128 > img.shape[0]:    #如果当前窗口超出图像范围，则结束判断
                continue
            if x + 64 > img.shape[0]:
                continue
            sub_img = img[y:y+128,x:x+64]    #将当前子窗口框中的地方构成子图
            """设置HOG法参数,对子图进行HOG法转换特征向量"""
            dect_win_size = (48,96)    #设置检测窗口的尺寸为64X128，即包含64X128个像素
            block_size = (16,16)    #定义块的大小为16X16
            cell_size = (8,8)    #定义胞元的大小为8X8，即块中包含4个胞元
            win_stride = (64,64)    #定义窗口的滑动步长为：宽度方向64，长度方向64
            block_stride = (8,8)    #定义块的滑动步长为：宽度方向8，长度方向8，即窗口之间存在两个重叠的胞元
            bins = 9    #定义直方图柱数为9，即将圆拆分成9等块供像素投影
            hog = cv2.HOGDescriptor(dect_win_size,block_size,block_stride,
                                cell_size,bins)    #生成一个HOG对象，并设置参数
            fea = np.array(hog.compute(sub_img,(64,64)))[:,0]
            y_pred = svc.predict(fea.reshape(1,-1))    #用训练好的svc模型检测子图是否有行人
            if y_pred == 1:    #如果y_pred为1
                found.append((y,x,128,64))
    return found


img_test = cv2.imread('test.jpg',0)
found = pedestrians_detect(img_test)
"""画出图像和边框"""
import matplotlib.pyplot as plt
from matplotlib import patches    #导入patches
fig = plt.figure()
ax = fig.add_subplot(111) 
plt.imshow(img_test,cmap='gray'),plt.axis('off')    #画出原始图像，同时去掉坐标轴
for f in found:    #将found中的每一个边框画出来
    ax.add_patch(patches.Rectangle((f[1],f[0]), f[3], f[2], color='b',linewidth=3,fill=False))
    #画出边框

                
                