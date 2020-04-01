# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 10:50:54 2020

@author: Zhuo
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits    #导入数据集MINIST
digits = load_digits()    #导入数据集
digits.images.shape    #查看图像的像素大小
#画出前十幅图
fig, axes = plt.subplots(5,5, figsize=(8, 8),subplot_kw={'xticks':[], 'yticks':[]},
                        gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray', interpolation='nearest')
    ax.text(0.07, 0.07, str(digits.target[i]),transform=ax.transAxes, color='white')
    
from sklearn.decomposition import PCA
pca = PCA(n_components=30)    #指定d=30的PCA类
digits_PCA = pca.fit_transform(digits.data)   #进行PCA降维

from sklearn.linear_model import LogisticRegression    #导入逻辑回归库
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
digits_zscore = scaler.fit_transform(digits_PCA)   #进行Zscore标准化
labels = digits.target   #将因变量y单独取出
from sklearn import model_selection
X_train,X_test,y_train,y_test = model_selection.train_test_split(
        digits_zscore,labels,test_size=0.3,random_state=1)    #拆分数据
softmax = LogisticRegression(multi_class = 'multinomial')   #使用softmax回归
softmax.fit(X_train,y_train)    #模型训练
y_train_pred = softmax.predict(X_train)
y_test_pred = softmax.predict(X_test)    #用回归模型得出预测值
#以下代码用于生成评价报表、算出精确率等
from sklearn.metrics import classification_report , accuracy_score	#引入评价用包
print('训练集结果报表')
print(classification_report(y_train,y_train_pred))
print('训练集中的精度为： ',accuracy_score(y_train,y_train_pred))
print('测试集结果报表')
print(classification_report(y_test,y_test_pred))
print('测试集集中的精度为： ',accuracy_score(y_test,y_test_pred))

 
