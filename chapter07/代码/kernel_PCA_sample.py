# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 19:15:11 2020

@author: Zhuo
"""
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import numpy as np
plt.rcParams['font.sans-serif']=['SimHei']    #画图时使用中文字体
plt.rcParams['axes.unicode_minus'] = False
X,y = make_circles(n_samples=500,random_state=1,noise=0.1,factor=0.1)   #产生线性不可分的分类数据集

"""画出图7.13的代码"""
plt.subplot(1,2,1)
plt.scatter(X[y==0,0],X[y==0,1],marker='x',color='black',s=50)
plt.scatter(X[y==1,0],X[y==1,1],marker='o')
plt.xlabel(u'降维前',fontsize=20)
pca = PCA(n_components=1)    #运用普通PCA进行降维，并画出图像
X_pca = pca.fit_transform(X)

mu,eig = np.linalg.eig(np.dot(X.T,X))    #求出特征值和特征向量
plt.quiver(0,0,0.5*eig[:,0],0.5*eig[0,1],units='xy',zorder=3,scale=0.3)
plt.text(0.7*eig[0,0],0.1*eig[0,1],r'$e_1$',
         fontsize=20,zorder=5,bbox=dict(facecolor='white',alpha=0.6))
plt.text(0.3*eig[1,0],0.3*eig[1,1],r'$e_2$',
         fontsize=20,zorder=5,bbox=dict(facecolor='white',alpha=0.6))

plt.subplot(1,2,2)
plt.axis([-1.5,1.5,-1.5,1.5])
plt.scatter(X_pca[y==0],np.zeros((250,1))+0.05,marker='x',color='black',alpha=0.95,s=50)
plt.scatter(X_pca[y==1],np.zeros((250,1))-0.05,marker='o')
plt.xlabel(u'降维后',fontsize=20)


"""画出图7.15的代码"""
plt.subplot(1,3,1)
plt.scatter(X[y==0,0],X[y==0,1],marker='x',color='black',s=50)
plt.scatter(X[y==1,0],X[y==1,1],marker='o')
plt.xlabel(u'初始数据',fontsize=20)
kpca = KernelPCA(n_components=2,kernel='rbf',gamma=2.0)    #运用核PCA进行降维，并画出图像
X_kpca = kpca.fit_transform(X)

plt.subplot(1,3,2)
plt.axis([-1.5,1.5,-1.5,1.5])
plt.scatter(X_kpca[y==0,0],X_kpca[y==0,1],marker='x',color='black',s=50)
plt.scatter(X_kpca[y==1,0],X_kpca[y==1,1],marker='o')
plt.xlabel(u'高维返二维',fontsize=20)

kpca = KernelPCA(n_components=1,kernel='rbf',gamma=2.0)    #运用核PCA进行降维，并画出图像
X_kpca = kpca.fit_transform(X)
plt.subplot(1,3,3)
plt.axis([-1.5,1.5,-1.5,1.5])
plt.scatter(X_kpca[y==0],np.zeros((250,1)),marker='x',color='black',alpha=0.5,s=50)
plt.scatter(X_kpca[y==1],np.zeros((250,1)),marker='o')
plt.xlabel(u'降维后',fontsize=20)