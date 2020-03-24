# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 17:24:56 2020

@author: Zhuo
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
"""画图7.11"""
plt.rcParams['font.sans-serif']=['SimHei']    #画图时使用中文字体
plt.rcParams['axes.unicode_minus'] = False
mean = [20,20]
cov = [[5,0],[25,25]]
x1,x2 = np.random.multivariate_normal(mean,cov,800).T    #产生示例数据
plt.plot(x1,x2,'o',MarkerSize=3)
plt.axis([0,40,0,40])
plt.xlabel(u'特征1')
plt.ylabel(u'特征2')
X = np.vstack((x1,x2)).T    #构建一个数据矩阵
mu,eig = np.linalg.eig(np.dot(X.T,X))    #求出特征值和特征向量
plt.quiver(mean[0],mean[1],eig[:,0],eig[0,1],units='xy',zorder=3,scale=0.2)
plt.text(mean[0]+8*eig[0,0],mean[1]+8*eig[0,1],r'$e_1$',
         fontsize=20,zorder=5,bbox=dict(facecolor='white',alpha=0.6))
plt.text(mean[0]+7*eig[1,0],mean[1]+7*eig[1,1],r'$e_2$',
         fontsize=20,zorder=5,bbox=dict(facecolor='white',alpha=0.6))

plt.show()

digits = load_digits()    #导入digits数据集，该数据集有64个特征
pca = PCA(n_components=30)    #指定d=30的PCA类
digits_PCA = pca.fit_transform(digits.data)   #进行PCA降维
evr = pca.explained_variance_ratio_
print(evr)    
"""画图7.12的代码"""
plt.figure(figsize=(12,4))
plt.subplots_adjust(left=0.125, bottom=None, right=0.9, top=None,
                wspace=0.3, hspace=None)
plt.subplot(1,2,1)
plt.bar(range(1,len(evr)+1),evr)   #画出贡献图
plt.axis([0,31,0,0.15])
plt.xlabel(u'特征x',fontsize=20)
plt.ylabel(u'贡献',fontsize=20)
evr_sum = evr.copy()
for i in range(1,len(evr)):
    evr_sum[i] = evr_sum[i-1]+evr_sum[i]
plt.subplot(1,2,2)
plt.bar(range(1,len(evr_sum)+1),evr_sum)   #画出贡献图
plt.axis([0,31,0,1])
plt.xlabel(u'特征x',fontsize=20)
plt.ylabel(u'累计贡献',fontsize=20)

