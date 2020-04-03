# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 09:23:01 2020

@author: Zhuo
"""

from load_dataset import data_generate
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def data_generate_final():
    waste_df,_ = data_generate()    #导入数据集
    y_cod = np.array(waste_df['产出COD'])    #提出输出因变量
    y_cod_rate = np.array(waste_df['COD去除率 %'])
    y_vfa = np.array(waste_df['产出VFA'])
    X = waste_df.iloc[:,[1,2,3,4,5,6,7,8,9,13,14,15,16]]    #提取出输入特征
    pca = PCA(n_components=7)    #指定d=7的PCA类
    X_PCA = pca.fit_transform(X)   #进行PCA降维
    return X_PCA,y_cod,y_cod_rate,y_vfa


    
if __name__ == '__main__':  # 主函数入口
    waste_df,_ = data_generate()    #导入数据集
    y_cod = waste_df['产出COD']    #提出因变量
    y_cod_rate = waste_df['COD去除率 %']
    y_vfa = waste_df['产出VFA']
    X = waste_df.iloc[:,[1,2,3,4,5,6,7,8,9,13,14,15,16]]    #提取出输入特征
    pca = PCA(n_components=7)    #指定d=7的PCA类
    X_PCA = pca.fit_transform(X)   #进行PCA降维
    corr_matrix = X.corr().abs()    #求出相关矩阵
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))    #取相关系数矩阵的上三角
    """画图7.12的代码"""
    evr = pca.explained_variance_ratio_
    print(evr)    #输出新特征的信息贡献
    plt.rcParams['font.sans-serif']=['SimHei']    #画图时使用中文字体
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(12,4))
    plt.subplots_adjust(left=0.125, bottom=None, right=0.9, top=None,
                    wspace=0.3, hspace=None)
    plt.subplot(1,2,1)
    plt.bar(range(1,len(evr)+1),evr)   #画出贡献图
    plt.axis([0,8,0,0.15])
    plt.xlabel(u'特征',fontsize=20)
    plt.ylabel(u'贡献',fontsize=20)
    evr_sum = evr.copy()
    for i in range(1,len(evr)):
        evr_sum[i] = evr_sum[i-1]+evr_sum[i]
    plt.subplot(1,2,2)
    plt.bar(range(1,len(evr_sum)+1),evr_sum)   #画出贡献图
    plt.axis([0,8,0,1])
    plt.xlabel(u'特征',fontsize=20)
    plt.ylabel(u'累计贡献',fontsize=20)

    




