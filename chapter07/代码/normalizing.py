# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:50:20 2020

@author: Zhuo
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif']=['SimHei']    #画图时使用中文字体
plt.rcParams['axes.unicode_minus'] = False
"""实现Zscore标准化"""
from sklearn.preprocessing import StandardScaler
np.random.seed(1)
#随机产生容量为1000个，三个特征的数据集
df = pd.DataFrame({
    'x1': np.random.normal(0, 2, 1000),
    'x2': np.random.normal(10, 4, 1000),
    'x3': np.random.normal(-10, 6, 1000)
})
scaler = StandardScaler()
df_zscore = scaler.fit_transform(df)    #进行Zscore标准化
df_zscore = pd.DataFrame(df_zscore, columns=['x1', 'x2', 'x3'])    #构成新的dataframe，以便观察。

fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(11,4))
"""以下为画图代码"""
ax1.set_title('Zscore标准化前',fontsize=20)
sns.kdeplot(df['x1'], ax=ax1,linestyle = '-')
sns.kdeplot(df['x2'], ax=ax1,linestyle = '--')
sns.kdeplot(df['x3'], ax=ax1,linestyle = ':')
plt.ylabel('频率',fontsize=20)
ax2.set_title('Zscore标准化后',fontsize=20)
sns.kdeplot(df_zscore['x1'], ax=ax2,linestyle = '-')
sns.kdeplot(df_zscore['x2'], ax=ax2,linestyle = '--')
sns.kdeplot(df_zscore['x3'], ax=ax2,linestyle = ':')
plt.ylabel('频率',fontsize=20)
plt.show()
#
"""最大最小值标准化"""
df = pd.DataFrame({
    # positive skew
    'x1': np.random.chisquare(8, 1000),    #生成卡方分布
    # negative skew 
    'x2': np.random.beta(8, 2, 1000) * 40,    #Beta分布
    # no skew
    'x3': np.random.normal(50, 3, 1000)
})
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df_minmax = scaler.fit_transform(df)    #进行最大最小值标准化
df_minmax = pd.DataFrame(df_minmax, columns=['x1', 'x2', 'x3'])

"""以下为画图代码"""
fig, (ax3, ax4) = plt.subplots(ncols=2,figsize=(11,4))
ax3.set_title('MinMax标准化前',fontsize=20)
sns.kdeplot(df['x1'], ax=ax3,linestyle = '-')
sns.kdeplot(df['x2'], ax=ax3,linestyle = '--')
sns.kdeplot(df['x3'], ax=ax3,linestyle = ':')
plt.ylabel('频率',fontsize=20)
ax4.set_title('MinMax标准化后',fontsize=20)
sns.kdeplot(df_minmax['x1'], ax=ax4,linestyle = '-')
sns.kdeplot(df_minmax['x2'], ax=ax4,linestyle = '--')
sns.kdeplot(df_minmax['x3'], ax=ax4,linestyle = ':')
plt.show()