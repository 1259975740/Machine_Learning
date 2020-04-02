# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:37:40 2020

@author: Administrator
"""

import warnings
warnings.filterwarnings('ignore')    #忽略warning信息
import pandas as pd
def replace_n(df):
    time = df.iloc[:,0]    #将数据集的第一列提取出来
    n = 363    #由于数据是从12月28日开始，故应将n设置从363开始
    for i in range(0,len(df)):
        time[i] = n    #将时间转换为整数序列
        n += 1    
        if n == 366:    #若n为365，即满一年时，从头开始（忽略闰年）
            n = 1
    return df    #返回修改后的数据集

def  tmp_replace(df):    
    low_tmp = df.iloc[:,16]    #提取当地最低气温
    high_tmp = df.iloc[:,15]    #提取当地最高气温
    nan_idx = low_tmp[low_tmp[:]!=low_tmp[:]].index.tolist()    #得到缺失值索引,这里根据nan不等于nan找出缺失值的位置
    n = df.iloc[:,0]    #获取时间序列
    def sub_model(x):    #用模型预测值替换缺失值
        y1 = -0.0009973*x**2+0.3388*x+1.429    #子模型
        y2 = -0.0009563*x**2+0.3405*x-10.85
        return y1,y2
    high_tmp.iloc[nan_idx],low_tmp.iloc[nan_idx] = sub_model(n.iloc[nan_idx])
    #用子模型的预测值替换数据的缺失值
    return df
def data_generate():
    waste_df = pd.read_excel(r'../datasets/data.xlsx')   #以Dataframe的形式读入数据集
    waste_df = replace_n(waste_df)
    waste_df = tmp_replace(waste_df)
    waste_df = waste_df.dropna()    #按行删除
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_zscore = scaler.fit_transform(waste_df)    #进行Zscore标准化
    from sklearn.neighbors import LocalOutlierFactor    #导入LOF库
    lof = LocalOutlierFactor(n_neighbors=3)    #进行LOF异常值检验，设置近邻个数为3
    labels = lof.fit_predict(data_zscore)    #输出异常标签，其中1代表正常，-1代表异常个体
    waste_df['是否异常'] = labels    #添加新的列，用于删除异常个体
    waste_df = waste_df.drop(waste_df.loc[waste_df['是否异常']==-1].index)    #按行删除异常个体
    waste_df = waste_df.drop(['是否异常'],axis=1)    #释放该列
    waste_df_before = waste_df.copy()    #拷贝一份未标准化的数据
    col_name = waste_df.columns.values.tolist()    #提取列名
    data_zscore_again = scaler.fit_transform(waste_df)    #再次进行标准化
    waste_df = pd.DataFrame(data=data_zscore_again,columns=col_name)
    
    
    return waste_df,waste_df_before

def main():
    waste_df,_ = data_generate()
    return waste_df
    
if __name__ == '__main__':  # 主函数入口
    waste_df = main()
    
    
        