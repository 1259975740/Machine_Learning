# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:39:43 2020

@author: Zhuo
"""
import load_dataset_neg
import load_dataset_pos
import cv2
data_neg,labels_neg = load_dataset_neg.data_generate()    #产生无行人数据集
data_pos,labels_pos = load_dataset_pos.data_generate()    #产生有行人数据集
"""画图代码"""
import numpy as np
import matplotlib.pyplot as plt
"""定义画图函数"""
def plot_data(data,row=2,col=2):     #画出图像，row为行数，col为列数
    fig, axes = plt.subplots(row,col, figsize=(10, 14),subplot_kw={'xticks':[], 'yticks':[]},
                            gridspec_kw=dict(hspace=0.1, wspace=0.1))    #生成画图窗口
    for _, ax in enumerate(axes.flat):    #在子窗口画图
        idx = np.random.choice(np.arange(0,900),1)    #随机选择图像
        ax.imshow(data[idx[0]], cmap='gray')


def data_generate_final():
    data_neg_resize = []
    for img_neg in data_neg:
        img_neg_resize = cv2.resize(img_neg,(64,128),interpolation=cv2.INTER_LINEAR)    #使用二线性插值法缩放图像
        data_neg_resize.append(img_neg_resize)    #构成缩放后数据集
            
    """将数据集结合在一起"""
    X = data_pos + data_neg_resize
    y = np.concatenate((labels_pos, labels_neg))
    
    """将所有图像转换为向量"""
    X_fea = []
    for img in X:
        img_fea = img.flatten()
        X_fea.append(img_fea)
        
    
    """类别不均衡"""
    from imblearn.over_sampling import BorderlineSMOTE  #导入边界SMOTE包
    over_sampling = BorderlineSMOTE()
    X_resample,y_resample = over_sampling.fit_resample(X_fea, y)
    """将list转换为array"""
    X_re = []
    for x_fea in X_resample:
        x_fea = np.array(x_fea,dtype=np.uint8)
        X_re.append(x_fea)
        
    
    """将向量X_fea反转为图像"""
    img_resample = []
    for x_fea in X_re:
        x_fea
        img_back = x_fea.reshape([128,64],order='C')
        img_resample.append(img_back)
    """使用HOG法将图像转换为梯度方向统计直方图构成的特征向量"""
    from img_transform import img_trans
    X_fea = img_trans(img_resample)   #img_trans函数将图像列表的每一张图像转换为HOG特征向量    
    y = np.array(y_resample,dtype=np.float32)    #转换数据格式为浮点型
    return X_fea,y

if __name__ == '__main__':  # 主函数入口
    plot_data(data_pos,1,6)    #画出图像
    plot_data(data_neg,1,6)
    plot_data(data_neg_resize,1,6)    #画出缩放后的部分图像
    """画出那些由过采样产生的新图片"""
    fig, axes = plt.subplots(1,5, figsize=(10, 14),subplot_kw={'xticks':[], 'yticks':[]},
                                gridspec_kw=dict(hspace=0.1, wspace=0.1))    #生成画图窗口
    for i, ax in enumerate(axes.flat):    #在子窗口画图
        ax.imshow(img_resample[1988+i], cmap='gray')
    """输出数据集"""
    X,y = data_generate_final()
