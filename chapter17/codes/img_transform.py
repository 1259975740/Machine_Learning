# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 11:26:53 2020

@author: Zhuo
"""
import cv2
def img_trans(img_list):    #输入参数为一个有多幅图像构成的列表
    """设置HOG法参数"""
    dect_win_size = (48,96)    #设置检测窗口的尺寸为64X128，即包含64X128个像素
    block_size = (16,16)    #定义块的大小为16X16
    cell_size = (8,8)    #定义胞元的大小为8X8，即块中包含4个胞元
    win_stride = (64,64)    #定义窗口的滑动步长为：宽度方向64，长度方向64
    block_stride = (8,8)    #定义块的滑动步长为：宽度方向8，长度方向8，即窗口之间存在两个重叠的胞元
    bins = 9    #定义直方图柱数为9，即将圆拆分成9等块供像素投影
    """使用HOG法将图像集转换为向量集"""
    img_fea_list = []
    for img in img_list:
        hog = cv2.HOGDescriptor(dect_win_size,block_size,block_stride,
                                cell_size,bins)    #生成一个HOG对象，并设置参数
        img_fea_hog = hog.compute(img,win_stride)    #用HOG法将图像转换为特征
        img_fea_list.append(img_fea_hog)
    return img_fea_list
