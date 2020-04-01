# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 07:57:05 2020

@author: Zhuo
"""

import os
import zipfile
import numpy as np

datadir = r'..\datasets\pedestrians128x64.zip'   #datadir为解压路径
extractdir = r'..\datasets\pic_pos_file'   #datadir为解压路径

def un_zip(zipfilename, unziptodir):
    if not os.path.exists(unziptodir):    #如果“解压到"目录不存在，则自动创建一个新的文件
        os.mkdir(unziptodir)
    zfobj = zipfile.ZipFile(zipfilename)    #打开压缩文件
    for name in zfobj.namelist():      #遍历压缩文件的所有子目录、文件
        if name.endswith('/'):    #如果读取到子目录，则需要创建一个子目录
            sub_path = os.path.join(unziptodir,name)
            if not os.path.exists(sub_path):
                os.mkdir(sub_path)    #创建子目录
        else:            
            ext_filename = os.path.join(unziptodir, name)    #若读取到文件（包括子目录里的文件），则将文件放置到当前路径中
            outfile = open(ext_filename, 'wb')   #读取文件
            outfile.write(zfobj.read(name))
            outfile.close()    #关闭文件，以读取下一个文件。
    i = 0;


# """函数说明:
#     read_single_file用于读取单一文件的内容
#     read_files用于从文件夹中读取所有文件的内容"""
import pandas as pd
import cv2
def read_single_file(filename):
    if os.path.isfile(filename):    
        img = cv2.imread(filename,0)
        
    return img


def read_files(path):
    for root,dirname,filenames in os.walk(path):   #不断进入文件夹，直到“遇到”单一文件
        for filename in filenames:
            filepath = os.path.join(root,filename)   
            yield read_single_file(filepath)    #执行函数read_single_file,读取文件内容

# """函数说明：
#    build_data_list：用于从读取的文件中构建数据集"""
def build_data_list(extractdir):
    img_list = []    
    for img in read_files(extractdir):    #读取文件的每一张图片，构成一个图片的list
        img_list.append(img)
    return img_list
   
def data_generate():    

    data = build_data_list(extractdir)
    data_del = [i for i in data if i is not None]    #删除那些读取不出的图片
    labels = np.ones(len(data_del),dtype=np.int32)
    return data_del,labels

import matplotlib.pyplot as plt
"""画图代码"""
def plot_data(data,row=2,col=2):     #画出图像，row为行数，col为列数
    fig, axes = plt.subplots(row,col, figsize=(8, 8),subplot_kw={'xticks':[], 'yticks':[]},
                            gridspec_kw=dict(hspace=0.1, wspace=0.1))    #生成画图窗口
    for i, ax in enumerate(axes.flat):    #在子窗口画图
        ax.imshow(data[i], cmap='gray')

def main():
    un_zip(datadir,extractdir) 
    """由于该文件中所有图像都为英文，故不用更名"""
    
main()

if __name__ == '__main__':  # 主函数入口
    data,labels = data_generate()    #产生数据集

    
    