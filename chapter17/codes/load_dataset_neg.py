# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 07:57:05 2020

@author: Zhuo
"""

import os
import zipfile
import numpy as np
files = ['\plants.zip','\load.zip','\car.zip','\houses.zip']
datadir = r'..\datasets'    #datadir为解压文件所在路径
extractdir = r'..\datasets\pic_neg_file'   #datadir为解压路径

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


"""函数说明:
    read_single_file用于读取单一文件的内容
    read_files用于从目录中读取所有文件的内容"""
import cv2
def read_single_file(filename):
    if os.path.isfile(filename):    #判断是否为单一文件
        img = cv2.imread(filename,0)    #图像以灰度图像的形式读取如Python中       
    return img


def read_files(path):
    for root,dirname,filenames in os.walk(path):   #遍历路径path
        for filename in filenames:    #找到path中的文件（不是目录）
            filepath = os.path.join(root,filename)   
            yield read_single_file(filepath)    #执行函数read_single_file,读取文件内容

"""函数说明：
    build_data_list：用于从读取的文件中构建数据集"""
def build_data_list(extractdir):
    img_list = []    #构建一个空列表，用于存放图像
    for img in read_files(extractdir):    #读取文件的每一张图片，构成一个图片的列表
        img_list.append(img)    #构成一个由图像填满的列表
    return img_list    #返回图像列表
   
"""产生数据集"""
def data_generate():    
    data = build_data_list(extractdir)
    data_del = [i for i in data if i is not None]    #删除那些读取不出的图片
    labels = np.zeros(len(data_del),dtype=np.int8)    #构建标签
    return data_del,labels

def main():
    for file in files:    #for循环遍历所有zip压缩文件
        datadir_file = datadir+file    #构成完整的文件路径
        un_zip(datadir_file,extractdir)     #使用自定义函数解压文件到目标路径extractdir中
    """由于cv2库不能识别中文路径，因此给文件内所有图像更名"""
    i = 0;    #将图像顺序更名为picx.jpg形式，故从i=0开始递增。
    for root_path, dir_names, file_names in os.walk(extractdir):    #遍历压缩路径的所有文件、子目录
        for file_name in file_names:    #对所有文件进行改名
            path = os.path.join(root_path, file_name)
            if not zipfile.is_zipfile(path):    #如果文件为压缩文件，则弹窗
                try:
                    file_name = 'pic'+str(i)+'.jpg'    #新文件名
                    new_path = os.path.join(root_path, file_name)    #得到新文件名的完整路径
                    os.rename(path, new_path)    #更改文件名
                    i=i+1
                except Exception as excep:
                        print('error:', excep)
    
main()

if __name__ == '__main__':  # 主函数入口
    data,labels = data_generate()    #产生数据集
