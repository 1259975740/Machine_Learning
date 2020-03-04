# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 22:28:50 2020

@author: Zhuo
"""
import tarfile
HAM = 0
SPAM = 1
datadir = r'..\数据集\datasets_q8'   #datadir为解压路径
sources = [
        ('beck-s.tar.gz',HAM),('farmer-d.tar.gz',HAM),('kitchen-l.tar.gz',HAM),('lokay-m.tar.gz',HAM),('williams-w3.tar.gz',HAM),
        ('BG.tar.gz',SPAM),
        ('GP.tar.gz',SPAM),
        ('SH.tar.gz',SPAM)]
"""函数说明：def extract_tar函数用于解压某个tar.gz的压缩文件。"""
def extract_tar(datafile,extractdir):   #定义一个解压缩函数
    tar = tarfile.open(datafile)   #加压缩文件datafile
    tar.extractall(path=extractdir)   #文件解压到 extractdir
    tar.close()
    print("%s 解压完成."%datafile)
    
for source,_ in sources:    #通过遍历解压文件夹datasets_q8中所有文件
    datafile = r'%s\%s' %(datadir,source)   #指定待解压缩文件
    extract_tar(datafile,datadir)    #将文件压缩到路径datadir
    
"""函数说明:
    read_single_file用于读取单一文件的内容
    read_files用于从文件夹中读取所有文件的内容"""
import pandas as pd
import os #导入操作系统包
def read_single_file(filename):
    past_header,lines = False, []    
    if os.path.isfile(filename):    #判断是否为单一文件
        f = open(filename,encoding='utf-8')    #用utf-8编码方式打开
        for line in f:
            if past_header:
                lines.append(line)
            elif line == '\n':
                past_header = True
        f.close()
    content = '\n'.join(lines)    #整合成一个字符串
    return filename, content
def read_files(path):
    for root,dirname,filenames in os.walk(path):   #不断进入文件夹，直到“遇到”单一文件
        for filename in filenames:
            filepath = os.path.join(root,filename)   
            yield read_single_file(filepath)    #执行函数read_single_file,读取文件内容

"""函数说明：
   build_data_frame：用于从读取的文件中构建数据集"""
def build_data_frame(extractdir,classification):
    rows = []    #读取每一行，以构建dataframe作为数据集
    index = []    #构建dataframe的索引
    for file_name, text in read_files(extractdir):
        rows.append({'text':text,'class':classification})
        index.append(file_name)
    data_frame = pd.DataFrame(rows,index=index)
    return data_frame
data = pd.DataFrame({'text':[],'class':[]})
for source, classification in sources:
    extractdir = '%s\%s' %(datadir,source[:-7])
    data = data.append(build_data_frame(extractdir,classification))
    
