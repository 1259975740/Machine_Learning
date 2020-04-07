# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:48:13 2020

@author: Zhuo
"""

import csv		#引出csv包，用于以csv的方式打开文件
smsdata = open(r'D:\桌面\我的书\chapter08\数据集\SMSSpamCollection.txt',
               'r',encoding='utf-8')		#以只读的方式打开数据文件
csv_reader = csv.reader(smsdata,delimiter='\t')		#使用csv读取txt文件，将其转化为表格
smsdata_data = [];		#初始化变量smsdata_data ，用以保存邮件数据
labels = [];		#初始化标签标量，用以保存邮件分类标签（spam/ham）
for line in csv_reader:	#以下代码将数据分别存入smsdata_data 、smsdata_labels中
    labels.append(line[0])
    smsdata_data.append(line[1])
smsdata.close()		#关闭文件
"""进行自然语言处理，将第五章"""

import nltk		#引入自然语言处理包，下面均为自然语言处理的常用包
from nltk.corpus import stopwords		#用于识别常见词的包
from nltk.tokenize import word_tokenize    #导入tokenize包
import string		#用于识别标点符号的包
import pandas as pd
from nltk.stem import PorterStemmer	#词干提取

def preprocessing(text):    #text为一条邮件，这里定义一个预处理函数
    text2 = " ".join("".join([" " if ch in string.punctuation else 	#识别标点符号，并用空格替代标点符号
                 ch for ch in text]).split())
    tokens = word_tokenize(text2)
 
    tokens = [word.lower() for word in tokens]		#将词全部转换为小写
    
    stopwds = stopwords.words('english')			#识别并删除常见词
    tokens_remove = [word for word in tokens if word not in stopwds]
    
    stemmer = PorterStemmer()		#词的删减，比如learns改为learn
    tokens_stem = [stemmer.stem(word) for word in tokens_remove]
    preproc_text = " ".join(tokens_stem)
    return preproc_text		#返回经自然语言处理的文本

data = []			#生成预处理后的数据集
for i in smsdata_data:		#对每一份邮件，均进行预处理并输出
    data.append(preprocessing(i))
    
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data
        ,labels,test_size=0.3,random_state=1)    #按7：3拆分数据集

from sklearn.feature_extraction.text import TfidfVectorizer    #导入相应的包
vectorizer = TfidfVectorizer()    #导入Tfidf方法模型
X_train = vectorizer.fit_transform(X_train).todense()
X_test = vectorizer.transform(X_test).todense()

from sklearn.naive_bayes import MultinomialNB    #引入朴素贝叶斯分类器模块
NB = MultinomialNB(alpha=1)    #导入多项式贝叶斯分类器，拉普拉斯修正系数为1
NB.fit(X_train,y_train)    #训练模型
y_train_pred = NB.predict(X_train)		#使用训练好的分类器对训练样本进行预测
y_test_pred = NB.predict(X_test)	#使用分类器对测试样本进行预测

from sklearn.metrics import classification_report , accuracy_score	#引入评价用包
#输出训练集、测试集的精度和结果摘要
print('训练集结果报表')
print(classification_report(y_train,y_train_pred))
print('测试集结果报表')
print(classification_report(y_test,y_test_pred))



        
    
    

