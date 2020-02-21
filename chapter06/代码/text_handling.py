# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:18:46 2020

@author: Administrator
"""
import numpy as np
import pandas as pd
import string
data = ['     Your Apple ID has been Locked!!!      ',
'    This Apple ID has been locked for security reasons.!!! It looks like your account is outdated and requires updated account ownership information,$$## so we can protect your account and improve our services to maintain your privacy.',
'    To continue using the Apple ID service, we advise you to update the information about your account ownership.']
data_strip = [string.strip() for string in data]   #采用strip方法去除两边的空格
#print(data)
#print(data_strip)   #展示处理前后的数据集
data_lower = [string.lower() for string in data_strip]    #将文本全部转换为小写
data_replace_pun = []    
for i in data_lower: 
    tmp = " ".join("".join([" " if ch in string.punctuation else 	#识别标点符号，并用空格替代
                 ch for ch in i]).split())

    data_replace_pun.append(tmp)
"""自然语言处理II"""
from nltk.tokenize import word_tokenize    #导入tokenize包
from nltk.corpus import stopwords    #导入常见词表
stopwds = stopwords.words('english')	   #导入英文常见词表到stopwds变量中
from nltk.stem.porter import PorterStemmer    #导入PorterStemmer包
from nltk import pos_tag    #导入标注词汇包
data2 = []    #用以构成经过自然语言处理II后的语料库
stemmer = PorterStemmer()
for s in data_replace_pun:     #对语料库中的每一条文本进行拆分处理
    s_tokens = word_tokenize(s)     #对s进行拆分
    print(s_tokens)     #输出s_tokens
    tokens_remove = [word for word in s_tokens if word not in stopwds]
    print(tokens_remove)
    tokens_stem = [stemmer.stem(word) for word in tokens_remove]
    print(tokens_stem)
    tokens_tag = pos_tag(s_tokens)    #对s_tokens进行标注
    print(tokens_tag)
    data2.append(" ".join(tokens_stem))    #构成新的字符串，每个单词以空格隔开
"""BOW方法"""
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()    #导入CountVectorizer模型
bow = vectorizer.fit_transform(data2)     #使用BOW方法转换语料库
data3 = bow.toarray()    #以列表形式输出语料库
data3_fea = vectorizer.get_feature_names()    #输出每一列的特征名称
data3_df = pd.DataFrame(data3,columns=data3_fea)    #以dataframe格式输出

"""2-grams"""
vectorizer = CountVectorizer(ngram_range=(2,2))    #导入CountVectorizer模型语料库
bow = vectorizer.fit_transform(data2)     #使用BOW方法转换语料库
data4 = bow.toarray()    #以列表形式输出语料库
data4_fea = vectorizer.get_feature_names()    #输出每一列的特征名称
data4_df = pd.DataFrame(data4,columns=data4_fea)    #以dataframe格式输出
"""tf-idf"""
from sklearn.feature_extraction.text import TfidfVectorizer    #导入相应的包
vectorizer = TfidfVectorizer()    #导入Tfidf方法模型
tf_idf = vectorizer.fit_transform(data2)    #使用Tfidf方法
data4 = tf_idf.toarray()    #以列表形式输出语料库
data4_fea = vectorizer.get_feature_names()    #输出每一列的特征名称
data4_df = pd.DataFrame(data4,columns=data4_fea)    #以dataframe格式输出
