# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 19:44:54 2020

@author: Zhuo
"""

"""运行代码可得古登堡计划的部分文本构成的预料库"""
from nltk.corpus import gutenberg
texts_list = gutenberg.fileids()
test_corpus = []
for title in texts_list:
    test_corpus.append(gutenberg.raw(title))