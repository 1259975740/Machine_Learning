# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 21:43:05 2020

@author: Zhuo
"""
from sklearn import datasets
X,y = datasets.make_blobs(100,2,centers=2,
                          random_state=1,cluster_std=2)    #产生数据集

from sklearn import naive_bayes

