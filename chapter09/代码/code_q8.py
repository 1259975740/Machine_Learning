# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 18:20:42 2020

@author: Administrator
"""
from sklearn.datasets import make_classification
X,y = make_classification(n_samples=200, n_features=20, n_informative=4, n_redundant=5,  
                    n_repeated=2, n_classes=4, n_clusters_per_class=2, weights=None,  
                    flip_y=0.01, class_sep=1.0, hypercube=True,shift=0.0, scale=1.0,   
                    shuffle=True, random_state=None)  #产生一个分类用数据集

#from sklearn.decomposition import PCA
#from sklearn.decomposition import KernelPCA
