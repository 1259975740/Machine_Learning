# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 23:38:27 2020

@author: Zhuo
"""
from sklearn.datasets import load_boston
datasets = load_boston()
X = datasets.data
y = datasets.targets
from sklearn.model_selection import train_test_split
X_train,y_train,X_test,y_test = 