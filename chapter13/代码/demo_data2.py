# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:45:21 2020

@author: Zhuo
"""
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
def data_generate():
    X = np.zeros(shape=(500,2),dtype=np.float32)
    y = np.zeros(shape=(500,),dtype=np.float32)
    t = 15*np.random.uniform(0,1,size=(250,1))
    X[0:250,:] = t*np.hstack([-np.cos(t),np.sin(t)]) 
    + np.random.uniform(0.0,1.8,size=(250,2))
    y[0:250]=1
    X[250:,:] = t*np.hstack([np.cos(t),-np.sin(t)]) 
    + np.random.uniform(0.0,1.8,size=(250,2))
    y[250:]=0
    X,y = shuffle(X,y,random_state=1)
    return X,y
