# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 19:09:57 2020

@author: Zhuo
"""


from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
def data_generate():
   x = np.arange(-1.5,1.5,0.03).reshape(-1,1)
   l = len(x)
   y = np.zeros(l)
   for i in range(0,l):
       y[i] = 2*x[i]**3+3*x[i]**2+x[i]+1.5+np.random.uniform(0,6)

   return x,y
#def __name__ == "__main__":
#    X0 = X[y.ravel()==0]
#    plt.scatter(X0[:, 0], X0[:, 1], marker='o',c='k')  
#    X1 = X[y.ravel()==1]
#    plt.scatter(X1[:, 0], X1[:, 1], marker='x') 