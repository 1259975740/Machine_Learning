# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 22:40:31 2020

@author: Zhuo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.subplots_adjust(left=0.125, bottom=None, right=0.9, top=None,
                wspace=0.3, hspace=None)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(1,2,1)
xValue = list(np.arange(0,1,0.001))
yValue = [np.random.rand() for x in xValue]
plt.title(u'无相关性')
plt.xlabel('x1')
plt.ylabel('y2')
plt.scatter(xValue, yValue, s=20, c="g", marker='o')

plt.subplot(1,2,2)
xValue = list(np.arange(0,1,0.001))
yValue = [x*np.random.rand() for x in xValue]
plt.title(u'存在一定相关性')
plt.xlabel('x1')
plt.ylabel('y2')
plt.scatter(xValue, yValue, s=20, c="r", marker='o')
plt.show()
