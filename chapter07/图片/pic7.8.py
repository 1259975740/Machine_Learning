# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:39:21 2020

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
X1 = np.random.normal(0, 2, 50)
X2 = np.random.normal(0, 3, 50)
X3 = np.random.normal(0, 0.5, 50)
plt.plot(X1,label='X1')
plt.plot(X2,'--',color='k',label='X2')
plt.plot(X3,':',linewidth=4,color='k',label='X3')

plt.legend(loc='best')
plt.show()