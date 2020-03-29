# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:28:11 2020

@author: Zhuo
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datesets import make_circles
X,y = make_circles(n_samples=500, factor=.5,noise=.05)
