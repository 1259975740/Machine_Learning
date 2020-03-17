# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 22:36:46 2020

@author: Zhuo
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 21:45:46 2020

@author: Zhuo
"""

"""产生一个实例数据"""
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-1.5,1.5,0.03).reshape(-1,1)
l = len(x)
y = np.zeros(l)
for i in range(0,l):
    y[i] = 2*x[i]**3+3*x[i]**2+x[i]+1.5+np.random.uniform(0,12)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)    #拆分数据

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=1)
dtr.fit(x_train,y_train)

import matplotlib.pyplot as plt   
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.scatter(x_train,y_train,s=15,c='k',label='Train')
plt.scatter(x_test,y_test,s=55,c='k',marker='x',label='Test')
plt.xlabel('x',fontsize=20)
y_pred = dtr.predict(x);    
plt.legend(prop=font1)
plt.plot(x,y_pred)

from sklearn.tree import DecisionTreeRegressor
dtr2 = DecisionTreeRegressor(random_state=2,max_leaf_nodes=10)
dtr2.fit(x_train,y_train)
plt.xlabel('x',fontsize=20)
y_pred = dtr2.predict(x);    
plt.legend(prop=font1)
plt.plot(x,y_pred,linestyle='dashed',linewidth=5)


from sklearn.tree import DecisionTreeRegressor
dtr3 = DecisionTreeRegressor(random_state=3,max_depth=3)
dtr3.fit(x_train,y_train)
plt.xlabel('x',fontsize=20)
y_pred = dtr3.predict(x);    
plt.legend(prop=font1)
plt.plot(x,y_pred,linestyle='--',linewidth=4)

