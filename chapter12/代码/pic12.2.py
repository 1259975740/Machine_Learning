# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:10:55 2020

@author: Zhuo
"""

import matplotlib.pyplot as plt
import numpy as np 
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.1, hspace=0.4)

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def relu(x):
    return np.where(x<0,0,x)

def tanh(x):
    return 2*sigmoid(2*x)-1

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}


def plot_tran_fun():   
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
     
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(221)
     
    x = np.arange(-10, 10)
    y = sigmoid(x)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.set_xticks([-10,-5,0,5,10])
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    ax.plot(x,y,label="Sigmoid",color = "blue")
    plt.legend(prop=font1,loc='lower right')
#    plt.show()
#
#    
    ax = fig.add_subplot(222)
    x = np.arange(-10, 10)
    y = tanh(x)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.set_xticks([-10,-5,0,5,10])
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    ax.plot(x,y,label="Tanh",color = "blue")
    plt.legend(prop=font1)
#    ax.show()   
#    
    ax = fig.add_subplot(223)
#     
    x = np.arange(-10, 10)
    y = relu(x)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.set_xticks([-10,-5,0,5,10])
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    ax.plot(x,y,label="ReLU",color = "blue")
    plt.legend(prop=font1)     
#    
    ax = fig.add_subplot(224)
#     
    x = np.arange(-10, 10)
    y = x
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.set_xticks([-10,-5,0,5,10])
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    ax.plot(x,y,label="Linear",color = "blue")
    plt.legend(prop=font1)
     
    
plot_tran_fun()

#def main():
#    plot_sigmoid()
#
#def __name__ == "__main__":
#    main()