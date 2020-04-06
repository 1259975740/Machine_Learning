# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:59:38 2020

@author: Zhuo
"""

PI = 3.14

def circle_area(r):    #定义一个能够计算圆面积的函数
    area = PI*r**2
    return area

if __name__ == '__main__':  # 主函数入口
    print('半径为2的圆的面积为：',circle_area(2))
    print('pi的大小为',PI)