# -*- coding: utf-8 -*-
"""
Spyder Editor

Create by Zhuo
"""

import cv2    #导入Opencv库
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('test.jpg',0)    #用灰度通道读取图像

"""提示：运行时可以将不需要的部分注释掉，并调整画图代码的参数显示图像"""

"""核处理"""
kernels = np.array([[0.0625,0.125,0.0625],
                    [0.125,0.25,0.125],
                    [0.0625,0.125,0.0625]])   #定义图像核
img_blur = cv2.filter2D(img,-1,kernels)

"""图像缩放"""
img_resize = cv2.resize(img,(96,96),interpolation=cv2.INTER_NEAREST)
  #通过第二个参数设置目标图像的尺寸

"""边缘检测"""
Gmax = np.max(img_resize)    #估算最大梯度
Tl = int(Gmax*1/3)    #计算低阀值
Th = int(Gmax*2/3)    #计算高阀值
img_canny = cv2.Canny(img,Tl,Th)   #Canny边缘检验
sobelXY = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)   #进行sobel检验
laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)   #进行拉普拉斯检验
sobelXY = cv2.convertScaleAbs(sobelXY)   #将结果有浮点数转化为整数
laplacian = cv2.convertScaleAbs(laplacian)
#
"""边角检测"""
img_corners = cv2.cornerHarris(img,2,5,0.06)


"""灰度图像转特征"""
img_rs = cv2.resize(img,(10,10),interpolation=cv2.INTER_LINEAR)
#使用双线性插值法缩放图像，目标图像为10X10
img_fea = img_rs.flatten()   #图像转特征

""""彩色图像转特征"""
img = cv2.imread('test.jpg',1)    #用三通道读取图像
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)   #将BGR编码转为RGB
channels = ["R","G","B"]
for i, channel in enumerate(channels):    #遍历三个通道
    hist = cv2.calcHist([img],[i],None,[256],[0,256])  #计算频数分布
    plt.plot(hist,color=channel,linewidth=(i+1)*2,label=channel) 
    #画出每个通道的频数分布
    plt.xlim([0,256])   #设置画图范围
plt.legend()    #显示图例
plt.show()
observation = np.array(img_fea).flatten()    #转为个体


"""设置HOGf法参数"""
dect_win_size = (64,128)    #设置检测窗口的尺寸为64X128，即包含64X128个像素
block_size = (16,16)    #定义块的大小为16X16
cell_size = (8,8)    #定义胞元的大小为8X8，即块中包含4个胞元
win_stride = (64,64)    #定义窗口的滑动步长为：宽度方向64，长度方向64
block_stride = (8,8)    #定义块的滑动步长为：宽度方向8，长度方向8，即窗口之间存在两个重叠的胞元
bins = 9    #定义直方图柱数为9，即将圆拆分成9等块供像素投影
"""使用HOG法将图像转换为向量"""
hog = cv2.HOGDescriptor(dect_win_size,block_size,block_stride,
                        cell_size,bins)    #生成一个HOG对象，并设置参数
img_fea_hog = hog.compute(img,win_stride)    #用HOG法将图像转换为特征
print(img_fea_hog)    #输出特征向量
print(img_fea_hog.shape)    #输出特征向量的长度

"""显示图像"""
plt.imshow(img,cmap="gray"),plt.axis('off')    #使用plt库打开
plt.show()
cv2.imshow("test picture",img)   #用cv2库显示图像，窗口标签为“test picture"
cv2.waitKey()
