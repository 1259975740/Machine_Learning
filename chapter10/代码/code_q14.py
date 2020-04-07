# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:20:27 2020

@author: Zhuo
"""

def data_generate():
    import gzip
    datadir = r'..\数据集\MNIST_data'   #datadir为解压路径
    sources = ['t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz',
               'train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz']
    
    """函数说明：def extract_tar函数用于解压某个tar.gz的压缩文件。"""
    def extract_tar(datafile,extractdir):   #定义一个解压缩函数
        file = datafile.replace(".gz","")
        g_file = gzip.GzipFile(datafile)
        #读取解压后的文件，并写入去掉后缀名的同名文件（即得到解压后的文件）
        open(file, "wb+").write(g_file.read())    #将文件加压缩到压缩文件所在的文件夹中
        g_file.close()
        print("%s 解压完成."%datafile)
        return file   #返回解压缩后文件的名称字符串
    data_file = []    
    for source in sources:    #通过遍历解压文件夹datasets_q8中所有文件
        datafile = r'%s\%s' %(datadir,source)   #指定待解压缩文件
        file = extract_tar(datafile,datadir)    #将文件压缩到路径datadir
        data_file.append(file)
    
    
    """很多类型的文件，其起始的几个字节的内容是固定的(或是有意填充，
    或是本就如此)。根据这几个字节的内容就可以确定文件类型，
    因此这几个字节的内容被称为魔数 (magic number)。"""
    import struct
    import numpy as np
    def decode_idx1_ubyte(idx1_ubyte_file):
        """
        解析idx1文件的通用函数
        :param idx1_ubyte_file: idx1文件路径
        :return: 数据集
        """
        # 读取二进制数据
        bin_data = open(idx1_ubyte_file, 'rb').read()
    
        # 解析文件头信息，依次为魔数和标签数
        offset = 0
        fmt_header = '>ii'
        magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
        print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))
    
        # 解析数据集
        offset += struct.calcsize(fmt_header)
        fmt_image = '>B'
        labels = np.empty(num_images)
        for i in range(num_images):
            if (i + 1) % 10000 == 0:
                print ('已解析 %d' % (i + 1) + '张')
            labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
            offset += struct.calcsize(fmt_image)
        return labels
    y_train = decode_idx1_ubyte(data_file[3])
    y_test = decode_idx1_ubyte(data_file[1])
    y_train = y_train.astype(np.int)
    y_test = y_test.astype(np.int)
    def decode_idx3_ubyte(idx3_ubyte_file):
        """
        解析idx3文件的通用函数
        :param idx3_ubyte_file: idx3文件路径
        :return: 数据集
        """
        # 读取二进制数据
        bin_data = open(idx3_ubyte_file, 'rb').read()
    
        # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
        offset = 0
        fmt_header = '>iiii' #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
        magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
        print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))
    
        # 解析数据集
        image_size = num_rows * num_cols
        offset += struct.calcsize(fmt_header)  #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
        print(offset)
        fmt_image = '>' + str(image_size) + 'B'  #图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
        print(fmt_image,offset,struct.calcsize(fmt_image))
        images = np.empty((num_images, num_rows, num_cols))
        #plt.figure()
        for i in range(num_images):
            if (i + 1) % 10000 == 0:
                print('已解析 %d' % (i + 1) + '张')
                print(offset)
            images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
            offset += struct.calcsize(fmt_image)
        return images
    X_train = decode_idx3_ubyte(data_file[2])
    X_test = decode_idx3_ubyte(data_file[0])
    
    """画图代码：读者可以运行该代码直观地查看数据集"""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5,5, figsize=(8, 8),subplot_kw={'xticks':[], 'yticks':[]},
                            gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_train[i], cmap='gray', interpolation='nearest')
        ax.text(0.07, 0.07, str(y_train[i]),transform=ax.transAxes, color='white')
    return X_train,y_train,X_test,y_test


 """参考答案"""
X_train,y_train,X_test,y_test = data_generate()
def flatten_img(X):
    X_new = []
    for i in range(0,X.shape[0]):
        x = X[i].flatten()
        X_new.append(x)
    return X_new
X_train = flatten_img(X_train)/255
X_test = flatten_img(X_test)/255

    
from sklearn.svm import SVC
svc = SVC(C=1.0,kernel='rbf',decision_function_shape='ovr')
svc.fit(X_train,y_train)
from sklearn.metrics import accuracy_score	#引入评价用包
y_train_pred = svc.predict(X_train)
y_test_pred = svc.predict(X_test)
print('训练集中的精确度为',accuracy_score(y_train,y_train_pred))
print('测试集中的精确度为',accuracy_score(y_test,y_test_pred))