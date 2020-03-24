# 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression,ElasticNet,Lasso


#创建数据集，并拆分成训练集、测试集。训练集容量为150，测试集为50.
X = np.arange(-5, 5, 0.05)
y = X + 2
y += np.random.uniform(0, 5, size=200)    #给数据添加噪声
for i in range(150, 200):    #给测试集数据添加噪声，使其偏离训练数据
    y[i] += np.random.uniform(-3, -2)   
X,y = X.reshape(-1,1),y.reshape(-1,1)    #将数据转换成行向量
X_train,y_train = X[0:150],y[0:150]    #拆分数据
X_test,y_test = X[150:200],y[150:200]



"""画出散点图像"""
font1 = {'family' : 'SimHei',
'weight' : 'normal',
'size'   : 16,
}
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.scatter(X_train, y_train,s=16,label='训练集')
plt.scatter(X_test, y_test, marker='x',s=40,label='测试集')
plt.xlabel('x',fontsize=16)
plt.ylabel('y',fontsize=16)
plt.legend(prop=font1)

lr = LinearRegression()
lr.fit(X_train,y_train)    #训练线性回归模型
lasso = Lasso(alpha=1)     #设置正则化系数为1
lasso.fit(X_train,y_train)    #训练Lasso回归
elastic = ElasticNet(alpha=1.0,l1_ratio=0.5)    #设置正则化系数alpha,beta=1,0.5
elastic.fit(X_train,y_train)    #训练ElasticNet回归
"""输出模型参数"""
print('线性回归模型: y = %.3fx + %.3f' % (lr.coef_, lr.intercept_))
print('Lasso回归模型: y = %.3fx + %.3f' % (lasso.coef_, lasso.intercept_))
print('ElasticNet回归模型: y = %.3fx + %.3f' % (elastic.coef_, elastic.intercept_))

"""画出函数图像"""
plt.subplot(1,2,2)
x_plot = np.arange(-5, 5, 0.05).reshape(-1,1)
y = lasso.predict(x_plot)
plt.plot(x_plot,y,linestyle='--',label='Lasso回归',linewidth=3,c='k')
y = lr.predict(x_plot)
plt.plot(x_plot,y,label='线性回归',c='k')
y = elastic.predict(x_plot)
plt.plot(x_plot,y,linestyle=':',linewidth=3,label='ElasticNet回归',c='k')
plt.legend(prop=font1)
plt.scatter(X_train, y_train,s=16)
plt.scatter(X_test, y_test, marker='x',s=40)
plt.xlabel('x',fontsize=16)
plt.ylabel('y',fontsize=16)
plt.show()


# if __name__ == '__main__':
       
#     # Show the dataset
#     show_dataset(X, Y)

#     # Create a linear regressor
#     lr = LinearRegression(normalize=True)
#     lr.fit(X.reshape(-1, 1), Y.reshape(-1, 1))
#     print('Standard regressor: y = %.3fx + %.3f' % (lr.coef_, lr.intercept_))

#     # Create RANSAC regressor
#     rs = RANSACRegressor(lr)
#     rs.fit(X.reshape(-1, 1), Y.reshape(-1, 1))
#     print('RANSAC regressor: y = %.3fx + %.3f' % (rs.estimator_.coef_, rs.estimator_.intercept_))