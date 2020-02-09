"""
+reated on Thu Feb  6 12:10:17 2020

@author: zhuo
"""
from sklearn.linear_model import Ridge 
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
boston = datasets.load_boston()    #导入boston数据集
X,y = boston.data, boston.target
alpha_vec = [1.0,0.1,0.01,0.005,0.0025,0.001,0.00025]   #生成参数向量
model = Ridge()    #选择模型
ridge_gs = GridSearchCV(model,param_grid={'alpha':alpha_vec},cv=5)
ridge_gs.fit(X,y)
print(ridge_gs.best_estimator_)    #输出最佳参数值