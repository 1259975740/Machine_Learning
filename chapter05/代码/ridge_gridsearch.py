"""
+reated on Thu Feb  6 12:10:17 2020

@author: zhuo
"""
from sklearn.linear_model import Ridge 
from sklearn.model_selection import GridSearchCV    #导入网格搜索模块
from sklearn.metrics import make_scorer,r2_score
from sklearn import datasets
boston = datasets.load_boston()    #导入boston数据集
X,y = boston.data, boston.target
grid = {'alpha':[0.01,0.1,0.5,1.0,1.5]}    #生成参数网格
r2_scorer = make_scorer(r2_score)    #定义一个scorer类，用于设置GridSearchCV的参数scoring
model = Ridge()    #选择模型
ridge_gs = GridSearchCV(model,param_grid=grid,cv=5,scoring=r2_scorer)    #cv=5设置交叉验证的折数
ridge_gs.fit(X,y)    #进行网格寻优
print(ridge_gs.best_params_)    #输出参数
