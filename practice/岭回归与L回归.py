# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 13:11:39 2019

@author: 64191
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,RidgeCV,LinearRegression,LassoCV,Lasso
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error
#读取数据
data=pd.read_excel('C:/Users/64191/Desktop/diabetes.xlsx')
data_target=data['Y']
data=data.iloc[:,2:-1]
X_train,X_test,Y_train,Y_test=train_test_split(data,data_target,test_size=0.2,random_state=1234)
#判断VIF
vif=pd.DataFrame()
vif['feature']=data.columns
vif['factor']=[variance_inflation_factor(data.values,i) for i in range(data.shape[1])]
#存在严重多重共线性
#用岭回归
lambdas=np.logspace(-5,2,200)
ridge_coff=[]
for i in lambdas:
    ridge=Ridge(alpha=i,normalize=True)
    ridge.fit(X_train,Y_train)
    ridge_coff.append(ridge.coef_)
#plt.plot(lambdas,ridge_coff)
#plt.xscale('log')
ridge_cv=RidgeCV(alphas=lambdas,cv=10,normalize=True,scoring='neg_mean_squared_error')
ridge_cv.fit(X_train,Y_train)
ridge_best=ridge_cv.alpha_
ridge=Ridge(alpha=ridge_best,normalize=True)
ridge.fit(X_train,Y_train)
#进行测试
y_pre=ridge.predict(X_test)
MSE=mean_squared_error(y_pre,Y_test)
#用Lasso回归
l_cv=LassoCV(alphas=lambdas,cv=10,max_iter=1000)
l_cv.fit(X_train,Y_train)
l_best=l_cv.alpha_
l=Lasso(alpha=l_best,max_iter=10000)
l.fit(X_train,Y_train)

