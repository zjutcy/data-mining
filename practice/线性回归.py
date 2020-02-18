# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 18:51:18 2019

@author: 64191
"""
import pandas as pd
from sklearn.model_selection import train_test_split 
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
Profit=pd.read_excel('C:/Users/64191/Desktop/1.xlsx')
#判断自变量间的多重共线性
X=Profit.loc[:,['RD_Spend','Marketing_Spend','Administration']]
vif=pd.DataFrame()
vif['feature']=X.columns
vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
#判断自变量与因变量间的线性关系
seaborn.pairplot(Profit.loc[:,['RD_Spend','Administration','Marketing_Spend','Profit']])
#对离散变量进行哑变量编码
dummies=pd.get_dummies(Profit.State)
Profit_new=pd.concat([Profit,dummies],axis=1)
Profit_new.drop(['State','New York'],axis=1,inplace=True)

#拆分数据集
train_data=Profit_new.loc[:,['RD_Spend','Administration','Marketing_Spend','California','Florida']]
train_target=Profit_new.Profit
X_train,X_test,Y_train,Y_test=train_test_split(train_data,train_target,test_size=0.3,random_state=1234)
#进行回归预测
re=LinearRegression()
re.fit(X_train,Y_train)
y_pred=re.predict(X_test)
y_pre=re.predict(X_train)
#进行回归效果评估
print(mean_squared_error(Y_train,y_pre))
print(mean_squared_error(Y_test,y_pred))






