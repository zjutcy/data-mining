# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:42:23 2019

@author: 64191
"""

import numpy as np 
import matplotlib.pyplot as plt 
#生成虚拟数据集
x = np.random.uniform(-3,3,size = 100)
X = x.reshape(-1,1)  
y = 0.5 * x**2 + x + 2 + np.random.normal(0,1,size = 100)

from sklearn.preprocessing import PolynomialFeatures
#degree :为数据添加几次幂
ploy = PolynomialFeatures(degree = 5)
ploy.fit(X)
X2 = ploy.transform(X)
#里面已经在第一列添加一列1了，所以不需要增加一列纯1的X0
from sklearn.linear_model import LinearRegression,Ridge
lin_reg2 = LinearRegression()
lin_reg2.fit(X2,y)
y_predict2 = lin_reg2.predict(X2)
ridge=Ridge(alpha=60)
ridge.fit(X2,y)
y_pre=ridge.predict(X2)
plt.scatter(x,y)
plt.plot(np.sort(x),y_predict2[np.argsort(x)],color = 'r')
plt.plot(np.sort(x),y_pre[np.argsort(x)],color = 'g')
plt.show()
print(lin_reg2.coef_)
print(ridge.coef_)


