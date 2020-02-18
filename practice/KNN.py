# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 19:22:10 2019

@author: 64191
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
import numpy as np
from sklearn.metrics import classification_report
data=pd.read_excel('C:/Users/64191/Desktop/Knowledge.xlsx')
X=data.iloc[:,:-1]
Y=data['UNS']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=1234)
knn=KNeighborsClassifier()
p={'n_neighbors':[1,2,3,4,5,6,7,8],'weights':['distance']}
grid=GridSearchCV(knn,param_grid=p,scoring='accuracy',cv=10)
grid.fit(X_train,Y_train)
print(grid.best_params_)
knn_class=KNeighborsClassifier(n_neighbors=6,weights='distance')
knn_class.fit(X_train,Y_train)
pred=knn_class.predict(X_test)
cm=pd.crosstab(pred,Y_test)

