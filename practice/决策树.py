# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 15:29:43 2019

@author: 64191
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plot
#进行一些预处理
data=pd.read_csv('C:/Users/64191/Desktop/Titanic.csv')
data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
age_mean=data.Age.groupby(data['Sex']).mean()
for i in data.Sex.unique():
   data.loc[data.Sex==i,'Age']=data.loc[data.Sex==i,'Age'].fillna(age_mean[i])
data.Embarked.fillna(data.Embarked.mode()[0],inplace=True)
lb=LabelEncoder()
data.Sex=lb.fit_transform(data.Sex)
data.Embarked=lb.fit_transform(data.Embarked)
X=data.iloc[:,1:]
Y=data.iloc[:,0]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=1234)
#构建模型
tree=DecisionTreeClassifier()
max_depth=[2,3,4,5,6]
min_samples_split=[2,3,4,5,6,7,8]
min_samples_leaf=[2,3,4,5,6,7,8,9]
parameters={'max_depth':max_depth,'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf}
grid=GridSearchCV(estimator=tree,param_grid=parameters,cv=10,scoring='accuracy')
grid.fit(X_train,Y_train)

new_tree=DecisionTreeClassifier(max_depth=6,min_samples_leaf=4,min_samples_split=6)
new_tree.fit(X_train,Y_train)
pred=new_tree.predict(X_test)
print(accuracy_score(Y_test,pred))
#构建随机森林模型
n=[i for i in range(100,200,10)]
rtree=RandomForestClassifier()
parameters={'n_estimators':n}
grid=GridSearchCV(estimator=rtree,param_grid=parameters,cv=10,scoring='accuracy')
grid.fit(X_train,Y_train)
new_rtree=RandomForestClassifier(n_estimators=100)
new_rtree.fit(X_train,Y_train)
pred=new_rtree.predict(X_test)
importance=new_tree.feature_importances_
ser=pd.Series(importance,index=X_train.columns)
ser.sort_values().plot('barh')

print(accuracy_score(Y_test,pred))

