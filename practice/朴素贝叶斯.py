# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 15:47:17 2019

@author: 64191
"""

import pandas as pd
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes  import GaussianNB,MultinomialNB
from sklearn.preprocessing import LabelEncoder
#高斯贝叶斯
#data=pd.read_excel('C:/Users/64191/Desktop/Skin_Segment.xlsx')
#
#X=data.iloc[:,:-1]
#Y=data.y
#X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=1234)
#gnb=GaussianNB()
#gnb.fit(X_train,Y_train)
#pred=gnb.predict(X_test)
#print(accuracy_score(Y_test,pred))
#print(classification_report(Y_test,pred))
#多项式贝叶斯
data=pd.read_csv('C:/Users/64191/Desktop/mushrooms.csv')
X=data.iloc[:,1:]
Y=data['type']

lb=LabelEncoder()
for i in X.columns:
    X[i]=pd.factorize(X[i])[0]



X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=10)
X_train['type']=Y_train

edible=X_train.loc[X_train['type']=='edible',:]

#mnb=MultinomialNB()
#mnb.fit(X_train,Y_train)
#pred=mnb.predict(X_test)
print(X_test.iloc[27,:])

data=pd.read_csv('C:/Users/64191/Desktop/mushrooms.csv')
X=data.iloc[:,1:]
Y=data['type']

lb=LabelEncoder()
for i in X.columns:
    X[i]=lb.fit_transform(X[i])



X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=10)
X_train['type']=Y_train

edible=X_train.loc[X_train['type']=='edible',:]

#mnb=MultinomialNB()
#mnb.fit(X_train,Y_train)
#new_pred=mnb.predict(X_test)
#p=pd.DataFrame({'a':pred,'b':new_pred})
print(X_test.iloc[27,:])


