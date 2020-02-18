# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:27:12 2019

@author: 64191
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#导入数据并预测
sports=pd.read_csv('C:/Users/64191/Desktop/1.csv')
X=sports.iloc[:,4:]
Y=sports.activity
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=1234)
log=LogisticRegression()
log.fit(X_train,Y_train)
pred=log.predict(X_test)
#进行性能评估
cm=confusion_matrix(Y_test,pred,labels=[0,1])
accuracy=accuracy_score(Y_test,pred)
y_score=log.predict_proba(X_test)[:,1]
fpr,tpr,thre=roc_curve(Y_test,y_score)
roc_auc=auc(fpr,tpr)

plt.stackplot(fpr,tpr,color='steelblue')
plt.plot(fpr,tpr,color='black',lw=1.5)
plt.plot([0,1],[0,1],color='red',linestyle='--')
plt.text(0.5,0.3,'Roc curve(area%0.2f)'%roc_auc)
plt.xlabel('fpr')
plt.ylabel('tpr')
c={'a':y_score,'b':pred}
d=pd.DataFrame(c)
print(d)
