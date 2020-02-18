# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:18:07 2020

@author: 64191
"""

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.model_selection import GridSearchCV,cross_val_score
import matplotlib.pyplot as plt
def train_model(model,X_train,Y_train,para=[],cv=6):
    if len(para)>0:
        gsearch = GridSearchCV(model, param_grid=para, cv=cv,
                               scoring='neg_mean_squared_error')
        gsearch.fit(X_train,Y_train)
        score=-gsearch.best_score_
        params=gsearch.best_params_
        a=gsearch.cv_results_
    else:
        result=cross_val_score(model,X_train,Y_train,scoring='neg_mean_squared_error',cv=cv)
        score=abs(np.mean(result))
        params=[]
        a=0
    return score,params,a
        
data_train=pd.read_table('C:/Users/64191/Desktop/zhengqi_train.txt',sep='\t')
data_test=pd.read_table('C:/Users/64191/Desktop/zhengqi_test.txt',sep='\t')
data_train["oringin"]="train"
data_test["oringin"]="test"
all_data=pd.concat([data_train,data_test],ignore_index=True,axis=0)
data1=all_data.loc[all_data['oringin']=='train',:].drop('oringin',axis=1)
#fcols=1
#frows=len(data1.columns)
#plt.figure(figsize=(5*fcols,4*frows))
#
#i=0
#for col in data1.columns:
#    i=i+1
#    ax=plt.subplot(frows,fcols,i)
#    
#    sns.regplot(data1[col],data1['target'],ax=ax)
#    plt.xlabel(col)
#    plt.ylabel('target')

corr=data1.corr()
col=corr.loc[abs(corr['target'])<0.1,:].index
all_data.drop(col,axis=1,inplace=True)
data2=all_data.drop(['oringin','target'],axis=1)
target=all_data.loc[all_data['oringin']=='train','target']
#from sklearn.preprocessing import MinMaxScaler
#scale=MinMaxScaler()
#all_data_new=pd.DataFrame(scale.fit_transform(data2),columns=data2.columns)
from sklearn.decomposition import PCA
pca=PCA(n_components=21)
all_data_new=pca.fit_transform(data2)
X_train=all_data_new[:len(data_train)]
X_test=all_data_new[len(data_train):]
model_name=[]
model_params={}
model_score=[]
lr=LinearRegression()
score,params,a=train_model(lr,X_train,target)
model_name.append('lr')
model_params['lr']=params
model_score.append(score)
ridge=Ridge()
score,params,a=train_model(ridge,X_train,target,para={'alpha':[0,5]})
print(a)

model_name.append('ridge')
model_score.append(score)
model_params['ridge']=params
result=pd.Series(model_score,index=model_name)
ridge=Ridge(alpha=5)
ridge.fit(X_train,target)
pred=ridge.predict(X_test)
pred=pd.DataFrame(pred)
pred.to_csv('result.txt',index=False,header=False)

 
