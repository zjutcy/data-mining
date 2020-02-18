# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 09:42:29 2020

@author: 64191
"""
import pandas as pd
train_data=pd.read_csv('C:/Users/64191/Desktop/happiness_train_complete.csv',encoding='latin-1')
test_data=pd.read_csv('C:/Users/64191/Desktop/happiness_test_complete.csv',encoding='latin-1')
train_data=train_data.loc[train_data['happiness']!=-8,:]
target=train_data['happiness']-1
del train_data['happiness']
data=pd.concat([train_data,test_data],axis=0,ignore_index=True)
data['survey_time']=pd.to_datetime(data['survey_time'])
data['survey_month']=data['survey_time'].dt.month
data['survey_day']=data['survey_time'].dt.day
data['survey_hour']=data['survey_time'].dt.hour

X=data.drop(columns=["survey_time","edu_other","property_other","invest_other"])
X["join_party"]=X["join_party"].apply(lambda x:0 if pd.isnull(x)  else 1)
X=X.fillna(0)
X_train=X.iloc[:7988,:]
X_train=X_train.drop(columns=['id'])
X_test=X.iloc[7988:,:]
X_test=X_test.reset_index()
del X_test['index']
a=X_test.drop(columns=['id'])


#from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import RandomForestClassifier
#RF= RandomForestClassifier(n_estimators=50,max_features=60)
##a={'n_estimators':[50,100,150],'max_features':[20,40,60]}
##grid=GridSearchCV(estimator=RF,param_grid=a,scoring='neg_mean_squared_error',cv=5)
##grid.fit(X_train,target)
##from sklearn.model_selection import cross_val_score
##scores = cross_val_score(RF,X_train,target,cv=10,scoring='neg_mean_squared_error')
#RF.fit(X_train,target)
#pred=RF.predict(a)
#X_test['happiness']=pred
#r=X_test.loc[:,['id','happiness']]
#r.to_csv("happiness_submit.csv",index=False)

from sklearn.decomposition import PCA
pca=PCA(n_components=8)
X_train=pca.fit_transform(X_train)
a=pca.transform(a)
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from sklearn.preprocessing import OneHotEncoder
one=OneHotEncoder()
target=one.fit_transform(target.values.reshape([-1,1])).toarray()
model = Sequential()
model.add(Dense(input_dim=8,output_dim=25))
model.add(Activation('sigmoid'))
model.add(Dense(input_dim=25,output_dim=5))
model.add(Activation('sigmoid'))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(X_train,target,nb_epoch=1000,batch_size=50)
pred=model.predict_classes(a)
X_test['happiness']=pred
X_test['happiness']=X_test['happiness']+2
r=X_test.loc[:,['id','happiness']]
r.to_csv('happy.csv')



