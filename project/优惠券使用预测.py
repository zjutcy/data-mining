# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 21:21:54 2020

@author: 64191
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from datetime import date
data=pd.read_csv('C:/Users/64191/Desktop/train.csv')
data_test=pd.read_csv('C:/Users/64191/Desktop/test.csv')
def covertrate(x):
    if pd.isnull(x):
        return 1.0
    elif ':' in x:
        w=x.split(':')
        return 1-float(w[1])/float(w[0])
    else:
        return float(x)
data['rate']=data['Discount_rate'].apply(covertrate)
data_test['rate']=data_test['Discount_rate'].apply(covertrate)
data['Distance'] = data['Distance'].fillna(-1).astype(int)
data_test['Distance']=data_test['Distance'].fillna(-1).astype(int)
#找每个用户领取优惠券的次数
temp=data.loc[(data['Date'].notnull())&(data['Date_received'].notnull())]
temp_=temp.groupby('User_id').size().reset_index(name='user_coupon')
data=pd.merge(data,temp_,on='User_id',how='left')
data['user_coupon'].fillna(0,inplace=True)

data_test=pd.merge(data_test,temp_,on='User_id',how='left')
data_test['user_coupon'].fillna(0,inplace=True)

#提取星期

def getWeekday(row):
    if row == 'nan':
        return np.nan
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1
data['weekday']=data['Date_received'].astype(str).apply(getWeekday)
data['weekday_type']=data['weekday'].apply(lambda x:1 if x in [6,7] else 0)
weekdaycols = ['weekday_' + str(i) for i in range(1,8)]
temp=pd.get_dummies(data['weekday'])
temp.columns=weekdaycols
data=pd.concat([data,temp],axis=1)
data_test['weekday']=data_test['Date_received'].astype(str).apply(getWeekday)
data_test['weekday_type']=data_test['weekday'].apply(lambda x:1 if x in [6,7] else 0)
weekdaycols = ['weekday_' + str(i) for i in range(1,8)]
temp=pd.get_dummies(data_test['weekday'])
temp.columns=weekdaycols
data_test=pd.concat([data_test,temp],axis=1)

#def label(x):
#    if pd.isnull(x['Date_received']):
#        return -1
#    if pd.notnull(x['Date']):
#        td=pd.to_datetime(str(x['Date'])[0:8]).date()-pd.to_datetime(str(x['Date_received'])[0:8]).date()
#        if td<pd.Timedelta(15, 'D'):
#            return 1
#    return 0
def label(row):
    if pd.isnull(row['Date_received']):
        return -1
    if pd.notnull(row['Date']):
        td = pd.to_datetime(row['Date'], format='%Y%m%d') -  pd.to_datetime(row['Date_received'], format='%Y%m%d')
        if td <= pd.Timedelta(15, 'D'):
            return 1
    return 0
data['label']=data.apply(label,axis=1)
df = data[data['label'] != -1].copy()
train = df[(df['Date_received'] < 20160516)].copy()
valid = df[(df['Date_received'] >= 20160516) & (df['Date_received'] <= 20160615)].copy()
log=LogisticRegression()
original_feature = ['rate','Distance', 'weekday', 'weekday_type','user_coupon'] + weekdaycols
log.fit(train[original_feature],train['label'])
pred=log.predict_proba(data_test[original_feature])
dftest1 = data_test[['User_id','Coupon_id','Date_received']]
dftest1['label'] = pred[:,1]
dftest1.to_csv('submit1.csv', index=False, header=False)


        


