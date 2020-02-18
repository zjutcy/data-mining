# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:00:09 2020

@author: 64191
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller as ADF
data=pd.read_csv('C:/Users/64191/Desktop/AirPassengers.csv')
data['Month']=pd.to_datetime(data['Month'])
data.index=data['Month']
del data['Month']
#检查稳定性
data.plot()
plot_acf(data)
print(ADF(data['#Passengers']))
#进行差分处理
D_data=data.diff().dropna()
D_data.plot()
plot_acf(D_data)
print(ADF(D_data['#Passengers']))
##进行白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
print(acorr_ljungbox(D_data,lags=1))
#建立模型
from statsmodels.tsa.arima_model import ARIMA
pmax=int(len(D_data)/10)
qmax=int(len(D_data)/10)
bic_matrix=[]
for p in range(pmax+1):
    temp=[]
    for q in range(qmax+1):
        try:
        
            temp.append(ARIMA(data['#Passengers'],(p,1,q)).fit().bic)
        except:
            continue
    bic_matrix.append(temp)
d=pd.DataFrame(bic_matrix)
model=ARIMA(data,(2,1,2)).fit()
print(model.resid)
pred=model.forecast(20)
index=pd.date_range(start='1960-12-02',periods=20)
pre=pd.Series(pred[0],index=index)
plt.plot(data)
plt.plot(pre)



        



