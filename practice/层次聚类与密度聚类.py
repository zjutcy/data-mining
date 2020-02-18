# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 19:22:03 2019

@author: 64191
"""
from sklearn.preprocessing import scale
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import DBSCAN,AgglomerativeClustering
data=pd.read_excel('C:/Users/64191/Desktop/Province.xlsx')
X=scale(data.loc[:,['Birth_Rate','Death_Rate']])
X=pd.DataFrame(X)
#选择密度聚类合适的超参数
res=[]
for eps in np.arange(0.001,1,0.05):
    for min_ in range(2,10):
        dbscan=DBSCAN(eps=eps,min_samples=min_)
        dbscan.fit(X)
        n_cluster=len([i for i in set(dbscan.labels_) if i !=-1])
        outline=np.sum(np.where(dbscan.labels_==-1,1,0))
        state=str(pd.Series([i for i in dbscan.labels_ if i !=-1]).value_counts().values)
        res.append({'eps':eps,'min_samples':min_,'n_cluster':n_cluster,'outline':outline,'state':state})
df=pd.DataFrame(res)
print(df.loc[df.n_cluster==3,:])
#构建密度聚类与层次聚类的模型
dbscan=DBSCAN(eps=0.801,min_samples=3)
dbscan.fit(X)
data['label']=dbscan.labels_
sns.lmplot(x='Birth_Rate',y='Death_Rate',hue='label',data=data,markers=['o','^','d','*'],fit_reg=False,legend=False)
agnes_min=AgglomerativeClustering(n_clusters=3,linkage='ward')
agnes_min.fit(X)
data['a_label']=agnes_min.labels_
sns.lmplot(x='Birth_Rate',y='Death_Rate',hue='a_label',data=data,markers=['o','^','d'],fit_reg=False,legend=False)

                                                                        
