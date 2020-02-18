# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 17:03:50 2019

@author: 64191
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import minmax_scale
import seaborn as sns

#折现拐点法选k
def K_cut(X,cluster):
  TSSE=[]
  k=range(1,cluster+1)
  for i in k:
      SSE=[]
      kmeans=KMeans(n_clusters=i)
      kmeans.fit(X)
      labels=kmeans.labels_
      centers=kmeans.cluster_centers_
      for label in set(labels):
         a=np.sum((X.loc[labels==label,:]-centers[label])**2)
         SSE.append(a)
      TSSE.append(np.sum(SSE))
  plt.plot(k,TSSE,'b*-')
#轮廓系数法选k
def k(X,cluster):
    k=range(2,cluster+1)
    TS=[]
    for i in k:
        kmeans=KMeans(n_clusters=i)
        kmeans.fit(X)
        labels=kmeans.labels_
        TS.append(silhouette_score(X,labels,metric='euclidean'))
    plt.plot(k,TS,'b*--')
#进行数据预处理
data=pd.read_csv('C:/Users/64191/Desktop/players.csv')
X=minmax_scale(data.loc[:,['得分','罚球命中率','命中率','三分命中率']])
new_data=pd.DataFrame(X,columns=['得分','罚球命中率','命中率','三分命中率'])
#K_cut(new_data,15)
#k(new_data,15)
#确定系数后进行分类
kmeans=KMeans(n_clusters=3)
kmeans.fit(new_data)
data['cluster']=kmeans.labels_
centers=[]
for i in data.cluster.unique():
    a=data.loc[data.cluster==i,['得分','罚球命中率','命中率','三分命中率']].mean()
    centers.append(a)
centers=np.array(centers)
sns.lmplot(x='得分',y='命中率',hue='cluster',data=data,fit_reg=False,scatter_kws={'alpha':0.8},legend=False)
plt.scatter(centers[:,0],centers[:,2],c='k',s=180)





    


