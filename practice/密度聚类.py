# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 12:50:44 2019

@author: 64191
"""

import pandas as pd
import numpy as np
from sklearn.datasets.samples_generator import make_moons,make_blobs
from sklearn.cluster import KMeans,DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt
#构造数据
X1,Y1=make_moons(n_samples=2000,noise=0.05,random_state=1234)
X2,Y2=make_blobs(n_samples=1000,centers=[[3,3]],cluster_std=0.5,random_state=1234)
Y2=np.where(Y2==0,2,0)
plot_data=pd.DataFrame(np.row_stack([np.column_stack((X1,Y1)),np.column_stack((X2,Y2))]),columns=['x1','x2','y'])
sns.lmplot('x1','x2',data=plot_data,hue='y',markers=['^','o','>'],fit_reg=False,legend=False)
#进行聚类比较
kmeans=KMeans(n_clusters=3,random_state=1234)
kmeans.fit(plot_data[['x1','x2']])
dbscan=DBSCAN(eps=0.3,min_samples=5)
dbscan.fit(plot_data[['x1','x2']])
plot_data['k']=kmeans.labels_
plot_data['d']=dbscan.labels_
plt.figure(figsize=(12,6))
ax1=plt.subplot2grid(shape=(1,2),loc=(0,0))
ax1.scatter(plot_data.x1,plot_data.x2,c=plot_data.k)
ax2=plt.subplot2grid(shape=(1,2),loc=(0,1))
ax2.scatter(plot_data.x1,plot_data.x2,c=plot_data.d.map({-1:2,0:0,1:3,2:1}))
print(plot_data.loc[plot_data.d==-1])
