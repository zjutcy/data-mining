# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import datetime
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB 
from sklearn.tree import DecisionTreeClassifier,export_graphviz
import pydotplus
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,OneHotEncoder
from sklearn import metrics
from sklearn.datasets import load_wine,load_iris,make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
import numpy as np
from sklearn.cluster import KMeans,DBSCAN
a=[['l1', 'l2', 'l5'], ['l2', 'l4'], ['l2', 'l3'],
            ['l1', 'l2', 'l4'], ['l1', 'l3'], ['l2', 'l3'],
            ['l1', 'l3'], ['l1', 'l2', 'l3', 'l5'], ['l1', 'l2', 'l3']]
#得到一项频繁集与它们的支持数
def get_L1(data,support):
    C1=[]
    for i in data:
        for j in i:
          C1.append(j)
    C1=set(C1)
    L1=[]
    L1_=[]
    L1_number=[]
    for i in C1:
        num=0
        for j in data:
            if(i in j):
                
                num=num+1
        
        if num>=support:
            L1.append(i)
            L1_number.append(num)
        else:
            L1_.append(i)
    return L1,L1_number
L1,L1_number=get_L1(a,2)
#得到二项频繁集与它们的支持数
def get_L2(a,data,support):
    C2=[]
    for i in data:
        for j in data:
            if i<j:
                C2.append(sorted([i,j]))
    L2=[]
    L2_=[]
    L2_number=[]
    for i in C2:
        num=0
        for j in a:
            if(set(i).issubset(set(j))):
                num=num+1
        if num>=support:
            L2.append(i)
            L2_number.append(num)
        else:
            L2_.append(i)
    return L2,L2_number
L2,L2_number=get_L2(a,L1,2)
#得到三项频繁集与它们的支持数
def get_L3(a,data,support):
    C3=[]
    for i in data:
        lenth=len(i)
        for j in data:
            set1=set(i[0:lenth-1])
            set2=set(j[0:lenth-1])
            if(list(set1.difference(set2))==[] and list(set(i).difference(set(j)))!=[]):
                C3_temp=set(i).union(set(j))
                C3.append(sorted(list(C3_temp)))
    C3_df = pd.DataFrame(C3)
    C3 = C3_df.drop_duplicates().values.tolist()
    
    L3=[]
    L3_number=[]
    for i in C3:
        num=0
        for j in a:
            if(set(i).issubset(set(j))):
                num=num+1
        if num>=support:
          L3.append(i)
          L3_number.append(num)
    return L3,L3_number        
L3,L3_number=get_L3(a,L2,2)
#计算置信度
def calconfidence(data1,data2,data1_number,data2_number):
    result=[]
    for num,i in enumerate(data1):
        if type(i)==str:
          for number,j in enumerate(data2):
              if(i in j):
                  confidence=data2_number[number]/data1_number[num]
                  for w in j:
                      if w!=i:
                          c=w
                  
                  result.append([i,c ,confidence])
        else:
            for number,j in enumerate(data2):
                if(set(i).issubset(set(j))):
                    confidence=data2_number[number]/data1_number[num]
                    result.append([i,list(set(j)-set(i)),confidence])
                    
            
            
    for i in result:
        print(i[0],'==>',i[1],'置信度',i[2])

calconfidence(L1,L2,L1_number,L2_number)


            



    
        


            
            
                
    



        
        



    
    
    
    
















 