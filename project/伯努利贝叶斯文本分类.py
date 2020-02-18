# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 18:26:54 2019

@author: 64191
"""
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#制作词向量表
data=pd.read_excel('C:/Users/64191/Desktop/Contents.xlsx',sheetname=0)
data.Content=data.Content.str.replace('[0-9a-zA-A]','')
jieba.load_userdict(r'C:/Users/64191/Desktop/all_words.txt')
with open(r'C:/Users/64191/Desktop/mystopwords.txt',encoding='UTF-8') as f:
    stop_words=[i.strip('\n') for i in f.readlines()]
def cut(x):
    words=[]
    for i in jieba.lcut(x):
        if i not in stop_words:
            words.append(i)
    result=' '.join(words)
    return result
word=data.Content.apply(cut)
counts=CountVectorizer(min_df=0.01)
data_matrix=counts.fit_transform(word).toarray()
#进行分类与测试
X=pd.DataFrame(data_matrix,columns=counts.get_feature_names())
Y=data.Type
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=1)
bnb=BernoulliNB()
bnb.fit(X_train,Y_train)
pred=bnb.predict(X_test)
print(classification_report(Y_test,pred))