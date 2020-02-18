# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:36:03 2019

@author: 64191
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

data=load_iris()
X=data['data']
Y=data['target']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
Y_train = keras.utils.to_categorical(Y_train, num_classes=3)

model=Sequential()
model.add(Dense(64,activation='relu',input_dim=4))
model.add(Dense(64, activation='relu'))
model.add(Dense(3,activation='sigmoid'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam')
model.fit(X_train,Y_train,nb_epoch=1000,batch_size=30)
pre=model.predict(X_test)
pred=model.predict_classes(X_test)

