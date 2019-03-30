# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 19:40:32 2019

@author: sarth_000
"""

#data preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelwala_1 = LabelEncoder()
x[:,1] = labelwala_1.fit_transform(x[:,1])
labelwala_2 = LabelEncoder()
x[:,2] = labelwala_2.fit_transform(x[:,2])

onehotencoder = OneHotEncoder(categorical_features=[1])
x = onehotencoder.fit_transform(x).toarray()

#dummy variable trap
x=x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# ANN
import keras

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

#input layer and first hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))

#second hidden layer. (optional)
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

#output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#integration of all layers (loss=== binary outcome- bianry_crossentropy, more than two outcome- categorical_crossentropy)
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])


classifier.fit(x_train,y_train,batch_size=10,epochs=200)

y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)



















