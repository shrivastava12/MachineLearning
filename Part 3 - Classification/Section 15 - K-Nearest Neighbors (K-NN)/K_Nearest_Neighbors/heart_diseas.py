# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 04:26:23 2020

@author: mithu
"""

# Importing library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing datasets
data = pd.read_csv('heart.csv')
X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values

# Splitting the data into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state = 0)

# Feature scalling of the dataset
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting model to training set
from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski',p = 2 )
classifier.fit(X_train,y_train)

# Predicting the result
y_pred = classifier.predict(X_test)

# Creating confusion metrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

total = sum(sum(cm))

