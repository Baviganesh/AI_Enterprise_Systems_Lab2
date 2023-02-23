# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 18:57:33 2023
import numpy as np
import pandas as pd
data = pd.read_csv("dataset.csv")
data.shape
data.head()
benign     = data[data["diagnosis"] == "B"]
malignant = data[data["diagnosis"] == "M"]
data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
data = data.drop_duplicates(keep='last')
y = data['diagnosis']
x = data.drop(['diagnosis'], axis = 1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() # 
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
def models(x_train,y_train):
    #Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators =20,criterion='entropy',random_state=42)
    forest.fit(x_train,y_train)
   
   
    print('Random Forest Classifier Training Accuracy:',forest.score(x_train,y_train))
   
    return forest
model = models(x_train,y_train)
from sklearn.metrics import confusion_matrix
for i in range(len(model)):
    cm=confusion_matrix(y_test,model[i].predict(x_test))

    TP=cm[0][0]
    TN=cm[1][1]
    FN=cm[1][0]
    FP=cm[0][1]

    print(cm)
    print('Testing Accuracy =',(TP+TN)/(TP+TN+FN+FP))

@author: bavig
"""

