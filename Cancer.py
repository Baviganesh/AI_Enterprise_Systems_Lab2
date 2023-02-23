# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 18:48:19 2023
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
   
    #Logistic Regression
     from sklearn.linear_model import LogisticRegression
     log = LogisticRegression(random_state=0)
     log.fit(x_train,y_train)
    
    #Decision Tree
     from sklearn.tree import DecisionTreeClassifier
     tree = DecisionTreeClassifier(criterion='entropy',random_state=42)
     tree.fit(x_train,y_train)
     
     #Print the model accuracy on the training data
     print('[1]Logistic Regression Training Accuracy:',log.score(x_train,y_train))
     print('[2]Decision Tree Training Accuracy:',tree.score(x_train,y_train))
     
    
     return log, tree
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

