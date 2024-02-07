# -*- coding: utf-8 -*-
"""RedWinePrediction

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wkufh_clEPIiakQDSxaCWWrvE64JJSxV

Import Libraries
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data=pd.read_csv("redwine.csv")
data.head()

data.shape

data.isnull().sum()

"""***Data Analysis & Data Visualization***"""

data.describe()

sns.catplot(x='quality',data=data,kind='count')

#volatile acidity vs Quality
plot=plt.figure(figsize=(4,4))
sns.barplot(x='quality',y='volatile acidity',data=data)

#citric acid vs quality
sns.barplot(x='quality', y='citric acid',data=data)

"""Correlations
1.Positive and Negative Correlations
"""

correlations=data.corr()

sns.heatmap(correlations,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')

"""***Data Preprocessing***"""

X = data.drop(['quality'], axis=1)

Y=data['quality'].apply(lambda value:1 if value>=7 else 0)
print(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)
print(Y.shape,Y_train.shape,Y_test.shape)

"""***Model Training***"""

model=RandomForestClassifier()
model.fit(X_train,Y_train)

"""***Model Evaluation***"""

prediction=model.predict(X_test)
test_accuracy=accuracy_score(prediction,Y_test)

print("Accuracy is: ",test_accuracy)

"""***Building a Predictive System***"""

input_data=(7.5,0.5,0.36,6.1,0.071,17.0,102.0,0.9978,3.35,0.8,10.5)
input_data_num=np.asarray(input_data)
input_data_reshape=input_data_num.reshape(1,-1)
predict_result=model.predict(input_data_reshape)
print(predict_result)
if (predict_result[0]==1):
  print('Good Quality Wine')
else:
    print('Bad Quality Wine')