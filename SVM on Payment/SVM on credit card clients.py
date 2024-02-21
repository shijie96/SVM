# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 08:30:24 2024

@author: shijie
"""

import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

#%%
# load data and check the structure and information of the data
data = pd.read_csv("C:/Users/shiji/OneDrive/python/DATASET/default of credit card clients.csv", header=1)
data.head()
data.info()
data.describe()
# To rename a column
data.rename({"default payment next month" : 'default'}, axis = 1, inplace = True)
data.info()
# To delete ID column
data.drop('ID', axis = 1, inplace = True)
data.info()
#%%
# check the levels of variables
data['SEX'].unique()
data['EDUCATION'].unique()
data['MARRIAGE'].unique()
len(data.loc[(data['EDUCATION'] == 0)| (data['MARRIAGE'] == 0)])
# filter the missing data off 
data_no_missing = data.loc[(data['EDUCATION'] !=0)&(data['MARRIAGE']!=0)]
data_no_missing
data_no_missing['EDUCATION'].unique()
len(data_no_missing)
data_no_missing.shape
#%%
# downsample the data
# seperate data based on independent variable
data_no_defalut = data_no_missing[data_no_missing['default']==0]
data_default = data_no_missing[data_no_missing['default'] == 1]
len(data_default)
len(data_no_defalut)
# dowmsample the data 
data_default_downsample = resample(data_default, replace = False, n_samples = 3000, random_state= 12)
data_no_defalut_downsample = resample(data_no_defalut, replace= False, n_samples= 3000, random_state= 12)
# concatenate two datasets to a conbined dataset having 6000 rows data
data_combine = pd.concat([data_default_downsample, data_no_defalut_downsample])
len(data_combine)

#%%
# we are going to split data into two parts. (dependent and independent)
# Then split the datasets into training data and testing 
x = data_combine.drop('default', axis = 1).copy()
y = data_combine['default'].copy()

#%%
# use one-hot coding to deal with categorical variables
pd.get_dummies(x, columns= ['MARRIAGE'])

X_code = pd.get_dummies(x, columns = ['SEX',
                                       'EDUCATION',
                                       'MARRIAGE',
                                       'PAY_0',
                                       'PAY_2',
                                       'PAY_3',
                                       'PAY_4',
                                       'PAY_5',
                                       'PAY_6'])
X_code
#%%
# split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(X_code, y, random_state= 23)
x_train_scale = scale(x_train)
x_test_scale = scale(x_test)
#%%
# We are going  to build a preliminary support vector machine
clf_svm = SVC(random_state=42)
clf_svm.fit(x_train_scale, y_train)
prediction = clf_svm.predict(x_test_scale)
confusion_matrix = confusion_matrix(y_test,prediction,
                                    labels=clf_svm.classes_)
clf_svm.classes_

accuracy = accuracy_score(y_test, prediction)
accuracy


disp = ConfusionMatrixDisplay(confusion_matrix, display_labels=['Did not default','default'])
disp.plot()
#%%
# Optimize parameters with cross validation and Gridseachcv()
parameters = [{'C': [0.5,1,10,100],
               'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
               'kernel': ['rbf']}]

optimal_parameters = GridSearchCV(SVC(), parameters,cv=5,scoring='accuracy', verbose= 2)
print(optimal_parameters)
optimal_parameters.fit(x_train_scale, y_train)
print(optimal_parameters.best_params_)
print(optimal_parameters.best_score_)
print(optimal_parameters.best_estimator_)


# the optimal parameters are c: 100 and gamma : 0.001
# we are going to set the parameters in our model to retrain data.

best_svm = SVC(random_state= 34, C=100, gamma = 0.001, kernel= 'rbf')
best_svm.fit(x_train_scale, y_train)
preditcions = best_svm.predict(x_test_scale)
cm = confusion_matrix(y_test, preditcions,
                      labels = best_svm.classes_)
disp2 = ConfusionMatrixDisplay(cm, display_labels=['Did not default', 'default'])
disp2.plot()

acu = accuracy_score(y_test, preditcions)
acu

tn = cm[0,0]
fp = cm[0,1]
fn = cm[1,0]
tp = cm[1,1]
sensitivity = tp / (tp+fn)
sensitivity
specificity = tn /(tn + fp)
specificity
