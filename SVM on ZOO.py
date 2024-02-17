# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 08:00:03 2024

@author: shiji
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
#%%
zoo = pd.read_csv("C:/Users/shiji/OneDrive/python/Zoo Project/zoo.data", header=0)
zoo
zoo.columns = ['animal names',
               'hair',
               'feathers',
               'eggs',
               'milk',
               'airborne',
               'aquatic',
               'predator',
               'toothed',
               'backbone',
               'breathes',
               'venomous',
               'fins',
               'legs',
               'tail',
               'domestic',
               'catsize',
               'type']

zoo = zoo.drop('animal names', axis= 1)
#%%
zoo['type'].unique()
zoo.info()
len(zoo.loc[zoo['type']==1])
len(zoo.loc[zoo['type'] == 2])
len(zoo.loc[zoo['type'] == 3])
len(zoo.loc[zoo['type'] == 4])
len(zoo.loc[zoo['type'] == 5])
len(zoo.loc[zoo['type'] == 6])
len(zoo.loc[zoo['type'] == 7])
# As we can see from the results, the target variable is imbalanced among multiple levels. Next, we are going
# to handle class imbalance to avoid our model becoming biased towards to majority class.

from imblearn.over_sampling import SMOTE
# Seperate dependent variables and independent variable
x = zoo.drop('type', axis = 1).copy()
y = zoo['type'].copy()

x_encoded = pd.get_dummies(x, columns= ['hair',
                                        'feathers',
                                        'eggs',
                                        'milk',
                                        'airborne',
                                        'aquatic',
                                        'predator',
                                        'toothed',
                                        'backbone',
                                        'breathes',
                                        'venomous',
                                        'fins',
                                        'legs',
                                        'tail',
                                        'domestic',
                                        'catsize'])


#%%
# Split data to training data and testing data
x_train, x_test, y_train, y_test = train_test_split(x_encoded,
                                                    y, random_state=23)
# scale data.
x_train_scale = scale(x_train)
x_test_scale = scale(x_test)

y_train.value_counts()
#%%
# we are going to use smote to oversample the data.
smote = SMOTE(k_neighbors=2,random_state=12)
x_oversample, y_oversample = smote.fit_resample(x_train_scale, y_train)
y_oversample.unique()
y_oversample.value_counts()
#%%
# built a model
svc = SVC(random_state=42)
svc.fit(x_oversample, y_oversample)
prediction = svc.predict(x_test_scale)
cm = confusion_matrix(y_test, prediction, labels = svc.classes_)
plt = ConfusionMatrixDisplay(cm)
plt.plot()
accuracy_score(y_test, prediction)
#%%
# optimize parameters with cross validation using Gridsearchcv
from sklearn.model_selection import GridSearchCV

parameters = [{'C': [0.5,1,10,100,1000],
               'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
               'kernel': ['rbf']}]

best_para = GridSearchCV(SVC(), parameters,
                         scoring='accuracy',cv = 10,verbose=0)

best_para.fit(x_train_scale, y_train)
print(best_para.best_params_)
# The best parameters for setting model select using Gridsearchcv are c:10, gamma: scale, kernel:rbf
# Remodel data through setting new parameters and to see the performance of our new model
clf_svc = SVC(C=10, gamma= 'scale', kernel='rbf',random_state=45)
clf_svc.fit(x_oversample, y_oversample)
predict = clf_svc.predict(x_test_scale)
cm2 = confusion_matrix(y_test, predict, labels= clf_svc.classes_)
plt = ConfusionMatrixDisplay(cm2)
plt.plot()

acu = accuracy_score(y_test, predict)
acu

