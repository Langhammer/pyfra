# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# Notebook 3
# ==============
# Modelling

import pandas as pd
import numpy as np
#uploaded = files.upload()
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn import svm
from sklearn import preprocessing

# Import Dataset
df = pd.read_csv('../data/df.csv')

data = df.drop(columns='grav',axis=1).select_dtypes(include=np.number).dropna(axis=1)
target = df.grav
X_train  , X_test, y_train ,y_test  = train_test_split(data, target, test_size = 0.2 ,random_state =23)


std_scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = std_scaler.transform(X_train)
X_test_scaled = std_scaler.transform(X_test)

kbest_selector = SelectKBest(k=6)
kbest_selector.fit(X_train_scaled,y_train);
X_train_scaled_selection = kbest_selector.transform(X_train_scaled)
X_test_scaled_selection = kbest_selector.transform(X_test_scaled)

var_selector.n_features_in_

svc = svm.SVC(tol=1e-2, cache_size=4000)
svc.fit(X_train_scaled_selection, y_train)

# # Model Random Forest

# +
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.ensemble import RandomForestClassifier

RFCLF = GridSearchCV(RandomForestClassifier(),param_grid = {
    'criterion': ['gini','entropy','log_loss'],
    'max_depth': [3,4,10,15,20],
    'min_samples_leaf':[1,3,5,10,15]
}, cv = RepeatedKFold())

RFCLF.fit(X_train_scaled,y_train))

print(RFCLF.best_params_)
print(RFCLF.best_score_)

# +
RFCLFbest = GridSearchCV(DecisionTreeClassifier(),param_grid = {
    'criterion': [],
    'max_depth': [],
    'min_samples_leaf':[]
    }, cv = RepeatedKFold())



RFCLFbest.fit(X_train_scaled,y_train))
y_pred = RFCLFbest.predict(X_test_scaled)
cm= pd.crosstab(y_test,y_pred, rownames=['Real'], colnames=['Prediction'])
print(cm)
# -

print('DT Score is:',RFCLFbest.score(X_test_scaled,y_test))

# # Decision Tree

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = "entropy", random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred) 
print(cm)
accuracy = metrics.accuracy_score(y_test, y_pred) 
print("Accuracy score:",accuracy)
precision = metrics.precision_score(y_test, y_pred) 
print("Precision score:",precision)
recall = metrics.recall_score(y_test, y_pred) 
print("Recall score:",recall)
