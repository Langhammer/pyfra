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
