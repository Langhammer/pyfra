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

# # Importing Packages and Data

# + vscode={"languageId": "python"}
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import f1_score
from time import sleep
from tqdm.notebook import tqdm


# + vscode={"languageId": "python"}
# Import Dataset
df = pd.read_csv('../data/df.csv')

# + vscode={"languageId": "python"}
# Create a sample of the data, because the whole dataset is too big for us to work with
relative_sample_size = 0.001
df_sample = df.sample(frac=relative_sample_size, random_state=23)
print(f'We are working on {len(df_sample)} data points, which represent {relative_sample_size*100}% of the original data,')

# + vscode={"languageId": "python"}
data = df_sample.drop(columns='grav',axis=1).select_dtypes(include=np.number).dropna(axis=1)
target = df_sample.grav
X_train, X_test, y_train, y_test  = train_test_split(data, target, test_size=0.2 ,random_state=23)
# -


# # Scaling the Data and Selecting Features

# + vscode={"languageId": "python"}
std_scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = std_scaler.transform(X_train)
X_test_scaled = std_scaler.transform(X_test)

# + vscode={"languageId": "python"}
k_features = 25
kbest_selector = SelectKBest(k=k_features)
kbest_selector.fit(X_train_scaled,y_train);
X_train_scaled_selection = kbest_selector.transform(X_train_scaled)
X_test_scaled_selection = kbest_selector.transform(X_test_scaled)
print(f'We use {k_features} of the original {df.shape[1]} features')
# -

# # Applying Machine Learning Models

# ## Support Vector Machine (SVM)

# + vscode={"languageId": "python"}
svc = svm.SVC(tol=1e-2, cache_size=4000)
svc.fit(X_train_scaled_selection, y_train)

# + vscode={"languageId": "python"}
y_svc = svc.predict(X_test_scaled_selection)
svc_score = f1_score(y_true=y_test, y_pred=y_svc, average='macro')
print(f'f1 score for the svm classifier: {svc_score:.04f}')
# -

# ## Random Forest

# + vscode={"languageId": "python"}
params = {
    'criterion': ['gini'],
    'max_depth': [3,10],
    'min_samples_leaf':[1,3,5],
    'n_estimators': [100,200,300]
    }

RFCLF = GridSearchCV(RandomForestClassifier(),param_grid = params, cv = RepeatedKFold(n_splits=4, n_repeats=1, random_state=23))
RFCLF.fit(X_train_scaled,y_train)

print(RFCLF.best_params_)
print(RFCLF.best_score_)

# + vscode={"languageId": "python"}
RFCLFbest = GridSearchCV(DecisionTreeClassifier(),param_grid = {
    'criterion': [],
    'max_depth': [],
    'min_samples_leaf':[]
    }, cv = RepeatedKFold())

RFCLFbest.fit(X_train_scaled,y_train)
y_pred = RFCLFbest.predict(X_test_scaled)
cm= pd.crosstab(y_test,y_pred, rownames=['Real'], colnames=['Prediction'])
print(cm)

# + vscode={"languageId": "python"}
print('DT Score is:',RFCLFbest.score(X_test_scaled,y_test))
