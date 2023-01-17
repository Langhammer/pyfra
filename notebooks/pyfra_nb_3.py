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
from sklearn import preprocessing
from time import sleep
from tqdm.notebook import tqdm


# + vscode={"languageId": "python"}
# Import Dataset
df = pd.read_csv('../data/df.csv')

# + vscode={"languageId": "python"}
data = df.drop(columns='grav',axis=1).select_dtypes(include=np.number).dropna(axis=1)
target = df.grav
X_train, X_test, y_train, y_test  = train_test_split(data, target, test_size=0.2 ,random_state=23)
# -


# # Scaling the Data and Selecting Features

# + vscode={"languageId": "python"}
std_scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = std_scaler.transform(X_train)
X_test_scaled = std_scaler.transform(X_test)

# + vscode={"languageId": "python"}
kbest_selector = SelectKBest(k=6)
kbest_selector.fit(X_train_scaled,y_train);
X_train_scaled_selection = kbest_selector.transform(X_train_scaled)
X_test_scaled_selection = kbest_selector.transform(X_test_scaled)
# -

# # Applying Machine Learning Models

# ## Support Vector Machine (SVM)

# + vscode={"languageId": "python"}
svc = svm.SVC(tol=1e-2, cache_size=4000)
svc.fit(X_train_scaled_selection, y_train)
# -

# ## Random Forest

# + vscode={"languageId": "python"}
params = {
    'criterion': ['gini','entropy','log_loss'],
    'max_depth': [3,4,10],
    'min_samples_leaf':[1,3,5]
    }

params_n_estimators = [100,200,300]

for i in tqdm(params_n_estimators):   
    RFCLF = GridSearchCV(RandomForestClassifier(n_estimators=i),param_grid = params, cv = RepeatedKFold())
    RFCLF.fit(X_train_scaled,y_train)

print(RFCLF.best_params_)
print(RFCLF.best_score_)

# + vscode={"languageId": "python"}
RFCLFbest = GridSearchCV(DecisionTreeClassifier(),param_grid = {
    'criterion': [],
    'max_depth': [],
    'min_samples_leaf':[]
    }, cv = RepeatedKFold())

RFCLFbest.fit(X_train_scaled,y_train))
y_pred = RFCLFbest.predict(X_test_scaled)
cm= pd.crosstab(y_test,y_pred, rownames=['Real'], colnames=['Prediction'])
print(cm)

# + vscode={"languageId": "python"}
print('DT Score is:',RFCLFbest.score(X_test_scaled,y_test))
