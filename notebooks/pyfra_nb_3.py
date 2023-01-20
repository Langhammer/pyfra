# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -LanguageId
#     formats: ipynb,py:light
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
# ---

# Notebook 3
# ==============
# Modelling

# # Importing Packages and Data

import pyfra
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score, recall_score, make_scorer
from imblearn.under_sampling import RandomUnderSampler
from time import sleep
from tqdm.notebook import tqdm


df = pd.read_pickle('../data/df.p')
n_rows_complete = len(df)

# Check whether or not the data is up-to-date (file can't be tracked on github because of it's file size)
pd.testing.assert_frame_equal(left=(pd.read_csv('../data/df_check_info.csv', index_col=0)), \
                         right=pyfra.df_testing_info(df),\
                         check_dtype=False, check_exact=False)

rus = RandomUnderSampler(random_state=23)

# Create a sample of the data, because the whole dataset is too big for us to work with
relative_sample_size = 0.01
df = df.sample(frac=relative_sample_size, random_state=23)

data = df.drop(columns='grav',axis=1).select_dtypes(include=np.number).dropna(axis=1)
target = df.grav
data, target = rus.fit_resample(X=data, y=target)



print(f'We are working on {len(target)} data points, which represent {len(target)/n_rows_complete*100:.04f}% of the original data,')

X_train, X_test, y_train, y_test  = train_test_split(data, target, test_size=0.2 ,random_state=23)

# # Scaling the Data and Selecting Features

std_scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = std_scaler.transform(X_train)
X_test_scaled = std_scaler.transform(X_test)

k_features = 25
kbest_selector = SelectKBest(k=k_features)
kbest_selector.fit(X_train_scaled,y_train);
X_train_scaled_selection = kbest_selector.transform(X_train_scaled)
X_test_scaled_selection = kbest_selector.transform(X_test_scaled)
print(f'We use {k_features} of the original {df.shape[1]} features')

# # Application of Machine Learning Models
# ## Setup of Metrics Table

# Creating a matrix to store the results
result_metrics = pd.DataFrame(columns=['model', 'f1', 'accuracy', 'recall'], index=['lr', 'svc', 'rf', 'dt'])
result_metrics


# Creating a function to compute and store the results for the respective model
def store_metrics(model_name, model, y_test, y_pred, result_df):
    result_df.loc[model_name, 'model'] = model 
    result_df.loc[model_name, 'f1'] = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')
    result_df.loc[model_name, 'accuracy'] = accuracy_score(y_true=y_test, y_pred=y_pred)
    result_df.loc[model_name, 'recall'] = recall_score(y_true=y_test, y_pred=y_pred, average='weighted')
    return result_df


# ## Setup of the Cross-Validator
# We will use a repeated stratified cross-validataion to make sure to pick the best parameters.
# The stratification will be used to ensure an equal distribution of the different categories in every bin.
# The repetition will be used in order ensure that the result is not an outlier. We will set a lower the number of repetitions, however, to save execution time (default would be 10 repetitions).

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=23)

# ## Support Vector Machine (SVM)
#
# A support vector machine classifier will be used with parameter optimization via grid search.
#
# ### Setup of the SVM and the Grid Search

# +
# Instantiation of the SVM Classifier
# We set the cache size to 1600 MB (default: 200 MB) to reduce the computing time.
# The other parameters will be set via grid search.
svc = svm.SVC(cache_size=1600)

# Choosing the parameters for the grid search
svc_params = {
    'kernel': ['poly', 'rbf', 'sigmoid'],
    'gamma': [0.1, 0.5, 'scale'],
    'C': [0.1, 0.5, 1, 2]
}

# Setup of the scoring. 
# We have to define the parameter 'average', because we are not dealing with a binary classification.
# Our sample is balanced, hence we can use a simple approach, using 'micro', which uses the global values of 
# true positives, false negatives and false positives.
f1_scoring = make_scorer(score_func=f1_score, average='micro')

# Instantiation of the GridSearchCv
# n_jobs is set to -1 to use all available threads for computation.
svc_grid = GridSearchCV(svc, param_grid=svc_params, scoring=f1_scoring, cv=cv, n_jobs=-1)
# -

# ### SVM Parameter Optimization, Training and Prediction

# +
# Fitting the grid search to find the best parameter combination
svc_grid.fit(X_train_scaled_selection, y_train)

# Print result of parameter optimization
print('Best parameter combination: ',svc_grid.best_params_)

# Predict target variable for the test set
y_svc = svc_grid.best_estimator_.predict(X_test_scaled_selection)

# -

# ### Metrics of SVM

# Calculate the metrics for the optimal svm model and store them in the result_metrics DataFrame 
# The model will be stored as well in the DataFrame
result_metrics = store_metrics(model=svc_grid, model_name='svc',
                               y_test=y_test, y_pred=y_svc,
                               result_df=result_metrics)
# Show the interim result                               
result_metrics

# ## Random Forest

# +
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

# +
RFCLFbest = GridSearchCV(DecisionTreeClassifier(),param_grid = {
    'criterion': [],
    'max_depth': [],
    'min_samples_leaf':[]
    }, cv = RepeatedKFold())

RFCLFbest.fit(X_train_scaled,y_train)
y_pred = RFCLFbest.predict(X_test_scaled)
cm= pd.crosstab(y_test,y_pred, rownames=['Real'], colnames=['Prediction'])
print(cm)
# -

print('DT Score is:',RFCLFbest.score(X_test_scaled,y_test))


# # Application of Advanced Models

#
# # Results and Conclusion
