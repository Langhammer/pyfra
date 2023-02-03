# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -LanguageId
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
# ---

# %% [markdown]
# Notebook 5
# ==============
# Further Analysis

# %% [markdown]
# # Outline
# The aim is to further investigate the models developed in the third notebook.
# We will
# 1. Identify the relationship between amount of training data and model performance
# 2. Compare the performance of our model with a naive approach of training on the un-stratified, imbalanced dataset

# %% [markdown]
# # Import Modules, Data and Model 

# %%
# Import the modules
import pyfra
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from joblib import load
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler


# %%
df = pd.read_pickle('../data/df.p')
n_rows_complete = len(df)

# %%
# Check whether or not the data is up-to-date (file can't be tracked on github because of it's file size)
pd.testing.assert_frame_equal(left=(pd.read_csv('../data/df_check_info.csv', index_col=0)), \
                         right=pyfra.df_testing_info(df),\
                         check_dtype=False, check_exact=False)

# %%
rus = RandomUnderSampler(random_state=23)

# %%
data = df.drop(columns='Severity',axis=1).select_dtypes(include=np.number).dropna(axis=1)
target = df['Severity']
data, target = rus.fit_resample(X=data, y=target)

# %%
target.value_counts()

# %%
print(f'We are working on {len(target)} data points, which represent {len(target)/n_rows_complete*100:.04f}% of the original data,')

# %%
data = df.drop(columns='Severity',axis=1).select_dtypes(include=np.number).dropna(axis=1)
target = df['Severity']

# %% [markdown]
# # Relation between Amount of Training Data and Model Performance

# %%
preprocessing_pipeline = load('../models/preprocessing_pipeline.joblib')
svc = load('../models/svc.joblib')
stacking_clf = load('../models/stacking_clf.joblib')

# %%
svc.verbose= 100
stacking_clf.verbose = 100

# %%
# Creating a matrix to store the results
result_metrics = pd.DataFrame(columns=['model', 'n_rows','f1', 'accuracy', 'recall'])
result_metrics.index.name = 'id'
result_metrics
result_metrics.shape

# %%
# Create a sample of the data, because the whole dataset is too big for us to work with
#df = df.sample(n=n_rows, random_state=23)
from sklearn.utils import random

# %%
# Creating a function to compute and store the results for the respective model
from sklearn.utils import random
from sklearn.metrics import f1_score, accuracy_score, recall_score
def store_metrics(model_label, model, n_rows, result_df):
    id = result_df.shape[0]
    result_df.loc[id, 'model_label'] = model_label
    result_df.loc[id, 'model'] = model
    result_df.loc[id, 'n_rows'] = n_rows
    print(f'Splitting {n_rows} rows of data...')
    sample_indices = random.sample_without_replacement(n_population=len(target), 
                                                       n_samples=n_rows)
    data_sample = data.iloc[sample_indices]
    target_sample = target.iloc[sample_indices]
    X_train, X_test, y_train, y_test = train_test_split(data_sample, 
                                                        target_sample, 
                                                        test_size=0.2, 
                                                        random_state=23, 
                                                        stratify=target_sample)
    print(f'Preprocessing Data...')
    X_train = preprocessing_pipeline.fit_transform(X_train, y_train)
    X_test = preprocessing_pipeline.transform(X_test)
    print(f'Fitting {model_label}...')
    model.fit(X_train, y_train)
    print(f'Predicting...')
    y_pred = model.predict(X_test)
    print(f'Computing scores...')
    result_df.loc[id, 'f1'] = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')
    result_df.loc[id, 'accuracy'] = accuracy_score(y_true=y_test, y_pred=y_pred)
    result_df.loc[id, 'recall'] = recall_score(y_true=y_test, y_pred=y_pred, average='weighted')
    return result_df


# %%

for n_rows in [500, 1_000, 2_000, 5_000, 10_000, 20_000]:
    result_metrics = store_metrics('stacking', stacking_clf, n_rows, result_metrics)

# %%
result_metrics

# %%
plt.plot(result_metrics['n_rows'],result_metrics['f1'],'x-', label='$f_1$');
#plt.plot(result_metrics['n_rows'],result_metrics['accuracy'], label='Accuracy');
plt.plot(result_metrics['n_rows'],result_metrics['recall'],'o-', label='Recall');
plt.title('Performance in Relation to the Amount of Input Data')
plt.xlabel('Number of Data Points')
plt.ylabel('Performance')
plt.ylim((0,0.5))
plt.legend();
