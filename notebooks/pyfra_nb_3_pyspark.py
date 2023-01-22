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

import pandas as pd
import numpy as np
#uploaded = files.upload()
from matplotlib import pyplot as plt
# %matplotlib inline

# Import Dataset
df = pd.read_csv('../data/df.csv')

df = df.select_dtypes(include=np.number).dropna(axis=1)


# # SVM

# ## With PySpark

# +
# Importing SparkSession and SParkContext
from pyspark.sql import SparkSession
from pyspark import SparkContext

# Creating a SparkContext
sc = SparkContext.getOrCreate()

# Creating a Spark session
spark = SparkSession \
    .builder \
    .appName("SVM") \
    .getOrCreate()
        
spark

# -

# Create PySpark DataFrame from pandas DataFrame 
data = data.reset_index(drop=True)
sdf = spark.createDataFrame(data)
sdf_sample = sdf.sample(False, 0.01, seed=23)
train_sample, test_sample = sdf_sample.randomSplit([.7,.3], seed=23)

from pyspark.ml.classification import LinearSVC
svm = LinearSVC(featuresCol='features', labelCol='label',maxIter=5,predictionCol='svm_prediction')
svm_model = svm.fit(train_sample)
result = svm_model.transform(test)
