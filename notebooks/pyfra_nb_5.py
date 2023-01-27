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
# # Import Modules and Data

# %%
import pyfra
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# %% [markdown]
# # Relation between Amount of Training Data and Model Performance

# %%
