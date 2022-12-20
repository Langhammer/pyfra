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

# # Cleaning and Feature Engineering
# 2<sup>nd</sup> Notebook

# # Import Modules and Data

import pandas as pd

df = pd.read_csv('../Data/df.csv', low_memory=False)

df.shape
