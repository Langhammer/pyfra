import pandas as pd

def df_testing_info(df):
    df_dtypes_and_na_counts = pd.DataFrame({'dtypes':df.dtypes, 'n_na': df.isna().sum()})
    return pd.concat([df.describe().T, df_dtypes_and_na_counts])
