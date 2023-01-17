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

# + [markdown] id="cb344b33"
# Notebook 3
# ==============
# Modelling

# + executionInfo={"elapsed": 1455, "status": "ok", "timestamp": 1673953084040, "user": {"displayName": "Saleh Saleh", "userId": "09280102644015922055"}, "user_tz": -60} id="a94ecf64"
import pandas as pd
import numpy as np
#uploaded = files.upload()
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn import svm
from sklearn import preprocessing
from tqdm.notebook import tqdm
from time import sleep

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 185, "status": "ok", "timestamp": 1673953084222, "user": {"displayName": "Saleh Saleh", "userId": "09280102644015922055"}, "user_tz": -60} id="PgLYOfyM2eRj" outputId="689e17ae-1b32-4543-a8b3-367315aa52c8"
# gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 215, "status": "ok", "timestamp": 1673953088342, "user": {"displayName": "Saleh Saleh", "userId": "09280102644015922055"}, "user_tz": -60} id="kaDud5922tCL" outputId="e2e0268a-e9a1-44c4-caaa-170f763e4ffc"
from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('Not using a high-RAM runtime')
else:
  print('You are using a high-RAM runtime!')

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 15800, "status": "ok", "timestamp": 1673953107669, "user": {"displayName": "Saleh Saleh", "userId": "09280102644015922055"}, "user_tz": -60} id="3QwS2hilE23t" outputId="05c8dacb-63a2-4bdd-b1dc-4fd59fbd4a4c"
from google.colab import drive
drive.mount('/content/drive')

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 23256, "status": "ok", "timestamp": 1673953130922, "user": {"displayName": "Saleh Saleh", "userId": "09280102644015922055"}, "user_tz": -60} id="c3300e42" outputId="e692a950-8361-4e86-d4e9-ce8f13ba5d1d"
# Import Dataset
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Google_Collab_DS_Project/data/df.csv')

# + id="UnZ7W1pvJ2_D"
df_temp = df 

# + executionInfo={"elapsed": 1071, "status": "ok", "timestamp": 1673953146207, "user": {"displayName": "Saleh Saleh", "userId": "09280102644015922055"}, "user_tz": -60} id="hgNi-yoyJQkD"
# FOR PROBLEM WITH VALUE COUNTS RUN,REMOVE TAG AND RUN BELOW CELLS ONLY ONCE
#df.grav.replace(to_replace=[-1,1,3,4],value=0,inplace=True)
#df.grav.replace(to_replace=2,value=1,inplace=True)
df.grav.value_counts()

DF_KNN = df
#DF_KNN.drop(columns=['id_vehicule','motor','num_veh'] ,axis=1,inplace=True)
DF_KNN = DF_KNN.select_dtypes(include=np.number)
temp = DF_KNN.dropna(axis=1)


# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 183, "status": "ok", "timestamp": 1673365371928, "user": {"displayName": "Saleh Saleh", "userId": "09280102644015922055"}, "user_tz": -60} id="rnqLNJeRKECr" outputId="ea715652-473f-4de5-c192-2afb5e12e795"

temp.columns

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 176, "status": "ok", "timestamp": 1673953156885, "user": {"displayName": "Saleh Saleh", "userId": "09280102644015922055"}, "user_tz": -60} id="lZkgrXYnKYcb" outputId="ff87fac3-da87-474b-adf1-bff927065678"
temp.grav.value_counts()

# + [markdown] id="eHQhZeIMft41"
# # New Section

# + executionInfo={"elapsed": 748, "status": "ok", "timestamp": 1673954748719, "user": {"displayName": "Saleh Saleh", "userId": "09280102644015922055"}, "user_tz": -60} id="1a52eae0"
data = temp.drop(columns='grav',axis=1).select_dtypes(include=np.number).dropna(axis=1)
target = temp.grav



# + executionInfo={"elapsed": 175, "status": "ok", "timestamp": 1673954750409, "user": {"displayName": "Saleh Saleh", "userId": "09280102644015922055"}, "user_tz": -60} id="X67w28rlhTYU"
data = data[:500000]
target = target[0:500000]
#100,000 32 seconds
#500,000 16 minutes

# + executionInfo={"elapsed": 179, "status": "ok", "timestamp": 1673954752572, "user": {"displayName": "Saleh Saleh", "userId": "09280102644015922055"}, "user_tz": -60} id="_YbEX6MpGlOF"
X_train  , X_test, y_train ,y_test  = train_test_split(data, target, test_size = 0.2 ,random_state =23)

# + executionInfo={"elapsed": 413, "status": "ok", "timestamp": 1673954632366, "user": {"displayName": "Saleh Saleh", "userId": "09280102644015922055"}, "user_tz": -60} id="23b2b73f"
std_scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = std_scaler.transform(X_train)
X_test_scaled = std_scaler.transform(X_test)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 694337, "status": "ok", "timestamp": 1673955452392, "user": {"displayName": "Saleh Saleh", "userId": "09280102644015922055"}, "user_tz": -60} id="scRI3fqxEnR0" outputId="1ea9f24e-3369-47d6-9305-88c3b897e65d"
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=3,n_jobs=-1)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

# + id="h8pKXZSlEoOE"


# + id="60f12736"
kbest_selector = SelectKBest(k=6)
kbest_selector.fit(X_train_scaled,y_train);
X_train_scaled_selection = kbest_selector.transform(X_train_scaled)
X_test_scaled_selection = kbest_selector.transform(X_test_scaled)

# + id="fc6ee8a1"
#svc = svm.SVC(tol=1e-2, cache_size=4000)
#svc.fit(X_train_scaled_selection, y_train)
