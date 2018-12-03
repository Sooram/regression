#----------------------------------------------------------------------------------
# Bind several sets into one set
# https://pandas.pydata.org/pandas-docs/stable/merging.html
# 1. row bind: pd.concat([df1, df2]) / df1.append(df2)
# 2. col bind: pd.concat([df1, df2], axis=1)/ df1.join(df2)
#----------------------------------------------------------------------------------
import pandas as pd
import numpy as np

data_1 = pd.read_csv("./data_origin/1.csv")
data_2_1 = pd.read_csv("./data_origin/2-1.csv")
data_2_2 = pd.read_csv("./data_origin/2-2.csv")
data_3_1 = pd.read_csv("./data_origin/3.csv")
data_3_2 = pd.read_csv("./data_origin/3-2.csv")

data_1.shape

data_2 = pd.concat([data_2_1, data_2_2])
data_2.shape
data_2_1.shape
data_2_2.shape
data_2_1.columns
data_2.to_csv("./data_origin/2.csv")

data_3 = pd.concat([data_3_1, data_3_2])
data_3.shape
data_3_1.shape
data_3_2.shape
data_3.to_csv("./data_origin/3.csv")

#----------------------------------------------------------------------------------
# Unify data types
#----------------------------------------------------------------------------------
data = pd.read_csv("file_path")
list(data)
data.shape

data_type_unified = data
data_type_unified = data_type_unified.drop('tag', axis=1)

obj_col = np.where(data_type_unified.dtypes==object)
obj_col = data_type_unified.columns[obj_col]    # the names of columns where have object type data

# replace object type cells with NA
for col in range(0,len(obj_col)):
    data_type_unified[obj_col[col]] =  pd.to_numeric(data_type_unified[obj_col[col]], errors='coerce')

np.where(data_type_unified.dtypes!=float)

data_type_unified = data_type_unified.join(data['tag'])
data_type_unified.to_csv("file_path")
