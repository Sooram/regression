#----------------------------------------------------------------------------------
# Handle NAs
#----------------------------------------------------------------------------------
import pandas as pd
import numpy as np

data_type_unified = pd.read_csv("file_path")

#------------------------------------------- function
def print_number_of_NAs(data):
    col_having_NA = data.columns[np.where(data.isnull().sum() > 0)]
    print('columns having NAs: ' + str(col_having_NA))
    print('number of NAs per column:\n' + str(data[col_having_NA].isnull().sum()))
    print('total number of NAs: ' + str(data.isnull().sum().sum()))

#------------------------------------------- examine NAs in the data
print_number_of_NAs(data_type_unified)

indices = [i for i, s in enumerate(list(data_type_unified['tag'])) if ' 5:00' == s[-5:]]
data_one = data_type_unified.iloc[indices]
data_one = data_one.reset_index(drop=True)
print_number_of_NAs(data_one)

data_one.to_csv("file_path")

#-------------------------------------------  1. drop all the rows with NAs
data_without_na = data_one.dropna()
print_number_of_NAs(data_without_na)

#-------------------------------------------  2. fill NAs with other values
# 2-1. fill with the previous value
data_type_unified = data_type_unified.fillna(method='ffill')

# 2-2. interpolate
data_without_na = data_type_unified.interpolate()
print_number_of_NAs(data_without_na)

data_without_na.to_csv("file_path")