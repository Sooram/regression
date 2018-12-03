#----------------------------------------------------------------------------------
# Drop rows with outliers
#----------------------------------------------------------------------------------
import pandas as pd

data = pd.read_csv("file_path")
data.shape
list(data)

data_outliers_dropped = data.drop(data.index[162])
data_outliers_dropped = data_outliers_dropped.drop(data.index[113])
data_outliers_dropped = data_outliers_dropped.reset_index(drop=True)
data.shape
data_outliers_dropped.shape
data_outliers_dropped

data_outliers_dropped.to_csv("file_path")