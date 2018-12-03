#----------------------------------------------------------------------------------
# Make a file of correlation matrix
#----------------------------------------------------------------------------------
import pandas as pd

data = pd.read_csv("file_path")
#data = data_outliers_dropped
list(data)
data = data.drop(['tag'], axis=1)

corr_mat = data.corr()
corr_y = corr_mat['y']
corr_mat.to_csv('file_path')
corr_y.to_csv('file_path')

#------------------------------------------- identify top 10

#------------------------------------------- identify highly correlated features
# Create correlation matrix
corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features
df.drop(df.columns[to_drop], axis=1)

