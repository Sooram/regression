#----------------------------------------------------------------------------------
# Run VIF test and leave out high correlated features
#----------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

#------------------------------------------- function
# At each turn, leave out a feature whose vif is the biggest.
# Keep removing features until there's no feature whose vif is over the threshold.
# x: dataframe of features
# thresh: the threshold
def calculate_vif(X, thresh):
    # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
    dropped = True
    while dropped:
        variables = X.columns
        dropped = False
        vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]

        max_vif = max(vif)
        if max_vif > thresh:
            maxloc = vif.index(max_vif)
            print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
            X = X.drop([X.columns.tolist()[maxloc]], axis=1)
            dropped = True
    return X

# def calculate_vif(x, thresh):
#     output = pd.DataFrame()
#     k = x.shape[1]
#     vif = [variance_inflation_factor(x.values, j) for j in range(x.shape[1])]
#     for i in range(1,k):
#         print("Iteration no.")
#         print(i)
#         print(vif)
#         x.columns[np.argmax(vif)]
#         x.columns[np.argsort(vif)[-2]]
#         a = np.argmax(vif)
#         # a = np.argsort(vif)[-2]
#         print("Max VIF is for variable no.:")
#         print(a)
#         if vif[a] <= thresh :
#             break
#         if i == 1 :
#             output = x.drop(x.columns[a], axis = 1)
#             vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
#         elif i > 1 :
#             output = output.drop(output.columns[a],axis = 1)
#             vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
#     return(output)

#------------------------------------------- vif test
data = pd.read_csv("file_path")
list(data)
x_data = data.drop(["tags_to_be_dropped"], axis=1)

train_out = calculate_vif(x_data, 10.0)
list(train_out)
train_out.columns

data_passed_vif = data[train_out.columns]
data_passed_vif.to_csv("file_path")