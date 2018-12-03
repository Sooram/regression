import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from math import sqrt
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

#------------------------------------ load data
data = pd.read_csv("file_path")
list(data)

x_data = data.drop(['tags_to_be_dropped'], axis=1)
y = data['y']

feature_selected = ['features']

x_data = data[feature_selected]
#------------------------------------ functions
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def sliding_predict_raw(model, windiow_size):
    actual_pred = pd.DataFrame(columns=['actual', 'pred'])
    reg = model

    for i in range(window_size, len(y)):
        # training data
        x_train = x_data.iloc[i - window_size:i]
        y_train = y.iloc[i - window_size:i]

        # test data
        x_test = pd.DataFrame(columns=list(x_data))
        x_test = x_test.append(x_data.iloc[i])
        y_test = y.iloc[i]  # true value of y at this point

        reg.fit(x_train, y_train)
        y_pred = reg.predict(x_test)
        y_pred = y_pred[0]  # predicted value of y at this point
        # y_pred = y_pred[0][0]  # pls

        # add current actual value and predicted value as a row
        curr_df = pd.DataFrame([[y_test, y_pred]], columns=['actual', 'pred'])
        actual_pred = actual_pred.append(curr_df)

        print(i)

    return (actual_pred)

#------------------------------------------- scale/smooth data in current window
def sliding_predict_raw(model, windiow_size):
    actual_pred = pd.DataFrame(columns=['actual', 'pred'])
    reg = model

    for i in range(window_size, len(y)):
        # x
        x = x_data.iloc[i - window_size:i+1].copy()
        for col in x.columns:
            x.loc[:,col] = smooth(preprocessing.scale(np.array(x.loc[:,col])), 3)
        x_train = x.iloc[:window_size]
        x_test = x.iloc[[window_size]]

        # y
        y_train = y.iloc[i - window_size:i]
        y_test = y.iloc[i]  # true value of y at this point

        reg.fit(x_train, y_train)
        y_pred = reg.predict(x_test)
        y_pred = y_pred[0]  # predicted value of y at this point

        # add current actual value and predicted value as a row
        curr_df = pd.DataFrame([[y_test, y_pred]], columns=['actual', 'pred'])
        actual_pred = actual_pred.append(curr_df)

        print(i)

    return (actual_pred)

window_size = 120

#------------------------------------------- sequential split
x_train = x_data.iloc[0:274]
x_test = x_data.iloc[274:]
y_train = y.iloc[0:274]
y_test = y.iloc[274:]
#------------------------------------------- random split
x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=0.3, random_state=10)

ridge = Ridge(alpha=1.0)
svr = SVR(kernel='linear')
reg = svr
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

print(sqrt(mean_squared_error(y_test, y_pred)))
print(r2_score(y_test, y_pred))
print(mean_absolute_percentage_error(y_test, y_pred))

#------------------------------------------- linear regerssion
linear = LinearRegression()
actual_pred_linear = sliding_predict_raw(linear, window_size)

actual = actual_pred_linear['actual']
pred = actual_pred_linear['pred']

print(sqrt(mean_squared_error(actual, pred))) # RMSE
print(r2_score(actual, pred)) # R2
print(mean_absolute_percentage_error(actual, pred))   # MAPE

actual_pred_linear.to_csv('file_path')

#------------------------------------------- randomforest
rf = RandomForestRegressor(max_depth=2, random_state=0)
actual_pred_rf = sliding_predict_raw(rf, window_size)

actual = actual_pred_rf['actual']
pred = actual_pred_rf['pred']

print(sqrt(mean_squared_error(actual, pred))) # RMSE
print(r2_score(actual, pred)) # R2
print(mean_absolute_percentage_error(actual, pred))   # MAPE

actual_pred_rf.to_csv('file_path')

#------------------------------------------- lasso
actual_pred_lasso = pd.DataFrame(columns=['actual','pred'])

for i in range(window_size,len(y)):
    # training data
    x_train = x_data.iloc[i-window_size:i]
    y_train = y.iloc[i-window_size:i]
    # y_train = y_splined.iloc[i-window_size:i]

    # test data
    x_test = pd.DataFrame(columns=list(x_data))
    x_test = x_test.append(x_data.iloc[i])
    y_test = y.iloc[i]  # true value of y at this point

    lasso = Lasso(max_iter=10000, normalize=False)
    lassocv = LassoCV(alphas=None, cv=10, max_iter=100000, normalize=False)

    lassocv.fit(x_train, y_train)
    lasso.set_params(alpha=lassocv.alpha_)

    lasso.fit(x_train, y_train)

    y_pred = lasso.predict(x_test)
    y_pred = y_pred[0]  # predicted value of y at this point

    # add current actual value and predicted value as a row
    curr_df = pd.DataFrame([[y_test,y_pred]], columns=['actual','pred'])
    actual_pred_lasso = actual_pred_lasso.append(curr_df)

    print(i)

actual = actual_pred_lasso['actual']
pred = actual_pred_lasso['pred']

print(sqrt(mean_squared_error(actual, pred))) # RMSE
print(r2_score(actual, pred)) # R2
print(mean_absolute_percentage_error(actual, pred))   # MAPE

actual_pred_lasso.to_csv('file_path')

#------------------------------------------- ridge
ridge = Ridge(alpha=1.0)
actual_pred_ridge = sliding_predict_raw(ridge, window_size)
actual = actual_pred_ridge['actual']
pred = actual_pred_ridge['pred']

print(sqrt(mean_squared_error(actual, pred))) # RMSE
print(r2_score(actual, pred)) # R2
print(mean_absolute_percentage_error(actual, pred))   # MAPE

actual_pred_ridge.to_csv('file_path')

#------------------------------------------- SVR

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr = SVR(gamma=1e-8)
actual_pred_svr = sliding_predict_raw(svr_rbf, window_size)

actual = actual_pred_svr['actual']
pred = actual_pred_svr['pred']

print(sqrt(mean_squared_error(actual, pred))) # RMSE
print(r2_score(actual, pred)) # R2
print(mean_absolute_percentage_error(actual, pred))   # MAPE

actual_pred_svr.to_csv('file_path')

#------------------------------------------- PLS

pls = PLSRegression(n_components=7, scale=False, max_iter=5000)
# actual_pred_pls = sliding_predict_splined(pls, window_size)

actual_pred_pls = pd.DataFrame(columns=['actual','pred'])
for i in range(window_size,len(y)):
    # training data
    x_train = x_data.iloc[i-window_size:i]
    y_train = y.iloc[i-window_size:i]
    # y_train = y_splined.iloc[i-window_size:i]

    # test data
    x_test = pd.DataFrame(columns=list(x_data))
    x_test = x_test.append(x_data.iloc[i])
    y_test = y.iloc[i]  # true value of y at this point

    pls.fit(x_train, y_train)
    y_pred = pls.predict(x_test)
    y_pred = y_pred[0][0]  # predicted value of y at this point

    # add current actual value and predicted value as a row
    curr_df = pd.DataFrame([[y_test,y_pred]], columns=['actual','pred'])
    actual_pred_pls = actual_pred_pls.append(curr_df)

    print(i)

actual = actual_pred_pls['actual']
pred = actual_pred_pls['pred']

print(sqrt(mean_squared_error(actual, pred))) # RMSE
print(r2_score(actual, pred)) # R2
print(mean_absolute_percentage_error(actual, pred))   # MAPE

actual_pred_pls.to_csv('file_path')