import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

#------------------------------------------- functions
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def make_plot(actual, pred):
    plt.figure(figsize=(20, 5))
    plt.plot(actual)
    plt.plot(pred)
    plt.grid()
    plt.savefig("file_path")
    plt.close()

def sliding_predict_raw(model, window_size, x_data):
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

#------------------------------------------- load data
data_all = pd.read_csv("file_path")

ranked_features=["features"]

x_data = data_all.drop(['y','tag'], axis=1)
y = data_all['y']

y = pd.read_csv("./data/df_y_raw.csv")
y = y['y']
y = y.drop(y.index[162])
y = y.drop(y.index[0:120])
y = y.reset_index(drop=True)

#------------------------------------------- modeling upon feature rank
window_size = 120

def modeling_by_ranking_sliding(model):
    accuracy = pd.DataFrame(columns=['r2', 'rmse', 'mape'])
    for i, x in enumerate(ranked_features):
        print(i)
        x_data_curr = x_data[ranked_features[:i+1]]
        actual_pred = sliding_predict_raw(model, window_size, x_data_curr)

        actual = actual_pred['actual']
        pred = actual_pred['pred']

        rmse = sqrt(mean_squared_error(actual, pred))
        r2 = r2_score(actual, pred)
        mape = mean_absolute_percentage_error(actual, pred)
        curr_accuracy = pd.DataFrame([[r2, rmse, mape]], columns=['r2', 'rmse', 'mape'])
        accuracy = accuracy.append(curr_accuracy)

        actual = actual.reset_index(drop=True)
        pred = pred.reset_index(drop=True)

        plt.figure(figsize=(20, 5))
        plt.plot(actual)
        plt.plot(pred)
        plt.grid()
        plt.savefig("file_path" + str(i+1) + ".png")
        plt.close()

    return(accuracy)


def modeling_by_ranking_random(reg):
    accuracy = pd.DataFrame(columns=['r2', 'rmse', 'mape'])
    for i, x in enumerate(ranked_features):
        print(i)
        x_data_curr = x_data[ranked_features[:i + 1]]
        x_train, x_test, y_train, y_test = train_test_split(x_data_curr, y, test_size=0.3, random_state=100)

        reg.fit(x_train, y_train)
        y_pred = reg.predict(x_test)

        rmse = sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        curr_accuracy = pd.DataFrame([[r2, rmse, mape]], columns=['r2', 'rmse', 'mape'])
        accuracy = accuracy.append(curr_accuracy)

        plt.figure(figsize=(10, 10))
        plt.plot(np.array(y_test), y_pred, 'o')
        plt.grid()
        plt.savefig("file_path" + str(i + 1) + ".png")
        plt.close()

    return (accuracy)

#-------------------------------------------
ridge = Ridge(alpha=0.5)
accuracy_ridge = modeling_by_ranking_sliding(ridge)
accuracy_ridge = modeling_by_ranking_random(ridge)
accuracy_ridge = accuracy_ridge.reset_index(drop=True)
accuracy_ridge.to_csv("file_path")

linear = LinearRegression()
accuracy_linear = modeling_by_ranking_sliding(linear)
accuracy_linear.to_csv("file_path")


