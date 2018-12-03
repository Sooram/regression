import pandas as pd
import numpy as np
from sklearn import preprocessing


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

data_all = pd.read_csv("file_path")
data_outlier_dropped = data_all.drop(data_all.index[162*144:163*144])
data_outlier_dropped = data_all.drop(data_all.index[0:120*144])
data_outlier_dropped = data_outlier_dropped.reset_index(drop=True)
list(data_outlier_dropped)
data_outlier_dropped = data_outlier_dropped.drop(['Unnamed: 0'], axis=1)

data_scaled = data_outlier_dropped
for i in range(3, data_scaled.shape[1]):
    data_scaled.iloc[:,i] = preprocessing.scale(np.array(data_scaled.iloc[:,i]))

data_smoothed = data_scaled
for i in range(3, data_smoothed.shape[1]):
    data_smoothed.iloc[:,i] = smooth(data_scaled.iloc[:,i],3)

data_smoothed.to_csv('file_path')

#------------------------------------ extract 5:00
indices = [i for i, s in enumerate(list(data_smoothed['tag'])) if ' 5:00' == s[-5:]]
data_5 = data_smoothed.iloc[indices]
data_5 = data_5.reset_index(drop=True)

data_5.to_csv("file_path")