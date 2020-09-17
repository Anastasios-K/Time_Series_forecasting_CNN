from sklearn import preprocessing
import numpy as np


def scaler(pd_frame):  # re-scale data from 0 to 1
    method = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled_array = method.fit_transform(pd_frame)
    return scaled_array


def sliding_data(array_data, window_length):  # converting data and labels to Sliding Window form
    sliding_data = np.array(list(map(lambda x: array_data[0:5, x:x + window_length],
                                     range(len(array_data.T) - window_length)
                                     )))
    labels = np.array(list(map(lambda x: x[0:1, -1:],
                               sliding_data
                               )))
    return sliding_data[:-1], labels[1:]


print("Data_Preprocessing imported")