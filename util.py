import pandas as pd
from sklearn import preprocessing
import numpy as np

history_points = 50


def csv_to_dataset(path):
    data = pd.read_csv(path)
    data = data.drop('date', axis=1)
    data = data.drop(0, axis=0)
    data = data.values
    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(data)

    data_histories_normalised = np.array([data_normalised[i:i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_normalised = np.array([data_normalised[:, 0][i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_normalised = np.expand_dims(next_day_normalised, -1)

    next_day = np.array([data[:, 0][i + history_points].copy() for i in range(len(data) - history_points)])
    next_day = np.expand_dims(next_day, -1)

    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit(next_day)

    technical_indicators = []
    for his in data_histories_normalised:
        sma = np.mean(his[:, 3])
        technical_indicators.append(np.array([sma]))

    technical_indicators = np.array(technical_indicators)

    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)

    return data_histories_normalised, technical_indicators_normalised, next_day_normalised, next_day, y_normaliser


