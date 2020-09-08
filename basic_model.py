import keras
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers
import numpy as np
np.random.seed(4)
import tensorflow as tf
tf.random.set_seed(4)
from util import csv_to_dataset, history_points
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_histories, _, next_day, unscaled_y, y_normaliser = csv_to_dataset('MSFT_daily.csv')

    test_split = 0.9
    n = int(data_histories.shape[0] * test_split)
    data_train = data_histories[:n]
    y_train = next_day[:n]
    data_test = data_histories[n:]
    y_test = next_day[n:]
    unscaled_y_test = unscaled_y[n:]

    print(data_train.shape)
    print(data_test.shape)


    # model code
    lstm_input = Input(shape=(history_points, 5), name='input')
    x = LSTM(50, name='lstm_layer')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout')(x)
    x = Dense(64, name='dense')(x)
    x = Activation('sigmoid', name='sigmoid')(x)
    x = Dense(1, name='dense_next')(x)
    output = Activation('linear', name='output')(x)

    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=data_train, y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)


    # evaluation

    y_test_predicted = model.predict(data_test)
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
    y_predicted = model.predict(data_histories)
    y_predicted = y_normaliser.inverse_transform(y_predicted)

    real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
    print(scaled_mse)


    plt.gcf().set_size_inches(22, 15, forward=True)
    real = plt.plot(unscaled_y_test[0:-1], label='real')
    pred = plt.plot(y_test_predicted[0:-1], label='predicted')
    plt.legend(['Real', 'Predicted'])
    plt.show()
    model.save(f'basic_model.h5')
