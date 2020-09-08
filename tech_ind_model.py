import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import numpy as np
np.random.seed(16)
import tensorflow as tf
tf.random.set_seed(16)
from util import csv_to_dataset, history_points
import matplotlib.pyplot as plt

# dataset
if __name__ == '__main__':

    data_histories, moving_av, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset('MSFT_daily.csv')
    test_split = 0.9
    n = int(data_histories.shape[0] * test_split)
    data_train = data_histories[:n]
    moving_av_train = moving_av[:n]
    y_train = next_day_open_values[:n]
    data_test = data_histories[n:]
    moving_av_test = moving_av[n:]
    y_test = next_day_open_values[n:]
    unscaled_y_test = unscaled_y[n:]

    #print(data_train.shape)
    #print(data_test.shape)


    # model code
    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    dense_input = Input(shape=(moving_av.shape[1],), name='moving_av_input')
    x = LSTM(50, name='lstm_layer')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout')(x)
    lstm_branch = Model(inputs=lstm_input, outputs=x)
    y = Dense(20, name='moving_av_dense')(dense_input)
    y = Activation("relu", name='moving_av_relu')(y)
    y = Dropout(0.2, name='moving_av_dropout')(y)
    moving_av_branch = Model(inputs=dense_input, outputs=y)
    combined = concatenate([lstm_branch.output, moving_av_branch.output], name='concatenate')
    z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
    z = Dense(1, activation="linear", name='dense_out')(z)
    model = Model(inputs=[lstm_branch.input, moving_av_branch.input], outputs=z)
    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=[data_train, moving_av_train], y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)

    # evaluation
    y_test_predicted = model.predict([data_test, moving_av_test])
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
    y_predicted = model.predict([data_histories, moving_av])
    y_predicted = y_normaliser.inverse_transform(y_predicted)
    real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
    print(scaled_mse)

    plt.gcf().set_size_inches(10, 10, forward=True)
    real = plt.plot(unscaled_y_test[0:-1], label='real')
    pred = plt.plot(y_test_predicted[0:-1], label='predicted')
    plt.legend(['Real', 'Predicted'])

    plt.show()
    model.save(f'branched_model.h5')
