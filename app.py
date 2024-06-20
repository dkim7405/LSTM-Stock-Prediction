import os
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Disable Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Dropout = form of regularization, drops out some of units or connection during training
# Removing too much dependency on one neuron

# Reading Data
data = pd.read_csv('data/samsung.csv')

# print("hello") # top five data (used to check if the data is read correctly)

# Computing Mid Price
high_prices = data['High'].values
low_prices = data['Low'].values
mid_prices = (high_prices + low_prices) / 2

# Creating Window - looking at recent # of data
window_size = 50
sequence_length = window_size + 1

result = []

for i in range(len(mid_prices) - sequence_length):
    result.append(mid_prices[i: i + sequence_length])
    # adds [ mid_prices[0], mid_prices[1], ..., mid_prices[sequence_length] ]

# Normalizing Data
normalized_data = []

for window in result:
    normalized_window = []

    for data_point in window:
        normalized_datapoint = (float(data_point) / float(window[0])) - 1
        normalized_window.append(normalized_datapoint)

    normalized_data.append(normalized_window)

result = np.array(normalized_data)

# Categorizing Train Data and Validation Data

# Training Data
num_train = int(round(result.shape[0] * 0.9)) # getting 90 percent of data as training
train = result[:num_train, :] # this gets every window from 0 - num_train, every data_point from window
np.random.shuffle(train)

x_train = train[:, :-1] # every data point, except for very last one

# x_train.shape[0] = rows in train, x_train.shape[1] = column in train, 1 adds dimension 1 at the end
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

y_train = train[:, -1] # the very last data point

# Validation Data
x_test = result[num_train:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[num_train:, -1]

# print(x_train.shape, x_test.shape)

# Building a Model

# return sequences = LSTM sequence data reading
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(50, 1)),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dense(1, activation='linear')
])

# Compiling Model
model.compile(loss='mse', optimizer='rmsprop') # mse = mean squared error
# model.summary()

# Training Model
model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size=10,
    epochs=20
)

# Visualization
prediction = model.predict(x_test)

visualizer = plt.figure(facecolor='white', figsize=(20, 10))
graph = visualizer.add_subplot(111)
graph.plot(y_test, label='Actual')
graph.plot(prediction, label='Prediction')

graph.legend()
plt.show()




