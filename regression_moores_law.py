# regression_moores_law.py - Linear regression in TF.

import requests
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks

# Get data
# VDATA_URL = "https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv"

# vresp = requests.get(VDATA_URL)
# open("moore.csv", "w+").write(vresp.text)


# Visualize data
vdata = pd.read_csv("data/moore.csv", header=None).values

print(type(vdata))
print(vdata[0:10])
print(vdata.shape)

# Convention is that X is a 2-D array of size N x D
# ...for tensorflow and keras.

X = vdata[:, 0].reshape(-1, 1)
print(X.shape)
print(X[0:10])

Y = vdata[:, 1]

print(Y[0:10])

# Take log of Y to make data linear rather than exponential
Y = np.log(Y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33)

# Scale X some
X = X - X.mean()  # Center around 0
print(X)

# Visualize
# plt.scatter(X, Y)
# plt.show()


# Build model
vmodel = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1, activation=None)
])


# Compile model
vmodel.compile(
    optimizer=tf.optimizers.SGD(0.001, 0.9),
    loss='mse',
    metrics=['accuracy'])


# Learning rate scheduler
def schedule(epoch, lr):
    if epoch >= 50:
        return 0.0001
    return 0.001


vscheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

# Train model
vres = vmodel.fit(X, Y, epochs=200, callbacks=[vscheduler])


# Evaluate model
plt.plot(vres.history['loss'], label='loss')
plt.show()

# plt.plot(vres.history['accuracy'], label='acc')
# plt.show()

# Visualize results
