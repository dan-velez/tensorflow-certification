"""
Neural network trained on arbitrary numerical data. 
"""

import tensorflow as tf
import numpy as np

# Init dataset.
xs = np.array([-1, 0, 1, 2, 3, 4], dtype=float)
ys = np.array([-3, -1, 1, 3, 5, 7], dtype=float)

# Build and train neural network.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(
    optimizer='sgd',
    loss='mean_squared_error')
hist = model.fit(xs, ys, epochs=500, verbose=1)

# Test model.
print(model.predict([10.0]))