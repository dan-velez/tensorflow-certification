"""
Linear regression in Tensorflow.
"""

import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt


def build_model(learning_rate:float=0.001) -> tf.keras.Model:
    """Create empty trainable neural network."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense()
    ])
    return model


def train_model(model:tf.keras.Model, feature:pd.DataFrame, 
                label:pd.DataFrame, epochs:int) -> tf.keras.Model:
    """Train net on examples."""
    return


if __name__ == "__main__":
    model = build_model()