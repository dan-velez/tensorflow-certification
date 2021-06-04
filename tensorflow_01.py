#!/usr/bin/python
"""
Learning Tensorflow for developer certificate exam.
https://ai.google/responsibilities/responsible-ai-practices/
https://www.tensorflow.org/resources/responsible-ai
"""

import os

import tensorflow as tf
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def building_models_demo():
    """Display how to put together a sequential model using 
    tf.keras (de facto)."""
    
    # Model architecture (binary classifier).
    vmodel = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(2,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Train model on data.
    vmodel.compile(optimizer='adam', # Gradient Descent 
                loss='binary_crossentropy', 
                metrics=['accuracy', 'recall', 'precision'])

    vtrained = vmodel.fit(X_train, y_train, 
                        validation_data=(X_test, y_test),
                        epochs=100)

    # Evaluate model.
    plt.plot(vtrained.history['loss'], label='loss')
    plt.plot(vtrained.history['val_loss'], label='val_loss')


def arrhythmia_dataset():
    """Test loading and plotting data."""

    vurl = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"+
           "arrhythmia/arrhythmia.data")

    tf.keras.utils.get_file(
        os.path.abspath(os.getcwd())+"\\arrhythmia.data",
        vurl)

    # Read in data
    vdf = pd.read_csv("arrhythmia.data", header=None)
    vdf_sample = vdf[[0, 1, 2, 3, 4, 5]]
    vdf_sample.columns = ['age', 'sex', 'height', 'weight', 'QRS duration', 
                          'P-R interval']
    
    print(vdf_sample.head())
    #vdf_sample.hist()
    #plt.show()

    scatter_matrix(vdf_sample)
    plt.show()


def auto_mpg_dataset():
    """Test loading and plotting data."""

    vurl = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"+
           "auto-mpg/auto-mpg.data")
    tf.keras.utils.get_file(
        os.path.abspath(os.getcwd())+"\\auto-mpg.data", vurl)
    print("[*] Auto MPG data downloaded...")
    vdf = pd.read_csv("auto-mpg.data", header=None, delim_whitespace=True)
    print(vdf.head())


def breast_cancer_classification():
    """Demo how to build a classifier on breast cancer data."""

    vdata = load_breast_cancer()
    print(type(vdata))
    print(vdata.keys())
    print(vdata.data.shape)

    # Split train and test set by 1/3 test size
    X_train, X_test, y_train, y_test = train_test_split(
        vdata.data, vdata.target, test_size=0.33)
    N, D = X_train.shape

    print(N, D)

    # Scale the data (normalize)
    vscaler = StandardScaler()
    X_train = vscaler.fit_transform(X_train)
    X_test = vscaler.fit_transform(X_test)

    print(X_train)

    # Build the model
    vmodel = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(D,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    vmodel.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    vres = vmodel.fit(X_train, y_train, 
        validation_data=(X_test, y_test), epochs=100)

    # Save model
    vmodel.save("model.hd5")

    # Eval model
    print("[*] Train score: ", vmodel.evaluate(X_train, y_train))
    print("[*] Test score: ", vmodel.evaluate(X_test, y_test))

    # Plot eval results
    # plt.plot(vres.history['loss'], label='loss')
    # plt.plot(vres.history['val_loss'], label='val_loss')
    # plt.legend()
    # plt.show()

    # Plot accuracy
    plt.plot(vres.history['accuracy'], label='acc')
    plt.plot(vres.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    breast_cancer_classification()
    # arrhythmia_dataset()