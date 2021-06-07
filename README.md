# Tensorflow Certification #
Learn to build neural networks using tensorflow.


## About Exam ##
* Need to build 5 models, from basic bin classifier, to time series forecasting
* No surprises, just be ready to use PyCharm and TensorFlow
  datasets efficiently
* https://github.com/lmoroney/dlaicourse
* No you only have to provide models, **no need to plot things**.
* regression, DNN, image CNN, NLP (LSTM, CNN), and time series


## Workflow ##
* Read input data
* Transform data
* Design model architecture (correct number input nodes)
* Compile model with loss and optimizer
* Fit the model with number of epochs  


## Neural Networks ##
Develop and experiment with models for arbitrary numerical data.

* Input * Weights + Bias.
* Then activate.
* Run several times.
* Update weights and do backpropagation.
* This is learning.
* Vectorize input.


## Model Development ##
* Pass data, make guess, use loss and optimizer to adjust.
* Make another guess, repeat
* Correct guess 1/3 times, then 2/3, then 3/3 (after optimization)
* `build`, `compile`, `fit`, `evaluate`, `predict`.


## Convolutions ##
* Used to extract **features** from images.
* For example, applying a certain convolution filter to an image,
  can produce a contour of the lines in the image. Therefore,
  the convolution was used to extract the line features of an image.
```python
model = tf.keras.Sequential([
  Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32,32,3)))
])
```

## Udemy ##
* Tensorflow 2.0
* Deep Learning A-Z: Several projects in TF2
* Deep Learning: The Complete Guide with ANN and CNN (good theory)
* Data Science: Deep Learning and Neural Networks in Python (good theory)
    * Backpropagation

* **X** Modern Deep Learning in Python: Uses TF1 and other libs
* **X** The Complete ML Course with Python: Uses keras
* [Google TF Course](https://developers.google.com/machine-learning/crash-course/first-steps-with-tensorflow)
* [Coursera dlaicourse](https://github.com/lmoroney/dlaicourse)


## Activation Functions ##
* Universal function approximator
* Can process any type of data (text, images, audio)
* **Sigmoid**
* **Hyperbolic Tangent**
* **ReLu**


## Hyperparameters ##
* Learning rate
* Regularization param
* Number of hidden layers
* Activation functions

* No precise way to choose hyperparameters.
* Can use autoML to test all values.
* Should develop intuition for hyperparameters.
* Dependant on many factors.


## Cross Validation ##
* General way to choose hyperparameters.
* Fit to signal, not noise of data (overfitting).
* Train - train on this data.
* Validation - validate on this data.
* Test - use data at very end.


## K-fold Cross Validation ##
* split data into k parts. EX k = 5:
    * 5 iterations
    * 1: train on 2-5, test on 1
    * 2: train on 1,3,4,5, test on 2
    * 3: train on 1,2,4,5, test on 3
    * etc...
* THEN take mean and variance of classification rate.


## Reinforcement Learning ##
* Is not just simple input/output (IN image OUT car/truck/etc)
* Keeps track of state of world (like in game AI)