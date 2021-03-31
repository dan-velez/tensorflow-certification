# Tensorflow Certification Study Notes #
Notes on machine learning and tensorflow.


## about exam ##
* need to build 5 models, from basic bin classifier, to time 
  series forecast

* No surprises, just be ready to use PyCharm and TensorFlow
  datasets efficiently

* https://github.com/lmoroney/dlaicourse
* No you only have to provide models! No need to plot things!

* regression, DNN, image CNN, NLP (LSTM, CNN), and time series


## workflow ##
* read input data
* transform data
* design model architecture (correct number input nodes)
* compile model with loss and optimizer
* fit the model with number of epochs  


## neural nets ##
* Input * Weights + Bias
* Then activate
* Run several times
* Update weights and do backpropagation
* This is learning
* Vectorize input


## models ##
* pass data, make guess, use loss and optimizer to adjust.
* make another guess, repeat
* correct guess 1/3 times, then 2/3, then 3/3 (after optimization)


## convolutions ##
* used to extract **features** from images.
* for example, applying a certain convolution filter to an image,
  can produce a contour of the lines in the image. Therefore,
  the convolution was used to extract the line features of an image.


## udemy ##
* Tensorflow 2.0
* Deep Learning A-Z: Several projects in TF2
* Deep Learning: The Complete Guide with ANN and CNN (good theory)
* Data Science: Deep Learning and Neural Networks in Python (good theory)
    * Backpropagation

* **X** Modern Deep Learning in Python: Uses TF1 and other libs
* **X** The Complete ML Course with Python: Uses keras


## activation functions ##
* Universal function approximator
* Can process any type of data (text, images, audio)
* **Sigmoid**
* **Hyperbolic Tangent**
* **ReLu**


## hyperparameters ##
* learning rate
* regularization param
* number of hidden layers
* activation functions

* No precise way to choose hyperparameters.
* Can use autoML to test all values.
* Should develop intuition for hyperparameters.
* Dependant on many factors.


## cross validation ##
* General way to choose hyperparameters.
* Fit to signal, not noise of data (overfitting).
* Train - train on this data.
* Validation - validate on this data.
* Test - use data at very end.


## k-fold cross validation ##
* split data into k parts. EX k = 5:
    * 5 iterations
    * 1: train on 2-5, test on 1
    * 2: train on 1,3,4,5, test on 2
    * 3: train on 1,2,4,5, test on 3
    * etc...
* THEN take mean and variance of classification rate.


## reinforcement learning ##
* is not just simple input/output (IN image OUT car/truck/etc)
* keeps track of state of world (like in game AI)
