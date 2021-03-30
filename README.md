# Tensorflow Certification Study Notes #


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


## models ##
* pass data, make guess, use loss and optimizer to adjust.
* make another guess, repeat
* correct guess 1/3 times, then 2/3, then 3/3 (after optimization)


## convolutions ##
* used to extract **features** from images.
* for example, applying a certain convolution filter to an image,
  can produce a contour of the lines in the image. Therefore,
  the convolution was used to extract the line features of an image.
