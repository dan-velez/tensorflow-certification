"""
Lerning tensorflow library for building and training models.
https://www.youtube.com/watch?v=tpCFfeUEGs8
Implement data science techniques using tensorflow.
Prepare for Google TF certification exam.
Write neural network code.
"""

import tensorflow as tf
import numpy as np

print()
print("*" * 64)
print("[* tf_learn] Init tf code.")
print(tf.__version__)


# Create tensor with shape (2, )
# Variable tensors created with Variable class.
vtensor = tf.Variable([10, 7])
print(vtensor.shape)

# Creating random tensors (tensors with random values).
# In ML project, input data must be converted into tensors,
# (numerical encoding).
# This gets passed into input layer of neural network.
# NN learns patterns / features / weights (representations).
# Convert output (decode tensors).

# Need to initialize input with random weights.
# Pseudo random numbers initiated with seed.
# Random weights are slowly adjusted, as NN learns on examples.
vrand1 = tf.random.Generator.from_seed(42).normal(shape=(3, 2))
vrand2 = tf.random.Generator.from_seed(42).normal(shape=(3, 2))
print(vrand1 == vrand2)

# Shape is (x, Y), where x is num rows, Y is num features.
# Shuffle order of elements in a tensor.
# Order is important and it must be random. E.g. if first 100
# examples are of one type, it will overfit to that type.
# NN can adjust internal paterns/weights in random order.
vtensor = tf.constant([[10,7], [3,4], [2,5]])
print(vtensor)
tf.random.shuffle(vtensor)
print(vtensor)

# Adding a seed to shuffle will always produce same order.
# Need to set global AND operation level seed.
# Can be used for NN experiments to produce same results.
vseed_num = 42
tf.random.set_seed(vseed_num)
tf.random.shuffle(vtensor, seed=vseed_num)

# Tensors can be created from **numpy arrays**.
vtensor_1s = tf.ones(shape=(10, 1), dtype='int32')
vtensor_0s = tf.zeros([10, 7], dtype='float32')

# Capital for matricies.
X = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Tensors can run on a GPU, numpy array CANNOT.
# Create numpy array from 1 - 25.
vnp_arr = np.arange(1, 25, dtype=np.int32)
vtensor = tf.constant(vnp_arr)

# Create from np array with different shape.
# (values need to add up to equal number of elements)
# EX: 2 * 3 * 4 = 24 elements.
vtensor = tf.constant(vnp_arr, shape=(2, 3, 4))

# Getting Tensor metadata.
# * shape
# * rank
# * axis / dim
# * size
vtensor = tf.constant([[10, 7], [3, 4]])
print( vtensor.shape )
print( vtensor.ndim )
print( vtensor[0] )
print( vtensor[:, 0] ) # Get first value in all rows.
print( tf.size(vtensor) )

# Rank 4 tensor (4 dimensions).
# Should get used to deducing tensor dimensions, from
# writing and creating tensors.
# When tensors are passed into network, it should be of a 
# certain shape. Should be familiar with tensor shapes.
vtensor = tf.zeros([2, 3, 4, 5])

# When probing and trying to visualize data, pretty print it:
def pprint_tensor (vtensor):
    print("")
    print("Datatype of Elements: %s" % vtensor.dtype)
    print("Number of dimensions: %s" % vtensor.ndim)
    print("Shape of tensor: %s" % vtensor.shape)
    print("Elements along 0 axis: %s" % vtensor.shape[0])
    print("Elements along last axis: %s" % vtensor.shape[-1])
    print("Total elements in tensor: %s" % 
            tf.size(vtensor).numpy())
    print("")

pprint_tensor(vtensor)

# Indexing Tensors can be done like Python lists.
# Ex: Get first 2 elements of each dimension.
# Multiple dimensions, seperate by columns.
vdims = vtensor[:2, :2, :2, :2]

# Omit a dimension.
# After encoding data, neural nets act on matricies and 
# perform many matrix operations (e.g. convolutions).
# Flattening through dimensions.
vdims = vtensor[:, :, :, -1]

# Get last item in each row.
vtensor2 = tf.constant([ [10,7], [1,2] ])
vtensor2[:, -1]

# Add dimension to tensor
# The '...' means include every axis.
vtensor3 = vtensor2[..., tf.newaxis]
vtensor3 = tf.expand_dims(vtensor2, axis=2) # Same as above.

# Tensor operations (+, -, *, /)
# **Element-Wise** operations.
# Finding patterns involves **MANIPULATING** tensors.
vten = tf.constant([ [10,7], [2,4] ])

# + 10 to every element. Then * everything by ten.
vten + 10
vten * 10
vten * vten # Shapes should be compatible.

# + 10 to every element in first row
tf.constant( [vten[0].numpy()+10, vten[1].numpy()] )

# Can use tf builtin functions for basic operations.
# Code will be sped up on GPU if using builtins.
tf.multiply(vten, 10)

# Matrix multiplication (used a lot in NN code).
# Most common tensor operation.
# NOTE: using mat * mat is **ELEMENT WISE** NOT DOT product.
tf.linalg.matmul( vten, vten ) # dot product.
tf.matmul(vten, vten)
vten @ vten # Using python operator.

# In matrix multiplication, one side must equal the other.
# This will NOT work...
try:
    ten = tf.constant([
        [1,2],
        [2,3]
    ])

    ten2 = tf.constant([
        [1,2,3,4],
        [2,3,4,5],
        [3,4,5,6]
    ])

    ten @ ten2
except Exception as e:
    print(e)
    print("Cannot multiply tensors without equal side")

# The **inner dimensions** of the matricies must match.
# [3,2] and [3,2] DONT match. [3,2] and [2,3] match.
