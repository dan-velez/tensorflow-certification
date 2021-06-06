"""
Learn how to use numpy and its features. Can use this file as a reference.

Numpy has **MANY** array and matrix functions for numerical computing.
"""

import numpy as np

arr = np.array([
    1.2, 2.3, 3.4, 4.5, 5.6
])

arr = np.random.randint(low=10, high=101, size=100)
arr = np.random.random(size=5) # Random **floats**.

# Linear algebra requires that matricies have same dimensions when performing
# addition or subtraction. Multiplication/division has stricter rules.
# Numpy uses **broadcasting** to expand the smaller dimension and allow for
# compatability between different sized matricies.
arr = arr + 2 # Broadcasting.
arr = arr * 2