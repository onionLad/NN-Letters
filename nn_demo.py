#
# nn_demo.py
#   Bill Xia
#   5/24/23
#
# This file tests our neural network functions on a smaller scale.
#

# Imports
import os
import numpy as np
import math

# Globals
RNG = 10

# Functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Sigmoid.
def sigmoid(arr):
    sig = lambda x: 1 / (1 + math.exp(-x[0]))
    return np.array( [sig(x) for x in arr] )

# Forward propagation.
def forwardProp(X, W0, b0, W1, b1):
    z1 = np.add(np.matmul(W0, X), b0)
    a1 = sigmoid(z1)
    z2 = np.add(np.matmul(W1, a1), b1)
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

# Body  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Initializing weights and biases. We are hardcoding these values to fit the
# example given by Prof Santini.
np.random.seed(RNG)
W0 = np.array([[-0.5, 1], [-0.5, 1]])
b0 = np.array([[-0.5], [-1]])
W1 = np.array([[-0.5, -0.5]])
b1 = np.array([[1]])

# Generating input and expected output.
x = [[0.0], [1.0]]
y = [[0.0]]

z1, a1, z2, a2 = forwardProp(x, W0, b0, W1, b1)

print('Initial Accuracy: ' + str(round(a2[0], 3)))

