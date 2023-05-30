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
ALPHA = 0.5

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

# Sigmoid derivative.
def sigmoidDerivative(arr):
    sig    = lambda x: 1 / (1 + math.exp(-x[0]))
    sigDer = lambda x: sig(x) * (1 - sig(x))
    return np.array( [sigDer(x) for x in arr] )

# Back propagation, taken from Santini's slides.
def backProp(X, y, b0, W1, b1, z1, a1, z2, a2):
    delta1 = np.multiply(sigmoidDerivative(z2), -2 * np.subtract(y, a2))
    dW1    = np.matmul(np.atleast_2d(a1).T, np.atleast_2d(delta1)).T
    db1    = np.multiply(np.atleast_2d(b1), np.atleast_2d(delta1))
    delta0 = np.multiply(sigmoidDerivative(z1), np.matmul(W1.T, delta1))
    dW0    = np.matmul(np.atleast_2d(X).T, np.atleast_2d(delta0)).T
    db0    = np.multiply(np.atleast_2d(b0), np.atleast_2d(delta0))
    return dW0, db0, dW1, db1

# One iteration of the neural network cycle.
def cycle(x, y, W0, b0, W1, b1):
    z1, a1, z2, a2 = forwardProp(x, W0, b0, W1, b1)

    dW0, db0, dW1, db1 = backProp(x, y, b0, W1, b1, z1, a1, z2, a2)

    W0 = np.subtract(W0, ALPHA * dW0)
    b0 = np.subtract(b0, ALPHA * db0)
    W1 = np.subtract(W1, ALPHA * dW1)
    b1 = np.subtract(b1, ALPHA * db1)

    return W0, b0, W1, b1


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
y = [[1.0]]

z1, a1, z2, a2 = forwardProp(x, W0, b0, W1, b1)

print('Initial Accuracy: ' + str(round(a2[0], 3)))

for i in range(10000):
    W0, b0, W1, b0 = cycle(x, y, W0, b0, W1, b1)

_, _, _, acc = forwardProp(x, W0, b0, W1, b1)

print('Final Accuracy: ' + str(round(acc[0], 3)))

