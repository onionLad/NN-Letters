#
# implementation.py
#   Bill Xia
#   5/17/23
#
# This file contains the implementation of my image classification neural
# network class.
#

# Imports
import numpy as np
import math

# Globals
INPUTSIZE  = 587    # Size of inputs (letter images)
LAYERSIZE  = 16     # Size of hidden layers
OUTPUTSIZE = 26     # Size of output vector

# The ImgClassifier Class
class ImgClassifier:

    # Constructor
    def __init__(self, alpha, randomState=10):

        # User Defined Variables
        self.alpha = alpha
        self.rand  = randomState

        # Randomly Generated Variables
        #   We are hardcoding the number of layers for simplicity's sake
        np.random.seed(self.rand)
        self.W0 = np.random.rand(LAYERSIZE, INPUTSIZE)
        self.b0 = np.random.rand(LAYERSIZE)
        self.W1 = np.random.rand(LAYERSIZE, LAYERSIZE)
        self.b1 = np.random.rand(LAYERSIZE)
        self.W2 = np.random.rand(OUTPUTSIZE, LAYERSIZE)
        self.b2 = np.random.rand(OUTPUTSIZE)

    # Training - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Helper function that divides data into batches.
    def batchData(self, X, y, numBatches, randomState=10):
        Xy_pairs = np.column_stack((X, y))
        np.random.seed(randomState)
        np.random.shuffle(Xy_pairs)
        return np.array_split(Xy_pairs, numBatches)

    # Sigmoid function. This is our activation function, used to normalize
    # np arrays of weighted sums.
    def sigmoid(self, arr):
        sig = lambda x: 1 / (1 + math.exp(-x))
        return np.array( [sig(x) for x in arr] )

    # Foward propogation. Performs matrix multiplication to obtain an output
    # vector from a single input vector.
    def forwardProp(self, pix):
        z1 = np.add(np.matmul(self.W0, pix), self.b0)
        a1 = self.sigmoid(z1)
        z2 = np.add(np.matmul(self.W1, a1), self.b1)
        a2 = self.sigmoid(z2)
        z3 = np.add(np.matmul(self.W2, a2), self.b2)
        a3 = self.sigmoid(z3)
        return z1, a1, z2, a2, z3, a3

    # Sigmoid derivative function. This function is used to compute deltas,
    # which represent the change in cost with respect to the weighted sums.
    def sigmoidDerivative(self, arr):
        sig    = lambda x: 1 / (1 + math.exp(-x))
        sigDer = lambda x: sig(x) * (1 - sig(x))
        return np.array( [sigDer(x) for x in arr] )

    # Back propogation. Performs calculus to obtain adjustment values for all
    # weights and biases.
    def backProp(self, y, z1, a1, z2, a2, z3, a3):

        # Obtaining expected output vector
        y_exp = np.zeros(OUTPUTSIZE)
        y_exp[ord(y) - 65] = 1

        # Computing necessary back prop values
        db2 = np.multiply(-2 * np.subtract(a3, y_exp), self.sigmoidDerivative(z3))
        dW2 = np.matmul(db2, a3.T)
        db1 = np.multiply(np.matmul(self.W2.T, db2), self.sigmoidDerivative(z2))
        dW1 = np.matmul(db1, a2.T)
        db0 = np.multiply(np.matmul(self.W1.T, db1), self.sigmoidDerivative(z1))
        dW0 = np.matmul(db0, a1.T)

        return dW0, db0, dW1, db1, dW2, db2

    # Uses forward and backward propogation to find average adjustment values
    # to apply to the classifier's weights and biases.
    def getAvgAdjustments(self, X, y):
        dW0, db0, dW1, db1, dW2, db2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        count = 0
        for idx, pix in enumerate(X):
            z1, a1, z2, a2, z3, a3 = self.forwardProp(pix)
            curr_dW0, curr_db0, curr_dW1, curr_db1, curr_dW2, curr_db2 = \
                self.backProp(y[idx], z1, a1, z2, a2, z3, a3)
            dW0 += curr_dW0
            db0 += curr_db0
            dW1 += curr_dW1
            db1 += curr_db1
            dW2 += curr_dW2
            db2 += curr_db2
            count += 1

        # Taking the averages of all changes to weights and biases
        dW0 /= count
        db0 /= count
        dW1 /= count
        db1 /= count
        dW2 /= count
        db2 /= count

        return dW0, db0, dW1, db1, dW2, db2

    # Uses the average adjustment values found by getAvgAdjustments to update
    # the classifier's weights and biases.
    def updateParams(self, dW0, db0, dW1, db1, dW2, db2):
        self.W0 -= self.alpha * dW0
        self.b0 -= self.alpha * db0
        self.W1 -= self.alpha * dW1
        self.b1 -= self.alpha * db1
        self.W2 -= self.alpha * dW2
        self.b2 -= self.alpha * db2

    # Primary Training Function.
    def fit(self, X, y, numBatches=50):
        # Randomly split data into batches
        batches = self.batchData(X, y, numBatches)

        # Loop over the batches
        # For each batch, perform forward prop, back prop, and update weights
        # and biases according to average adjustments found using back prop
        for idx, batch in enumerate(batches):
            curr_X = np.array([pair[:INPUTSIZE] for pair in batch], dtype=float)
            curr_y = np.array([pair[INPUTSIZE] for pair in batch])

            dW0, db0, dW1, db1, dW2, db2 = self.getAvgAdjustments(curr_X, curr_y)
            self.updateParams( dW0, db0, dW1, db1, dW2, db2 )
            print('Processed batch ' + str(idx + 1) + '/' + str(numBatches))

    # Classifing - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Function that uses forward propogation to find the label with the
    # highest likelihood of being correct.
    def findLabel(self, pix):
        _, _, _, _, _, a3 = self.forwardProp(pix)
        return chr(np.argmax(a3) + 65)

    # Primary Classification Function.
    def predict(self, X):
        return np.array([self.findLabel(x) for x in X])

