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
INPUTSIZE  = 588    # Size of inputs (letter images)
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
        #   We are hardcoding the number of layers for simplicity's sake. 
        np.random.seed(self.rand)
        self.W0 = np.random.rand(LAYERSIZE, INPUTSIZE)
        self.b0 = np.random.rand(LAYERSIZE, 1)
        self.W1 = np.random.rand(LAYERSIZE, LAYERSIZE)
        self.b1 = np.random.rand(LAYERSIZE, 1)
        self.W2 = np.random.rand(LAYERSIZE, OUTPUTSIZE)
        self.b2 = np.random.rand(OUTPUTSIZE, 1)

    # Training - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Helper function that divides data into batches.
    def batchData(self, X, y, numBatches, randomState=10):
        Xy_pairs = np.column_stack((X, y))
        np.random.seed(randomState)
        np.random.shuffle(Xy_pairs)
        return np.array_split(Xy_pairs, numBatches)

    # Foward propogation. Performs matrix multiplication to obtain an output
    # vector from an input vector.
    def forwardProp(self, pix):
        z1 = np.matmul(self.W0, pix)

    # Uses forward and backward propogation to find average adjustment values
    # to apply to the classifier's weights and biases.
    def getAvgAdjustments(self, X, y):
        dW0, db0, dW1, db1, dW2, db2 = 0, 0, 0, 0, 0, 0

        # for idx, pix in np.ndenumerate(X):
        #     print()

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

    # Primary Training Function
    def fit(self, X, y, numBatches=50):
        # Randomly split data into batches.
        batches = self.batchData(X, y, numBatches)

        # Loop over the batches.
        # For each batch, perform forward prop, back prop, and update weights
        # and biases according to average adjustments found using back prop.
        for batch in batches:
            dW0, db0, dW1, db1, dW2, db2 = self.getAvgAdjustments(X, y)
            self.updateParams( dW0, db0, dW1, db1, dW2, db2 )

    # Classifing - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Primary Classification Function
    def predict(self, X):
        # Step 1: Run all elements of X through the model.
        return None

