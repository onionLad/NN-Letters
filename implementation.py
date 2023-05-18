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

    # Primary Training Function
    def fit(self, X, y, numBatches=20, iterations=50):
        # Step 1: Randomly split data into batches.
        batches = self.batchData(X, y, numBatches)

        # Step 2: Perform forward prop on a data sample.
        

        # Step 3: Perform back prop on the same data sample.
        # Step 4: Record values obtained from back prop.
        # Step 5: Repeat steps 2-4 on all elements in the first batch.
        # Step 6: Update weights and biases according to average adjustments
        #         found in Step 6.
        # Step 7: Repeat steps 5-6 on each batch until you've performed the
        #         desired number of iterations.
        return None

    # Classifing - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Primary Classification Function
    def predict(self, X):
        # Step 1: Run all elements of X through the model.
        return None

