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
INPUTSIZE  = 4      # Size of inputs (letter images)
LAYERSIZE  = 32     # Size of hidden layers
OUTPUTSIZE = 3      # Size of output vector
LIMITER    = 0.001  # Limits the range of initial weights and biases

# The ImgClassifier Class
class ImgClassifier:

    # Construction - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Helper function that normalizes initial parameter values so they all
    # have the same variance. This is supposed to prevent vanishing and
    # exploding gradients.
    def normalizeParams(self):
        self.W0 = (self.W0 - np.mean(self.W0)) / np.std(self.W0)
        self.b0 = (self.b0 - np.mean(self.b0)) / np.std(self.b0)
        self.W1 = (self.W1 - np.mean(self.W1)) / np.std(self.W1)
        self.b1 = (self.b1 - np.mean(self.b1)) / np.std(self.b1)
        self.W2 = (self.W2 - np.mean(self.W2)) / np.std(self.W2)
        self.b2 = (self.b2 - np.mean(self.b2)) / np.std(self.b2)

    # Constructor Function.
    def __init__(self, alpha, randomState=10):

        # User Defined Variables
        self.alpha = alpha
        self.rand  = randomState

        # Randomly Generated Variables
        #   We are hardcoding the number of layers for simplicity's sake
        np.random.seed(self.rand)
        self.W0 = (np.random.rand(LAYERSIZE,  INPUTSIZE) - 0.5) * LIMITER
        self.b0 = (np.random.rand(LAYERSIZE,  1) - 0.5)         * LIMITER
        self.W1 = (np.random.rand(LAYERSIZE,  LAYERSIZE) - 0.5) * LIMITER
        self.b1 = (np.random.rand(LAYERSIZE,  1) - 0.5)         * LIMITER
        self.W2 = (np.random.rand(OUTPUTSIZE, LAYERSIZE) - 0.5) * LIMITER
        self.b2 = (np.random.rand(OUTPUTSIZE, 1) - 0.5)         * LIMITER

        self.normalizeParams()

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
        sig = lambda x: 1 / (1 + math.exp(-x[0]))
        return np.array( [sig(x) for x in arr] )

    # ReLU, an alternative activation function.
    def ReLU(self, arr):
        return np.maximum(arr, 0)

    # Softmax, a function that normalizes the output layer when using the ReLU
    # activation function.
    # def Softmax(self, arr):
    #     # print(np.sum(np.exp(arr)), end='')
    #     softmax = np.exp(arr) / np.sum(np.exp(arr))
    #     return softmax
    def Softmax(self, arr):
        with np.errstate(over='raise'):
            try:
                softmax = np.exp(arr) / np.sum(np.exp(arr))
            except FloatingPointError:
                # Handle overflow error
                softmax = np.zeros_like(arr)
                print(arr)
                exit()
        return softmax

    # Foward propogation. Performs matrix multiplication to obtain an output
    # vector from a single input vector.
    def forwardProp(self, pix):
        # z1 = np.add(np.matmul(self.W0, np.atleast_2d(pix).T), self.b0)
        # a1 = self.sigmoid(z1)
        # z2 = np.add(np.matmul(self.W1, a1), self.b1)
        # a2 = self.sigmoid(z2)
        # z3 = np.add(np.matmul(self.W2, a2), self.b2)
        # a3 = self.sigmoid(z3)

        # ReLU Code
        z1 = np.add(np.matmul(self.W0, np.atleast_2d(pix).T), self.b0)
        a1 = self.ReLU(z1)
        z2 = np.add(np.matmul(self.W1, np.atleast_2d(a1)), self.b1)
        a2 = self.ReLU(z2)
        z3 = np.add(np.matmul(self.W2, np.atleast_2d(a2)), self.b2)
        a3 = self.Softmax(z3)

        return z1, a1, z2, a2, z3, a3

    # Sigmoid derivative function. This function is used to compute deltas,
    # which represent the change in cost with respect to the weighted sums.
    def sigmoidDerivative(self, arr):
        sig    = lambda x: 1 / (1 + math.exp(-x[0]))
        sigDer = lambda x: sig(x) * (1 - sig(x))
        return np.array( [sigDer(x) for x in arr] )

    # ReLU derivative function. Same application as above, but used for ReLU
    # activation function.
    def ReLUDerivative(self, arr):
        return np.array([float(x > 0) for x in arr])

    # Softmax "derivative" function. It serves to undo the softmax function.
    def SoftmaxDerivative(self, arr):
        return np.array([np.sum(arr) for x in arr])

    # Back propogation. Performs calculus to obtain adjustment values for all
    # weights and biases.
    def backProp(self, X, y, z1, a1, z2, a2, z3, a3):

        # Obtaining expected output vector
        y_exp = np.zeros(OUTPUTSIZE)
        # y_exp[ord(y) - 65] = 1

        # Obtaining expected output vector for flower data
        if y == 'Iris-setosa':
            y_exp[0] = 1
        elif y == 'Iris-versicolor':
            y_exp[1] = 1
        else:
            y_exp[2] = 1

        # Computing change values
        # db2 = np.multiply(-2 * np.subtract(y_exp, a3), self.sigmoidDerivative(z3))
        # dW2 = np.matmul(np.atleast_2d(db2).T, np.atleast_2d(a2))
        # db1 = np.multiply(np.matmul(self.W2.T, db2), self.sigmoidDerivative(z2))
        # dW1 = np.matmul(np.atleast_2d(db1).T, np.atleast_2d(a1))
        # db0 = np.multiply(np.matmul(self.W1.T, db1), self.sigmoidDerivative(z1))
        # dW0 = np.matmul(np.atleast_2d(db0).T, np.atleast_2d(X))

        # New Sigmoid Code
        # delta2 = np.multiply(self.sigmoidDerivative(z3), np.subtract(y_exp, a3))
        # dW2 = np.matmul(np.atleast_2d(a2).T, np.atleast_2d(delta2)).T
        # db2 = self.SoftmaxDerivative(delta2)
        # delta1 = np.multiply(self.sigmoidDerivative(z2), np.matmul(self.W2.T, delta2))
        # dW1 = np.matmul(np.atleast_2d(a1).T, np.atleast_2d(delta1)).T
        # db1 = self.SoftmaxDerivative(delta1)
        # delta0 = np.multiply(self.sigmoidDerivative(z1), np.matmul(self.W1.T, delta1))
        # dW0 = np.matmul(np.atleast_2d(X).T, np.atleast_2d(delta0)).T
        # db0 = self.SoftmaxDerivative(delta0)

        # ReLU Code
        dz3 = np.subtract(a3, np.atleast_2d(y_exp).T)
        db2 = self.SoftmaxDerivative(dz3)
        dW2 = np.matmul(np.atleast_2d(dz3), np.atleast_2d(a2).T)
        dz2 = np.matmul(self.W2.T, dz3) * np.atleast_2d(self.ReLUDerivative(z2)).T
        db1 = self.SoftmaxDerivative(z2)
        dW1 = np.matmul(np.atleast_2d(dz2), np.atleast_2d(a1).T)
        dz1 = np.matmul(self.W1.T, dz2) * np.atleast_2d(self.ReLUDerivative(z1)).T
        db0 = self.SoftmaxDerivative(z1)
        dW0 = np.matmul(np.atleast_2d(dz1), np.atleast_2d(X))

        # print(f'Changes:\n  dW0 = {dW0}\n  db0 = {db0}\n  dW1 = {dW1}\n  db1 = {db1}\n  dW2 = {dW2}\n  db2 = {db2}')
        # exit()

        return dW0, db0, dW1, db1, dW2, db2

    # Uses forward and backward propogation to find average adjustment values
    # to apply to the classifier's weights and biases.
    def getAvgAdjustments(self, X, y):
        dW0, db0, dW1, db1, dW2, db2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        count = 0
        for idx, pix in enumerate(X):
            z1, a1, z2, a2, z3, a3 = self.forwardProp(pix)
            curr_dW0, curr_db0, curr_dW1, curr_db1, curr_dW2, curr_db2 = \
                self.backProp(pix, y[idx], z1, a1, z2, a2, z3, a3)
            dW0 = np.add(dW0, curr_dW0)
            db0 = np.add(db0, curr_db0)
            dW1 = np.add(dW1, curr_dW1)
            db1 = np.add(db1, curr_db1)
            dW2 = np.add(dW2, curr_dW2)
            db2 = np.add(db2, curr_db2)
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
        self.W0 = np.subtract(self.W0, self.alpha * dW0)
        self.b0 = np.subtract(self.b0, self.alpha * np.atleast_2d(db0).T)
        self.W1 = np.subtract(self.W1, self.alpha * dW1)
        self.b1 = np.subtract(self.b1, self.alpha * np.atleast_2d(db1).T)
        self.W2 = np.subtract(self.W2, self.alpha * dW2)
        self.b2 = np.subtract(self.b2, self.alpha * np.atleast_2d(db2).T)

    # Debugging function that prints the classifier's member variables.
    def printMembers(self):
        print(f'\nself.W0 = \n{self.W0}')

    # Debugging function that prints the average magnitude of the change values.
    def printChanges(self, dW0, db0, dW1, db1, dW2, db2):
        print(f'  Avg dW0: {np.mean(abs(dW0))}')
        print(f'  Avg db0: {np.mean(abs(db0))}')
        print(f'  Avg dW1: {np.mean(abs(dW1))}')
        print(f'  Avg db1: {np.mean(abs(db1))}')
        print(f'  Avg dW2: {np.mean(abs(dW2))}')
        print(f'  Avg db2: {np.mean(abs(db2))}\n')

    # Primary Training Function.
    def fit(self, X, y, numBatches=1, iterations=500):
        # Randomly split data into batches
        batches = self.batchData(X, y, numBatches, self.rand)
        print('Initial Training Accuracy: ' + str(self.accuracy(X, y)))
        # self.printMembers()
        initB0 = self.b0[0]

        for i in range(iterations):
            # self.printMembers()

            # Loop over the batches
            # For each batch, perform forward prop, back prop, and update weights
            # and biases according to average adjustments found using back prop
            for idx, batch in enumerate(batches):
                curr_X = np.array([pair[:INPUTSIZE] for pair in batch], dtype=float)
                curr_y = np.array([pair[INPUTSIZE] for pair in batch])

                dW0, db0, dW1, db1, dW2, db2 = self.getAvgAdjustments(curr_X, curr_y)
                # self.printChanges(dW0, db0, dW1, db1, dW2, db2)
                self.updateParams( dW0, db0, dW1, db1, dW2, db2 )
                # print('Processed batch ' + str(idx) + '/' + str(numBatches))
                # print(f'Changes:\n  dW0 = {dW0}\n  db0 = {db0}\n  dW1 = {dW1}\n  db1 = {db1}\n  dW2 = {dW2}\n  db2 = {db2}')

            # print('- - - - - - - - - - -')
            # self.printMembers()
            if i % 10 == 0:
                print('Processed iteration ' + str(i) + '/' + str(iterations))
            # if i % 100 == 0:
                print('  Current Accuracy: ' + str(self.accuracy(X, y)))
        # self.printMembers()
        print(f'\nChange in b0[0]: {(initB0 - self.b0[0])[0]}\n')


    # Classifying - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Function that uses forward propogation to find the label with the
    # highest likelihood of being correct.
    def findLabel(self, pix):
        _, _, _, _, _, a3 = self.forwardProp(pix)
        max_idx = np.argmax(a3)
        # pred = chr(max_idx + 65)

        # Obtaining flower data prediction
        if max_idx == 0:
            pred = 'Iris-setosa'
        elif max_idx == 1:
            pred = 'Iris-versicolor'
        else:
            pred = 'Iris-virginica'
        return pred

    # Primary Classification Function.
    def predict(self, X):
        return np.array([self.findLabel(x) for x in X])

    # Accuracy Function.
    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.sum([predictions[i] == x for i, x in enumerate(y)]) / float(len(predictions))

