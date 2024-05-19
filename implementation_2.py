#
# implementation_2.py
#   Bill Xia
#   5/11/24
#
# Purpose: Our second attempt at implementing a neural network from scratch.
#          This time, the plan is to just classify flowers. Maybe it'll turn
#          out this time.
#
# Architecture: The plan is to construct a neural network with one hidden
#               layers. That is, we'll need to learn weights and biases for
#               2 separate translations.
#

# Imports
import numpy as np

# Globals
INPUTSIZE  = 4      # Size of input (number of features)
LAYERSIZE  = 32     # Size of hidden layers
OUTPUTSIZE = 3      # Size of output (number of categories)

# Class
class FlowersClassifier:
    '''
    Class implementing a neural network that classifies iris data.
    '''

    # Constructor ----------------------------------------------------------- #
    def __init__(self, alpha=0.1, randomState=69):

        # Store parameters values.
        self.alpha = alpha          # Learning rate
        self.rand  = randomState    # Random state
        np.random.seed(self.rand)

        # Initializing weights and biases.
        #   Recall that the formula for each transformation is (wx + b).
        self.W0 = (np.random.rand(LAYERSIZE,  INPUTSIZE) - 0.5)
        self.b0 = (np.random.rand(LAYERSIZE,  1) - 0.5)
        self.W1 = (np.random.rand(OUTPUTSIZE, LAYERSIZE) - 0.5)
        self.b1 = (np.random.rand(OUTPUTSIZE, 1) - 0.5)

        # TODO (optional): Normalize weights and biases.

    # ----------------------------------------------------------------------- #

    # Train ----------------------------------------------------------------- #
    def ReLU(self, arr):
        return np.maximum(arr, 0)

    def sigmoid(self, arr):
        return (np.ones_like(arr) / (1 + np.exp(-1 * arr)))

    def forwardProp(self, sample):
        '''
        Performs matrix multiplication on the sample to obtain an output
        vector.
        '''
        # print('Forward Prop')
        # print(f'  W0 shape: {self.W0.shape}')
        # print(f'  x shape:  {sample[:,np.newaxis].shape}')
        # print(f'  b0 shape: {self.b0.shape}')
        z0 = np.matmul(self.W0, sample[:,np.newaxis]) + self.b0
        a0 = self.ReLU(z0)
        # print(f'  z0 shape: {z0.shape}')
        # print()

        # print(f'  W1 shape: {self.W1.shape}')
        # print(f'  a0 shape: {a0.shape}')
        # print(f'  b1 shape: {self.b1.shape}')
        z1 = np.matmul(self.W1, a0) + self.b1
        a1 = self.sigmoid(z1)
        # print(f'  a1 shape: {a1.shape}')
        # print()

        return z0, a0, z1, a1

    def ReLUDerivative(self, arr):
        return np.array([float(x > 0) for x in arr])

    def sigmoidDerivative(self, arr):
        return self.sigmoid(arr) * (1 - self.sigmoid(arr))

    def backProp(self, X, y, z0, a0, z1, a1):
        '''
        The beast. This is the biggest headache in the class.
        '''

        # Initializing expected output vector.
        y_expected = np.zeros(OUTPUTSIZE)
        if y == 'Iris-setosa':
            y_expected[0] = 1
        elif y == 'Iris-versicolor':
            y_expected[1] = 1
        else:
            y_expected[2] = 1

        # print(a1.shape)
        # print(y_expected[:,np.newaxis].shape)

        # TODO: Calculus.
        delta1 = (a1 - y_expected[:,np.newaxis]) * self.sigmoidDerivative(a1)  # Output error
        dW1 = np.matmul(delta1, a0.T)  # Adjustments for W1
        db1 = np.sum(delta1, axis=0, keepdims=True)  # Adjustments for b1

        delta0 = np.matmul(self.W1.T, delta1) * self.ReLUDerivative(a0)[:,np.newaxis]  # Hidden layer 1 error
        
        # print(delta1.shape)
        # print(a0.shape)

        # print(f'  {np.matmul(self.W1.T, delta1).shape}')
        # print(f'  {self.ReLUDerivative(a0)[:,np.newaxis].shape}')
        # print(delta0.shape)
        # print(X[:,np.newaxis].shape)
        
        dW0 = np.matmul(delta0, X[:,np.newaxis].T)  # Adjustments for W0
        db0 = np.sum(delta0, axis=0, keepdims=True)  # Adjustments for b0

        # print(dW0.shape)
        # print(db0.shape)
        # print(dW1.shape)
        # print(db1.shape)

        return dW0, db0, dW1, db1

    def getAdjustments(self, X, y):
        '''
        For each training example, find the best adjustments you can make for
        each of our parameters. Then, return the averages of each parameter's
        adjustments.
        '''
        dW0, db0, dW1, db1 = 0.0, 0.0, 0.0, 0.0
        count = 0
        for idx, sample in enumerate(X):

            # Forward and back propogation.
            z0, a0, z1, a1 = self.forwardProp(sample)
            c_dW0, c_db0, c_dW1, c_db1= self.backProp(
                sample, y[idx], z0, a0, z1, a1
            )

            # Update adjustments.
            dW0 += c_dW0
            db0 += c_db0
            dW1 += c_dW1
            db1 += c_db1
            count += 1

        # Taking the average of all adjustments.
        dW0 /= count
        db0 /= count
        dW1 /= count
        db1 /= count

        return dW0, db0, dW1, db1

    def updateParams(self, dW0, db0, dW1, db1):
        '''
        Apply adjustments to parameters.
        '''
        self.W0 -= self.alpha * dW0
        self.b0 -= self.alpha * db0
        self.W1 -= self.alpha * dW1
        self.b1 -= self.alpha * db1

    def fit(self, X, y, iters=500):
        '''
        Primary training function. For each iteration, perform a step of
        gradient descent. Ideally, the neural network converges to something
        good.
        '''

        # For each training iteration...
        for i in range(iters):

            # Get adjustments.
            dW0, db0, dW1, db1 = self.getAdjustments(X, y)

            # Update parameters.
            self.updateParams( dW0, db0, dW1, db1 )

            # Every so often, check the current accuracy.
            if i % 100 == 0 or i == iters - 1:
                # print(f'dW2: {dW2[0,0]}')
                # print(f'W2:  {self.W2}')
                print(f'Iteration: {i}/{iters}')
                print(f'Train Acc: {self.accuracy(X, y)}')
    # ----------------------------------------------------------------------- #

    # Predicting ------------------------------------------------------------ #
    def findLabel(self, sample):
        _, _, _, a1, = self.forwardProp(sample)
        max_idx = np.argmax(a1)

        # Obtaining iris prediction
        if max_idx == 0:
            pred = 'Iris-setosa'
        elif max_idx == 1:
            pred = 'Iris-versicolor'
        elif max_idx == 2:
            pred = 'Iris-virginica'
        else:
            raise Exception

        return pred

    def predict(self, X):
        '''
        Primary prediction function.
        '''
        return np.array( [self.findLabel(x) for x in X] )

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.sum( [preds[i] == actual for i, actual in enumerate(y)] ) / len(preds)
    # ----------------------------------------------------------------------- #