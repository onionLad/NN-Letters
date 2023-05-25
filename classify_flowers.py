#
# calssify_flowers.py
#   Bill Xia
#   5/25/23
#
# This file uses our neural network implementation to classify iris data.
#

# Impmorts
import os
import numpy as np
import pandas as pd
from implementation import ImgClassifier

# Globals
RNG = 10

# Body - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Getting data
data = pd.read_csv('data/flowers.csv', header=None)

# Shuffling and seperating data into a training set (80%) and a testing set (20%)
data = data.sample(frac=1, random_state=RNG)

split_idx = int((0.8 * len(data)))
data_tr = data[:split_idx]
data_te = data[split_idx:]

x_train = data_tr[[0, 1, 2, 3]].values
y_train = data_tr[4].values
x_test  = data_te[[0, 1, 2, 3]].values
y_test  = data_te[4].values

# Generating and training the neural network classifier
classifier = ImgClassifier(alpha=0.01, randomState=RNG)
classifier.fit(x_train, y_train, numBatches=1, iterations=500)

# Testing the trained classifier and displaying its accuracy
print('Train Accuracy: ' + str(round(classifier.accuracy(x_train, y_train) * 100, 2)) + '%')
print('Test Accuracy: ' + str(round(classifier.accuracy(x_test, y_test) * 100, 2)) + '%')

