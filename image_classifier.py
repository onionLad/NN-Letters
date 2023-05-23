#
# image_classifier.py
#   Bill Xia
#   5/17/23
#
# This file uses my neural network implementation to classify images of
# handwritten capital letters from the English alphabet. 
#

# Imports
import os
import numpy as np
import pandas as pd
from implementation import ImgClassifier

# Globals
RANDOMSTATE = 10

# Functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# A function that organizes all of our text files into an array of pairs,
# where each pair contains 588 pixel values and a letter label.
def unpackData():

    # This list will store image-label pairs and will be returned at the end
    # of the function
    data = []

    # Obtaining an array of image names and labels
    labels  = pd.read_csv('data/labels.csv', delimiter=',')
    nlArray = labels.values

    # Use image names to obtain pixel arrays and store those arrays with their
    # labels in data.
    for pair in nlArray:
        px = pd.read_csv(pair[0]).values.flatten()
        data.append((px, pair[1]))

    return data

# A function that splits an array of feature-label pairs into two feature
# arrays and two label arrays.
def splitData(data, randomState=10):

    xtr, ytr, xte, yte = np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)

    np.random.seed(randomState)
    np.random.shuffle(data)

    trainTestSplit = int(len(data) * 0.8)
    train = data[:trainTestSplit]
    test  = data[trainTestSplit:]

    xtr = np.array([ px for px, lb in train ])
    ytr = np.array([ lb for px, lb in train ])
    xte = np.array([ px for px, lb in test ])
    yte = np.array([ lb for px, lb in test ])

    return xtr, ytr, xte, yte

# A function that determines the accuracy of a set of our neural network's
# outputs compared to the corresponding actual values.
def calcAccuracy(actual, expected):
    # We return (# of correct labels) / (# of labels).
    # return np.sum([expected[i] == x for i, x in enumerate(actual)]) / float(len(expected))
    return np.sum(actual == expected) / float(len(expected))

# Body  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Obtaining and splitting data into training and testing sets
data = unpackData()
# print('Unpacked Data')
X_train, y_train, X_test, y_test = splitData(data, randomState=RANDOMSTATE)
# print('Split Data')

# Generating and training the neural network classifier
classifier = ImgClassifier(alpha=0.1, randomState=RANDOMSTATE)
classifier.fit(X_train, y_train, numBatches=1, iterations=500)

# # Testing the trained classifier and displaying its accuracy
# predictions = classifier.predict(X_test)
print('Train Accuracy: ' + str(round(classifier.accuracy(X_train, y_train) * 100, 2)) + '%')
print('Test Accuracy: ' + str(round(classifier.accuracy(X_test, y_test) * 100, 2)) + '%')

