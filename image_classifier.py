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
from implementation import ImgClassifier

# Globals
DATADIR = 'data/Img_txts'

# Functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# A function that organizes all of our text files into an array of pairs,
# where each pair contains 588 pixel values and a letter label.
def unpackData():
    return None

# A function that splits an array of feature-label pairs into two feature
# arrays and two label arrays.
def splitData(data):
    return np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)

# A function that determines the accuracy of a set of our neural network's
# outputs compared to the corresponding actual values.
def calcAccuracy(actual, expected):
    return None

# Body  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Obtaining and splitting data into training and testing sets
data = unpackData()
X_train, y_train, X_test, y_test = splitData(data)

# Generating and training the neural network classifier
classifier = ImgClassifier(alpha=0.1)
classifier.fit(X_train, y_train)

# # Testing the trained classifier and displaying its accuracy
# predictions = classifier.predict(X_test)
# print('Classifier Accuracy: ' + calcAccuracy(predictions, y_test))

