#
# compress_data.py
#   Bill Xia
#   5/17/23
#
# This program is meant to shrink the 1200px x 900px images in data/Img into
# smaller, more useable images.
#
# This file was created back when we were trying to classify image data rather
# than iris data.
#

# Imports
import os
from PIL import Image
import numpy as np

# Globals
FINSIZE = (28, 21)

# Functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Name:    compressImage
# Purpose: Shrinks a given image to a desired size and exports it to the
#          desired location.
# Inputs:  An image's filepath, desired size, and export destination.
# Output:  A compressed image.
def compressImage(fp, finalSize, dest):
    img = Image.open(fp)
    img = img.resize(finalSize)
    img.save(dest, 'PNG')

# Name:    toTextFile
# Purpose: Converts a given image into a text file containing a list of that
#          image's pixels, where each pixel is represented by a float value
#          between 0 and 1. 
# Inputs:  An image's name, filepath, and export destination.
# Output:  A text file created using an image.
def toTextFile(fn, fp, dest):

    # Creating an array representing the original image.
    img = Image.open(fp)
    imgAsArray = np.array(img)

    # Converting that array into a 1D array.
    img_1D = np.zeros(0)
    for row in imgAsArray:
        for pix in row:
            img_1D = np.append(img_1D, (pix[0] / 255))

    # Saving 1D array to a file.
    destFp = os.path.join(dest, fn.split('.')[0] + '.csv')
    np.savetxt(destFp, img_1D, delimiter=',', fmt='%.8f')

# Body  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Obtaining relevant directories
srcDirName  = os.path.join('data', 'Img')
sourceDir   = os.listdir(srcDirName)
destDirName = os.path.join('data', 'Img_small')
destDir     = os.listdir(destDirName)
txtDirName  = os.path.join('data', 'Img_txts')

# COMPRESSION
#   Looping across all images in the source directory, compressing all images
#   that we want to keep in the final dataset. 
#   We want to keep images whose names start with 'img011' to 'img036'.
for fn in sourceDir:
    label = int(fn.split("-")[0][3:])
    if label >= 11 and label <= 36:
        fp   = os.path.join(srcDirName, fn)
        dest = os.path.join(destDirName, fn)
        compressImage(fp, FINSIZE, dest)

# TEXT CONVERSION
#   Looping across all images in the destination directory, converting all
#   images into text files. These text files will each contain a list of 
#   every pixel in a given image. Each pixel will be represented as a float
#   value between 0 and 1.
for fn in destDir:
    fp = os.path.join(destDirName, fn)
    toTextFile(fn, fp, txtDirName)

