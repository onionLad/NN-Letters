#
# compress_data.py
#   Bill Xia
#   5/17/23
#
# This program is meant to shrink the 1200px x 900px images in data/Img into
# smaller, more useable images. 
#

# Imports
import os
from PIL import Image

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

# Body  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Obtaining relevant directories
srcDirName  = os.path.join('data', 'Img')
sourceDir   = os.listdir(srcDirName)
destDirName = os.path.join('data', 'Img_small')

# Looping across all images in the source directory, compressing all images
# that we want to keep in the final dataset. 
# We want to keep images whose names start with 'img011' to 'img036'.
for fn in sourceDir:
    label = int(fn.split("-")[0][3:])
    if label >= 11 and label <= 36:
        fp   = os.path.join(srcDirName, fn)
        dest = os.path.join(destDirName, fn)
        compressImage(fp, FINSIZE, dest)

