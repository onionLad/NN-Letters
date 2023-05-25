#
# verify_txts.py
#   Bill Xia
#   5/24/23
#
# This file reads in a txt file created using an image file and confirms that
# the txt file is an accurate representation of the image file.
#

# Impmorts
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Body - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

sample = pd.read_csv('data/Img_txts/img011-001.csv').values.flatten()
sample = np.atleast_2d(np.append(sample, [1])).reshape(21,28)
print(sample)
plt.imshow(sample)
plt.show()

