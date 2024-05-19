# NN-Letters
A neural network designed to classify flowers found in the iris dataset. We started this project in May of 2023. Back then, we began with the much more ambitious task of classifying hand-written letters (hence this repo's name). We reattempted the project in May of 2024 using the iris dataset instead.

## Inspiration
This project was inspired by Tufts University's CS 131 final project, where students are tasked with implementing a neural network from scratch to classify flowers in the iris dataset. Although we never took the course, we were interested in the assignment and decided to give it a shot ourselves.

## Performance
Our model from 2023, whose purpose was to classify images, was never able to successfully perform classification. In our reattempt, we were able to build a complete, single-hidden-layer neural network that classifies flowers with a 96.67% accuracy rate.

## Files
- `implementation_2.py` contains our neural network class, called `FlowersClassifier`. Our neural network architecture has a single, 32-node hidden layer used to separate our input and output layers.
- `eval_imp_2.ipynb` contains code used to evaluate our neural network's performance. It also prepocesses our data before it gets used to train and test out model.
- The `data` directory contains our flower data, used to train the neural network class. It once contained images used by the 2023 model, but those have been removed since the new model doesn't use them.
- The `old files` directory contains files we wrote in 2023. While ineffective for classification, they served as a solid foundation for us when we restarted the project in May 2024. 
