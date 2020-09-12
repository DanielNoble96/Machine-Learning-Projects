# Machine-Learning-Projects
Class projects implementing different ML techniques. I highlight some of my better ones below.

## ICA.py
Implementation of independent component analysis on 3 sound files to mix them and then separate them. Reconstructed the independent sounds using the ICA algorithm. I've attached a write-up of the project that explains the code and analyzes the results in depth.

## knn.py

Compared the results of using the k-nearest neighbours algorithm alone vs using KNN after dimensionality reduction using principal component analysis to classify the MNIST dataset (loaded using keras/tf). A write-up analyzes the two approahces, concluding slightly higher accuracy without PCA but significantly faster runtimes.

## Backpropagation.py

Implements a simple feed-forward neural network from scratch that utilizes the backprop algorithm to classify the MNIST dataset. I wrote this when I was first learning about NN's so didn't know about nuances I could incorporate to try and improve the accuracy beyond the basic feedforward and backprop, but I achieved a test accurracy of 89% in much faster time than I did with the KNN algorithm, so I'm quite happy with it. My write-up analyzes it all in detail.

## RL_GridWorld.py 

Builds a grid world with an agent that implements the Q-learning algorithm to navigate the gridworld with different objectives. The write-up compares the different policies: the agent goes through the world without going off the sidewalk, the agent goes through without hitting obstacles, and the agent goes through picking up litter. The objective is to combine all of these policies into some sub-optimal policy that considers all of them.

## GP.ipynb

Implementation of Guassian processes to model the motion data of a positional coordinate on a person's body. Ran a prediction using a global kernel and a local kerrnel and compared the results of the two in detail in the write-up!
