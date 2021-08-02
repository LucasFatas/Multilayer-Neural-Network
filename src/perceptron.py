import numpy as np


class Perceptron(object):

    def __init__(self, p_weights, p_bias):
        self.weights = p_weights
        self.bias = p_bias

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, new_weights, new_bias):
        self.weights = new_weights
        self.bias = new_bias

    def prediction(self, var):
        temp = np.dot(self.weights, var) + self.bias
        sigma = sigmoid(temp)
        return sigma

    def __str__(self):
        return f"Num of weights: {len(self.weights)}\tWeights{self.weights}\tBias: {self.bias}"


# Simple loss function
def loss_function(expected, predicted):
    temp = (expected - predicted)
    return 0.5 * temp * temp


# Should take a vector nx1 where n is the number of labels and output
# the vector of probabilities for each of the labels
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
