import numpy as np


class LogicGateNetwork(object):

    def __init__(self):
        self.bias = np.random.randn()
        self.weights = [np.random.randn(), np.random.randn()]

    def feedforward(self, a):
        return activation_f(self.weights[0] * a[0] + self.weights[1] * a[1] + self.bias)

    def training(self, training_data, epochs, step, test_data):
        training_data = list(training_data)

        for j in range(epochs):
            for x, y in training_data:
                z = self.weights[0] * x[0] + self.weights[1] * x[1] + self.bias
                activation = activation_f(z)
                l = y - activation
                self.weights[0] = self.weights[0] + step * x[0] * l
                self.weights[1] = self.weights[1] + step * x[1] * l
                self.bias = self.bias + step * l
            print("Epoch {} : Accuracy {}".format(j, self.accuracy(test_data)))

    def accuracy(self, test_data):
        test_results = [(self.feedforward(x), y) for (x, y) in test_data]
        accuracy = 0
        for x, y in test_results:
            if x == y:
                accuracy += 1

        return accuracy / len(test_data) * 100


def activation_f(z):
    return 1 if z > 0 else 0
