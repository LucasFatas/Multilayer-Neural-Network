from sklearn.model_selection import train_test_split
from Assignment_1.src.cross_validation import cross_validation
import neural_network
import logic_gates_perceptron

import numpy as np


def test_network_and():
    print("AND")
    training_data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
    test_data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1), ([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1),
                 ([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1), ([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1),
                 ([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1), ([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1),
                 ([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1), ([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1),
                 ([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1), ([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1),
                 ([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1), ([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1),
                 ([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1), ([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1),
                 ([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1), ([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1),
                 ([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1), ([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
    net = logic_gates_perceptron.LogicGateNetwork()
    net.training(training_data, 10, 1, test_data)


def test_network_or():
    print("OR")
    training_data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)]
    test_data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1), ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1),
                 ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1), ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1),
                 ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1), ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1),
                 ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1), ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1),
                 ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1), ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1),
                 ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1), ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1),
                 ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1), ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1),
                 ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1), ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1),
                 ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1), ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)]
    net = logic_gates_perceptron.LogicGateNetwork()
    net.training(training_data, 10, 1, test_data)


def test_network_xor():
    print("XOR")
    training_data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
    test_data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0), ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0),
                 ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0), ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0),
                 ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0), ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0),
                 ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0), ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0),
                 ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0), ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0),
                 ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0), ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0),
                 ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0), ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0),
                 ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0), ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0),
                 ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0), ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
    net = logic_gates_perceptron.LogicGateNetwork()
    net.training(training_data, 10, 1, test_data)


def network_assignment1():
    features = np.genfromtxt("../data/features.txt", delimiter=",")
    targets = np.genfromtxt("../data/targets.txt", delimiter=",")

    test_fraction = 0.15  # Feel free to change
    random_seed = 30

    x, x_test, y, y_test = train_test_split(features, targets, test_size=test_fraction)

    network = neural_network.Network([10, 23, 7], random_seed)  # creates network

    cross_validation(x, x_test, y, y_test, network, 2)  # trains network

    exit()

    # network.feedforward("PUT UNKNOWN DATA HERE") # uses network

    # unknown_predictions(network)
    # acc, cm = network.accuracy(list(zip(x_test, y_test)))     # Used when we need the confusion matrix


def unknown_predictions(network):
    unknown = np.genfromtxt("../data/unknown.txt", delimiter=",")
    unknown_ys = [network.feedforward(x) for x in unknown]
    np.savetxt("Group_23_classes.txt", unknown_ys, fmt='%i', newline=",")


if __name__ == '__main__':
    network_assignment1()
