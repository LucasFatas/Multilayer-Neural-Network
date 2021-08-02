import numpy as np
from perceptron import Perceptron


class Network(object):

    def __init__(self, size, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)

        self.size = size[1:]
        self.num_inputs = size[0]
        self.num_outputs = size[-1]

        network = list()
        # Sets the weights of each perceptron in each layer of the network
        for i in range(len(self.size)):
            temp = list()
            for j in range(self.size[i]):
                if i == 0:
                    temp.append(Perceptron(np.random.randn(self.num_inputs), 0))
                else:
                    temp.append(Perceptron(np.random.randn(self.size[i - 1]), 0))

            network.append(np.array(temp))
        self.network = network

    def feedforward(self, input_variable):
        x = input_variable

        for layer in self.network:
            intermediate_results = list()   # Stores results between layers

            for perceptron in layer:
                intermediate_results.append(perceptron.prediction(x))

            x = intermediate_results
        x = x / np.sum(x)

        return np.argmax(x) + 1

    def intermediate_feedforward(self, input_variable):
        result_list = list()

        temp_result = input_variable
        result_list.append(temp_result)

        for layer in self.network:
            intermediate_results = list()

            for perceptron in layer:
                intermediate_results.append(perceptron.prediction(temp_result))

            temp_result = intermediate_results
            result_list.append(intermediate_results)

        return result_list

    def training(self, training_data, alpha, test_data):
        for x_train, y_train in training_data:
            intermediate_results = self.intermediate_feedforward(x_train)
            prediction_array = intermediate_results[-1]  # Last layer output
            prediction_array = prediction_array / np.sum(prediction_array)  # Normalizes the array

            reverse_network = self.network[:: -1]  # Reverses the network
            reverse_iv = intermediate_results[:: -1]  # Reverses the intermediate results

            delta_matrix = list()

            for layer_index, layer in enumerate(reverse_network):

                delta_array = list()

                for p_index, p in enumerate(layer):
                    w = p.get_weights()  # Gets the weights of a given perceptron at the layer

                    # Retrieves the result of by the previous layer (reversed)
                    i_result = np.array(reverse_iv[layer_index + 1])

                    # Computes the y - y_hat (necessary only for the results of the last layer)
                    if layer_index == 0:
                        derivative = vectorize(y_train, self.num_outputs)[p_index] - prediction_array[p_index]
                        derivative *= sigmoid_prime(prediction_array[p_index])
                        delta_array.append(derivative)
                    else:
                        weights = list()

                        # Gets the weights for each perceptron from the previous layer
                        for weight_perceptron in reverse_network[layer_index - 1]:
                            weights.append(weight_perceptron.get_weights()[p_index])

                        weights = np.array(weights)
                        previous_deltas = delta_matrix[layer_index - 1]  # Stores the deltas of the previous layer
                        weights_delta = np.dot(weights, previous_deltas)

                        # Result this perceptron produced
                        current_perceptron_result = reverse_iv[layer_index][p_index]

                        # Computes the delta for this perceptron
                        delta_value = sigmoid_prime(current_perceptron_result) * weights_delta

                        delta_array.append(delta_value)
                        derivative = delta_value

                    new_weights = w + alpha * derivative * i_result
                    new_bias = p.get_bias() + alpha * derivative
                    p.update(new_weights, new_bias)

                # Calculates the average derivative of the layer and increases the overall derivative by that
                delta_array = np.array(delta_array)
                delta_matrix.append(np.array(delta_array))

        return self.accuracy(test_data)

    def accuracy(self, test_data):
        accuracy = 0

        # confusion_matrix = np.zeros((7, 7))

        for predict, expected in test_data:
            temp_train = predict

            for index, layer in enumerate(self.network):
                temp_train = np.array([p.prediction(temp_train) for p in layer])

            y_hat = np.argmax(temp_train) + 1  # Finds the index of the max element

            if y_hat == expected:
                accuracy += 1
            # Computes the confusion matrix
            # else:
            #     confusion_matrix[int(expected - 1)][int(y_hat - 1)] += 1

        return accuracy / len(test_data) * 100  # , confusion_matrix / len(test_data)


def vectorize(value, size):
    array = np.zeros((size,))
    array[int(value) - 1] = 1
    return array


def sigmoid_prime(z):
    return z * (1 - z)
