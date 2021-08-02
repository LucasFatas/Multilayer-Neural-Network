from sklearn.model_selection import RepeatedKFold


# this function takes inputs X(features/inputs), Y(results/labels) and k(the number of folds the validation will do)
# Using 10 folds is quite common.
# this function will call a function predict k times with a different validation and training set every time
# this predict function should create and train the neural network and then test and return its accuracy
# the k different accuracies will be averaged and printed
def cross_validation(x, x_test, y, y_test, network, epochs):
    kf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=None)

    for epoch in range(epochs):
        count = 0
        epoch_sum = 0

        for train_i, test_i in kf.split(x):
            count += 1
            x_train = x[train_i]
            x_validation = x[test_i]
            y_train = y[train_i]
            y_validation = y[test_i]
            accuracy = network.training(list(zip(x_train, y_train)), 0.001,
                                        list(zip(x_validation, y_validation)))
            epoch_sum += accuracy

        print("\nAverage Accuracy: ", epoch_sum / 10, " for epoch: ", epoch + 1)

        performance = network.accuracy(list(zip(x_test, y_test)))
        print("Performance on test set: ", performance)
