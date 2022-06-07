import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate
import random
from random import seed
from numpy.random import rand
from IntervalValuedKNN.interval_valued_fuzzy_set_k_neighbours import IntervalValuedFuzzyKNN

seed(1)


def prepare_data(attempts, add_missing=False, missing_rate=0.5, test_size=0.5, file_name=False):
    if not file_name:
        x, y = load_breast_cancer(return_X_y=True)
        x = x[:, :10]
    else:
        x = np.load('../data/tests/x' + str(file_name) + '.npy')
        y = np.load('../data/tests/y' + str(file_name) + '.npy')
    x = MinMaxScaler().fit_transform(x)
    print(tabulate(x))
    discretized_x = np.zeros(shape=(x.shape[0], x.shape[1], 2))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            discretized_x[i, j] = IntervalValuedFuzzyKNN.change_nr_to_interval(x[i, j])
    for i in range(attempts):
        x_train, x_test, y_train, y_test = train_test_split(discretized_x, y, test_size=test_size)
        missings = ''
        if add_missing:
            x_train = add_missing_values(x_train, missing_rate)
            x_test = add_missing_values(x_test, missing_rate)
            missings = 'Missing/'
        missings += '[test' + str(test_size) + ']'
        np.save('../data/tests/' + missings + 'X_train' + str(missing_rate) + str(i), x_train)
        np.save('../data/tests/' + missings + 'X_test' + str(missing_rate) + str(i), x_test)
        np.save('../data/tests/' + missings + 'y_train' + str(missing_rate) + str(i), y_train)
        np.save('../data/tests/' + missings + 'y_test' + str(missing_rate) + str(i), y_test)


def add_missing_values(data_set, missing_rate):
    n_samples, n_features, n = data_set.shape

    n_missing_samples = int(n_samples * missing_rate)

    missing_samples = np.zeros(n_samples, dtype=bool)
    missing_samples[: n_missing_samples] = True

    random.shuffle(missing_samples)
    missing_features = []
    for i in range(n_missing_samples):
        f2 = int(rand() * n_features)
        missing_features.append(f2)

    x_missing = data_set.copy()
    x_missing[missing_samples, missing_features] = np.nan

    return x_missing


def prep_small_data(class_obj_number):
    X, y = load_breast_cancer(return_X_y=True)
    X = X[:, :10]
    zeros, ones = 0, 0
    new_x = np.zeros(shape=(class_obj_number * 2, X.shape[1]))
    new_y = np.zeros(class_obj_number * 2)
    counts = 0
    print(new_x.shape, new_y.shape)
    for i in range(y.shape[0]):
        if (y[i] == 0 and zeros < class_obj_number) or (y[i] == 1 and ones < class_obj_number):
            new_x[counts] = X[i]
            new_y[counts] = y[i]
            counts += 1
            if y[i] == 0:
                zeros += 1
            else:
                ones += 1
    np.save('../data/tests/x' + str(class_obj_number), new_x)
    np.save('../data/tests/y' + str(class_obj_number), new_y)


if __name__ == '__main__':
    # X, y = load_breast_cancer(return_X_y=True)
    # print(type(X), X.shape, tabulate(X))
    # X = X[:, :10]
    # print(type(X), X.shape, tabulate(X))
    # prepare_data(10)

    class_obj_num = 25
    # prep_small_data(class_obj_num)

    prepare_data(1, add_missing=True, missing_rate=0.05, test_size=0.2, file_name=class_obj_num)
    prepare_data(1, add_missing=True, missing_rate=0.20, test_size=0.3, file_name=class_obj_num)
    prepare_data(1, add_missing=True, missing_rate=0.5, file_name=class_obj_num)

    # x = np.load('../data/tests/x25.npy')
    # y = np.load('../data/tests/y25.npy')
    # print(tabulate(x))
    # print(y)
    # print(x.shape, y.shape)