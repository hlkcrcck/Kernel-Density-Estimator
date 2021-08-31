import gzip
import sys
import pickle
import numpy as np


def read_mnist_data(mnist_path, size=10000):
    f = gzip.open(mnist_path, 'rb')
    if sys.version_info < (3,):
        data = pickle.load(f)
    else:
        data = pickle.load(f, encoding="bytes")

    training_data = data[0][0]
    np.random.shuffle(training_data)
    test_data = data[2][0]

    return training_data[:size], test_data[:size]
