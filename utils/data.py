"""
    data.py
    Open data
"""

from .util import shuffle
import numpy as np

def call_mnist(one_hot=True):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/mnist", one_hot=True)
    train_data, train_label = shuffle(np.concatenate((mnist.train.images, mnist.validation.images),axis=0),
                                     np.concatenate((mnist.train.labels, mnist.validation.labels),axis=0))
    test_data, test_label = shuffle(mnist.test.images, mnist.test.labels)
    return train_data.reshape(-1, 28,28,1), train_label, test_data.reshape(-1, 28,28,1), test_label
