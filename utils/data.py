"""
    data.py
    Open data
"""


from .util import shuffle
import numpy as np
import sklearn

def one_hot(y, classes=None):
    if classes is None:
        classes = np.max(y)+1
    y = y.reshape(-1)
    N = y.shape[0]
    y_one_hot = np.zeros((N, classes))
    y_one_hot[np.arange(N),y]=1
    return y_one_hot

def call_mnist(one_hot_coding=True):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/mnist", one_hot=False)
    train_data, train_label = shuffle(np.concatenate((mnist.train.images, mnist.validation.images),axis=0),
                                     np.concatenate((mnist.train.labels, mnist.validation.labels),axis=0))
    test_data, test_label = shuffle(mnist.test.images, mnist.test.labels)
    if one_hot_coding:
        train_label = one_hot(train_label)
    return {'train_x':train_data.reshape(-1, 28,28,1),
            'train_y':train_label,
            'test_x': test_data.reshape(-1, 28,28,1),
            'test_y': test_label}

def call_cifar10(one_hot_coding=True):
    from tensorflow.python.keras.datasets.cifar10 import load_data
    (train_data, train_label), (test_data, test_label) = load_data()
    if one_hot_coding:
        train_label = one_hot(train_label)
    train_data, train_label = shuffle(train_data, train_label)
    return {'train_x' : train_data,
            'train_y' : train_label,
            'test_x' : test_data,
            'test_y' : test_label}

def call_wine(one_hot_coding=True, train_ratio=0.7):
    from sklearn.datasets import load_wine
    dataset = load_wine()
    data = dataset['data']
    label = dataset['target']
    data, label = shuffle(data, label)
    train = int(len(data)*train_ratio)
    train_data = data[:train]
    train_label = one_hot(label[:train]) if one_hot_coding else label[:train]
    test_data = data[train:]
    test_label = label[train:]

    return {'train_x':train_data,
            'train_y':train_label,
            'test_x':test_data,
            'test_y':test_label}

def call_iris(one_hot_coding=True, train_ratio=0.7):
    from sklearn.datasets import load_iris
    dataset = load_iris()
    data = dataset['data']
    label = dataset['target']
    data, label = shuffle(data, label)
    train = int(len(data)*train_ratio)
    train_data = data[:train]
    train_label = one_hot(label[:train]) if one_hot_coding else label[:train]
    test_data = data[train:]
    test_label = label[train:]

    return {'train_x':train_data,
            'train_y':train_label,
            'test_x':test_data,
            'test_y':test_label}
