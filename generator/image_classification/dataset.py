#  import pandas as pd  # type: ignore
import numpy as np

from keras.datasets import cifar10, mnist
from keras.preprocessing.image import ImageDataGenerator

from ..generator_config import GeneratorConfig


class Dataset(GeneratorConfig):
    NAME = 'Dataset'
    LABELS = ['label']
    IMAGE_SIZE = (256, 256)  # ('height', 'width')

    def load_data(self):
        ...


class MNIST(Dataset):
    NAME = 'MNIST'
    LABELS = '0 1 2 3 4 5 6 7 8 9'.split()
    IMAGE_SIZE = (28, 28)

    def load_data(self):
        (train_x, train_y), (test_x, test_y) = mnist.load_data()
        return ((np.expand_dims(train_x, 3), np.expand_dims(train_y, 1)),
                (np.expand_dims(test_x, 3), np.expand_dims(test_y, 1)))


class CIFAR10(Dataset):
    NAME = 'CIFAR10'
    LABELS = ('airplane automobile bird cat deer '
              'dog frog horse ship truck ').split()
    IMAGE_SIZE = (32, 32)

    def load_data(self):
        return cifar10.load_data()
