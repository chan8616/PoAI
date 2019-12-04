#  import pandas as pd  # type: ignore
import numpy as np

from keras.datasets import cifar10, mnist
from sklearn.datasets import fetch_olivetti_faces
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


class OlivettFaces(Dataset):
    NAME = 'OlivettiFaces'
    LABELS = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09',
              '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
              '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
              '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
              ]
    IMAGE_SIZE = (64, 64)

    def load_data(self):
        data = fetch_olivetti_faces()

        label = data.target.reshape(40, 10)
        images = data.images.reshape(40, 10, 64, 64)
        X_train = images[:, :9].reshape(-1, 64, 64)*255
        y_train = label[:, :9].reshape(-1)
        X_test = images[:, 9:].reshape(-1, 64, 64)*255
        y_test = label[:, 9:].reshape(-1)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        return ((np.array(np.expand_dims(X_train, 3), dtype=int),
                 np.expand_dims(y_train, 1)),
                (np.array(np.expand_dims(X_test, 3), dtype=int),
                 np.expand_dims(y_test, 1)))
