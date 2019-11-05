#  import pandas as pd  # type: ignore

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

from ..generator_config import GeneratorConfig


class Dataset(GeneratorConfig):
    NAME = 'Dataset'
    LABELS = ['label']
    IMAGE_SIZE = (256, 256)  # ('height', 'width')

    def load_data(self):
        ...


class CIFAR10(Dataset):
    NAME = 'CIFAR10'
    LABELS = ('airplane automobile bird cat deer '
              'dog frog horse ship truck ').split()
    IMAGE_SIZE = (32, 32)

    def load_data(self):
        return cifar10.load_data()
