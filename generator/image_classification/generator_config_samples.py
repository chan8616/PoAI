from .image_classification_generator_config import (
        COLOR_MODES,
        ImageClassificationGeneratorConfig,
        )

#  from .dataset_samples.cifar10 import CIFAR10 as CIFAR10sample
from . import dataset


class GEN_CIFAR10(dataset.CIFAR10,
                  ImageClassificationGeneratorConfig):
    def __init__(self):
        super(GEN_CIFAR10, self).__init__()
        self.TARGET_SIZE = self.IMAGE_SIZE


class GEN_MNIST(dataset.MNIST,
                ImageClassificationGeneratorConfig):
    def __init__(self):
        super(GEN_MNIST, self).__init__()
        self.TARGET_SIZE = self.IMAGE_SIZE
        self.COLOR_MODE = COLOR_MODES[0]


"""
class ICGC_CIFAR100(ImageClassificationGeneratorConfig):
    TARGET_SIZE = (32, 32)


class ICGC_FashionMNIST(ImageClassificationGeneratorConfig):
    TARGET_SIZE = (28, 28)
    COLOR_MODE = COLOR_MODES[0]
"""

class GEN_OlivettiFaces(dataset.OlivettFaces,
                        ImageClassificationGeneratorConfig):
    def __init__(self):
        super(GEN_OlivettiFaces, self).__init__()
        self.TARGET_SIZE = self.IMAGE_SIZE
        self.COLOR_MODE = COLOR_MODES[0]
