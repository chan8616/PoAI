from .image_classification_generator_config import (
        COLOR_MODES,
        ImageClassificationGeneratorConfig,
        )

from .generator_config_samples import (
        GEN_CIFAR10, GEN_MNIST, GEN_OlivettiFaces)
from .directory_dataset.dataset_config_samples import (
        DIR_CIFAR10, DIR_MNIST, DIR_OlivettiFaces)


class DIR_GEN_CIFAR10(DIR_CIFAR10,
                      GEN_CIFAR10):
    def __init__(self):
        super(DIR_GEN_CIFAR10, self).__init__()


class DIR_GEN_MNIST(DIR_MNIST,
                    GEN_MNIST):
    def __init__(self):
        super(DIR_GEN_MNIST, self).__init__()


"""
class ICGC_CIFAR100(ImageClassificationGeneratorConfig):
    TARGET_SIZE = (32, 32)


class ICGC_FashionMNIST(ImageClassificationGeneratorConfig):
    TARGET_SIZE = (28, 28)
    COLOR_MODE = COLOR_MODES[0]
"""

class DIR_GEN_OlivettiFaces(DIR_OlivettiFaces,
                            GEN_OlivettiFaces):
    def __init__(self):
        super(DIR_GEN_OlivettiFaces, self).__init__()
