import os

from model.keras_applications.train_config import (WEIGHTS,
                                                   LOSSES, OPTIMIZERS)
from generator.image_classification.config_samples import (DIR_GEN_CIFAR10,
                                                           DIR_GEN_OlivettiFaces,
                                                           COLOR_MODES,)
from .train_config import DenseNet201TrainConfig
from .build_config import DenseNet201Config
from model.keras_applications.test_config import TestConfig


class DenseNet201TestConfig(TestConfig):
    NAME = 'DenseNet201'


class DenseNet201ImagenetConfig(
        DenseNet201Config, DenseNet201TrainConfig):
    NAME = os.path.join(DenseNet201Config.NAME, 'imagenet')
    CLASSES = 1000

    LOSS = LOSSES[2]


class DenseNet201CIFAR10Config(
        DenseNet201Config,
        DenseNet201TrainConfig,
        DIR_GEN_CIFAR10,
        ):
    NAME = os.path.join(DenseNet201Config.NAME, DIR_GEN_CIFAR10.NAME)

    WEIGHT = WEIGHTS[0]  # type: ignore
    #  EPOCHS = 10

    LOSS = LOSSES[2]

    def __init__(self):
        super(DenseNet201CIFAR10Config, self).__init__()
        self.CLASSES = len(self.LABELS)
        self.TARGET_SIZE = self.INPUT_SHAPE[:2]

class DenseNet201OlivettiFacesConfig(
        DenseNet201Config,
        DenseNet201TrainConfig,
        DIR_GEN_OlivettiFaces,
        ):
    NAME = os.path.join(DenseNet201Config.NAME, DIR_GEN_OlivettiFaces.NAME)

    INPUT_SHAPE = (64, 64, 3)  # type: ignore

    HIDDEN_LAYERS = [256, 256]

    WEIGHT = WEIGHTS[0]  # type: ignore
    EPOCHS = 20
    #  VALIDATION_STEPS = 10

    LOSS = LOSSES[2]

    OPTIMIZER = OPTIMIZERS[1]
    LEARNING_RATE = 1e-4
    LEARNING_MOMENTUM = 0.0

    def __init__(self):
        super(DenseNet201OlivettiFacesConfig, self).__init__()
        self.CLASSES = len(self.LABELS)
        self.TARGET_SIZE = self.INPUT_SHAPE[:2]
        self.COLOR_MODE = COLOR_MODES[1]