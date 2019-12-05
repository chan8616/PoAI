import os

from model.keras_applications.train_config import (WEIGHTS,
                                                   LOSSES, OPTIMIZERS)
from generator.image_classification.config_samples import (DIR_GEN_CIFAR10,
                                                           DIR_GEN_OlivettiFaces,
                                                           COLOR_MODES,)
from .train_config import XceptionTrainConfig
from .build_config import XceptionConfig

from model.keras_applications.test_config import TestConfig


class XceptionTestConfig(TestConfig):
    NAME = 'Xception'


class XceptionImagenetConfig(
        XceptionConfig, XceptionTrainConfig):
    NAME = os.path.join(XceptionConfig.NAME, 'imagenet')
    CLASSES = 1000

    LOSS = LOSSES[2]


class XceptionCIFAR10Config(
        XceptionConfig,
        XceptionTrainConfig,
        DIR_GEN_CIFAR10,
        ):
    NAME = os.path.join(XceptionConfig.NAME, DIR_GEN_CIFAR10.NAME)

    WEIGHT = WEIGHTS[0]  # type: ignore
    #  EPOCHS = 10
    LOSS = LOSSES[2]

    def __init__(self):
        super(XceptionCIFAR10Config, self).__init__()
        self.CLASSES = len(self.LABELS)
        self.TARGET_SIZE = self.INPUT_SHAPE[:2]


class XceptionOlivettiFacesConfig(
        XceptionConfig,
        XceptionTrainConfig,
        DIR_GEN_OlivettiFaces,
        ):
    NAME = os.path.join(XceptionConfig.NAME, DIR_GEN_OlivettiFaces.NAME)

    INPUT_SHAPE = (128, 128, 3)  # type: ignore

    HIDDEN_LAYERS = [256, 256]

    WEIGHT = WEIGHTS[0]  # type: ignore
    EPOCHS = 20
    #  VALIDATION_STEPS = 10

    LOSS = LOSSES[2]

    OPTIMIZER = OPTIMIZERS[1]
    LEARNING_RATE = 1e-4
    LEARNING_MOMENTUM = 0.0

    def __init__(self):
        super(XceptionOlivettiFacesConfig, self).__init__()
        self.CLASSES = len(self.LABELS)
        self.TARGET_SIZE = self.INPUT_SHAPE[:2]
        self.COLOR_MODE = COLOR_MODES[1]