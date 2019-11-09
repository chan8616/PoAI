import os

from model.keras_applications.train_config import (WEIGHTS,
                                                   LOSSES)
from generator.image_classification.config_samples import DIR_GEN_CIFAR10
from .train_config import VGG19TrainConfig
from .build_config import VGG19Config


class VGG19ImagenetConfig(
        VGG19Config, VGG19TrainConfig):
    NAME = os.path.join(VGG19Config.NAME, 'imagenet')
    CLASSES = 1000

    LOSS = LOSSES[2]


class VGG19CIFAR10Config(
        VGG19Config,
        VGG19TrainConfig,
        DIR_GEN_CIFAR10,
        ):
    NAME = os.path.join(VGG19Config.NAME, DIR_GEN_CIFAR10.NAME)

    WEIGHT = WEIGHTS[0]  # type: ignore
    #  EPOCHS = 10

    LOSS = LOSSES[2]

    def __init__(self):
        super(VGG19CIFAR10Config, self).__init__()
        self.CLASSES = len(self.LABELS)
        self.TARGET_SIZE = self.INPUT_SHAPE[:2]
