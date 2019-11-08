import os

from model.keras_applications.train_config import (WEIGHTS,
                                                   LOSSES)
from generator.image_classification.config_samples import DIR_GEN_CIFAR10
from .train_config import ResNet50TrainConfig
from .build_config import ResNet50Config


class ResNet50ImagenetConfig(
        ResNet50Config, ResNet50TrainConfig):
    NAME = os.path.join(ResNet50Config.NAME, 'imagenet')
    CLASSES = 1000

    LOSS = LOSSES[2]


class ResNet50CIFAR10Config(
        ResNet50Config,
        ResNet50TrainConfig,
        DIR_GEN_CIFAR10,
        ):
    NAME = os.path.join(ResNet50Config.NAME, DIR_GEN_CIFAR10.NAME)

    WEIGHT = WEIGHTS[0]  # type: ignore
    #  EPOCHS = 10

    LOSS = LOSSES[2]

    def __init__(self):
        super(ResNet50CIFAR10Config, self).__init__()
        self.CLASSES = len(self.LABELS)
        self.TARGET_SIZE = self.INPUT_SHAPE[:2]
