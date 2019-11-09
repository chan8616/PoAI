import os

from model.keras_applications.train_config import (WEIGHTS,
                                                   LOSSES)
from generator.image_classification.config_samples import DIR_GEN_CIFAR10
from .train_config import InceptionV3TrainConfig
from .build_config import InceptionV3Config


class InceptionV3ImagenetConfig(
        InceptionV3Config, InceptionV3TrainConfig):
    NAME = os.path.join(InceptionV3Config.NAME, 'imagenet')
    CLASSES = 1000

    LOSS = LOSSES[2]


class InceptionV3CIFAR10Config(
        InceptionV3Config,
        InceptionV3TrainConfig,
        DIR_GEN_CIFAR10,
        ):
    NAME = os.path.join(InceptionV3Config.NAME, DIR_GEN_CIFAR10.NAME)

    WEIGHT = WEIGHTS[0]  # type: ignore
    #  EPOCHS = 10

    LOSS = LOSSES[2]

    def __init__(self):
        super(InceptionV3CIFAR10Config, self).__init__()
        self.CLASSES = len(self.LABELS)
        self.TARGET_SIZE = self.INPUT_SHAPE[:2]
