import os

from .build_config import XceptionConfig
from model.keras_applications.train_config import (TrainConfig,
                                                   WEIGHTS,
                                                   LOSSES)
from generator.image_classification.config_samples import DIR_GEN_CIFAR10


class XceptionImagenetConfig(
        XceptionConfig, TrainConfig):
    NAME = os.path.join(XceptionConfig.NAME, 'imagenet')
    CLASSES = 1000

    LOSS = LOSSES[2]


class XceptionCIFAR10Config(
        XceptionConfig,
        TrainConfig,
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
