import os

from model.keras_applications.train_config import (WEIGHTS,
                                                   LOSSES)
from generator.image_classification.config_samples import DIR_GEN_CIFAR10
from .train_config import MobileNetTrainConfig
from .build_config import MobileNetConfig


class MobileNetImagenetConfig(
        MobileNetConfig, MobileNetTrainConfig):
    NAME = os.path.join(MobileNetConfig.NAME, 'imagenet')
    CLASSES = 1000

    LOSS = LOSSES[2]


class MobileNetCIFAR10Config(
        MobileNetConfig,
        MobileNetTrainConfig,
        DIR_GEN_CIFAR10,
        ):
    NAME = os.path.join(MobileNetConfig.NAME, DIR_GEN_CIFAR10.NAME)

    WEIGHT = WEIGHTS[0]  # type: ignore
    #  EPOCHS = 10

    LOSS = LOSSES[2]

    def __init__(self):
        super(MobileNetCIFAR10Config, self).__init__()
        self.CLASSES = len(self.LABELS)
        self.TARGET_SIZE = self.INPUT_SHAPE[:2]
