import os

from model.keras_applications.train_config import (WEIGHTS,
                                                   LOSSES)
from generator.image_classification.config_samples import DIR_GEN_CIFAR10
from .train_config import VGG16TrainConfig
from .build_config import VGG16Config


class VGG16ImagenetConfig(
        VGG16Config, VGG16TrainConfig):
    NAME = os.path.join(VGG16Config.NAME, 'imagenet')
    CLASSES = 1000

    LOSS = LOSSES[2]


class VGG16CIFAR10Config(
        VGG16Config,
        VGG16TrainConfig,
        DIR_GEN_CIFAR10,
        ):
    NAME = os.path.join(VGG16Config.NAME, DIR_GEN_CIFAR10.NAME)

    WEIGHT = WEIGHTS[0]  # type: ignore
    #  EPOCHS = 10

    LOSS = LOSSES[2]

    def __init__(self):
        super(VGG16CIFAR10Config, self).__init__()
        self.CLASSES = len(self.LABELS)
        self.TARGET_SIZE = self.INPUT_SHAPE[:2]
