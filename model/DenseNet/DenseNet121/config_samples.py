import os

from model.keras_applications.train_config import (WEIGHTS,
                                                   LOSSES)
from generator.image_classification.config_samples import DIR_GEN_CIFAR10
from .train_config import DenseNet121TrainConfig
from .build_config import DenseNet121Config


class DenseNet121ImagenetConfig(
        DenseNet121Config, DenseNet121TrainConfig):
    NAME = os.path.join(DenseNet121Config.NAME, 'imagenet')
    CLASSES = 1000

    LOSS = LOSSES[2]


class DenseNet121CIFAR10Config(
        DenseNet121Config,
        DenseNet121TrainConfig,
        DIR_GEN_CIFAR10,
        ):
    NAME = os.path.join(DenseNet121Config.NAME, DIR_GEN_CIFAR10.NAME)

    WEIGHT = WEIGHTS[0]  # type: ignore
    #  EPOCHS = 10

    LOSS = LOSSES[2]

    def __init__(self):
        super(DenseNet121CIFAR10Config, self).__init__()
        self.CLASSES = len(self.LABELS)
        self.TARGET_SIZE = self.INPUT_SHAPE[:2]
