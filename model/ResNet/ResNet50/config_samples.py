import os
from .build_config import ResNet50Config
from model.keras_applications.train_config import (TrainConfig,
                                                   WEIGHTS,
                                                   LOSSES)
#  from ..keras_applications.generator import TrainConfig, LOSSES
#  from ..keras_applications.config_samples import XceptionImagenetConfig
#  from generator.image_classification.directory.config_samples import (
#          DirectoryGeneratorConfig,
#          DGC_CIFAR10,
#          DGC_CIFAR100,
#          DGC_MNIST,
#          DGC_FashionMNIST,
#          )

#  from generator.image_classification.dataset \
#          import CIFAR10
from generator.image_classification.config_samples import DIR_GEN_CIFAR10
#  from generator.image_classification.directory_dataset.config_samples \
#          import CIFAR10 as CIFAR10


class ResNet50ImagenetConfig(
        ResNet50Config, TrainConfig):
    NAME = os.path.join(ResNet50Config.NAME, 'imagenet')
    CLASSES = 1000

    LOSS = LOSSES[2]


class ResNet50CIFAR10Config(
        ResNet50Config,
        TrainConfig,
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
