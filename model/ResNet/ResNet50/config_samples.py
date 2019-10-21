from .build_config import ResNet50Config
from model.keras_applications.train_config import (TrainConfig,
                                                   WEIGHTS,
                                                   LOSSES)
#  from ..keras_applications.generator import TrainConfig, LOSSES
#  from ..keras_applications.config_samples import XceptionImagenetConfig
#  from generator.image_classification.image_classification_generator \
#          import DGC_CIFAR10


class ImagenetConfig(
        ResNet50Config, TrainConfig):
    CLASSES = 1000

    LOSS = LOSSES[2]


class CIFAR10Config(ResNet50Config, TrainConfig):
    CLASSES = 10

    WEIGHT = WEIGHTS[0]  # type: ignore
    EPOCHS = 200

    LOSS = LOSSES[2]

    #  def __init__(self):
    #      super(CIFAR10Config, self).__init__()
    #      self.TARGET_SIZE = self.INPUT_SHAPE[:2]
