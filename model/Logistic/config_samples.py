import os

from generator.image_classification.config_samples import DIR_GEN_MNIST
from .train_config import (LogisticTrainConfig,
                           WEIGHTS, LOSSES, OPTIMIZERS)
from .build_config import LogisticConfig, POOLINGS


class LogisticMNISTConfig(
        LogisticConfig,
        LogisticTrainConfig,
        DIR_GEN_MNIST,
        ):
    NAME = os.path.join(LogisticConfig.NAME, DIR_GEN_MNIST.NAME)

    INPUT_SHAPE = (28, 28, 1)  # type: ignore

    POOLING = POOLINGS[0]
    HIDDEN_LAYERS = [256, 256]

    WEIGHT = WEIGHTS[0]  # type: ignore
    EPOCHS = 5

    LOSS = LOSSES[1]

    OPTIMIZER = OPTIMIZERS[0]
    LEARNING_RATE = 1e-1
    LEARNING_MOMENTUM = 0.0

    def __init__(self):
        super(LogisticMNISTConfig, self).__init__()
        self.CLASSES = len(self.LABELS)
        self.TARGET_SIZE = self.INPUT_SHAPE[:2]
