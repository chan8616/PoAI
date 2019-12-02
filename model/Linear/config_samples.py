import os

from .build_config import LinearBuildConfig
from .train_config import (LinearTrainConfig,
                           WEIGHTS, LOSSES, OPTIMIZERS)
from .generator_config import LinearGeneratorConfig

class LinearSINEBuildConfig(
        LinearBuildConfig,
        ):
    BUILD_NAME = os.path.join(LinearBuildConfig.BUILD_NAME, 'sine')

    FLATTEN_INPUT_SHAPE = [1]  # type: ignore

    #  HIDDEN_LAYERS = []
    def __init__(self):
        super(LinearSINEBuildConfig, self).__init__()
        self.TARGET_SIZE = 1


class LinearSINETrainConfig(
        LinearTrainConfig,
        ):
    TRAIN_NAME = os.path.join(LinearTrainConfig.TRAIN_NAME, 'sine')

    WEIGHT = WEIGHTS[0]
    EPOCHS = 5

    LOSS = LOSSES[0]

    OPTIMIZER = OPTIMIZERS[0]
    LEARNING_RATE = 1e-1
    LEARNING_MOMENTUM = 0.0

    def __init__(self):
        super(LinearSINETrainConfig, self).__init__()


class LinearSINEConfig(
        LinearGeneratorConfig,
        ):
    NAME = os.path.join(LinearGeneratorConfig.NAME, 'sine')
    def __init__(self):
        self.X_COL = 'x'
        self.Y_COL = 'y'
        self.DATAFRAME_PATH = 'dataset/sine.csv'
