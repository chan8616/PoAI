import os

from .build_config import LinearBuildConfig
from .train_config import (LinearTrainConfig,
                           WEIGHTS, LOSSES, OPTIMIZERS)
from .generator_config import LinearGeneratorConfig


class LinearBostonHousePricesBuildConfig(
        LinearBuildConfig,
        ):
    NAME = os.path.join(LinearBuildConfig.NAME, 'boston_house_prices')
    BUILD_NAME = os.path.join(LinearBuildConfig.BUILD_NAME, 'boston_house_prices')

    FLATTEN_INPUT_SHAPE = [13]  # type: ignore

    HIDDEN_LAYERS = [25, 25]

    def __init__(self):
        super(LinearBostonHousePricesBuildConfig, self).__init__()
        self.TARGET_SIZE = 1


class LinearBostonHousePricesTrainConfig(
        LinearTrainConfig,
        ):
    TRAIN_NAME = os.path.join(LinearTrainConfig.TRAIN_NAME, 'boston_house_prices')

    WEIGHT = WEIGHTS[0]
    EPOCHS = 20

    LOSS = LOSSES[0]

    OPTIMIZER = OPTIMIZERS[1]
    LEARNING_RATE = 1e-2
    LEARNING_MOMENTUM = 0.0

    def __init__(self):
        super(LinearBostonHousePricesTrainConfig, self).__init__()


class LinearBostonHousePricesConfig(
        LinearGeneratorConfig,
        ):
    NAME = os.path.join(LinearGeneratorConfig.NAME, 'boston_house_prices')
    def __init__(self):
        self.X_COL = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]
        self.Y_COL = ['MEDV']
        self.DATAFRAME_PATH = 'dataset/boston_house_prices.csv'
        self.VALID_DATAFRAME_PATH = 'dataset/boston_house_prices.csv'
