import os

from .build_config import RidgeBuildConfig
from .train_config import RidgeTrainConfig
from .generator_config import RidgeGeneratorConfig


class RidgeBostonHousePricesBuildConfig(
        RidgeBuildConfig,
        ):
    NAME = os.path.join(RidgeBuildConfig.NAME, 'boston_house_prices')
    BUILD_NAME = os.path.join(RidgeBuildConfig.BUILD_NAME, 'boston_house_prices')

    def __init__(self):
        super(RidgeBostonHousePricesBuildConfig, self).__init__()
        self.TARGET_SIZE = 1


class RidgeBostonHousePricesTrainConfig(
        RidgeTrainConfig,
        ):
    TRAIN_NAME = os.path.join(RidgeTrainConfig.TRAIN_NAME, 'boston_house_prices')

    def __init__(self):
        super(RidgeBostonHousePricesTrainConfig, self).__init__()


class RidgeBostonHousePricesConfig(
        RidgeGeneratorConfig,
        ):
    NAME = os.path.join(RidgeGeneratorConfig.NAME, 'boston_house_prices')
    def __init__(self):
        self.X_COL = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]
        self.Y_COL = ['MEDV']
        self.DATAFRAME_PATH = 'dataset/boston_house_prices.csv'
        self.VALID_DATAFRAME_PATH = 'dataset/boston_house_prices.csv'
