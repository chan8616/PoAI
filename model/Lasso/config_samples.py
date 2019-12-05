import os

from .build_config import LassoBuildConfig
from .train_config import LassoTrainConfig
from .generator_config import LassoGeneratorConfig


class LassoBostonHousePricesBuildConfig(
        LassoBuildConfig,
        ):
    NAME = os.path.join(LassoBuildConfig.NAME, 'boston_house_prices')
    BUILD_NAME = os.path.join(LassoBuildConfig.BUILD_NAME, 'boston_house_prices')

    def __init__(self):
        super(LassoBostonHousePricesBuildConfig, self).__init__()
        self.TARGET_SIZE = 1


class LassoBostonHousePricesTrainConfig(
        LassoTrainConfig,
        ):
    TRAIN_NAME = os.path.join(LassoTrainConfig.TRAIN_NAME, 'boston_house_prices')

    def __init__(self):
        super(LassoBostonHousePricesTrainConfig, self).__init__()


class LassoBostonHousePricesConfig(
        LassoGeneratorConfig,
        ):
    NAME = os.path.join(LassoGeneratorConfig.NAME, 'boston_house_prices')
    def __init__(self):
        self.X_COL = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]
        self.Y_COL = ['MEDV']
        self.DATAFRAME_PATH = 'dataset/boston_house_prices.csv'
        self.VALID_DATAFRAME_PATH = 'dataset/boston_house_prices.csv'
