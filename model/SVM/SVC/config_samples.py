import os

from .build_config import SVCBuildConfig
from .train_config import SVCTrainConfig
from .generator_config import SVCGeneratorConfig


class SVCIRISBuildConfig(
        SVCBuildConfig,
        ):
    BUILD_NAME = os.path.join(SVCBuildConfig.BUILD_NAME, 'iris')

    def __init__(self):
        super(SVCIRISBuildConfig, self).__init__()


class SVCIRISTrainConfig(
        SVCTrainConfig,
        ):
    TRAIN_NAME = os.path.join(SVCTrainConfig.TRAIN_NAME, 'iris')

    def __init__(self):
        super(SVCIRISTrainConfig, self).__init__()


class SVCIRISConfig(
        SVCGeneratorConfig,
        ):
    NAME = os.path.join(SVCGeneratorConfig.NAME, 'iris')
    def __init__(self):
        self.X_COL = ['sepal length (cm)', 'sepal width (cm)',
                      'petal length (cm)', 'petal width (cm)'],
        self.Y_COL = ['label']
        self.DATAFRAME_PATH = 'dataset/iris/iris.csv'
        self.VALID_DATAFRAME_PATH = 'dataset/iris/iris.csv'
