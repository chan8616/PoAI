from argparse import Namespace
from pathlib import Path
from typing import Union
from gooey import Gooey, GooeyParser

import pandas as pd
from sklearn.preprocessing import StandardScaler

from .generator_config import LinearGeneratorConfig
from model.Linear.dataframe_iterator import DataFrameIterator
from model.Linear.config_samples import LinearBostonHousePricesConfig


class LinearGeneratorConfigList():
    def __init__(self,
                 linear_generator_config_list=[
                     LinearBostonHousePricesConfig(),
                     LinearGeneratorConfig(),
                     ],
                 ):
       self.linear_generator_config_list = linear_generator_config_list

    def config(self, cmd, args):
        for config in self.linear_generator_config_list:
            if config.NAME == cmd:
                config.update(args)
                return config

    def _sub_parser(self, subs=GooeyParser().add_subparsers()):
        for config in self.linear_generator_config_list:
            parser = subs.add_parser(config.NAME)
            config._parser(parser)

    def _parser(self, parser=GooeyParser()):
        subs = parser.add_subparsers()
        self._sub_parser(subs)
        return parser


generator_parser = LinearGeneratorConfigList()._parser


class LinearGenerator():
    def __init__(self, linear_generator_config=LinearGeneratorConfig()):
        self.config = linear_generator_config

    def train_valid_generator(self):
        try:
            dataframe = pd.read_csv(self.config.DATAFRAME_PATH)
        except Exception as e:
            print(e)
            assert False, "wrong dataframe path {}.".format(
                    self.config.DATAFRAME_PATH)
        train_generator = DataFrameIterator(
                dataframe=dataframe,
                x_col=self.config.X_COL,
                y_col=self.config.Y_COL,
                class_mode=self.config.CLASS_MODE,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                )
        if self.config.VALID_DATAFRAME_PATH:
            try:
                valid_dataframe = pd.read_csv(self.config.VALID_DATAFRAME_PATH)
            except Exception as e:
                print(e)
                assert False, "wrong dataframe path {}.".format(
                    self.config.VALID_DATAFRAME_PATH)
            valid_generator = DataFrameIterator(
                dataframe=valid_dataframe,
                x_col=self.config.X_COL,
                y_col=self.config.Y_COL,
                class_mode=self.config.CLASS_MODE,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                )
        else:
            valid_generator = None
        return (train_generator, valid_generator)
