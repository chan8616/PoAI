import time
from argparse import Namespace
from pathlib import Path
from typing import Union
from gooey import Gooey, GooeyParser

import model as modellib
from sklearn.linear_model import Ridge
from model.Ridge.build_config import RidgeBuildConfig
from model.Ridge.config_samples import RidgeBostonHousePricesBuildConfig


class RidgeBuildConfigList():
    def __init__(self,
                 ridge_build_config_list=[
                     RidgeBostonHousePricesBuildConfig(),
                     RidgeBuildConfig(),
                     ],
                 ):
       self.ridge_build_config_list = ridge_build_config_list

    def config(self, cmd, args):
        #  print(self.ridge_build_config_list, cmd, args)
        for config in self.ridge_build_config_list:
            if config.BUILD_NAME == cmd:
                config.update(args)
                return config
        assert False, "cmd {} not found".format(cmd)

    def _sub_parser(self, subs=GooeyParser().add_subparsers()):
        for config in self.ridge_build_config_list:
            parser = subs.add_parser(config.BUILD_NAME)
            config._parser(parser)

    def _parser(self, parser=GooeyParser()):
        subs = parser.add_subparsers()
        self._sub_parser(subs)
        return parser


class RidgeBuild():
    def __init__(self, ridge_build_config=RidgeBuildConfig()):
        self.config = ridge_build_config

    def update(self, args):
        self.config(args)
        if args.print_model:
            model = self.build()
            print(model)

    def build(self):
        model = Ridge(self.config.ALPHA)
        return model

build_parser = RidgeBuildConfigList()._parser
