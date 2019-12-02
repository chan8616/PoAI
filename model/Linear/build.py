from argparse import Namespace
from pathlib import Path
from typing import Union
from gooey import Gooey, GooeyParser

import model as modellib
from .build_config import LinearBuildConfig
from model.Linear.model import LinearModel
from model.Linear.config_samples import LinearSINEBuildConfig


class LinearBuildConfigList():
    def __init__(self,
                 linear_build_config_list=[
                     LinearSINEBuildConfig(),
                     LinearBuildConfig(),
                     ],
                 ):
       self.linear_build_config_list = linear_build_config_list

    def config(self, cmd, args):
        #  print(self.linear_build_config_list, cmd, args)
        for config in self.linear_build_config_list:
            if config.BUILD_NAME == cmd:
                config.update(args)
                return config
        assert False, "cmd {} not found".format(cmd)

    def _sub_parser(self, subs=GooeyParser().add_subparsers()):
        for config in self.linear_build_config_list:
            parser = subs.add_parser(config.BUILD_NAME)
            config._parser(parser)

    def _parser(self, parser=GooeyParser()):
        subs = parser.add_subparsers()
        self._sub_parser(subs)
        return parser


class LinearBuild():
    def __init__(self, linear_build_config=LinearBuildConfig()):
        self.config = linear_build_config

    def update(self, args):
        self.config(args)
        if args.print_model_summary:
            K.clear_session()
            model = self.build()
            print(model.summary())
            K.clear_session()

    def build(self):
        model = LinearModel()
        model.build(self.config)
        return model

build_parser = LinearBuildConfigList()._parser
