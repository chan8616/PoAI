import time
from argparse import Namespace
from pathlib import Path
from typing import Union
from gooey import Gooey, GooeyParser

import model as modellib
from sklearn.svm import SVC
from model.SVM.SVC.build_config import SVCBuildConfig
from model.SVM.SVC.config_samples import SVCIRISBuildConfig


class SVCBuildConfigList():
    def __init__(self,
                 svc_build_config_list=[
                     SVCIRISBuildConfig(),
                     SVCBuildConfig(),
                     ],
                 ):
       self.svc_build_config_list = svc_build_config_list

    def config(self, cmd, args):
        #  print(self.svc_build_config_list, cmd, args)
        for config in self.svc_build_config_list:
            if config.BUILD_NAME == cmd:
                config.update(args)
                return config
        assert False, "cmd {} not found".format(cmd)

    def _sub_parser(self, subs=GooeyParser().add_subparsers()):
        for config in self.svc_build_config_list:
            parser = subs.add_parser(config.BUILD_NAME)
            config._parser(parser)

    def _parser(self, parser=GooeyParser()):
        subs = parser.add_subparsers()
        self._sub_parser(subs)
        return parser


class SVCBuild():
    def __init__(self, svc_build_config=SVCBuildConfig()):
        self.config = svc_build_config

    def update(self, args):
        self.config(args)
        if args.print_model:
            model = self.build()
            print(model)

    def build(self):
        model = SVC(
                kernel=self.config.KERNEL,
                degree=self.config.DEGREE,
                gamma=self.config.GAMMA)
        return model

build_parser = SVCBuildConfigList()._parser
