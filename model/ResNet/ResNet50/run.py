from collections import OrderedDict
from gooey import Gooey, GooeyParser

from .model import Model
from model.keras_applications import run as runlib
#  from ..keras_applications.run import run
from .config_samples import (TrainConfig,
                             ImagenetConfig,
                             CIFAR10Config)


def run_parser(
        parser: GooeyParser = GooeyParser(),
        title="Train Setting",
        ) -> GooeyParser:

    return runlib.run_parser(parser,
                             title,
                             train_config=TrainConfig(),
                             train_configs=OrderedDict([
                                 ('train_cifar10', CIFAR10Config()),
                                 ('train_imagenet', ImagenetConfig()),
                             ]))


def run(config):
    return runlib.run(Model(), config)
