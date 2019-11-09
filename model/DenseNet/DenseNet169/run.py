from collections import OrderedDict
from gooey import Gooey, GooeyParser

from .model import Model
from model.keras_applications import run as runlib
from .config_samples import (DenseNet169TrainConfig,
                             DenseNet169ImagenetConfig,
                             DenseNet169CIFAR10Config)


def run_parser(
        parser: GooeyParser = GooeyParser(),
        title="Train Setting",
        ) -> GooeyParser:

    return runlib.run_parser(parser,
                             title,
                             train_config=DenseNet169TrainConfig(),
                             train_configs=OrderedDict([
                                 ('train_cifar10', DenseNet169CIFAR10Config()),
                                 ('train_imagenet', DenseNet169ImagenetConfig()),
                             ]))


def run(config):
    return runlib.run(Model(), config)
