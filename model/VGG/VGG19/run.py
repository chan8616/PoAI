from collections import OrderedDict
from gooey import Gooey, GooeyParser

from .model import Model
from model.keras_applications import run as runlib
from .config_samples import (VGG19TrainConfig,
                             #  VGG19ImagenetConfig,
                             VGG19CIFAR10Config)


def run_parser(
        parser: GooeyParser = GooeyParser(),
        title="Train Setting",
        ) -> GooeyParser:

    return runlib.run_parser(parser,
                             title,
                             train_config=VGG19TrainConfig(),
                             train_configs=OrderedDict([
                                 ('train_cifar10', VGG19CIFAR10Config()),
                                 #  ('train_imagenet', VGG19ImagenetConfig()),
                             ]))


def run(config):
    return runlib.run(Model(), config)
