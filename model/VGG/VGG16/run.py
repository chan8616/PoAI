from collections import OrderedDict
from gooey import Gooey, GooeyParser

from .model import Model
from model.keras_applications import run as runlib
from .config_samples import (VGG16TrainConfig,
                             #  VGG16ImagenetConfig,
                             VGG16CIFAR10Config)


def run_parser(
        parser: GooeyParser = GooeyParser(),
        title="Train Setting",
        ) -> GooeyParser:

    return runlib.run_parser(parser,
                             title,
                             train_config=VGG16TrainConfig(),
                             train_configs=OrderedDict([
                                 ('train_cifar10', VGG16CIFAR10Config()),
                                 #  ('train_imagenet', VGG16ImagenetConfig()),
                             ]))


def run(config):
    return runlib.run(Model(), config, VGG16TrainConfig())
