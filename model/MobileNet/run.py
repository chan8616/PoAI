from collections import OrderedDict
from gooey import Gooey, GooeyParser

from .model import Model
from model.keras_applications import run as runlib
from .config_samples import (MobileNetTrainConfig,
                             MobileNetImagenetConfig,
                             MobileNetCIFAR10Config)


def run_parser(
        parser: GooeyParser = GooeyParser(),
        title="Train Setting",
        ) -> GooeyParser:

    return runlib.run_parser(parser,
                             title,
                             train_config=MobileNetTrainConfig(),
                             train_configs=OrderedDict([
                                 ('train_cifar10', MobileNetCIFAR10Config()),
                                 ('train_imagenet', MobileNetImagenetConfig()),
                             ]))


def run(config):
    return runlib.run(Model(), config)
