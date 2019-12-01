from collections import OrderedDict
from gooey import Gooey, GooeyParser

from .model import Model
from model.keras_applications import run as runlib
from .config_samples import (XceptionTrainConfig,
                             XceptionImagenetConfig,
                             XceptionCIFAR10Config)


def run_parser(
        parser: GooeyParser = GooeyParser(),
        title="Train Setting",
        ) -> GooeyParser:

    return runlib.run_parser(parser,
                             title,
                             train_config=XceptionTrainConfig(),
                             train_configs=OrderedDict([
                                 ('train_cifar10', XceptionCIFAR10Config()),
                                 ('train_imagenet', XceptionImagenetConfig()),
                             ]))


def run(config):
    return runlib.run(Model(), config, XceptionTrainConfig())
